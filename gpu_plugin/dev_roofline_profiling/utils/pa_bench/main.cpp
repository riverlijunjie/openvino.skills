/**
 * PA Benchmark — Qwen3-8B PagedAttention with FP16/INT8 KV cache
 *
 * Usage: ./pa_bench <mode> <S_q> <S_kv> [iters] [warmup] [num_bufs] [kv_type] [impl] [flush_mb]
 *   mode: "decode" (S_q=1) or "prefill" (S_q=S, S_kv=0)
 *   kv_type: "f16" (default) or "i8"
 *   impl: "ocl" (default) or "cm"
 *     - ocl: standard OCL/micro-kernel PA path (block_size=16)
 *     - cm:  XAttention CM kernel path (block_size=256). Requires Xe2/Xe3
 *            (BMG, PTL, LNL, etc.) with CM-JIT support. Enables the XAttention
 *            transformations pipeline by exposing xattention_threshold/
 *            xattention_block_size/xattention_stride as named dynamic
 *            Parameters (matching SDPAToPagedAttention's contract — see
 *            src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp
 *            use_xattention detection).
 *   flush_mb: size of the L2/L3 flush buffer in MiB (default 64). Pass 0 to disable.
 *             Env override: PA_BENCH_FLUSH_MB. Between every PA infer we run a
 *             Parameter→Relu→Result over a `flush_mb`-MB f16 USM_DEVICE tensor;
 *             reading+writing 2*flush_mb MiB of VRAM evicts any KV-cache lines
 *             that the previous PA infer left in the GPU's L2/L3, so each
 *             measured iteration actually streams KV from DRAM. Without this
 *             flush, rotating-buffer alone is not enough: when per-buf KV ≈ LLC
 *             size (e.g. 16 MiB at KV=4096 / NKV=8 / HD=128 / f16 on PTL 12Xe),
 *             intra-iteration kernel reuse + partial LLC residency inflates
 *             measured BW past the DRAM peak (reported eff% > 100%).
 *             The flush kernel compiles to an `activation_*` primitive whose
 *             name is excluded by parse_logs.py KERNEL_EXCLUDES, so it never
 *             contaminates the per-kernel PA breakdown.
 *   Run separately for each input size per SKILL.md requirement.
 *
 * Uses OpenVINO's built-in PagedAttentionExtension op directly (not decomposed
 * gemm+softmax+gemm), so the GPU plugin exercises its native PA kernels.
 *
 * The op requires exactly 28 inputs; unused optional inputs are 0-shaped constants.
 * KV cache parameters use rank-3 dynamic shapes with element::dynamic;
 * the ConvertPagedAttnInputs pass adjusts shapes/types at compile time.
 */
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include "openvino/op/paged_attention.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace ov;

// Default Qwen3-8B attention config; overridable via env (PA_NH/PA_NKV/PA_HD) or trailing CLI args.
static int NH  = 32;   // num_attention_heads
static int NKV = 8;    // num_kv_heads
static int HD  = 128;  // head_dim
static constexpr int BLOCK_SIZE = 16;  // KV cache block size

// Peaks printed are informational only; actual roofline analysis uses cliloader kernel data
// and generate_metrics.py with platform-specific hardware specs.

/**
 * Build a Model with PagedAttentionExtension.
 *
 * PA inputs (28 total):
 *  [0] query:  [batch_tokens, NH * HD],  f16
 *  [1] key:    [batch_tokens, NKV * HD], f16
 *  [2] value:  [batch_tokens, NKV * HD], f16
 *  [3] key_cache:   [num_blocks, NKV, HD*BLOCK_SIZE], dynamic type (rank-3)
 *  [4] value_cache: [num_blocks, NKV, BLOCK_SIZE*HD], dynamic type (rank-3)
 *  [5] past_lens:           [batch_seqs], i32
 *  [6] subsequence_begins:  [batch_seqs + 1], i32
 *  [7] block_indices:       [total_blocks], i32
 *  [8] block_indices_begins:[batch_seqs + 1], i32
 *  [9] scale:               scalar, f32
 *  [10] sliding_window:     scalar, i32
 *  [11] alibi_slopes:       [0], f32 (disabled)
 *  [12] max_context_len:    scalar, i32
 *  [13..27] optional:       0-shaped constants
 */
static std::shared_ptr<ov::Model> build_pa_model(int num_blocks, bool use_cm, int block_size_for_context) {
    // query dim[1] MUST be static (NH*HD) — GPU plugin reads it to compute heads_num
    auto query       = std::make_shared<op::v0::Parameter>(element::f16,
                           PartialShape{Dimension::dynamic(), int64_t(NH * HD)});
    auto key         = std::make_shared<op::v0::Parameter>(element::f16,
                           PartialShape{Dimension::dynamic(), int64_t(NKV * HD)});
    auto value       = std::make_shared<op::v0::Parameter>(element::f16,
                           PartialShape{Dimension::dynamic(), int64_t(NKV * HD)});

    // KV cache: rank-3, dynamic element type — ConvertPagedAttnInputs pass reshapes to rank-4
    auto key_cache   = std::make_shared<op::v0::Parameter>(element::dynamic,
                           PartialShape{Dimension::dynamic(), int64_t(NKV), Dimension::dynamic()});
    auto value_cache = std::make_shared<op::v0::Parameter>(element::dynamic,
                           PartialShape{Dimension::dynamic(), int64_t(NKV), Dimension::dynamic()});

    auto past_lens        = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto subseq_begins    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto block_indices    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto block_idx_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    query->set_friendly_name("query");
    key->set_friendly_name("key");
    value->set_friendly_name("value");
    key_cache->set_friendly_name("key_cache.0");
    value_cache->set_friendly_name("value_cache.0");
    past_lens->set_friendly_name("past_lens");
    subseq_begins->set_friendly_name("subsequence_begins");
    block_indices->set_friendly_name("block_indices");
    block_idx_begins->set_friendly_name("block_indices_begins");

    // Fixed constants for required inputs
    float scale_val = 1.0f / std::sqrt(float(HD));
    auto scale          = op::v0::Constant::create(element::f32, Shape{}, {scale_val});
    auto sliding_window = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto alibi_slopes   = op::v0::Constant::create(element::f32, Shape{0}, {});
    int total_context = num_blocks * block_size_for_context;
    auto max_context_len = op::v0::Constant::create(element::i32, Shape{}, {total_context});

    // Optional inputs (13..27) — constants to satisfy the 28-input requirement
    auto score_agg     = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto rot_blk_idx   = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rot_deltas    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rot_trig_lut  = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});

    // XAttention inputs: when use_cm=true, expose them as named dynamic Parameters so the
    // GPU plugin's transformations_pipeline detects "xattention_block_size" by friendly_name
    // and switches to XAttention path (block_size=256, CM-PA validate_impl returns true).
    // Otherwise emit zero-shaped constants (legacy OCL path).
    std::shared_ptr<ov::Node> xattn_thresh, xattn_blk_sz, xattn_stride;
    std::shared_ptr<op::v0::Parameter> xattn_thresh_param, xattn_blk_sz_param, xattn_stride_param;
    if (use_cm) {
        // PagedAttentionExtension validates these per-input ranks (see
        // src/core/src/op/paged_attention.cpp): xattention_threshold=rank 1,
        // xattention_block_size=rank 0, xattention_stride=rank 0. Use named
        // dynamic Parameters so the GPU plugin's transformations_pipeline
        // detects use_xattention=true via friendly_name lookup.
        xattn_thresh_param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
        xattn_blk_sz_param = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
        xattn_stride_param = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
        xattn_thresh_param->set_friendly_name("xattention_threshold");
        xattn_blk_sz_param->set_friendly_name("xattention_block_size");
        xattn_stride_param->set_friendly_name("xattention_stride");
        xattn_thresh = xattn_thresh_param;
        xattn_blk_sz = xattn_blk_sz_param;
        xattn_stride = xattn_stride_param;
    } else {
        xattn_thresh = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});
        xattn_blk_sz = op::v0::Constant::create(element::i32, Shape{}, {64});
        xattn_stride = op::v0::Constant::create(element::i32, Shape{}, {8});
    }
    auto sinks         = op::v0::Constant::create(element::f16, Shape{0}, std::vector<ov::float16>{});
    auto arkv_start    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto arkv_evict    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto arkv_div_idx  = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto arkv_div_beg  = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto token_type    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto qq_bias       = op::v0::Constant::create(element::u8, Shape{0}, std::vector<uint8_t>{});
    auto qq_bias_beg   = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});

    // Build the PagedAttentionExtension node — newer OV expects 28 inputs (adds qq_bias,
    // qq_bias_beg at the tail), older OV (pre-2026-03-25) expects 26. Try 28 first, fall back to 26.
    ov::OutputVector pa_inputs_28{
        query, key, value,
        key_cache, value_cache,
        past_lens, subseq_begins, block_indices, block_idx_begins,
        scale, sliding_window, alibi_slopes, max_context_len,
        score_agg,
        rot_blk_idx, rot_deltas, rot_trig_lut,
        xattn_thresh, xattn_blk_sz, xattn_stride,
        sinks,
        arkv_start, arkv_evict, arkv_div_idx, arkv_div_beg,
        token_type,
        qq_bias, qq_bias_beg
    };
    std::shared_ptr<ov::op::PagedAttentionExtension> pa;
    try {
        pa = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs_28);
    } catch (const std::exception& e) {
        // Older OV (<2026-03-25) used a 26-input PagedAttentionExtension. Fall back
        // only after surfacing the original failure so contract drift is visible.
        std::cerr << "[PA_BENCH WARN] 28-input PagedAttentionExtension failed: "
                  << e.what() << " \u2014 retrying with 26 inputs (legacy contract)" << std::endl;
        ov::OutputVector pa_inputs_26(pa_inputs_28.begin(), pa_inputs_28.begin() + 26);
        pa = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs_26);
    }

    // Set rt_info for ConvertPagedAttnInputs pass to properly adjust KV cache shapes
    pa->get_rt_info()["num_k_heads"] = Dimension::value_type(NKV);
    pa->get_rt_info()["k_head_size"] = Dimension::value_type(HD);
    pa->get_rt_info()["num_v_heads"] = Dimension::value_type(NKV);
    pa->get_rt_info()["v_head_size"] = Dimension::value_type(HD);

    auto result0 = std::make_shared<op::v0::Result>(pa->output(0));
    // Note: Only use output(0) (data). Including output(1) (scores) sets num_outputs>=2,
    // which makes has_scores_output() return true and:
    //   1) adds paged_attention_opt__scores_calculation kernel
    //   2) DISABLES sdpa_micro__prefill (via supports_micro_sdpa() check)
    ParameterVector params{query, key, value, key_cache, value_cache,
                           past_lens, subseq_begins, block_indices, block_idx_begins};
    if (use_cm) {
        params.push_back(xattn_thresh_param);
        params.push_back(xattn_blk_sz_param);
        params.push_back(xattn_stride_param);
    }
    return std::make_shared<Model>(OutputVector{result0}, params, "pa_bench");
}

// L2/L3 cache flush model: Parameter(f16,[N]) -> Relu -> Result.
// Running it streams 2*N*2 bytes of VRAM (read + write), evicting any KV-cache
// lines that the previous PA infer left in the GPU's on-die caches so the next
// PA iteration reads KV fresh from DRAM. Mirrors fc_bench's flush helper.
static std::shared_ptr<ov::Model> build_flush_model(size_t n_elems) {
    auto p = std::make_shared<op::v0::Parameter>(element::f16, Shape{n_elems});
    p->set_friendly_name("flush_input");
    auto r = std::make_shared<op::v0::Relu>(p);
    r->set_friendly_name("flush_relu");
    return std::make_shared<Model>(OutputVector{r}, ParameterVector{p}, "pa_l2_flush");
}

static void fill_f16(Tensor& t, std::mt19937& rng) {
    auto* p = t.data<ov::float16>();
    for (size_t i = 0; i < t.get_size(); i++)
        p[i] = ov::float16(float(rng() % 200 - 100) / 100.0f);
}

static void fill_i8(Tensor& t, std::mt19937& rng) {
    auto* p = t.data<int8_t>();
    for (size_t i = 0; i < t.get_size(); i++)
        p[i] = int8_t(rng() % 256 - 128);
}

int main(int argc, char* argv[]) {
    try {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mode:decode|prefill> <S_q> <S_kv>"
                  << " [iters=100] [warmup=10] [num_bufs=4] [kv_type=f16|i8] [impl=ocl|cm] [flush_mb=64]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int S_q  = std::atoi(argv[2]);
    int S_kv = std::atoi(argv[3]);
    int iters    = argc > 4 ? std::atoi(argv[4]) : 100;
    int warmup   = argc > 5 ? std::atoi(argv[5]) : 10;
    int num_bufs_cli = argc > 6 ? std::atoi(argv[6]) : 0;  // 0 = auto-size
    bool use_i8  = (argc > 7 && std::string(argv[7]) == "i8");
    bool use_cm  = (argc > 8 && std::string(argv[8]) == "cm");
    // L2/L3 flush size in MiB. CLI arg has priority; otherwise env var PA_BENCH_FLUSH_MB;
    // otherwise default 64 MiB (>=4x typical Intel iGPU LLC of 16 MiB on PTL 12Xe).
    int flush_mb = 64;
    if (argc > 9) {
        flush_mb = std::atoi(argv[9]);
    } else if (const char* e = std::getenv("PA_BENCH_FLUSH_MB")) {
        flush_mb = std::atoi(e);
    }

    // Auto-size num_bufs so the rotating-buffer set comfortably exceeds the GPU
    // last-level cache, defeating cross-iteration cache reuse that would
    // otherwise inflate achieved-BW figures past the DRAM peak.
    //
    // History: previous code assumed Xe3 L3=18 MB and capped num_bufs at 8. On
    // PTL 12Xe / Arc B390 the LLC reported by clinfo is 16 MiB, and per-buf KV
    // for KV=8192 INT8 is exactly 16 MiB — a stride-(num_bufs) access pattern
    // with bufs near L3-multiples produces partial cache hits that inflate
    // measured PA BW above DRAM peak (e.g. CM PA decode KV=8192 reported 187
    // GB/s on a 97 GB/s DRAM). We now target a much larger working set.
    //
    // Default target = 256 MiB (>= 8x any current Intel iGPU LLC). Override via
    // env PA_BENCH_TARGET_MB. The cap is 16 buffers to bound host RAM at very
    // long context. We size on logical KV (S_q + S_kv tokens), which matches
    // what the kernel actually streams from DRAM regardless of physical block
    // size (16 for OCL PA, 256 for CM XAttention).
    int num_bufs;
    if (num_bufs_cli > 0) {
        num_bufs = num_bufs_cli;  // explicit override
    } else {
        long long target_mb = 256;  // default: 256 MiB resident KV across all bufs
        if (const char* e = std::getenv("PA_BENCH_TARGET_MB")) {
            long long v = std::atoll(e);
            if (v > 0) target_mb = v;
        }
        const long long kv_bytes_per_buf =
            2LL * std::max(1, S_q + S_kv) * NKV * HD * (use_i8 ? 1 : 2);
        const long long target_resident = target_mb * 1024 * 1024;
        num_bufs = std::max(2, int((target_resident + kv_bytes_per_buf - 1) /
                                   std::max<long long>(1, kv_bytes_per_buf)));
        num_bufs = std::min(num_bufs, 16);
        std::cout << "[auto-bufs] kv_per_buf=" << (kv_bytes_per_buf / (1024*1024))
                  << " MiB, target_resident=" << target_mb
                  << " MiB -> num_bufs=" << num_bufs << std::endl;
    }

    // Optional model-shape overrides via env vars (kept simple to preserve back-compat with
    // existing scripts that pass positional args). Set PA_NH / PA_NKV / PA_HD to e.g. Qwen3.5-MoE
    // (16/2/256) or any other model.
    if (const char* e = std::getenv("PA_NH"))  NH  = std::atoi(e);
    if (const char* e = std::getenv("PA_NKV")) NKV = std::atoi(e);
    if (const char* e = std::getenv("PA_HD"))  HD  = std::atoi(e);

    // For decode: past_lens = S_kv, new tokens = S_q, total_context = S_kv + S_q
    // For prefill: S_q = S (input_len), S_kv = 0 (no past), total_context = S_q
    //   Force S_kv=0 for prefill so that:
    //   1. past_lens[0] = 0 → GPU plugin detects PREFILL stage (not MIXED)
    //   2. KV cache blocks match S_q (no oversized allocation)
    //   3. max_context_len = S_q (correct partitioning)
    if (mode == "prefill") {
        if (S_kv != 0) {
            std::cout << "[prefill] Overriding S_kv from " << S_kv << " to 0 (no past context for first-time prefill)" << std::endl;
            S_kv = 0;
        }
    }
    int total_context = S_kv + S_q;
    int num_blocks = (total_context + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "=== PA Benchmark ===" << std::endl;
    std::cout << "NH=" << NH << " NKV=" << NKV << " HD=" << HD << std::endl;
    std::cout << "Mode=" << mode << " S_q=" << S_q << " S_kv=" << S_kv
              << " blocks=" << num_blocks << " kv_type=" << (use_i8 ? "i8" : "f16")
              << " impl=" << (use_cm ? "cm" : "ocl")
              << " iters=" << iters << " warmup=" << warmup
              << " bufs=" << num_bufs
              << " flush_mb=" << flush_mb << std::endl;

    // CM XAttention uses a different KV-cache block size internally (block_size_xattn=256)
    // — see src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp.
    // Recompute block count so the cache covers `total_context` tokens.
    if (use_cm) {
        const int xattn_block = 256;
        num_blocks = (total_context + xattn_block - 1) / xattn_block;
        if (num_blocks < 1) num_blocks = 1;
    }

    int block_size_for_context = use_cm ? 256 : BLOCK_SIZE;
    auto model = build_pa_model(num_blocks, use_cm, block_size_for_context);

    Core core;
    ov::AnyMap props;
    // ALWAYS set kv_cache_precision explicitly. The GPU plugin defaults to i8 for
    // PagedAttention models (see src/plugins/intel_gpu/src/runtime/execution_config.cpp
    // ~line 293: `m_kv_cache_precision = ov::element::i8;` for is_paged_attention_model).
    // If we only set it for use_i8=true, then the f16 branch silently runs INT8 KV
    // and the bytes/BW figures become wrong (kernel streams ~half the assumed bytes,
    // pushing apparent eff% above 100%).
    if (use_i8) {
        props[ov::hint::kv_cache_precision.name()] = ov::element::i8;
    } else {
        props[ov::hint::kv_cache_precision.name()] = ov::element::f16;
    }
    // Note: GPU_USE_CM is internal-only (cannot be set via string AnyMap from public
    // API). Its default is true; if your environment exports OV_GPU_USE_CM=0 the
    // CM PA path will be unreachable and the GPU plugin will throw at compile time.
    // OCL PA validate_impl() rejects has_xattention=true (see
    // src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.hpp),
    // so once XAttention is enabled here CM PA is the only valid candidate.
    auto compiled = core.compile_model(model, "GPU", props);

    // Query compiled model for actual KV cache shapes and types
    // (ConvertPagedAttnInputs pass adjusts rank-3 dynamic → rank-4 with correct dims)
    auto kc_type = compiled.input(3).get_element_type();
    auto vc_type = compiled.input(4).get_element_type();
    auto kc_ps = compiled.input(3).get_partial_shape();
    auto vc_ps = compiled.input(4).get_partial_shape();

    std::cout << "KV cache key type=" << kc_type << " shape=" << kc_ps << std::endl;
    std::cout << "KV cache val type=" << vc_type << " shape=" << vc_ps << std::endl;

    // Build actual KV cache shapes: replace dynamic dims with num_blocks
    auto resolve_shape = [&](const PartialShape& ps) -> Shape {
        Shape s(ps.size());
        for (size_t i = 0; i < ps.size(); i++) {
            if (ps[i].is_static()) {
                s[i] = ps[i].get_length();
            } else {
                s[i] = (size_t)num_blocks;  // first dim is num_blocks
            }
        }
        return s;
    };

    Shape kc_shape = resolve_shape(kc_ps);
    Shape vc_shape = resolve_shape(vc_ps);
    std::cout << "Resolved KC shape: [";
    for (size_t i = 0; i < kc_shape.size(); i++) std::cout << (i?",":"") << kc_shape[i];
    std::cout << "]" << std::endl;

    // Create multiple infer requests with different buffers to avoid L3 cache reuse
    std::vector<InferRequest> reqs;
    std::mt19937 rng(42);

    for (int b = 0; b < num_bufs; b++) {
        auto req = compiled.create_infer_request();

        // Fill query [batch_tokens, NH*HD], key [batch_tokens, NKV*HD], value [batch_tokens, NKV*HD]
        {
            auto t = Tensor(element::f16, Shape{(size_t)S_q, (size_t)(NH * HD)});
            fill_f16(t, rng);
            req.set_input_tensor(0, t);
        }
        {
            auto t = Tensor(element::f16, Shape{(size_t)S_q, (size_t)(NKV * HD)});
            fill_f16(t, rng);
            req.set_input_tensor(1, t);
        }
        {
            auto t = Tensor(element::f16, Shape{(size_t)S_q, (size_t)(NKV * HD)});
            fill_f16(t, rng);
            req.set_input_tensor(2, t);
        }

        // key_cache and value_cache with resolved shapes
        {
            auto t = Tensor(kc_type, kc_shape);
            if (kc_type == element::i8) fill_i8(t, rng); else fill_f16(t, rng);
            req.set_input_tensor(3, t);
        }
        {
            auto t = Tensor(vc_type, vc_shape);
            if (vc_type == element::i8) fill_i8(t, rng); else fill_f16(t, rng);
            req.set_input_tensor(4, t);
        }

        // past_lens = [past_context_length]:
        //   decode:  past_lens[0] = S_kv (existing KV tokens before this decode step)
        //   prefill: past_lens[0] = 0    (first-time prefill: no past context)
        // When past_lens[0] != 0 AND query_tokens > batch_seqs, the GPU plugin
        // detects MIXED stage (not PREFILL), causing it to dispatch
        // sdpa_micro__generate instead of sdpa_micro__prefill.
        {
            auto t = Tensor(element::i32, Shape{1});
            t.data<int32_t>()[0] = (mode == "prefill") ? 0 : S_kv;
            req.set_input_tensor(5, t);
        }
        // subsequence_begins = [0, S_q]
        {
            auto t = Tensor(element::i32, Shape{2});
            t.data<int32_t>()[0] = 0;
            t.data<int32_t>()[1] = S_q;
            req.set_input_tensor(6, t);
        }
        // block_indices = [0, 1, ..., num_blocks-1]
        {
            auto t = Tensor(element::i32, Shape{(size_t)num_blocks});
            auto* p = t.data<int32_t>();
            for (int i = 0; i < num_blocks; i++) p[i] = i;
            req.set_input_tensor(7, t);
        }
        // block_indices_begins = [0, num_blocks]
        {
            auto t = Tensor(element::i32, Shape{2});
            t.data<int32_t>()[0] = 0;
            t.data<int32_t>()[1] = num_blocks;
            req.set_input_tensor(8, t);
        }

        // Optional XAttention inputs (only present when use_cm=true). Indexes 9..11 in
        // the order they were appended to ParameterVector in build_pa_model. Each is
        // a scalar (rank-0) tensor.
        if (use_cm) {
            // xattention_threshold (rank 1, f32): set to 1.0 so bypass_xattn() returns
            // true (see src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.cpp
            // bypass_xattn() — bypass = xattn_thresh >= 1.0). This skips the XAttention
            // block-selection pipeline (xattn_estimate_gemmqk / find_block / post_proc)
            // and runs the plain CM pa_multi_token_1 kernel during prefill. Without this
            // bypass, the CM XAttention GEMMQK kernel asserts N >= M for cold prefill
            // (paged_attention_gen.cpp:673) and prefill cannot complete on CM. This
            // matches the user-facing CM-PA contract: threshold=1.0 -> dense CM PA.
            {
                auto t = Tensor(element::f32, Shape{1});
                t.data<float>()[0] = 1.0f;
                req.set_input_tensor(9, t);
            }
            // xattention_block_size (rank 0, i32): selection block size (matches test 128)
            {
                auto t = Tensor(element::i32, Shape{});
                t.data<int32_t>()[0] = 128;
                req.set_input_tensor(10, t);
            }
            // xattention_stride (rank 0, i32): stride (matches test 16)
            {
                auto t = Tensor(element::i32, Shape{});
                t.data<int32_t>()[0] = 16;
                req.set_input_tensor(11, t);
            }
        }

        reqs.push_back(std::move(req));
    }

    // Build + compile the L2/L3 cache flush helper. Single InferRequest streaming
    // a `flush_mb`-MB f16 buffer through Relu evicts cached KV from on-die caches
    // between iterations so each measured PA infer reads KV from DRAM. The flush
    // kernel name (`activation_*`) is on parse_logs.py's KERNEL_EXCLUDES list and
    // does not contaminate the PA per-kernel breakdown.
    InferRequest flush_req;
    bool flush_enabled = flush_mb > 0;
    if (flush_enabled) {
        size_t flush_elems = static_cast<size_t>(flush_mb) * 1024ull * 1024ull / 2ull; // f16 = 2B
        auto fmodel = build_flush_model(flush_elems);
        auto fcompiled = core.compile_model(fmodel, "GPU");
        flush_req = fcompiled.create_infer_request();
        // Use plain host-side Tensor (no RemoteContext) to avoid coupling pa_bench
        // to USM_DEVICE / RemoteContext path; the GPU plugin still places the
        // Relu input in VRAM and the kernel reads/writes 2*flush_mb MiB of DRAM.
        auto fin_port = fcompiled.input();
        ov::Tensor flush_in(fin_port.get_element_type(), fin_port.get_shape());
        flush_req.set_input_tensor(flush_in);
        flush_req.infer();  // warm up the flush primitive
        std::cout << "L2/L3 flush kernel: " << flush_mb
                  << " MB f16 Relu between every PA infer." << std::endl;
    } else {
        std::cout << "L2/L3 flush kernel DISABLED (flush_mb=0)." << std::endl;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (flush_enabled) flush_req.infer();
        reqs[i % num_bufs].infer();
    }

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(iters);
    for (int i = 0; i < iters; i++) {
        if (flush_enabled) flush_req.infer();  // evict KV from L2/L3 before measurement
        auto& req = reqs[i % num_bufs];
        auto t0 = std::chrono::high_resolution_clock::now();
        req.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double min_lat = latencies.front();
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    // FLOPs: QK(2*NH*attn_pairs*HD) + AV(2*NH*attn_pairs*HD) + softmax(~25*NH*attn_pairs)
    // For prefill with causal mask, each query i attends to tokens 0..i, so
    // effective attention pairs = S*(S+1)/2 (triangular). For decode (S_q=1),
    // the single query attends to all S_kv_total tokens (no masking effect).
    double S_kv_total = double(total_context);
    double attn_pairs;  // effective (query, key) pairs computed
    if (mode == "prefill") {
        // Causal: sum_{i=0}^{Sq-1} (i+1) = Sq*(Sq+1)/2
        attn_pairs = double(S_q) * (double(S_q) + 1.0) / 2.0;
    } else {
        // Decode: single query attends to all past + current tokens
        attn_pairs = double(S_q) * S_kv_total;
    }
    double qk_flops = 2.0 * NH * attn_pairs * HD;
    double av_flops = 2.0 * NH * attn_pairs * HD;
    double sm_flops = 25.0 * NH * attn_pairs;
    double flops = qk_flops + av_flops + sm_flops;

    // Bytes: Q(f16) + K_new(f16) + V_new(f16) + K_cache + V_cache + output(f16).
    // Use *logical* past_lens KV bytes (S_kv tokens for decode, S_q for prefill);
    // this matches what the kernel actually streams from DRAM for the attention
    // compute, regardless of the underlying physical block_size (16 for OCL,
    // 256 for the CM XAttention path).
    //
    // CRITICAL: derive the KV element size from the *compiled* model type
    // (`kc_type` queried earlier), not from the user's CLI flag. The GPU plugin
    // can silently override the requested precision (e.g. defaults to i8 for PA
    // models unless `ov::hint::kv_cache_precision` is set explicitly). Using
    // `use_i8` here when the compiled cache is actually i8 would double the
    // assumed bytes and inflate measured BW past DRAM peak.
    const bool compiled_kv_is_int8 = (kc_type == element::i8 || kc_type == element::u8);
    int kv_elem_size = kc_type.bitwidth() / 8;
    if (kv_elem_size < 1) kv_elem_size = 1;
    double q_bytes      = double(S_q) * NH * HD * 2;
    double kv_new_bytes = double(S_q) * NKV * HD * 2 * 2;  // K+V new tokens (always f16 input)
    double kv_logical_tokens = double(total_context);
    double kv_cache_bytes = 2.0 * kv_logical_tokens * NKV * HD * kv_elem_size;  // K+V logical data
    if (compiled_kv_is_int8) {
        // INT8 KV cache carries per-(head,HD,block) and per-(head,token) scale+zero_point
        // pairs (2 × fp16 = 4 bytes). The actual GPU layout reports rank-4 shapes
        // [num_blocks, NKV, HD, BLOCK+4] for K (BY_CHANNEL) and
        // [num_blocks, NKV, BLOCK, HD+4] for V (BY_TOKEN). The +4 padding adds
        // 4/BLOCK bytes per K element-row and 4/HD bytes per V element-row of
        // overhead; total overhead = kv_logical_tokens * NKV * (HD * 4/BLOCK + 4)
        // for K (BY_CHANNEL) + kv_logical_tokens * NKV * 4 for V (BY_TOKEN).
        // Use BLOCK_SIZE here (OCL PA path); for CM XAttention the block size is
        // 256 and the per-token overhead becomes negligible.
        const double k_overhead = kv_logical_tokens * NKV * (double(HD) * 4.0 / BLOCK_SIZE);
        const double v_overhead = kv_logical_tokens * NKV * 4.0;
        kv_cache_bytes += k_overhead + v_overhead;
    }
    double out_bytes    = double(S_q) * NH * HD * 2;
    double total_bytes  = q_bytes + kv_new_bytes + kv_cache_bytes + out_bytes;

    double gflops = flops / (median * 1e-3) / 1e9;
    double bw = total_bytes / (median * 1e-3) / 1e9;

    std::cout << "Median_ms: " << median << std::endl;
    std::cout << "Min_ms: " << min_lat << std::endl;
    std::cout << "Avg_ms: " << avg << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    std::cout << "BW_GBs: " << bw << std::endl;
    std::cout << "AI: " << (flops / total_bytes) << std::endl;
    std::cout << "TotalFLOPs: " << flops << std::endl;
    std::cout << "TotalBytes: " << total_bytes << std::endl;

    return 0;
    } catch (const std::exception& e) {
        std::cerr << "[PA_BENCH ERROR] " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "[PA_BENCH ERROR] unknown exception" << std::endl;
        return 3;
    }
}
