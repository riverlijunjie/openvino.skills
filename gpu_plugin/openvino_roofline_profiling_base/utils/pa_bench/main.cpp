/**
 * PA Benchmark — Qwen3-8B PagedAttention with FP16/INT8 KV cache
 *
 * Usage: ./pa_bench <mode> <S_q> <S_kv> [iters] [warmup] [num_bufs] [kv_type]
 *   mode: "decode" (S_q=1) or "prefill" (S_q=S, S_kv=0)
 *   kv_type: "f16" (default) or "i8"
 *   Run separately for each input size per SKILL.md requirement
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

// Qwen3-8B attention config
static constexpr int NH  = 32;   // num_attention_heads
static constexpr int NKV = 8;    // num_kv_heads
static constexpr int HD  = 128;  // head_dim
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
static std::shared_ptr<ov::Model> build_pa_model(int num_blocks) {
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
    int total_context = num_blocks * BLOCK_SIZE;
    auto max_context_len = op::v0::Constant::create(element::i32, Shape{}, {total_context});

    // Optional inputs (13..27) — constants to satisfy the 28-input requirement
    auto score_agg     = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto rot_blk_idx   = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rot_deltas    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto rot_trig_lut  = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});
    auto xattn_thresh  = op::v0::Constant::create(element::f32, Shape{0}, std::vector<float>{});
    auto xattn_blk_sz  = op::v0::Constant::create(element::i32, Shape{}, {64});
    auto xattn_stride  = op::v0::Constant::create(element::i32, Shape{}, {8});
    auto sinks         = op::v0::Constant::create(element::f16, Shape{0}, std::vector<ov::float16>{});
    auto arkv_start    = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto arkv_evict    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto arkv_div_idx  = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto arkv_div_beg  = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto token_type    = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});
    auto qq_bias       = op::v0::Constant::create(element::u8, Shape{0}, std::vector<uint8_t>{});
    auto qq_bias_beg   = op::v0::Constant::create(element::i32, Shape{0}, std::vector<int32_t>{});

    // Build the PagedAttentionExtension node with all 28 inputs
    auto pa = std::make_shared<ov::op::PagedAttentionExtension>(ov::OutputVector{
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
    });

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
    return std::make_shared<Model>(
        OutputVector{result0},
        ParameterVector{query, key, value, key_cache, value_cache,
                        past_lens, subseq_begins, block_indices, block_idx_begins},
        "pa_bench");
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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mode:decode|prefill> <S_q> <S_kv>"
                  << " [iters=100] [warmup=10] [num_bufs=4] [kv_type=f16|i8]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int S_q  = std::atoi(argv[2]);
    int S_kv = std::atoi(argv[3]);
    int iters    = argc > 4 ? std::atoi(argv[4]) : 100;
    int warmup   = argc > 5 ? std::atoi(argv[5]) : 10;
    int num_bufs = argc > 6 ? std::atoi(argv[6]) : 4;
    bool use_i8  = (argc > 7 && std::string(argv[7]) == "i8");

    // For decode: past_lens = S_kv, new tokens = S_q, total_context = S_kv + S_q
    // For prefill: S_q = S (input_len), S_kv = 0 (no past), total_context = S_q
    int total_context = S_kv + S_q;
    int num_blocks = (total_context + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << "=== PA Benchmark (Qwen3-8B) ===" << std::endl;
    std::cout << "Mode=" << mode << " S_q=" << S_q << " S_kv=" << S_kv
              << " blocks=" << num_blocks << " kv_type=" << (use_i8 ? "i8" : "f16")
              << " iters=" << iters << " warmup=" << warmup
              << " bufs=" << num_bufs << std::endl;

    auto model = build_pa_model(num_blocks);

    Core core;
    ov::AnyMap props;
    if (use_i8) {
        props[ov::hint::kv_cache_precision.name()] = ov::element::i8;
    }
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

        // past_lens = [S_kv]
        {
            auto t = Tensor(element::i32, Shape{1});
            t.data<int32_t>()[0] = S_kv;
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

        reqs.push_back(std::move(req));
    }

    // Warmup
    for (int i = 0; i < warmup; i++)
        reqs[i % num_bufs].infer();

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(iters);
    for (int i = 0; i < iters; i++) {
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

    // FLOPs: QK(2*NH*Sq*Skv_total*HD) + AV(2*NH*Sq*Skv_total*HD) + softmax(~25*NH*Sq*Skv_total)
    double S_kv_total = double(total_context);
    double qk_flops = 2.0 * NH * S_q * S_kv_total * HD;
    double av_flops = 2.0 * NH * S_q * S_kv_total * HD;
    double sm_flops = 25.0 * NH * S_q * S_kv_total;
    double flops = qk_flops + av_flops + sm_flops;

    // Bytes: Q(f16) + K_new(f16) + V_new(f16) + K_cache + V_cache + output(f16)
    int kv_elem_size = use_i8 ? 1 : 2;
    double q_bytes      = double(S_q) * NH * HD * 2;
    double kv_new_bytes = double(S_q) * NKV * HD * 2 * 2;  // K+V new tokens
    double kv_cache_bytes = double(num_blocks) * BLOCK_SIZE * NKV * HD * kv_elem_size * 2;  // K+V cache
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
}
