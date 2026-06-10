/**
 * GatedDeltaNet (linear-attention) Benchmark — PAGED variant
 *
 * Exercises ov::op::internal::PagedGatedDeltaNet directly so the GPU plugin runs
 * its OPTIMIZED OCL kernel
 * (src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_gated_delta_net_opt.cl).
 *
 * WHY paged (not the standalone GatedDeltaNet op): real LLM inference lowers the
 * GatedDeltaNet subgraph into PagedGatedDeltaNet via the sdpa_to_paged_attention
 * pass (PagedGatedDeltaNetFusion). The non-paged ov::op::internal::GatedDeltaNet
 * has ONLY a reference GPU impl (gated_delta_net_ref), so a bench built on it can
 * never measure the optimized kernel. The paged op registers BOTH opt + ref with
 * opt first; opt is selected when k_head_dim%16==0 && v_head_dim%16==0 (128 -> ok).
 *
 * Inputs (11, match single_op/paged_gated_delta_net.cpp SetUp layout):
 *   query  [tokens, qk_heads, qk_head_dim]                  f16
 *   key    [tokens, qk_heads, qk_head_dim]                  f16
 *   value  [tokens, v_heads,  v_head_dim]                   f16
 *   state  [num_blocks, v_heads, v_head_dim, qk_head_dim]   f16  (recurrent state table)
 *   gate   [tokens, v_heads]                                f16
 *   beta   [tokens, v_heads]                                f16
 *   subsequence_begins   [num_seq+1]                        i32
 *   block_indices        [num_blocks]                       i32
 *   block_indices_begins [num_seq+1]                        i32
 *   past_lens            [num_seq]                          i32
 *   cache_interval       [num_seq]                          i32
 *
 * For Qwen3.5/3.6-MoE linear-attn layers: qk_heads=16, v_heads=32 (GQA group=2),
 * qk_head_dim=v_head_dim=128. conv1d k=4 pre-step is NOT modeled (tiny vs SSM core).
 *
 * CLI (back-compat positional): ./gdn_bench <B> <T> <HK> <H> <K> [iters=200] [warmup=10] [bufs=4] [cache_interval=256]
 *   B  : ignored (single sequence is used)
 *   T  : tokens   (decode -> 1, prefill -> S)
 *   HK : qk_heads (query/key heads, 16 for Qwen3.6)
 *   H  : v_heads  (value heads, 32 for Qwen3.6)
 *   K  : head dim (== qk_head_dim == v_head_dim, 128)
 *   cache_interval : recurrent-state snapshot stride (runtime la.cache_interval).
 *                    >=T => single final snapshot (matches non-paged ref state store).
 */
#include <openvino/openvino.hpp>
#include <openvino/op/paged_gated_delta_net.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ov;

// Recurrent-state cache stride (paged "block size" for linear attention).
// State is written to a new block every LA_BLOCK tokens; num_blocks scales with T.
static constexpr int LA_BLOCK = 256;

static std::shared_ptr<ov::Model> build_pgdn_model(int qk_heads, int v_heads, int head_dim) {
    auto q = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(qk_heads), int64_t(head_dim)});
    auto k = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(qk_heads), int64_t(head_dim)});
    auto v = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(v_heads), int64_t(head_dim)});
    auto state = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(v_heads), int64_t(head_dim), int64_t(head_dim)});
    auto g    = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(v_heads)});
    auto beta = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{Dimension::dynamic(), int64_t(v_heads)});
    auto subseq       = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto block_idx    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto block_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto past_lens    = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto cache_intv   = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    q->set_friendly_name("query");
    k->set_friendly_name("key");
    v->set_friendly_name("value");
    state->set_friendly_name("state");
    g->set_friendly_name("gate");
    beta->set_friendly_name("beta");

    auto pgdn = std::make_shared<ov::op::internal::PagedGatedDeltaNet>(
        q, k, v, state, g, beta,
        subseq, block_idx, block_begins, past_lens, cache_intv,
        /*use_qk_l2norm=*/true,
        /*q_l2_norm_eps=*/1e-6f,
        /*k_l2_norm_eps=*/1e-6f);

    ov::ResultVector results{std::make_shared<op::v0::Result>(pgdn->output(0))};
    return std::make_shared<Model>(results,
                                   ParameterVector{q, k, v, state, g, beta,
                                                   subseq, block_idx, block_begins, past_lens, cache_intv},
                                   "pgdn_bench");
}

static void fill_f16(Tensor& t, std::mt19937& rng, float lo, float hi) {
    auto* p = t.data<ov::float16>();
    float r = hi - lo;
    for (size_t i = 0; i < t.get_size(); i++) {
        float u = float(rng() % 1000) / 1000.0f;
        p[i] = ov::float16(lo + u * r);
    }
}

static Tensor make_i32(const std::vector<int32_t>& vals) {
    Tensor t(element::i32, Shape{vals.size()});
    std::copy(vals.begin(), vals.end(), t.data<int32_t>());
    return t;
}

int main(int argc, char* argv[]) {
    try {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <B> <T> <HK> <H> <K> [iters=200] [warmup=10] [num_bufs=4] [cache_interval=256]" << std::endl;
        return 1;
    }
    int T  = std::atoi(argv[2]);            // tokens
    int HK = std::atoi(argv[3]);            // qk_heads
    int H  = std::atoi(argv[4]);            // v_heads
    int K  = std::atoi(argv[5]);            // head dim (qk == v)
    int iters    = argc > 6 ? std::atoi(argv[6]) : 200;
    int warmup   = argc > 7 ? std::atoi(argv[7]) : 10;
    int num_bufs = argc > 8 ? std::atoi(argv[8]) : 4;
    // cache_interval: how often the recurrent state is snapshotted to a fresh paged
    // block. Runtime-set (la.cache_interval) in real inference; expose it here to
    // quantify the paging state-write overhead. interval>=T => single final snapshot
    // (matches the non-paged ref's one state store).
    int interval_arg = argc > 9 ? std::atoi(argv[9]) : LA_BLOCK;

    // Single sequence, past_len=0, state cached every `interval` tokens.
    const int interval     = interval_arg > 0 ? interval_arg : (T > 0 ? T : 1);
    const int write_blocks = std::max(1, (T + interval - 1) / interval);
    const int num_blocks   = 1 + write_blocks;   // 1 read block + write blocks

    std::cout << "=== Paged GDN Benchmark ===" << std::endl;
    std::cout << "T=" << T << " qk_heads=" << HK << " v_heads=" << H << " head_dim=" << K
              << " num_blocks=" << num_blocks << " interval=" << interval
              << " iters=" << iters << " warmup=" << warmup << " bufs=" << num_bufs << std::endl;

    auto model = build_pgdn_model(HK, H, K);
    Core core;
    auto compiled = core.compile_model(model, "GPU");

    // index inputs (i32) are identical across buffers
    std::vector<int32_t> block_indices(num_blocks);
    std::iota(block_indices.begin(), block_indices.end(), 0);

    std::vector<InferRequest> reqs;
    std::mt19937 rng(42);
    for (int b = 0; b < num_bufs; b++) {
        auto req = compiled.create_infer_request();
        Shape qkS{(size_t)T, (size_t)HK, (size_t)K};
        Shape vS {(size_t)T, (size_t)H,  (size_t)K};
        Shape stS{(size_t)num_blocks, (size_t)H, (size_t)K, (size_t)K};
        Shape gvS{(size_t)T, (size_t)H};

        Tensor tq(element::f16, qkS); fill_f16(tq, rng, -1, 1); req.set_input_tensor(0, tq);
        Tensor tk(element::f16, qkS); fill_f16(tk, rng, -1, 1); req.set_input_tensor(1, tk);
        Tensor tv(element::f16, vS);  fill_f16(tv, rng, -1, 1); req.set_input_tensor(2, tv);
        Tensor tst(element::f16, stS); fill_f16(tst, rng, -1, 1); req.set_input_tensor(3, tst);
        Tensor tg(element::f16, gvS); fill_f16(tg, rng, -2, 0);  req.set_input_tensor(4, tg);  // exp(g) -> decay
        Tensor tb(element::f16, gvS); fill_f16(tb, rng, 0, 1);   req.set_input_tensor(5, tb);

        req.set_input_tensor(6, make_i32({0, T}));               // subsequence_begins
        req.set_input_tensor(7, make_i32(block_indices));        // block_indices
        req.set_input_tensor(8, make_i32({0, num_blocks}));      // block_indices_begins
        req.set_input_tensor(9, make_i32({0}));                  // past_lens
        req.set_input_tensor(10, make_i32({interval}));          // cache_interval
        reqs.push_back(std::move(req));
    }

    for (int i = 0; i < warmup; i++) reqs[i % num_bufs].infer();

    std::vector<double> latencies; latencies.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto& req = reqs[i % num_bufs];
        auto t0 = std::chrono::high_resolution_clock::now();
        req.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double avg    = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    // FLOPs estimate per timestep per (h,v_idx):
    //   K reads + dot(state·k) [2K] + decay [K] + update [3K] + dot(state·q) [2K]
    //   ≈ 8*K per (T,H,V) cell  → total ≈ 8 * T * H * V * K  (V==K)
    double flops = 8.0 * T * H * K * K;

    // Bytes: state load+store (2 * num_blocks*H*K*K*2) + q,k,v,g,beta + output
    double state_bytes = 2.0 * num_blocks * H * K * K * 2;
    double q_bytes  = double(T) * HK * K * 2;
    double k_bytes  = double(T) * HK * K * 2;
    double v_bytes  = double(T) * H  * K * 2;
    double g_bytes  = double(T) * H  * 2;
    double b_bytes  = double(T) * H  * 2;
    double out_bytes= double(T) * H  * K * 2;
    double total_bytes = state_bytes + q_bytes + k_bytes + v_bytes + g_bytes + b_bytes + out_bytes;

    double gflops = flops / (median * 1e-3) / 1e9;
    double bw     = total_bytes / (median * 1e-3) / 1e9;

    std::cout << "Median_ms: " << median << std::endl;
    std::cout << "Avg_ms: "    << avg    << std::endl;
    std::cout << "GFLOPS: "    << gflops << std::endl;
    std::cout << "BW_GBs: "    << bw     << std::endl;
    std::cout << "AI: "        << (flops / total_bytes) << std::endl;
    std::cout << "TotalFLOPs: "<< flops  << std::endl;
    std::cout << "TotalBytes: "<< total_bytes << std::endl;
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "[GDN_BENCH ERROR] " << e.what() << std::endl;
        return 2;
    }
}
