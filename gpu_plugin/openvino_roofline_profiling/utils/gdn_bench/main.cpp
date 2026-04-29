/**
 * GatedDeltaNet (linear-attention) Benchmark
 *
 * Exercises ov::op::internal::GatedDeltaNet directly so the GPU plugin runs
 * its native gated_delta_net_ref OCL kernel
 * (src/plugins/intel_gpu/src/graph/impls/ocl_v2/gated_delta_net_ref.cl).
 *
 * Inputs (matches gated_delta_net.cpp unit test layout):
 *   q     [B, T, HK, K]
 *   k     [B, T, HK, K]
 *   v     [B, T, H,  V]   (V == K)
 *   state [B, H, K, V]
 *   g     [B, T, H]
 *   beta  [B, T, H]
 *
 * For Qwen3.5-MoE linear-attn layers: HK=16, H=32, K=V=128, conv1d k=4 (NOT modeled here —
 * conv1d is a tiny pre-step and the SSM core is the dominant cost).
 *
 * CLI: ./gdn_bench <B> <T> <HK> <H> <K> [iters=200] [warmup=10] [num_bufs=4]
 */
#include <openvino/openvino.hpp>
#include <openvino/op/gated_delta_net.hpp>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ov;

static std::shared_ptr<ov::Model> build_gdn_model(int HK, int H, int K) {
    auto q     = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), Dimension::dynamic(), int64_t(HK), int64_t(K)});
    auto k     = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), Dimension::dynamic(), int64_t(HK), int64_t(K)});
    auto v     = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), Dimension::dynamic(), int64_t(H),  int64_t(K)});
    auto state = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), int64_t(H), int64_t(K), int64_t(K)});
    auto g     = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), Dimension::dynamic(), int64_t(H)});
    auto beta  = std::make_shared<op::v0::Parameter>(element::f16,
                     PartialShape{Dimension::dynamic(), Dimension::dynamic(), int64_t(H)});

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");
    state->set_friendly_name("state");
    g->set_friendly_name("g");
    beta->set_friendly_name("beta");

    auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(
        OutputVector{q, k, v, state, g, beta},
        /*fuse_qk_l2norm=*/true,
        /*q_l2_norm_eps=*/1e-6f,
        /*k_l2_norm_eps=*/1e-6f);

    auto out0 = std::make_shared<op::v0::Result>(gdn->output(0));
    // Newer GDN op (post-GQA support, PR #35472) always produces 2 outputs:
    // (0) attention output, (1) updated recurrent state. We must materialize both,
    // otherwise the GPU runtime sees more output kernel-args than Result tensors
    // and fails with "allocated output memory is necessary to set kernel arguments".
    ov::ResultVector results{out0};
    if (gdn->get_output_size() > 1) {
        results.push_back(std::make_shared<op::v0::Result>(gdn->output(1)));
    }
    return std::make_shared<Model>(results,
                                    ParameterVector{q, k, v, state, g, beta},
                                    "gdn_bench");
}

static void fill_f16(Tensor& t, std::mt19937& rng, float lo, float hi) {
    auto* p = t.data<ov::float16>();
    float r = hi - lo;
    for (size_t i = 0; i < t.get_size(); i++) {
        float u = float(rng() % 1000) / 1000.0f;
        p[i] = ov::float16(lo + u * r);
    }
}

int main(int argc, char* argv[]) {
    try {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <B> <T> <HK> <H> <K> [iters=200] [warmup=10] [num_bufs=4]" << std::endl;
        return 1;
    }
    int B = std::atoi(argv[1]);
    int T = std::atoi(argv[2]);
    int HK = std::atoi(argv[3]);
    int H  = std::atoi(argv[4]);
    int K  = std::atoi(argv[5]);
    int iters    = argc > 6 ? std::atoi(argv[6]) : 200;
    int warmup   = argc > 7 ? std::atoi(argv[7]) : 10;
    int num_bufs = argc > 8 ? std::atoi(argv[8]) : 4;

    std::cout << "=== GDN Benchmark ===" << std::endl;
    std::cout << "B=" << B << " T=" << T << " HK=" << HK << " H=" << H << " K=" << K
              << " iters=" << iters << " warmup=" << warmup << " bufs=" << num_bufs << std::endl;

    auto model = build_gdn_model(HK, H, K);
    Core core;
    auto compiled = core.compile_model(model, "GPU");

    std::vector<InferRequest> reqs;
    std::mt19937 rng(42);
    for (int b = 0; b < num_bufs; b++) {
        auto req = compiled.create_infer_request();
        Shape qkS{(size_t)B, (size_t)T, (size_t)HK, (size_t)K};
        Shape vS {(size_t)B, (size_t)T, (size_t)H,  (size_t)K};
        Shape stS{(size_t)B, (size_t)H, (size_t)K,  (size_t)K};
        Shape gS {(size_t)B, (size_t)T, (size_t)H};

        Tensor tq(element::f16, qkS); fill_f16(tq, rng, -1, 1); req.set_input_tensor(0, tq);
        Tensor tk(element::f16, qkS); fill_f16(tk, rng, -1, 1); req.set_input_tensor(1, tk);
        Tensor tv(element::f16, vS);  fill_f16(tv, rng, -1, 1); req.set_input_tensor(2, tv);
        Tensor tst(element::f16, stS); fill_f16(tst, rng, -1, 1); req.set_input_tensor(3, tst);
        Tensor tg(element::f16, gS);  fill_f16(tg, rng, -2, 0);  req.set_input_tensor(4, tg);  // exp(g) → decay
        Tensor tb(element::f16, gS);  fill_f16(tb, rng, 0, 1);   req.set_input_tensor(5, tb);
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

    // FLOPs estimate per timestep per (b,h,v_idx):
    //   K reads + dot(state·k) [2K] + decay [K] + update [3K] + dot(state·q) [2K]
    //   ≈ 8*K per (B,T,H,V) cell  → total ≈ 8 * B * T * H * V * K
    //   (V==K)
    double flops = 8.0 * B * T * H * K * K;

    // Bytes per call: state load+store (2 * B*H*K*K*2) + q,k,v,g,beta + output
    double state_bytes = 2.0 * B * H * K * K * 2;          // f16 read + write
    double q_bytes  = double(B) * T * HK * K * 2;
    double k_bytes  = double(B) * T * HK * K * 2;
    double v_bytes  = double(B) * T * H  * K * 2;
    double g_bytes  = double(B) * T * H  * 2;
    double b_bytes  = double(B) * T * H  * 2;
    double out_bytes= double(B) * T * H  * K * 2;
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
