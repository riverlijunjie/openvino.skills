/**
 * VLSDPA Benchmark — variable-length SDPA via the GPU plugin's CM kernel.
 *
 * Usage: ./vlsdpa_bench <head_size> <num_heads> <cu_seqlens_csv>
 *                      [iters=200] [warmup=20] [num_bufs=4]
 *
 *   head_size       : per-head dimension (64, 72, 80, 128, ...). Must follow the
 *                     CM-VLSDPA kernel constraints (multiples of 8/16; see
 *                     src/plugins/intel_gpu/src/graph/impls/cm/vl_sdpa_opt.cpp).
 *   num_heads       : Q/K/V head count.
 *   cu_seqlens_csv  : cumulative sequence-length boundaries, e.g. "0,16" for one
 *                     16-token window or "0,1024,2048" for two 1024-token windows.
 *                     The first element MUST be 0.
 *
 * VLSDPA is the variable-length SDPA used in ViTs (Qwen2-VL/Qwen2.5-VL); the GPU
 * plugin currently has only a CM implementation (cm_sdpa_vlen.cm). This bench
 * builds the canonical [seq_len, num_heads, head_size] f16 layout that the CM
 * kernel expects (bfyx) and feeds cu_seqlens directly — matching the reference
 * test in src/plugins/intel_gpu/tests/unit/test_cases/vlsdpa_gpu_test.cpp.
 *
 * Requirements (mirrors VLSDPAOptImplementationManager::validate_impl):
 *   - GPU arch == Xe2 (BMG / PTL / LNL)
 *   - check_cm_jit_support() == true (CMC compiler available)
 *   - GPU_USE_CM enabled (default true)
 *
 * The bench prints median/min/avg per-call latency plus a compute-side roofline
 * (GFLOPS, GB/s). Use cliloader to harvest per-kernel timings of cm_sdpa_vlen.
 */

#include <openvino/openvino.hpp>
#include "ov_ops/vl_sdpa.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ov;

static std::vector<int32_t> parse_csv(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

static void fill_f16(Tensor& t, std::mt19937& rng) {
    auto* p = t.data<ov::float16>();
    for (size_t i = 0; i < t.get_size(); ++i)
        p[i] = ov::float16(float(int(rng() % 200) - 100) / 100.0f);
}

static std::shared_ptr<ov::Model> build_vlsdpa_model(int num_heads, int head_size) {
    // Q/K/V: [seq_len, num_heads, head_size], dynamic seq_len.
    auto q = std::make_shared<op::v0::Parameter>(element::f16,
                  PartialShape{Dimension::dynamic(), int64_t(num_heads), int64_t(head_size)});
    auto k = std::make_shared<op::v0::Parameter>(element::f16,
                  PartialShape{Dimension::dynamic(), int64_t(num_heads), int64_t(head_size)});
    auto v = std::make_shared<op::v0::Parameter>(element::f16,
                  PartialShape{Dimension::dynamic(), int64_t(num_heads), int64_t(head_size)});
    // cu_seqlens: [num_seqs + 1] i32.
    auto cu = std::make_shared<op::v0::Parameter>(element::i32,
                  PartialShape{Dimension::dynamic()});

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");
    cu->set_friendly_name("cu_seqlens");

    // Match vlsdpa_gpu_test create_topology(): order_q/k/v/out = {1, 0, 2}.
    // The transpose-fusion pass and CM kernel expect this triplet to be active
    // when Q/K/V are emitted in [seq_len, num_heads, head_size] layout.
    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_out = {1, 0, 2};

    auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(
        OutputVector{q, k, v, cu}, order_q, order_k, order_v, order_out);

    auto result = std::make_shared<op::v0::Result>(vlsdpa->output(0));
    return std::make_shared<Model>(OutputVector{result},
                                   ParameterVector{q, k, v, cu},
                                   "vlsdpa_bench");
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <head_size> <num_heads> <cu_seqlens_csv>"
                      << " [iters=200] [warmup=20] [num_bufs=4]"
                      << std::endl;
            std::cerr << "Examples:" << std::endl;
            std::cerr << "  " << argv[0] << " 80 16 0,1024              # ViT 1-image window 1024"
                      << std::endl;
            std::cerr << "  " << argv[0] << " 80 16 0,1024,2048         # 2 windows of 1024"
                      << std::endl;
            return 1;
        }

        const int head_size = std::atoi(argv[1]);
        const int num_heads = std::atoi(argv[2]);
        const std::vector<int32_t> cu_seqlens = parse_csv(argv[3]);
        const int iters    = argc > 4 ? std::atoi(argv[4]) : 200;
        const int warmup   = argc > 5 ? std::atoi(argv[5]) : 20;
        const int num_bufs = argc > 6 ? std::atoi(argv[6]) : 4;

        if (cu_seqlens.size() < 2 || cu_seqlens.front() != 0) {
            std::cerr << "[VLSDPA_BENCH ERROR] cu_seqlens_csv must start with 0 and have >=2 entries"
                      << std::endl;
            return 1;
        }
        const int total_tokens = cu_seqlens.back();

        std::cout << "=== VLSDPA Benchmark (CM kernel) ===" << std::endl;
        std::cout << "head_size=" << head_size << " num_heads=" << num_heads
                  << " total_tokens=" << total_tokens
                  << " num_seqs=" << (cu_seqlens.size() - 1)
                  << " iters=" << iters << " warmup=" << warmup
                  << " bufs=" << num_bufs << std::endl;

        auto model = build_vlsdpa_model(num_heads, head_size);

        Core core;
        // Note: VLSDPA only has a CM impl in the GPU plugin (cm_sdpa_vlen.cm).
        // GPU_USE_CM defaults to true and is internal-only; if the environment
        // exports OV_GPU_USE_CM=0 the primitive will have no implementation
        // manager and compile_model will throw \u2014 the message below makes that
        // failure mode self-evident.
        auto compiled = core.compile_model(model, "GPU");

        std::vector<InferRequest> reqs;
        std::mt19937 rng(123);

        for (int b = 0; b < num_bufs; ++b) {
            auto req = compiled.create_infer_request();
            // Q/K/V tensors of shape [total_tokens, num_heads, head_size]
            for (int idx : {0, 1, 2}) {
                Tensor t(element::f16,
                         Shape{(size_t)total_tokens, (size_t)num_heads, (size_t)head_size});
                fill_f16(t, rng);
                req.set_input_tensor(idx, t);
            }
            // cu_seqlens
            Tensor cu_t(element::i32, Shape{cu_seqlens.size()});
            std::copy(cu_seqlens.begin(), cu_seqlens.end(), cu_t.data<int32_t>());
            req.set_input_tensor(3, cu_t);
            reqs.push_back(std::move(req));
        }

        for (int i = 0; i < warmup; ++i)
            reqs[i % num_bufs].infer();

        std::vector<double> latencies;
        latencies.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            auto& req = reqs[i % num_bufs];
            auto t0 = std::chrono::high_resolution_clock::now();
            req.infer();
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::sort(latencies.begin(), latencies.end());
        const double median = latencies[latencies.size() / 2];
        const double min_lat = latencies.front();
        const double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

        // Roofline numbers \u2014 dense-equivalent FLOPs aggregated per cu_seqlens chunk.
        // Per chunk (with chunk length L): QK = 2 * num_heads * L*L*head_size ;
        //                                  AV = 2 * num_heads * L*L*head_size ;
        //                                  softmax \u2248 25 * num_heads * L * L
        // VLSDPA preserves block-diagonal structure so cross-chunk attention is skipped \u2014
        // the formulas match the kernel's actual work for each window.
        double flops = 0.0;
        double bytes_qkv = 0.0;
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            const double L = double(cu_seqlens[i] - cu_seqlens[i - 1]);
            flops += 2.0 * num_heads * L * L * head_size;  // QK
            flops += 2.0 * num_heads * L * L * head_size;  // AV
            flops += 25.0 * num_heads * L * L;             // softmax
            bytes_qkv += L * num_heads * head_size * 2.0 * 3.0;  // Q+K+V f16 reads (per chunk)
        }
        const double bytes_out = double(total_tokens) * num_heads * head_size * 2.0;
        const double total_bytes = bytes_qkv + bytes_out;

        const double gflops = flops / (median * 1e-3) / 1e9;
        const double bw = total_bytes / (median * 1e-3) / 1e9;

        std::cout << "Median_ms: " << median << std::endl;
        std::cout << "Min_ms: "    << min_lat << std::endl;
        std::cout << "Avg_ms: "    << avg << std::endl;
        std::cout << "GFLOPS: "    << gflops << std::endl;
        std::cout << "BW_GBs: "    << bw << std::endl;
        std::cout << "AI: "        << (flops / total_bytes) << std::endl;
        std::cout << "TotalFLOPs: " << flops << std::endl;
        std::cout << "TotalBytes: " << total_bytes << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[VLSDPA_BENCH ERROR] " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "[VLSDPA_BENCH ERROR] unknown exception" << std::endl;
        return 3;
    }
}
