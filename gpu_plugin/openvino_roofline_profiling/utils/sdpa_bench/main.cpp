/**
 * SDPA Benchmark — OpenVINO ScaledDotProductAttention (opset13)
 *
 * Usage: ./sdpa_bench <mode> <S_q> <S_kv> [iters] [warmup] [num_bufs] [causal]
 *   mode:    "decode" (S_q=1) or "prefill" (S_q=S, S_kv=S typical)
 *   causal:  "0" (default for decode) or "1" (default for prefill)
 *
 * Layout is the canonical PyTorch SDPA layout:
 *   Q : [B, NH,  S_q,  HD]  f16
 *   K : [B, NKV, S_kv, HD]  f16
 *   V : [B, NKV, S_kv, HD]  f16
 *   attn_mask: [S_q, S_kv]  f16 (optional, all-zero for non-causal)
 *
 * Notes
 * -----
 * - SDPA here is the *uncompressed* attention path. KV-cache compression
 *   (e.g. INT8) and paging belong to PagedAttention; use pa_bench for those.
 * - For GQA (NH > NKV), the GPU plugin broadcasts K/V along the head dim;
 *   we just need to feed [B, NKV, S_kv, HD].
 * - We rotate <num_bufs> independent input sets per iteration to avoid the
 *   GPU plugin re-using the previous Q/K/V from L3, in line with the SKILL
 *   guidance for performance-metrics collection.
 *
 * Default model shape is Qwen3-8B-style attention (NH=32, NKV=8, HD=128) and
 * is overridable via env vars: SDPA_NH, SDPA_NKV, SDPA_HD.
 */
#include <openvino/openvino.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/result.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

using namespace ov;

static int NH  = 32;   // num attention heads
static int NKV = 8;    // num kv heads (GQA)
static int HD  = 128;  // head dim
static constexpr int B = 1;

static std::shared_ptr<ov::Model> build_sdpa_model(bool causal, bool with_mask) {
    auto q = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{B, NH, Dimension::dynamic(), HD});
    auto k = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{B, NKV, Dimension::dynamic(), HD});
    auto v = std::make_shared<op::v0::Parameter>(element::f16,
                 PartialShape{B, NKV, Dimension::dynamic(), HD});
    q->set_friendly_name("query");
    k->set_friendly_name("key");
    v->set_friendly_name("value");

    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    ParameterVector params{q, k, v};
    if (with_mask) {
        auto mask = std::make_shared<op::v0::Parameter>(element::f16,
                        PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        mask->set_friendly_name("attn_mask");
        params.push_back(mask);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask, causal);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);
    }

    auto result = std::make_shared<op::v0::Result>(sdpa->output(0));
    return std::make_shared<Model>(OutputVector{result}, params, "sdpa_bench");
}

static void fill_f16(Tensor& t, std::mt19937& rng) {
    auto* p = t.data<ov::float16>();
    for (size_t i = 0; i < t.get_size(); i++)
        p[i] = ov::float16(float(rng() % 200 - 100) / 100.0f);
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <mode:decode|prefill> <S_q> <S_kv>"
                      << " [iters=200] [warmup=20] [num_bufs=4] [causal=auto]"
                      << std::endl;
            return 1;
        }

        std::string mode = argv[1];
        int S_q  = std::atoi(argv[2]);
        int S_kv = std::atoi(argv[3]);
        int iters    = argc > 4 ? std::atoi(argv[4]) : 200;
        int warmup   = argc > 5 ? std::atoi(argv[5]) : 20;
        int num_bufs = argc > 6 ? std::atoi(argv[6]) : 4;

        // causal: default = (mode == "prefill"); explicit "0"/"1" overrides
        bool causal = (mode == "prefill");
        if (argc > 7) causal = std::string(argv[7]) == "1";

        // Optional model-shape overrides
        if (const char* e = std::getenv("SDPA_NH"))  NH  = std::atoi(e);
        if (const char* e = std::getenv("SDPA_NKV")) NKV = std::atoi(e);
        if (const char* e = std::getenv("SDPA_HD"))  HD  = std::atoi(e);

        // For decode: S_q==1, S_kv==context_len (past tokens).
        // For prefill: typical S_q==S_kv==S.
        if (S_q <= 0 || S_kv <= 0) {
            std::cerr << "S_q and S_kv must be positive" << std::endl;
            return 1;
        }

        std::cout << "=== SDPA Benchmark ===" << std::endl;
        std::cout << "NH=" << NH << " NKV=" << NKV << " HD=" << HD << std::endl;
        std::cout << "Mode=" << mode << " S_q=" << S_q << " S_kv=" << S_kv
                  << " causal=" << causal << " iters=" << iters
                  << " warmup=" << warmup << " bufs=" << num_bufs << std::endl;

        auto model = build_sdpa_model(causal, /*with_mask=*/false);

        Core core;
        auto compiled = core.compile_model(model, "GPU");

        std::vector<InferRequest> reqs;
        std::mt19937 rng(42);
        for (int b = 0; b < num_bufs; b++) {
            auto req = compiled.create_infer_request();
            // Q : [B, NH,  S_q,  HD]
            {
                auto t = Tensor(element::f16, Shape{(size_t)B, (size_t)NH,  (size_t)S_q,  (size_t)HD});
                fill_f16(t, rng);
                req.set_input_tensor(0, t);
            }
            // K : [B, NKV, S_kv, HD]
            {
                auto t = Tensor(element::f16, Shape{(size_t)B, (size_t)NKV, (size_t)S_kv, (size_t)HD});
                fill_f16(t, rng);
                req.set_input_tensor(1, t);
            }
            // V : [B, NKV, S_kv, HD]
            {
                auto t = Tensor(element::f16, Shape{(size_t)B, (size_t)NKV, (size_t)S_kv, (size_t)HD});
                fill_f16(t, rng);
                req.set_input_tensor(2, t);
            }
            reqs.push_back(req);
        }

        // Warmup
        for (int i = 0; i < warmup; i++) {
            reqs[i % num_bufs].infer();
        }

        // Timed loop. cliloader collects per-kernel ns; this wall-time is
        // informational only.
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            reqs[i % num_bufs].infer();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Total wall ms=" << ms << " per_iter_ms=" << (ms / iters) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
