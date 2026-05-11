/**
 * FC Benchmark — FullyConnected with compressed (INT4/INT8) or plain FP16 weights
 *
 * Usage: ./fc_bench <M> <K> <N> [group_size=128] [iters=100] [warmup=10] [num_bufs=8] [precision=u4|u8|f16] [flush_mb=64]
 *   - Run separately for each (M,K,N) per SKILL.md requirement
 *   - Multiple input tensors rotate to avoid L3 reuse of activations
 *   - Between every infer, a large Relu on a `flush_mb`-MB USM_DEVICE buffer is
 *     enqueued to evict the FC's weight constants from L2/L3. This is needed
 *     because a single body-FC weight footprint (8-25 MB on Qwen3-8B) otherwise
 *     fits inside BMG B580's ~18 MB L2 and silently measures cache bandwidth,
 *     not VRAM bandwidth. cliloader reports kernels by name; the flush kernel
 *     runs under a different CompiledModel (single `activation` primitive) so
 *     parsing by FC kernel name naturally excludes it from the FC timing avg.
 *   - flush_mb must exceed the largest on-die cache of the target. BMG L2=18 MB,
 *     PTL shares CPU LLC ~30 MB; 64 MB default covers both.
 *   - Pass flush_mb=0 to disable the flush (regression / ablation study).
 *
 * Uses the standard dequantization pattern with public OV ops:
 *   Weights(u4, {N, n_groups, group_size})
 *       → Convert(f16) → Subtract(zp) → Multiply(scale) → Reshape({N, K})
 *   Input(f16, {M, K}) → MatMul(input, decompressed_weights, false, true)
 *
 * The GPU plugin's KeepConstPrecision pass matches this dequantization pattern
 * and marks the u4 constant to prevent ConvertPrecision from converting u4→u8.
 * Then ConvertMatMulToFullyConnected and ConvertFCToCompressed transform
 * the MatMul into FullyConnectedCompressed with native u4 weights.
 *
 * Set dynamic_quantization_group_size for prefill (INT8 XMX path).
 */
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

using namespace ov;

static std::shared_ptr<ov::Model> build_fc_deq_model(int M, int K, int N,
                                                      int group_size = 128,
                                                      bool u8_weights = false) {
    size_t sM = static_cast<size_t>(M);
    size_t sK = static_cast<size_t>(K);
    size_t sN = static_cast<size_t>(N);
    size_t n_groups = static_cast<size_t>(K / group_size);
    size_t sG = static_cast<size_t>(group_size);

    auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{sM, sK});
    input->set_friendly_name("input");

    std::mt19937 rng(42);

    element::Type wtype = u8_weights ? element::u8 : element::u4;
    size_t w_elems = sN * n_groups * sG;
    size_t w_bytes = u8_weights ? w_elems : (w_elems + 1) / 2;
    std::vector<uint8_t> w_packed(w_bytes);
    for (auto& v : w_packed) v = uint8_t(rng() % 256);
    auto weights = op::v0::Constant::create(wtype,
                                             Shape{sN, n_groups, sG}, w_packed.data());
    weights->set_friendly_name("weights");

    auto w_convert = std::make_shared<op::v0::Convert>(weights, element::f16);
    w_convert->set_friendly_name("w_convert");

    auto zp_const = op::v0::Constant::create(element::u8, Shape{1, 1, 1},
                                              std::vector<uint8_t>{8});
    auto zp_convert = std::make_shared<op::v0::Convert>(zp_const, element::f16);
    zp_convert->set_friendly_name("zp_convert");

    auto subtract = std::make_shared<op::v1::Subtract>(w_convert, zp_convert);
    subtract->set_friendly_name("w_subtract_zp");

    std::vector<ov::float16> scale_data(sN * n_groups);
    for (auto& v : scale_data) v = ov::float16(float(rng() % 100 + 1) / 1000.0f);
    auto scale_const = op::v0::Constant::create(element::f16,
                                                  Shape{sN, n_groups, 1}, scale_data.data());
    scale_const->set_friendly_name("weight_scale");

    auto multiply = std::make_shared<op::v1::Multiply>(subtract, scale_const);
    multiply->set_friendly_name("w_mul_scale");

    auto reshape_shape = op::v0::Constant::create(element::i32, Shape{2},
                                                    std::vector<int32_t>{-1, static_cast<int32_t>(sK)});
    auto reshape = std::make_shared<op::v1::Reshape>(multiply, reshape_shape, false);
    reshape->set_friendly_name("w_reshape");

    auto matmul = std::make_shared<op::v0::MatMul>(input, reshape, false, true);
    matmul->set_friendly_name("fc_matmul");

    return std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input}, "fc_int4");
}

// Plain FP16 MatMul model — no weight compression, used when FC_QKV/FC_O are uncompressed.
// Builds: Input(f16,[M,K]) × Constant(f16,[N,K]) → MatMul → Result
static std::shared_ptr<ov::Model> build_fc_f16_model(int M, int K, int N) {
    size_t sM = static_cast<size_t>(M);
    size_t sK = static_cast<size_t>(K);
    size_t sN = static_cast<size_t>(N);

    auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{sM, sK});
    input->set_friendly_name("input");

    std::mt19937 rng(42);
    std::vector<ov::float16> w_data(sN * sK);
    for (auto& v : w_data) v = ov::float16(float(rng() % 200 - 100) / 1000.0f);
    auto weights = op::v0::Constant::create(element::f16, Shape{sN, sK}, w_data.data());
    weights->set_friendly_name("weights_f16");

    auto matmul = std::make_shared<op::v0::MatMul>(input, weights, false, true);
    matmul->set_friendly_name("fc_matmul");

    return std::make_shared<Model>(OutputVector{matmul}, ParameterVector{input}, "fc_f16");
}

// L2/L3 cache flush model: Parameter(f16,[N]) -> Relu -> Result.
// Running it reads+writes 2*N*2 bytes of VRAM, evicting any smaller cached data
// (weights, scales) that the FC under test just loaded into on-die caches.
static std::shared_ptr<ov::Model> build_flush_model(size_t n_elems) {
    auto p = std::make_shared<op::v0::Parameter>(element::f16, Shape{n_elems});
    p->set_friendly_name("flush_input");
    auto r = std::make_shared<op::v0::Relu>(p);
    r->set_friendly_name("flush_relu");
    return std::make_shared<Model>(OutputVector{r}, ParameterVector{p}, "l2_flush");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <K> <N> [group_size=128] [iters=100] [warmup=10] [num_bufs=8] [precision=u4|u8|f16] [flush_mb=64]"
                  << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int K = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);
    int group_size = argc > 4 ? std::atoi(argv[4]) : 128;
    int iters      = argc > 5 ? std::atoi(argv[5]) : 100;
    int warmup     = argc > 6 ? std::atoi(argv[6]) : 10;
    int num_bufs   = argc > 7 ? std::atoi(argv[7]) : 8;
    std::string precision_str = (argc > 8) ? std::string(argv[8]) : "u4";
    bool u8_weights  = (precision_str == "u8");
    bool f16_weights = (precision_str == "f16");
    int flush_mb   = argc > 9 ? std::atoi(argv[9]) : 64;

    std::cout << "=== FC Benchmark (precision=" << precision_str << ") ===" << std::endl;
    std::cout << "M=" << M << " K=" << K << " N=" << N
              << " group_size=" << group_size
              << " iters=" << iters << " warmup=" << warmup
              << " bufs=" << num_bufs
              << " flush_mb=" << flush_mb << std::endl;

    std::shared_ptr<ov::Model> model;
    if (f16_weights) {
        model = build_fc_f16_model(M, K, N);
    } else {
        model = build_fc_deq_model(M, K, N, group_size, u8_weights);
    }

    Core core;
    ov::AnyMap props;
    if (!f16_weights) {
        if (M > 1) {
            props[ov::hint::dynamic_quantization_group_size.name()] = group_size;
            std::cout << "Dynamic quantization: ON (group_size=" << group_size << ")" << std::endl;
        } else {
            props[ov::hint::dynamic_quantization_group_size.name()] = 0;
            std::cout << "Dynamic quantization: OFF (decode mode, group_size=0)" << std::endl;
        }
    } else {
        std::cout << "F16 plain MatMul: no dynamic quantization." << std::endl;
    }
    auto compiled = core.compile_model(model, "GPU", props);

    auto remote_ctx = core.get_default_context("GPU");
    auto input_port = compiled.input();

    std::vector<InferRequest> reqs;
    std::vector<ov::Tensor> device_tensors;
    for (int b = 0; b < num_bufs; b++) {
        auto req = compiled.create_infer_request();
        ov::Tensor dev_tensor = remote_ctx.create_tensor(
            input_port.get_element_type(), input_port.get_shape(), {});
        req.set_input_tensor(dev_tensor);
        device_tensors.push_back(dev_tensor);
        reqs.push_back(std::move(req));
    }
    std::cout << "Input tensors allocated in USM_DEVICE (VRAM) via RemoteContext." << std::endl;

    // Build + compile the L2/L3 cache flush helper (single InferRequest, single
    // big VRAM buffer). Running it between FC infers evicts cached FC weights
    // so each measured FC iteration reads weights from DRAM, not L2.
    InferRequest flush_req;
    bool flush_enabled = flush_mb > 0;
    if (flush_enabled) {
        size_t flush_elems = static_cast<size_t>(flush_mb) * 1024ull * 1024ull / 2ull; // f16 = 2B
        auto fmodel = build_flush_model(flush_elems);
        auto fcompiled = core.compile_model(fmodel, "GPU");
        flush_req = fcompiled.create_infer_request();
        auto fin_port = fcompiled.input();
        ov::Tensor flush_in = remote_ctx.create_tensor(
            fin_port.get_element_type(), fin_port.get_shape(), {});
        flush_req.set_input_tensor(flush_in);
        flush_req.infer();  // warm up the flush primitive
        std::cout << "L2/L3 flush kernel: " << flush_mb
                  << " MB f16 Relu between every FC infer." << std::endl;
    } else {
        std::cout << "L2/L3 flush kernel DISABLED (flush_mb=0)." << std::endl;
    }

    for (int i = 0; i < warmup; i++) {
        if (flush_enabled) flush_req.infer();
        reqs[i % num_bufs].infer();
    }

    std::vector<double> latencies;
    latencies.reserve(iters);
    for (int i = 0; i < iters; i++) {
        if (flush_enabled) flush_req.infer();  // evict L2/L3 before measurement
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

    int n_groups = K / group_size;
    double flops = 2.0 * M * K * N;
    double weight_bytes;
    double scale_bytes;
    if (f16_weights) {
        weight_bytes = double(N) * K * 2;  // FP16 = 2 bytes/element, no compression
        scale_bytes  = 0.0;
    } else {
        weight_bytes = u8_weights ? double(N) * K * 1.0 : double(N) * K * 0.5;
        scale_bytes  = double(N) * n_groups * 2;
    }
    double io_bytes     = double(M) * K * 2 + double(M) * N * 2;
    double total_bytes  = weight_bytes + scale_bytes + io_bytes;

    double gflops = flops / (median * 1e-3) / 1e9;
    double bw = total_bytes / (median * 1e-3) / 1e9;

    std::cout << "Median_ms: " << median << std::endl;
    std::cout << "Min_ms: " << min_lat << std::endl;
    std::cout << "Avg_ms: " << avg << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;
    std::cout << "BW_GBs: " << bw << std::endl;
    std::cout << "AI: " << (flops / total_bytes) << std::endl;
    std::cout << "WeightBytes: " << weight_bytes << std::endl;
    std::cout << "TotalBytes: " << total_bytes << std::endl;
    std::cout << "TotalFLOPs: " << flops << std::endl;

    return 0;
}
