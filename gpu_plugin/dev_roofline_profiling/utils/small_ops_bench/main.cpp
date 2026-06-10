/**
 * Small Ops Benchmark — Qwen3-8B element-wise / normalization / activation ops
 *
 * Usage: ./small_ops_bench <op> <M> [params...] [--iters N] [--warmup N] [--bufs N]
 *
 * Supported ops (shapes match the real Qwen3-8B dump; see verbose log analysis):
 *   rmsnorm    <M> <H>            — RMSNorm on [M, H], gamma [H]          (input_layernorm, post_attn_layernorm: H=4096)
 *   rmsnorm3d  <M> <NH> <HD>      — RMSNorm on [M, NH, HD] reduced on last, gamma [HD]  (q_norm NH=32 HD=128, k_norm NH=8 HD=128)
 *   add        <M> <H>            — Eltwise Add [M, H] + [M, H]           (residual: H=4096)
 *   rope       <M> <NH> <HD>      — RoPE on [1, M, NH, HD] with cos/sin [1, M, 1, HD]
 *   swish      <M> <H>            — SiLU/Swish: x * sigmoid(x), f16 [M, H]  (SwiGLU gate: H=12288)
 *   multiply   <M> <H>            — Eltwise Multiply [M, H] * [M, H]      (SwiGLU gate*up: H=12288)
 *
 * Decomposed patterns are used so GPU plugin fusion passes produce native kernels:
 *   rmsnorm → RMSFusion → rms GPU primitive
 *   add     → eltwise GPU primitive
 *   rope    → element-wise ops (may or may not fuse to rope primitive)
 *
 * Multiple input buffers rotate to avoid L3 cache reuse (same as fc_bench).
 * Run with cliloader for accurate GPU kernel timing.
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
#include <cstring>

using namespace ov;

// ============================================================================
// RMSNorm: decomposed as Power(2) → ReduceMean → Add(eps) → Sqrt → Divide → Mul(gamma)
// GPU plugin's RMSFusion will fuse this into the native rms primitive.
// ============================================================================
static std::shared_ptr<Model> build_rmsnorm_model(int M, int H) {
    auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    input->set_friendly_name("input");

    // Gamma weight [H]
    std::mt19937 rng(42);
    std::vector<ov::float16> gamma_data(H);
    for (auto& v : gamma_data) v = ov::float16(1.0f);
    auto gamma = op::v0::Constant::create(element::f16, Shape{size_t(H)}, gamma_data.data());
    gamma->set_friendly_name("gamma");

    // x^2
    auto two = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(2.0f)});
    auto power = std::make_shared<op::v1::Power>(input, two);

    // ReduceMean over last axis, keepdims=true
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto mean = std::make_shared<op::v1::ReduceMean>(power, axes, true);

    // + eps
    auto eps = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(1e-5f)});
    auto add_eps = std::make_shared<op::v1::Add>(mean, eps);

    // sqrt
    auto sqrt_op = std::make_shared<op::v0::Sqrt>(add_eps);

    // 1/sqrt (rsqrt)
    auto one = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(1.0f)});
    auto inv = std::make_shared<op::v1::Divide>(one, sqrt_op);

    // x * rsqrt
    auto norm = std::make_shared<op::v1::Multiply>(input, inv);

    // * gamma
    auto output = std::make_shared<op::v1::Multiply>(norm, gamma);
    output->set_friendly_name("rmsnorm_out");

    return std::make_shared<Model>(OutputVector{output}, ParameterVector{input}, "rmsnorm");
}

// ============================================================================
// RMSNorm 3D: matches Qwen3 per-head q_norm / k_norm.
//   input  [M, NH, HD], gamma [HD], reduce over last axis.
//   rms_gpu_bfyx_opt handles this shape natively.
// ============================================================================
static std::shared_ptr<Model> build_rmsnorm3d_model(int M, int NH, int HD) {
    auto input = std::make_shared<op::v0::Parameter>(element::f16,
                     Shape{size_t(M), size_t(NH), size_t(HD)});
    input->set_friendly_name("input");

    std::vector<ov::float16> gamma_data(HD, ov::float16(1.0f));
    auto gamma = op::v0::Constant::create(element::f16, Shape{size_t(HD)}, gamma_data.data());

    auto two = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(2.0f)});
    auto power = std::make_shared<op::v1::Power>(input, two);
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto mean = std::make_shared<op::v1::ReduceMean>(power, axes, true);
    auto eps = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(1e-5f)});
    auto add_eps = std::make_shared<op::v1::Add>(mean, eps);
    auto sqrt_op = std::make_shared<op::v0::Sqrt>(add_eps);
    auto one = op::v0::Constant::create(element::f16, Shape{}, std::vector<ov::float16>{ov::float16(1.0f)});
    auto inv = std::make_shared<op::v1::Divide>(one, sqrt_op);
    auto norm = std::make_shared<op::v1::Multiply>(input, inv);
    auto output = std::make_shared<op::v1::Multiply>(norm, gamma);
    output->set_friendly_name("rmsnorm3d_out");
    return std::make_shared<Model>(OutputVector{output}, ParameterVector{input}, "rmsnorm3d");
}

// ============================================================================
// Swish / SiLU: y = x * sigmoid(x).  Matches Qwen3 MLP gate activation.
// ============================================================================
static std::shared_ptr<Model> build_swish_model(int M, int H) {
    auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    input->set_friendly_name("input");
    auto output = std::make_shared<op::v4::Swish>(input);
    output->set_friendly_name("swish_out");
    return std::make_shared<Model>(OutputVector{output}, ParameterVector{input}, "swish");
}

// ============================================================================
// Eltwise Multiply: input1 * input2, both [M, H] f16.  SwiGLU: swish(gate) * up.
// ============================================================================
static std::shared_ptr<Model> build_multiply_model(int M, int H) {
    auto a = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    auto b = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    a->set_friendly_name("input1");
    b->set_friendly_name("input2");
    auto output = std::make_shared<op::v1::Multiply>(a, b);
    output->set_friendly_name("multiply_out");
    return std::make_shared<Model>(OutputVector{output}, ParameterVector{a, b}, "multiply");
}

// ============================================================================
// Attention output gate: out = x * sigmoid(y), both [M, H] f16.
// Matches Qwen3.5/Qwen3.6 gated attention: attn_output * sigmoid(gate), where
// the gate is the second chunk of q_proj (attn_output_gate=true). H = num_heads*head_dim.
// ============================================================================
static std::shared_ptr<Model> build_gate_model(int M, int H) {
    auto x = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    auto y = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    x->set_friendly_name("attn_output");
    y->set_friendly_name("gate");
    auto sig = std::make_shared<op::v0::Sigmoid>(y);
    sig->set_friendly_name("gate_sigmoid");
    auto output = std::make_shared<op::v1::Multiply>(x, sig);
    output->set_friendly_name("gate_out");
    return std::make_shared<Model>(OutputVector{output}, ParameterVector{x, y}, "attn_gate");
}

// ============================================================================
// Eltwise Add: input1 + input2, both [M, H] f16
// ============================================================================
static std::shared_ptr<Model> build_add_model(int M, int H) {
    auto input1 = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    auto input2 = std::make_shared<op::v0::Parameter>(element::f16, Shape{size_t(M), size_t(H)});
    input1->set_friendly_name("input1");
    input2->set_friendly_name("input2");

    auto output = std::make_shared<op::v1::Add>(input1, input2);
    output->set_friendly_name("add_out");

    return std::make_shared<Model>(OutputVector{output}, ParameterVector{input1, input2}, "eltwise_add");
}

// ============================================================================
// RoPE (decomposed): x * cos + rotate_half(x) * sin
// Input: [1, M, NH, HD], cos/sin: [1, M, 1, HD]
// rotate_half: concat(-x[..., half:], x[..., :half], axis=-1)
//
// Note: May or may not fuse into the dedicated rope GPU kernel depending on
// the exact pattern. Cliloader will show the actual kernel(s) executed.
// ============================================================================
static std::shared_ptr<Model> build_rope_model(int M, int NH, int HD) {
    size_t sM = size_t(M), sNH = size_t(NH), sHD = size_t(HD);
    size_t half = sHD / 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, sM, sNH, sHD});
    auto cos_in = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, sM, 1, sHD});
    auto sin_in = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, sM, 1, sHD});
    input->set_friendly_name("input");
    cos_in->set_friendly_name("cos");
    sin_in->set_friendly_name("sin");

    // x * cos
    auto x_cos = std::make_shared<op::v1::Multiply>(input, cos_in);

    // Split x into first half and second half along last dim
    // x1 = x[..., :half], x2 = x[..., half:]
    auto ss_begin_1 = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 0, 0, 0});
    auto ss_end_1 = op::v0::Constant::create(element::i64, Shape{4},
                                              std::vector<int64_t>{1, int64_t(sM), int64_t(sNH), int64_t(half)});
    auto ss_stride_1 = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
    auto x1 = std::make_shared<op::v1::StridedSlice>(input, ss_begin_1, ss_end_1, ss_stride_1,
                                                      std::vector<int64_t>{0, 0, 0, 0},
                                                      std::vector<int64_t>{0, 0, 0, 0});

    auto ss_begin_2 = op::v0::Constant::create(element::i64, Shape{4},
                                                std::vector<int64_t>{0, 0, 0, int64_t(half)});
    auto ss_end_2 = op::v0::Constant::create(element::i64, Shape{4},
                                              std::vector<int64_t>{1, int64_t(sM), int64_t(sNH), int64_t(sHD)});
    auto ss_stride_2 = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
    auto x2 = std::make_shared<op::v1::StridedSlice>(input, ss_begin_2, ss_end_2, ss_stride_2,
                                                      std::vector<int64_t>{0, 0, 0, 0},
                                                      std::vector<int64_t>{0, 0, 0, 0});

    // rotate_half = concat(-x2, x1, axis=-1)
    auto neg_x2 = std::make_shared<op::v0::Negative>(x2);
    auto rotate = std::make_shared<op::v0::Concat>(OutputVector{neg_x2, x1}, -1);

    // rotate_half * sin
    auto rot_sin = std::make_shared<op::v1::Multiply>(rotate, sin_in);

    // output = x * cos + rotate_half * sin
    auto output = std::make_shared<op::v1::Add>(x_cos, rot_sin);
    output->set_friendly_name("rope_out");

    return std::make_shared<Model>(OutputVector{output},
                                   ParameterVector{input, cos_in, sin_in}, "rope");
}

// ============================================================================
// Main benchmark harness
// ============================================================================
static void fill_random_f16(ov::Tensor& t, std::mt19937& rng) {
    auto* ptr = t.data<ov::float16>();
    for (size_t i = 0; i < t.get_size(); i++)
        ptr[i] = ov::float16(float(rng() % 200 - 100) / 100.0f);
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <op> <params...> [--iters N] [--warmup N] [--bufs N]\n"
              << "\nOps:\n"
              << "  rmsnorm    <M> <H>        RMSNorm [M,H] with gamma [H]     (input_layernorm H=4096)\n"
              << "  rmsnorm3d  <M> <NH> <HD>  RMSNorm [M,NH,HD] reduced on HD   (q_norm NH=32 HD=128; k_norm NH=8 HD=128)\n"
              << "  add        <M> <H>        Eltwise Add [M,H] + [M,H]        (residual H=4096)\n"
              << "  rope       <M> <NH> <HD>  RoPE [1,M,NH,HD] with cos/sin\n"
              << "  swish      <M> <H>        SiLU/Swish f16 [M,H]             (SwiGLU gate H=12288)\n"
              << "  multiply   <M> <H>        Eltwise Multiply [M,H] * [M,H]   (SwiGLU gate*up H=12288)\n"
              << "  gate       <M> <H>        Attn output gate x*sigmoid(y)    (Qwen3.6 attn gate H=4096)\n"
              << "\nOptions:\n"
              << "  --iters N    Timed iterations (default 150)\n"
              << "  --warmup N   Warmup iterations (default 20)\n"
              << "  --bufs N     Rotating buffers (default 8)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string op_name = argv[1];
    int iters = 150, warmup = 20, num_bufs = 8;

    // Parse trailing options
    auto parse_opts = [&](int start) {
        for (int i = start; i < argc; i++) {
            if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--bufs") == 0 && i + 1 < argc) num_bufs = std::atoi(argv[++i]);
        }
    };

    std::shared_ptr<Model> model;
    std::string desc;
    double total_bytes = 0;  // for BW calculation

    if (op_name == "rmsnorm") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        parse_opts(4);
        model = build_rmsnorm_model(M, H);
        desc = "RMSNorm M=" + std::to_string(M) + " H=" + std::to_string(H);
        // Bytes: read input [M,H]*2 + gamma [H]*2 + write output [M,H]*2
        total_bytes = double(M) * H * 2 + double(H) * 2 + double(M) * H * 2;
    }
    else if (op_name == "add") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        parse_opts(4);
        model = build_add_model(M, H);
        desc = "Add M=" + std::to_string(M) + " H=" + std::to_string(H);
        // Bytes: read 2 inputs [M,H]*2 + write output [M,H]*2
        total_bytes = 3.0 * M * H * 2;
    }
    else if (op_name == "rope") {
        if (argc < 5) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int NH = std::atoi(argv[3]);
        int HD = std::atoi(argv[4]);
        parse_opts(5);
        model = build_rope_model(M, NH, HD);
        desc = "RoPE M=" + std::to_string(M) + " NH=" + std::to_string(NH) + " HD=" + std::to_string(HD);
        // Bytes: read input [1,M,NH,HD]*2 + cos [1,M,1,HD]*2 + sin [1,M,1,HD]*2 + write output [1,M,NH,HD]*2
        total_bytes = double(M) * NH * HD * 2 + 2.0 * M * HD * 2 + double(M) * NH * HD * 2;
    }
    else if (op_name == "rmsnorm3d") {
        if (argc < 5) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int NH = std::atoi(argv[3]);
        int HD = std::atoi(argv[4]);
        parse_opts(5);
        model = build_rmsnorm3d_model(M, NH, HD);
        desc = "RMSNorm3D M=" + std::to_string(M) + " NH=" + std::to_string(NH) + " HD=" + std::to_string(HD);
        // Bytes: read input [M,NH,HD]*2 + gamma [HD]*2 + write output [M,NH,HD]*2
        total_bytes = 2.0 * M * NH * HD * 2 + double(HD) * 2;
    }
    else if (op_name == "swish") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        parse_opts(4);
        model = build_swish_model(M, H);
        desc = "Swish M=" + std::to_string(M) + " H=" + std::to_string(H);
        // Bytes: read input [M,H]*2 + write output [M,H]*2
        total_bytes = 2.0 * M * H * 2;
    }
    else if (op_name == "multiply") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        parse_opts(4);
        model = build_multiply_model(M, H);
        desc = "Multiply M=" + std::to_string(M) + " H=" + std::to_string(H);
        // Bytes: read 2 inputs [M,H]*2 + write output [M,H]*2
        total_bytes = 3.0 * M * H * 2;
    }
    else if (op_name == "gate") {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        int M = std::atoi(argv[2]);
        int H = std::atoi(argv[3]);
        parse_opts(4);
        model = build_gate_model(M, H);
        desc = "Gate M=" + std::to_string(M) + " H=" + std::to_string(H);
        // Bytes: read 2 inputs [M,H]*2 + write output [M,H]*2
        total_bytes = 3.0 * M * H * 2;
    }
    else {
        std::cerr << "Unknown op: " << op_name << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Small Ops Benchmark ===" << std::endl;
    std::cout << desc << " iters=" << iters << " warmup=" << warmup << " bufs=" << num_bufs << std::endl;

    Core core;
    auto compiled = core.compile_model(model, "GPU");

    // Allocate ALL input tensors in USM_DEVICE (VRAM) via RemoteContext so
    // the kernel under measurement reads activations from VRAM, not PCIe
    // usm_host. On PTL (iGPU) this is equivalent; on BMG (dGPU) this is
    // essential to get true kernel bandwidth instead of PCIe ceiling.
    auto remote_ctx = core.get_default_context("GPU");

    // Create rotating infer request buffers
    std::vector<InferRequest> reqs;
    std::vector<std::vector<ov::Tensor>> dev_tensors_per_req;
    for (int b = 0; b < num_bufs; b++) {
        auto req = compiled.create_infer_request();
        std::vector<ov::Tensor> req_dev_tensors;
        size_t n_inputs = compiled.inputs().size();
        for (size_t idx = 0; idx < n_inputs; idx++) {
            auto input_port = compiled.input(idx);
            ov::Tensor dev_tensor = remote_ctx.create_tensor(
                input_port.get_element_type(), input_port.get_shape(), {});
            req.set_input_tensor(idx, dev_tensor);
            req_dev_tensors.push_back(dev_tensor);
        }
        dev_tensors_per_req.push_back(std::move(req_dev_tensors));
        reqs.push_back(std::move(req));
    }
    std::cout << "Input tensors allocated in USM_DEVICE (VRAM) via RemoteContext." << std::endl;

    // Warmup
    for (int i = 0; i < warmup; i++)
        reqs[i % num_bufs].infer();

    // Timed iterations
    std::vector<double> latencies;
    latencies.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto& req = reqs[i % num_bufs];
        auto t0 = std::chrono::high_resolution_clock::now();
        req.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::sort(latencies.begin(), latencies.end());
    double median = latencies[latencies.size() / 2];
    double min_lat = latencies.front();
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    double bw = total_bytes / (median * 1e-6) / 1e9;  // GB/s using median

    std::cout << "Median_us: " << median << std::endl;
    std::cout << "Min_us: " << min_lat << std::endl;
    std::cout << "Avg_us: " << avg << std::endl;
    std::cout << "BW_GBs: " << bw << std::endl;
    std::cout << "TotalBytes: " << total_bytes << std::endl;

    return 0;
}
