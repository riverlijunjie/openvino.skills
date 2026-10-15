/**
 * Conv Benchmark — 3D Convolution for video VAE encoder/decoder resnet blocks.
 *
 * Covers the two OpenVINO op families SeedVR2's VAE encoder/decoder actually
 * compile to (confirmed from the exported IR, `SeedVR2/ir/vae_{encoder,decoder}*.xml`,
 * via `grep type="Convolution\|type="CausalConv3DCarry"`):
 *
 *   - "plain"  : ov::op::v1::Convolution, 5D NCDHW. Used for kernel_t==1 taps
 *                (1x1x1 channel-projection shortcuts/upscale convs, and the
 *                spatial-only stride-2 downsamplers whose temporal kernel is 1).
 *   - "causal" : ov::op::internal::CausalConv3DCarry (dev_api-only internal op,
 *                see thirdparty/openvino/src/core/dev_api/openvino/op/causal_conv3d_carry.hpp).
 *                Used for every kernel_t==3 resnet conv1/conv2/conv_in/conv_out,
 *                which needs cross-segment temporal history (`carry_in`,
 *                updated in place by the kernel -- no separate carry_out output).
 *                The GPU plugin lowers this to its oneDNN-backed
 *                `causal_conv3d` primitive (onednn_causal_conv3d.cpp), which
 *                requires an immad (oneDNN GPU integration) capable device.
 *
 * Usage: ./conv_bench <plain|causal> <c_in> <c_out> <kt> <kh> <kw>
 *                      <t_body> <h> <w>
 *                      [stride_t=1] [stride_h=1] [stride_w=1]
 *                      [iters=50] [warmup=5] [num_bufs=4] [has_bias=1] [init_mode=0]
 *
 * Shapes should come from `models/seedvr2/ops_mapping.json`, derived from a
 * real measured run (`SeedVR2/orbix-seedvr2/mem_report_1080P_full_utf8.log`,
 * SKILL.md's "benchmark what OpenVINO really executes" + "input should be
 * fsv16 layout" guidance -- fsv16 vs plain layout selection for the "plain"
 * Convolution path is entirely the GPU plugin's layout_optimizer's job, not
 * something this bench controls directly; see
 * src/plugins/intel_gpu/src/graph/layout_optimizer.cpp for the selection rule).
 *
 * H/W padding is SAME-style (pad = (k-1)/2 for odd kernels, 0 for k=1) so a
 * stride-1 conv preserves H/W and a stride-2 conv halves it, matching the
 * resnet/downsampler shapes observed in the real log. Causal (time-axis)
 * history always comes from carry_in, never from padding (matches the op's
 * own contract).
 *
 * `--flush-mb` cache-flush convention mirrors fc_bench (see its header
 * comment) since this op's filters are often MB-scale too.
 */
#include <openvino/openvino.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/causal_conv3d_carry.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ov;

// Hard ceiling on this bench's own device allocations (weights + num_bufs x
// (inputs+outputs)), independent of whatever num_bufs was requested on the
// command line -- large full-resolution convs (e.g. 1088x1920) can otherwise
// request tens of GB once multiplied by num_bufs. Leaves headroom under a
// 20 GiB target so the driver/other processes aren't starved.
constexpr uint64_t kGpuMemBudgetBytes = 18ull * 1024 * 1024 * 1024;

namespace {

// Shrinks num_bufs (down to a minimum of 1) so that
// shared_bytes + num_bufs * per_buf_bytes stays within budget_bytes. Returns
// the (possibly reduced) buf count; prints a warning if even 1 buf overflows.
int cap_num_bufs_for_budget(int requested_bufs, double per_buf_bytes, double shared_bytes,
                             uint64_t budget_bytes) {
    int bufs = std::max(1, requested_bufs);
    while (bufs > 1 && shared_bytes + double(bufs) * per_buf_bytes > double(budget_bytes)) {
        bufs--;
    }
    const double estimated = shared_bytes + double(bufs) * per_buf_bytes;
    if (bufs < requested_bufs) {
        std::cout << "[CONV_BENCH] num_bufs reduced " << requested_bufs << " -> " << bufs
                  << " to stay under " << (budget_bytes / 1024.0 / 1024.0 / 1024.0) << " GiB budget"
                  << " (estimated " << (estimated / 1024.0 / 1024.0 / 1024.0) << " GiB)" << std::endl;
    }
    if (estimated > double(budget_bytes)) {
        std::cout << "[CONV_BENCH] WARNING: even num_bufs=1 is estimated at "
                  << (estimated / 1024.0 / 1024.0 / 1024.0) << " GiB, over budget -- proceeding anyway"
                  << std::endl;
    }
    return bufs;
}

void print_peak_gpu_memory(ov::Core& core) {
    try {
        auto stats = core.get_property("GPU", ov::intel_gpu::memory_statistics);
        uint64_t usm_device = stats.count("usm_device") ? stats.at("usm_device") : 0;
        uint64_t usm_host = stats.count("usm_host") ? stats.at("usm_host") : 0;
        std::cout << "PeakUsmDeviceBytes: " << usm_device << std::endl;
        std::cout << "PeakUsmDeviceMB: " << (double(usm_device) / 1024.0 / 1024.0) << std::endl;
        std::cout << "PeakUsmHostBytes: " << usm_host << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[CONV_BENCH] memory_statistics query failed: " << e.what() << std::endl;
    }
}

}  // namespace

namespace {

std::shared_ptr<ov::Model> build_plain_model(int64_t c_in, int64_t c_out,
                                              int64_t kt, int64_t kh, int64_t kw,
                                              int64_t t_body, int64_t h, int64_t w,
                                              int64_t st, int64_t sh, int64_t sw,
                                              bool /*has_bias*/) {
    auto data = std::make_shared<op::v0::Parameter>(
        element::f16, Shape{1, (size_t)c_in, (size_t)t_body, (size_t)h, (size_t)w});
    data->set_friendly_name("data");

    std::mt19937 rng(42);
    std::vector<ov::float16> w_data((size_t)(c_out * c_in * kt * kh * kw));
    for (auto& v : w_data) v = ov::float16(float(rng() % 200 - 100) / 1000.0f);
    auto filters = op::v0::Constant::create(
        element::f16, Shape{(size_t)c_out, (size_t)c_in, (size_t)kt, (size_t)kh, (size_t)kw}, w_data.data());
    filters->set_friendly_name("filters");

    const int64_t pad_t = 0;  // plain-op usage here is always kt==1 (no temporal context needed)
    const int64_t pad_h = (kh - 1) / 2;
    const int64_t pad_w = (kw - 1) / 2;

    auto conv = std::make_shared<op::v1::Convolution>(
        data, filters,
        Strides{(size_t)st, (size_t)sh, (size_t)sw},
        CoordinateDiff{pad_t, pad_h, pad_w},
        CoordinateDiff{pad_t, pad_h, pad_w},
        Strides{1, 1, 1});
    conv->set_friendly_name("conv");

    auto result = std::make_shared<op::v0::Result>(conv->output(0));
    return std::make_shared<Model>(OutputVector{result}, ParameterVector{data}, "conv_bench_plain");
}

std::shared_ptr<ov::Model> build_causal_model(int64_t c_in, int64_t c_out,
                                               int64_t kt, int64_t kh, int64_t kw,
                                               int64_t t_body, int64_t t_carry,
                                               int64_t h, int64_t w,
                                               int64_t st, int64_t sh, int64_t sw,
                                               bool has_bias, bool init_mode) {
    auto body = std::make_shared<op::v0::Parameter>(
        element::f16, Shape{1, (size_t)c_in, (size_t)t_body, (size_t)h, (size_t)w});
    body->set_friendly_name("body");
    auto carry_in = std::make_shared<op::v0::Parameter>(
        element::f16, Shape{1, (size_t)c_in, (size_t)t_carry, (size_t)h, (size_t)w});
    carry_in->set_friendly_name("carry_in");

    std::mt19937 rng(42);
    std::vector<ov::float16> w_data((size_t)(c_out * c_in * kt * kh * kw));
    for (auto& v : w_data) v = ov::float16(float(rng() % 200 - 100) / 1000.0f);
    auto filters = op::v0::Constant::create(
        element::f16, Shape{(size_t)c_out, (size_t)c_in, (size_t)kt, (size_t)kh, (size_t)kw}, w_data.data());
    filters->set_friendly_name("filters");

    const int64_t pad_h = (kh - 1) / 2;
    const int64_t pad_w = (kw - 1) / 2;
    const Strides strides{(size_t)st, (size_t)sh, (size_t)sw};
    const CoordinateDiff pads_begin{pad_h, pad_w};
    const CoordinateDiff pads_end{pad_h, pad_w};
    const Strides dilations{1, 1, 1};

    std::shared_ptr<ov::op::internal::CausalConv3DCarry> conv;
    if (has_bias) {
        std::vector<ov::float16> b_data((size_t)c_out);
        for (auto& v : b_data) v = ov::float16(float(rng() % 100) / 1000.0f);
        auto bias = op::v0::Constant::create(element::f16, Shape{1, (size_t)c_out, 1, 1, 1}, b_data.data());
        bias->set_friendly_name("bias");
        conv = std::make_shared<ov::op::internal::CausalConv3DCarry>(
            body, carry_in, filters, bias, strides, pads_begin, pads_end, dilations, init_mode);
    } else {
        conv = std::make_shared<ov::op::internal::CausalConv3DCarry>(
            body, carry_in, filters, strides, pads_begin, pads_end, dilations, init_mode);
    }
    conv->set_friendly_name("causal_conv");

    auto result = std::make_shared<op::v0::Result>(conv->output(0));
    return std::make_shared<Model>(OutputVector{result}, ParameterVector{body, carry_in}, "conv_bench_causal");
}

std::shared_ptr<ov::Model> build_flush_model(size_t n_elems) {
    auto p = std::make_shared<op::v0::Parameter>(element::f16, Shape{n_elems});
    p->set_friendly_name("flush_input");
    auto r = std::make_shared<op::v0::Relu>(p);
    r->set_friendly_name("flush_relu");
    return std::make_shared<Model>(OutputVector{r}, ParameterVector{p}, "l2_flush");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <plain|causal> <c_in> <c_out> <kt> <kh> <kw> <t_body> <h> <w>"
                  << " [stride_t=1] [stride_h=1] [stride_w=1]"
                  << " [iters=50] [warmup=5] [num_bufs=4] [has_bias=1] [init_mode=0] [flush_mb=64]"
                  << std::endl;
        return 1;
    }

    try {
        const std::string kind = argv[1];
        const int64_t c_in  = std::atoll(argv[2]);
        const int64_t c_out = std::atoll(argv[3]);
        const int64_t kt = std::atoll(argv[4]);
        const int64_t kh = std::atoll(argv[5]);
        const int64_t kw = std::atoll(argv[6]);
        const int64_t t_body = std::atoll(argv[7]);
        const int64_t h = std::atoll(argv[8]);
        const int64_t w = std::atoll(argv[9]);

        int argi = 10;
        auto next_or = [&](int64_t defv) -> int64_t {
            if (argi < argc) return std::atoll(argv[argi++]);
            return defv;
        };
        const int64_t stride_t = next_or(1);
        const int64_t stride_h = next_or(1);
        const int64_t stride_w = next_or(1);
        const int iters    = (int)next_or(50);
        const int warmup   = (int)next_or(5);
        const int num_bufs = (int)next_or(4);
        const bool has_bias = next_or(1) != 0;
        const bool init_mode = next_or(0) != 0;
        const int flush_mb = (int)next_or(64);

        const bool is_causal = (kind == "causal");
        const int64_t t_carry = is_causal ? std::max<int64_t>(kt - stride_t, 0) : 0;
        const int64_t t_total = is_causal ? (t_body + t_carry) : t_body;
        const int64_t t_out = (t_total - kt) / stride_t + 1;
        const int64_t h_out = (h + 2 * ((kh - 1) / 2) - kh) / stride_h + 1;
        const int64_t w_out = (w + 2 * ((kw - 1) / 2) - kw) / stride_w + 1;

        std::cout << "=== Conv Benchmark (kind=" << kind << ") ===" << std::endl;
        std::cout << "c_in=" << c_in << " c_out=" << c_out
                  << " k=(" << kt << "," << kh << "," << kw << ")"
                  << " t_body=" << t_body << " t_carry=" << t_carry
                  << " h=" << h << " w=" << w
                  << " stride=(" << stride_t << "," << stride_h << "," << stride_w << ")"
                  << " out=(" << t_out << "," << h_out << "," << w_out << ")"
                  << " has_bias=" << has_bias << " init_mode=" << init_mode
                  << " iters=" << iters << " warmup=" << warmup << " bufs=" << num_bufs
                  << std::endl;

        std::shared_ptr<ov::Model> model;
        if (is_causal) {
            model = build_causal_model(c_in, c_out, kt, kh, kw, t_body, t_carry, h, w,
                                        stride_t, stride_h, stride_w, has_bias, init_mode);
        } else if (kind == "plain") {
            model = build_plain_model(c_in, c_out, kt, kh, kw, t_body, h, w,
                                       stride_t, stride_h, stride_w, has_bias);
        } else {
            std::cerr << "[CONV_BENCH ERROR] <plain|causal> expected, got '" << kind << "'" << std::endl;
            return 1;
        }

        Core core;
        auto compiled = core.compile_model(model, "GPU");
        auto remote_ctx = core.get_default_context("GPU");

        // Budget check: weights (shared across requests) + num_bufs x (input(s) + output).
        const double weight_bytes_est = double(c_out) * c_in * kt * kh * kw * 2.0;
        const double in_bytes_est = double(c_in) * t_body * h * w * 2.0
                                   + (is_causal ? double(c_in) * t_carry * h * w * 2.0 : 0.0);
        const double out_bytes_est = double(c_out) * t_out * h_out * w_out * 2.0;
        const int capped_bufs = cap_num_bufs_for_budget(
            num_bufs, in_bytes_est + out_bytes_est, weight_bytes_est, kGpuMemBudgetBytes);

        std::vector<InferRequest> reqs;
        for (int b = 0; b < capped_bufs; b++) {
            auto req = compiled.create_infer_request();
            for (size_t idx = 0; idx < compiled.inputs().size(); idx++) {
                auto port = compiled.input(idx);
                // RemoteTensor::data<T>() is not implemented (matches fc_bench's existing
                // pattern) -- uninitialized USM_DEVICE content is fine, this bench only
                // measures kernel performance, not numerical correctness.
                ov::Tensor dev_tensor = remote_ctx.create_tensor(port.get_element_type(), port.get_shape(), {});
                req.set_input_tensor(idx, dev_tensor);
            }
            reqs.push_back(std::move(req));
        }
        std::cout << "Input tensors allocated in USM_DEVICE (VRAM) via RemoteContext." << std::endl;

        InferRequest flush_req;
        const bool flush_enabled = flush_mb > 0;
        if (flush_enabled) {
            const size_t flush_elems = (size_t)flush_mb * 1024ull * 1024ull / 2ull;
            auto fmodel = build_flush_model(flush_elems);
            auto fcompiled = core.compile_model(fmodel, "GPU");
            flush_req = fcompiled.create_infer_request();
            auto fin_port = fcompiled.input();
            ov::Tensor flush_in = remote_ctx.create_tensor(fin_port.get_element_type(), fin_port.get_shape(), {});
            flush_req.set_input_tensor(flush_in);
            flush_req.infer();
        }

        for (int i = 0; i < warmup; i++) {
            if (flush_enabled) flush_req.infer();
            reqs[i % capped_bufs].infer();
        }

        std::vector<double> latencies;
        latencies.reserve(iters);
        for (int i = 0; i < iters; i++) {
            if (flush_enabled) flush_req.infer();
            auto& req = reqs[i % capped_bufs];
            auto t0 = std::chrono::high_resolution_clock::now();
            req.infer();
            auto t1 = std::chrono::high_resolution_clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::sort(latencies.begin(), latencies.end());
        const double median = latencies[latencies.size() / 2];
        const double min_lat = latencies.front();
        const double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

        const double flops = 2.0 * c_out * c_in * kt * kh * kw * t_out * h_out * w_out;
        const double weight_bytes = double(c_out) * c_in * kt * kh * kw * 2.0;
        const double in_bytes = double(c_in) * t_body * h * w * 2.0
                               + (is_causal ? double(c_in) * t_carry * h * w * 2.0 : 0.0);
        const double out_bytes = double(c_out) * t_out * h_out * w_out * 2.0;
        const double total_bytes = weight_bytes + in_bytes + out_bytes;

        const double gflops = flops / (median * 1e-3) / 1e9;
        const double bw = total_bytes / (median * 1e-3) / 1e9;

        std::cout << "Median_ms: " << median << std::endl;
        std::cout << "Min_ms: " << min_lat << std::endl;
        std::cout << "Avg_ms: " << avg << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;
        std::cout << "BW_GBs: " << bw << std::endl;
        std::cout << "AI: " << (flops / total_bytes) << std::endl;
        std::cout << "TotalFLOPs: " << flops << std::endl;
        std::cout << "TotalBytes: " << total_bytes << std::endl;
        print_peak_gpu_memory(core);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[CONV_BENCH ERROR] " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "[CONV_BENCH ERROR] unknown exception" << std::endl;
        return 3;
    }
}
