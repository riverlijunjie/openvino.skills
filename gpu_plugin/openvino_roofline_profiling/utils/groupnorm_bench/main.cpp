/**
 * GroupNorm Benchmark — ov::op::internal::GroupNormalizationZSplit
 *
 * SeedVR2's exported VAE encoder/decoder IR uses this SeedVR2-project-specific
 * internal op (NOT the standard opset `ov::op::v12::GroupNormalization`) for
 * every ResnetBlock/attention-prenorm GroupNorm call -- confirmed via
 * `grep type="GroupNormalizationZSplit" SeedVR2/ir/vae_{encoder,decoder}*.xml`
 * (22-30 occurrences per IR). Per SKILL.md's "benchmark what OpenVINO really
 * executes" rule, this bench targets that op directly rather than the plain
 * opset12 GroupNormalization.
 *
 * Header: thirdparty/openvino/src/core/include/openvino/op/internal/
 * group_normalization_zsplit.hpp (header-only, part of the installed core
 * include tree -- no OV_SRC_DIR needed, unlike CausalConv3DCarry/VLSDPA which
 * live under dev_api/ov_ops and require it).
 *
 * Contract (see the header's own doc comment): data is 5D (B,C,Z,Y,X), Z is
 * an independent per-slice split dim (here: the temporal/frame axis -- every
 * (batch,frame) slice gets its own group statistics, never reduced jointly
 * across frames). The GPU plugin lowers this to `cldnn::group_normalization_zsplit`,
 * which accepts either a b_fs_zyx_fsv16 (blocked, inherited from an upstream
 * Convolution) or a plain bfzyx activation input, and always writes a plain
 * bfzyx output -- this bench feeds a Parameter directly (no upstream conv),
 * which the layout optimizer will keep as plain bfzyx (same situation the
 * existing group_norm_stats_partial_op_test.cpp's isolated op test hits).
 *
 * Usage: ./groupnorm_bench <c> <z> <h> <w> [num_groups=32]
 *                          [iters=50] [warmup=5] [num_bufs=4]
 *
 * Shapes should come from `models/seedvr2/ops_mapping.json` (derived from a
 * real measured run, same source as conv_bench).
 */
#include <openvino/openvino.hpp>
#include <openvino/op/internal/group_normalization_zsplit.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/result.hpp>
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

// Hard ceiling on this bench's own device allocations (num_bufs x (input+output)),
// independent of whatever num_bufs was requested -- a full-resolution activation
// (e.g. 128x4x1088x1920) can otherwise request tens of GB once multiplied by
// num_bufs. Leaves headroom under a 20 GiB target.
constexpr uint64_t kGpuMemBudgetBytes = 18ull * 1024 * 1024 * 1024;

namespace {

int cap_num_bufs_for_budget(int requested_bufs, double per_buf_bytes, uint64_t budget_bytes) {
    int bufs = std::max(1, requested_bufs);
    while (bufs > 1 && double(bufs) * per_buf_bytes > double(budget_bytes)) {
        bufs--;
    }
    const double estimated = double(bufs) * per_buf_bytes;
    if (bufs < requested_bufs) {
        std::cout << "[GROUPNORM_BENCH] num_bufs reduced " << requested_bufs << " -> " << bufs
                  << " to stay under " << (budget_bytes / 1024.0 / 1024.0 / 1024.0) << " GiB budget"
                  << " (estimated " << (estimated / 1024.0 / 1024.0 / 1024.0) << " GiB)" << std::endl;
    }
    if (estimated > double(budget_bytes)) {
        std::cout << "[GROUPNORM_BENCH] WARNING: even num_bufs=1 is estimated at "
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
        std::cout << "[GROUPNORM_BENCH] memory_statistics query failed: " << e.what() << std::endl;
    }
}

std::shared_ptr<ov::Model> build_groupnorm_model(int64_t c, int64_t z, int64_t h, int64_t w,
                                                  int64_t num_groups) {
    auto data = std::make_shared<op::v0::Parameter>(
        element::f16, Shape{1, (size_t)c, (size_t)z, (size_t)h, (size_t)w});
    data->set_friendly_name("data");

    std::mt19937 rng(42);
    std::vector<ov::float16> scale_data((size_t)c);
    std::vector<ov::float16> bias_data((size_t)c);
    for (auto& v : scale_data) v = ov::float16(float(rng() % 100 + 50) / 100.0f);
    for (auto& v : bias_data) v = ov::float16(float(rng() % 100 - 50) / 100.0f);
    auto scale = op::v0::Constant::create(element::f16, Shape{(size_t)c}, scale_data.data());
    scale->set_friendly_name("scale");
    auto bias = op::v0::Constant::create(element::f16, Shape{(size_t)c}, bias_data.data());
    bias->set_friendly_name("bias");

    auto gn = std::make_shared<ov::op::internal::GroupNormalizationZSplit>(
        data, scale, bias, num_groups, 1e-6);
    gn->set_friendly_name("group_norm_zsplit");

    auto result = std::make_shared<op::v0::Result>(gn->output(0));
    return std::make_shared<Model>(OutputVector{result}, ParameterVector{data}, "groupnorm_bench");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <c> <z> <h> <w> [num_groups=32] [iters=50] [warmup=5] [num_bufs=4]"
                  << std::endl;
        return 1;
    }

    try {
        const int64_t c = std::atoll(argv[1]);
        const int64_t z = std::atoll(argv[2]);
        const int64_t h = std::atoll(argv[3]);
        const int64_t w = std::atoll(argv[4]);

        int argi = 5;
        auto next_or = [&](int64_t defv) -> int64_t {
            if (argi < argc) return std::atoll(argv[argi++]);
            return defv;
        };
        const int64_t num_groups = next_or(32);
        const int iters    = (int)next_or(50);
        const int warmup   = (int)next_or(5);
        const int num_bufs = (int)next_or(4);

        std::cout << "=== GroupNorm Benchmark (GroupNormalizationZSplit) ===" << std::endl;
        std::cout << "c=" << c << " z=" << z << " h=" << h << " w=" << w
                  << " num_groups=" << num_groups
                  << " iters=" << iters << " warmup=" << warmup << " bufs=" << num_bufs
                  << std::endl;

        auto model = build_groupnorm_model(c, z, h, w, num_groups);

        Core core;
        auto compiled = core.compile_model(model, "GPU");
        auto remote_ctx = core.get_default_context("GPU");

        // Budget check: num_bufs x (input + output), both f16, same elem count.
        const double n_elems_est = double(c) * z * h * w;
        const double per_buf_bytes_est = n_elems_est * 2.0 * 2.0;  // in + out
        const int capped_bufs = cap_num_bufs_for_budget(num_bufs, per_buf_bytes_est, kGpuMemBudgetBytes);

        std::vector<InferRequest> reqs;
        for (int b = 0; b < capped_bufs; b++) {
            auto req = compiled.create_infer_request();
            auto port = compiled.input();
            // RemoteTensor::data<T>() is not implemented (matches fc_bench's existing
            // pattern) -- uninitialized USM_DEVICE content is fine for a perf-only bench.
            ov::Tensor dev_tensor = remote_ctx.create_tensor(port.get_element_type(), port.get_shape(), {});
            req.set_input_tensor(dev_tensor);
            reqs.push_back(std::move(req));
        }
        std::cout << "Input tensors allocated in USM_DEVICE (VRAM) via RemoteContext." << std::endl;

        for (int i = 0; i < warmup; i++)
            reqs[i % capped_bufs].infer();

        std::vector<double> latencies;
        latencies.reserve(iters);
        for (int i = 0; i < iters; i++) {
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

        // Reduction pass (mean+var, ~4 flop/elem: sum, sumsq, 2 for combine) +
        // normalize+affine pass (~4 flop/elem: sub, mul(inv_std), mul(scale), add(bias)).
        const double n_elems = double(c) * z * h * w;
        const double flops = n_elems * 8.0;
        const double io_bytes = n_elems * 2.0 * 2.0;  // read once + write once, f16
        const double param_bytes = double(c) * 2.0 * 2.0;  // scale+bias, f16
        const double total_bytes = io_bytes + param_bytes;

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
        std::cerr << "[GROUPNORM_BENCH ERROR] " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "[GROUPNORM_BENCH ERROR] unknown exception" << std::endl;
        return 3;
    }
}
