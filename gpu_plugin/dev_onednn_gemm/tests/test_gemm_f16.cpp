// =============================================================================
// GEMM f16 OpenCL Test: Correctness + Performance
//
// Tests C[M,N] = A[M,K] * B[K,N] with half precision.
// Uses OpenCL profiling events for accurate kernel timing.
// Includes L3 cache flush between performance iterations.
//
// B580 f16 peak: 96 TFLOPS
// Required compute: 2*M*N*K FLOPs per GEMM
// =============================================================================

#include <CL/cl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Half-float conversion helpers (IEEE 754 half <-> float)
static uint16_t float_to_half(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint16_t h;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0) {
        h = sign;
    } else if (exp >= 31) {
        h = sign | 0x7C00;
    } else {
        h = sign | (exp << 10) | (mant >> 13);
    }
    return h;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return *(float*)&f;
}

#define CHECK_CL(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d (%s)\n", (err), __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

static std::string load_file(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Build kernel with compile options
static cl_program build_program(cl_context ctx, cl_device_id dev, const char* src, const char* opts) {
    cl_int err;
    const char* srcs[] = {src};
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, nullptr, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        fprintf(stderr, "Build error:\n%s\n", log.data());
        exit(1);
    }
    return prog;
}

struct TestConfig {
    int M, N, K;
    const char* label;
};

void run_test(cl_context ctx, cl_command_queue queue, cl_device_id dev,
              cl_kernel gemm_kernel, cl_kernel flush_kernel,
              cl_mem flush_buf, int flush_count,
              const TestConfig& cfg) {
    cl_int err;
    int M = cfg.M, N = cfg.N, K = cfg.K;
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    printf("\n========================================\n");
    printf("Test: %s\n", cfg.label);
    printf("  A[%d, %d] * B[%d, %d] = C[%d, %d]\n", M, K, K, N, M, N);
    printf("========================================\n");

    // -------------------------------------------------------------------------
    // Initialize matrices with small random values
    // -------------------------------------------------------------------------
    std::vector<uint16_t> h_A(size_A), h_B(size_B), h_C(size_C);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < size_A; i++) h_A[i] = float_to_half(dist(rng));
    for (size_t i = 0; i < size_B; i++) h_B[i] = float_to_half(dist(rng));

    // Create OpenCL buffers
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                size_A * sizeof(uint16_t), h_A.data(), &err);
    CHECK_CL(err, "create d_A");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                size_B * sizeof(uint16_t), h_B.data(), &err);
    CHECK_CL(err, "create d_B");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                size_C * sizeof(uint16_t), nullptr, &err);
    CHECK_CL(err, "create d_C");

    // Set kernel args
    err  = clSetKernelArg(gemm_kernel, 0, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(gemm_kernel, 1, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(gemm_kernel, 2, sizeof(cl_mem), &d_C);
    err |= clSetKernelArg(gemm_kernel, 3, sizeof(int), &M);
    err |= clSetKernelArg(gemm_kernel, 4, sizeof(int), &N);
    err |= clSetKernelArg(gemm_kernel, 5, sizeof(int), &K);
    CHECK_CL(err, "setKernelArg gemm");

    // Compute dispatch sizes
    // WG local size = SG_SIZE * WG_M * WG_N = 16 * 4 * 8 = 512
    const int TILE_M = 32, TILE_N = 32, WG_M = 4, WG_N = 8;
    const int WG_TILE_M = WG_M * TILE_M;  // 128
    const int WG_TILE_N = WG_N * TILE_N;  // 128
    const int SG_SIZE = 16;

    int grid_m = (M + WG_TILE_M - 1) / WG_TILE_M;
    int grid_n = (N + WG_TILE_N - 1) / WG_TILE_N;

    size_t local_size[2]  = {(size_t)(SG_SIZE * WG_M * WG_N), 1};
    size_t global_size[2] = {(size_t)(grid_m * local_size[0]), (size_t)grid_n};

    // -------------------------------------------------------------------------
    // Correctness test
    // -------------------------------------------------------------------------
    printf("  Running correctness test...\n");
    err = clEnqueueNDRangeKernel(queue, gemm_kernel, 2, nullptr,
                                  global_size, local_size, 0, nullptr, nullptr);
    CHECK_CL(err, "enqueue gemm (correctness)");
    clFinish(queue);

    // Read back C
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                              size_C * sizeof(uint16_t), h_C.data(), 0, nullptr, nullptr);
    CHECK_CL(err, "read C");

    // Reference computation (in f32)
    double max_err = 0.0, avg_err = 0.0;
    int check_count = std::min((int)size_C, 10000);  // check subset for speed
    std::mt19937 check_rng(123);
    for (int idx = 0; idx < check_count; idx++) {
        int c_idx = (check_count == (int)size_C) ? idx : (check_rng() % size_C);
        int row = c_idx / N;
        int col = c_idx % N;
        float ref = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = half_to_float(h_A[row * K + k]);
            float b = half_to_float(h_B[k * N + col]);
            ref += a * b;
        }
        float got = half_to_float(h_C[c_idx]);
        double err_val = fabs((double)got - (double)ref);
        double rel_err = err_val / (fabs((double)ref) + 1e-6);
        max_err = std::max(max_err, rel_err);
        avg_err += rel_err;
    }
    avg_err /= check_count;
    printf("  Correctness: max_rel_err=%.6f, avg_rel_err=%.6f [%s]\n",
           max_err, avg_err,
           (max_err < 0.05) ? "PASS" : "FAIL");

    // -------------------------------------------------------------------------
    // Performance test
    // -------------------------------------------------------------------------
    printf("  Running performance test (warmup=10, iterations=200)...\n");

    // Warmup
    for (int i = 0; i < 10; i++) {
        clEnqueueNDRangeKernel(queue, gemm_kernel, 2, nullptr,
                               global_size, local_size, 0, nullptr, nullptr);
    }
    clFinish(queue);

    // Timed iterations with L3 flush between runs
    double total_time_us = 0.0;
    const int NUM_ITERS = 200;
    size_t flush_global = (size_t)flush_count;
    size_t flush_local  = 256;

    for (int i = 0; i < NUM_ITERS; i++) {
        // Flush L3 cache
        clEnqueueNDRangeKernel(queue, flush_kernel, 1, nullptr,
                               &flush_global, &flush_local, 0, nullptr, nullptr);
        clFinish(queue);

        // Run GEMM with profiling event
        cl_event ev;
        err = clEnqueueNDRangeKernel(queue, gemm_kernel, 2, nullptr,
                                      global_size, local_size, 0, nullptr, &ev);
        CHECK_CL(err, "enqueue gemm (perf)");
        clFinish(queue);

        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
        total_time_us += (double)(t_end - t_start) / 1000.0;  // ns → µs
        clReleaseEvent(ev);
    }

    double avg_time_us = total_time_us / NUM_ITERS;
    double avg_time_ms = avg_time_us / 1000.0;

    // Compute metrics
    double flops = 2.0 * M * N * K;
    double tflops_achieved = (flops / (avg_time_us * 1e-6)) / 1e12;
    double peak_tflops = 96.0;  // B580 f16 peak
    double efficiency = (tflops_achieved / peak_tflops) * 100.0;

    printf("\n  +---------------------------+------------------+\n");
    printf("  | Metric                    | Value            |\n");
    printf("  +---------------------------+------------------+\n");
    printf("  | Matrix size               | [%d,%d]*[%d,%d] |\n", M, K, K, N);
    printf("  | Required FLOPs            | %.3f GFLOP      |\n", flops / 1e9);
    printf("  | Avg kernel time           | %.3f ms         |\n", avg_time_ms);
    printf("  | Achieved throughput       | %.3f TFLOPS     |\n", tflops_achieved);
    printf("  | Peak throughput (B580)    | %.1f TFLOPS      |\n", peak_tflops);
    printf("  | Roofline efficiency       | %.2f%%           |\n", efficiency);
    printf("  +---------------------------+------------------+\n");

    // Cleanup buffers
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
}

int main(int argc, char** argv) {
    cl_int err;

    // -------------------------------------------------------------------------
    // Platform / device / context / queue setup
    // -------------------------------------------------------------------------
    // Iterate over all platforms to find one that exposes a GPU device.
    // Systems with multiple OpenCL runtimes (e.g. Intel CPU + Intel GPU)
    // may have the GPU on a platform other than index 0.
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_CL(err, "getPlatformCount");

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CHECK_CL(err, "getPlatforms");

    cl_device_id device = nullptr;
    cl_platform_id gpu_platform = nullptr;
    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
        cl_int ret = clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (ret == CL_SUCCESS) { gpu_platform = platforms[pi]; break; }
        device = nullptr;
    }
    if (!device) { fprintf(stderr, "No GPU device found on any platform\n"); return 1; }

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    printf("Device: %s\n", dev_name);

    // Pass the platform explicitly to avoid context/platform mismatch
    // when the GPU is on a non-default platform.
    cl_context_properties ctx_props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)gpu_platform, 0
    };
    cl_context ctx = clCreateContext(ctx_props, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "createContext");

    // Queue with profiling enabled
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, props, &err);
    CHECK_CL(err, "createQueue");

    // -------------------------------------------------------------------------
    // Build programs
    // -------------------------------------------------------------------------
    std::string kernel_src = load_file("gemm_f16.cl");
    const char* build_opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -Dcl_intel_subgroup_matrix_multiply_accumulate -Dcl_intel_subgroup_2d_block_io";
    cl_program prog = build_program(ctx, device, kernel_src.c_str(), build_opts);

    cl_kernel gemm_kernel = clCreateKernel(prog, "gemm_f16", &err);
    CHECK_CL(err, "createKernel gemm_f16");
    cl_kernel flush_kernel = clCreateKernel(prog, "flush_l3", &err);
    CHECK_CL(err, "createKernel flush_l3");

    // -------------------------------------------------------------------------
    // Create L3 flush buffer (>= 32 MB for B580)
    // 32 MB / 4 bytes = 8M floats
    // -------------------------------------------------------------------------
    int flush_count = 8 * 1024 * 1024;  // 8M floats = 32 MB
    cl_mem flush_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                      (size_t)flush_count * sizeof(float), nullptr, &err);
    CHECK_CL(err, "create flush_buf");
    err  = clSetKernelArg(flush_kernel, 0, sizeof(cl_mem), &flush_buf);
    err |= clSetKernelArg(flush_kernel, 1, sizeof(int), &flush_count);
    CHECK_CL(err, "setKernelArg flush");

    // -------------------------------------------------------------------------
    // Run tests for specified configurations
    // -------------------------------------------------------------------------
    TestConfig configs[] = {
        {2048, 2048, 2560, "A[2048,2560] * B[2560,2048] = C[2048,2048]"},
        {2048, 2048, 4096, "A[2048,4096] * B[4096,2048] = C[2048,2048]"},
    };

    for (auto& cfg : configs) {
        run_test(ctx, queue, device, gemm_kernel, flush_kernel, flush_buf, flush_count, cfg);
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    clReleaseMemObject(flush_buf);
    clReleaseKernel(gemm_kernel);
    clReleaseKernel(flush_kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    printf("\nDone.\n");
    return 0;
}
