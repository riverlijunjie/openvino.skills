// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone test for mlp_gate_up kernel (moe_decode.cl)
// Tests correctness (f16 reference) and performance on PTL 12Xe GPU.
//
// Build (Windows): cl /EHsc /O2 /std:c++17 moe_decode_test.cpp /I<OpenCL_Headers_path> OpenCL.lib
// Build (Linux):   g++ -O2 -std=c++17 moe_decode_test.cpp -lOpenCL -o moe_decode_test

#include <CL/cl.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ===================== Half-float conversion helpers =====================
// IEEE 754 half-precision (binary16)
static uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp_val = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exp_val > 15) {
        return sign | 0x7C00;  // inf
    } else if (exp_val < -14) {
        // subnormal or zero
        if (exp_val < -24) return sign;
        mantissa |= 0x800000;
        int shift = -exp_val - 1;
        mantissa >>= shift;
        return sign | (mantissa >> 13);
    }
    return sign | ((exp_val + 15) << 10) | (mantissa >> 13);
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp_val = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exp_val == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        // subnormal
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exp_val--;
        }
        exp_val++;
        mantissa &= 0x3FF;
    } else if (exp_val == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }

    uint32_t result = sign | ((exp_val + 112) << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ===================== Model parameters =====================
static constexpr int MAX_TOPK = 8;
static constexpr int EXPERT_NUM = 128;
static constexpr int HIDDEN_SIZE = 2048;
static constexpr int INTERMEDIATE_SIZE = 768;
static constexpr int GROUP_SIZE = 128;  // GATE_UP_GROUP_SIZE
static constexpr int NUM_GROUPS = HIDDEN_SIZE / GROUP_SIZE;  // 16

// Buffer sizes
static constexpr size_t EXPERT_WEI_SIZE = (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE / 2;  // u4: N*K/2 bytes
static constexpr size_t EXPERT_SCALE_SIZE = (size_t)INTERMEDIATE_SIZE * NUM_GROUPS;      // f16: N*num_groups
static constexpr size_t EXPERT_ZP_SIZE = (size_t)INTERMEDIATE_SIZE * NUM_GROUPS / 2;     // u4: N*num_groups/2 bytes

static constexpr size_t TOTAL_WEI_SIZE = EXPERT_NUM * EXPERT_WEI_SIZE;
static constexpr size_t TOTAL_SCALE_SIZE = EXPERT_NUM * EXPERT_SCALE_SIZE;
static constexpr size_t TOTAL_ZP_SIZE = EXPERT_NUM * EXPERT_ZP_SIZE;

// ===================== Utility: Read kernel source =====================
static std::string read_kernel_source(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open kernel file: %s\n", path.c_str());
        exit(1);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ===================== OpenCL error check =====================
#define CL_CHECK(err)                                                          \
    do {                                                                        \
        cl_int _e = (err);                                                     \
        if (_e != CL_SUCCESS) {                                                \
            fprintf(stderr, "OpenCL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
            exit(1);                                                            \
        }                                                                      \
    } while (0)

// ===================== Reference implementation (f16 precision) =====================
// Dequantize u4 weight: val = (nibble - zp) * scale
// Weight layout: [N, K] packed as u4, so each byte holds 2 consecutive K-elements
// Scale layout: [N, num_groups] stored per-row then per-group
// ZP layout: [N, num_groups] packed as u4, 2 N-elements per byte

static void reference_gate_up(
    const int* expert_list,
    const std::vector<uint8_t>& gate_weight_all,   // [128][N*K/2]
    const std::vector<uint16_t>& gate_scale_all,   // [128][N*num_groups]
    const std::vector<uint8_t>& gate_zp_all,       // [128][N*num_groups/2]
    const std::vector<uint8_t>& up_weight_all,
    const std::vector<uint16_t>& up_scale_all,
    const std::vector<uint8_t>& up_zp_all,
    const std::vector<uint16_t>& x_f16,            // [HIDDEN_SIZE]
    std::vector<uint16_t>& y_f16)                  // [MAX_TOPK * INTERMEDIATE_SIZE]
{
    // Convert x to float for accumulation
    std::vector<float> x_f(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        x_f[i] = half_to_float(x_f16[i]);
    }

    for (int slot = 0; slot < MAX_TOPK; slot++) {
        int expert_id = expert_list[slot];
        float* out = new float[INTERMEDIATE_SIZE];

        // --- UP projection (no activation) ---
        const uint8_t* up_w = up_weight_all.data() + (size_t)expert_id * EXPERT_WEI_SIZE;
        const uint16_t* up_s = up_scale_all.data() + (size_t)expert_id * EXPERT_SCALE_SIZE;
        const uint8_t* up_z = up_zp_all.data() + (size_t)expert_id * EXPERT_ZP_SIZE;

        for (int n = 0; n < INTERMEDIATE_SIZE; n++) {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                int group_idx = k / GROUP_SIZE;
                // Weight: row n, col k. Packed u4: byte at [n * K/2 + k/2]
                int byte_idx = n * (HIDDEN_SIZE / 2) + k / 2;
                uint8_t byte_val = up_w[byte_idx];
                float w_val;
                if (k % 2 == 0) {
                    w_val = (float)(byte_val & 0x0F);
                } else {
                    w_val = (float)(byte_val >> 4);
                }

                // Scale: [N, num_groups] -- scale for row n, group group_idx
                float scale = half_to_float(up_s[n + group_idx * INTERMEDIATE_SIZE]);

                // ZP: [N, num_groups] packed u4, 2 N per byte
                // The zp layout matches the kernel: zps + n/2, offset = group_idx * N/2
                int zp_byte_idx = n / 2 + group_idx * (INTERMEDIATE_SIZE / 2);
                uint8_t zp_byte = up_z[zp_byte_idx];
                float zp;
                if (n % 2 == 0) {
                    zp = (float)(zp_byte & 0x0F);
                } else {
                    zp = (float)(zp_byte >> 4);
                }

                sum += x_f[k] * (w_val - zp) * scale;
            }
            out[n] = sum;
        }

        // --- GATE projection (with SwiGLU) ---
        const uint8_t* gate_w = gate_weight_all.data() + (size_t)expert_id * EXPERT_WEI_SIZE;
        const uint16_t* gate_s = gate_scale_all.data() + (size_t)expert_id * EXPERT_SCALE_SIZE;
        const uint8_t* gate_z = gate_zp_all.data() + (size_t)expert_id * EXPERT_ZP_SIZE;

        for (int n = 0; n < INTERMEDIATE_SIZE; n++) {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                int group_idx = k / GROUP_SIZE;
                int byte_idx = n * (HIDDEN_SIZE / 2) + k / 2;
                uint8_t byte_val = gate_w[byte_idx];
                float w_val;
                if (k % 2 == 0) {
                    w_val = (float)(byte_val & 0x0F);
                } else {
                    w_val = (float)(byte_val >> 4);
                }

                float scale = half_to_float(gate_s[n + group_idx * INTERMEDIATE_SIZE]);

                int zp_byte_idx = n / 2 + group_idx * (INTERMEDIATE_SIZE / 2);
                uint8_t zp_byte = gate_z[zp_byte_idx];
                float zp;
                if (n % 2 == 0) {
                    zp = (float)(zp_byte & 0x0F);
                } else {
                    zp = (float)(zp_byte >> 4);
                }

                sum += x_f[k] * (w_val - zp) * scale;
            }
            // SwiGLU: out[n] = up[n] * swish(gate[n])
            float swish = sum / (1.0f + expf(-sum));
            out[n] = out[n] * swish;
        }

        // Write output
        for (int n = 0; n < INTERMEDIATE_SIZE; n++) {
            y_f16[slot * INTERMEDIATE_SIZE + n] = float_to_half(out[n]);
        }
        delete[] out;
    }
}

// ===================== Main =====================
int main(int argc, char** argv) {
    printf("=== MoE Gate-Up Kernel Test (PTL 12Xe) ===\n");
    printf("Config: %d experts, top-%d, hidden=%d, intermediate=%d, group=%d\n",
           EXPERT_NUM, MAX_TOPK, HIDDEN_SIZE, INTERMEDIATE_SIZE, GROUP_SIZE);

    // --- Find kernel file path ---
    std::string kernel_path = "moe_decode.cl";
    if (argc > 1) kernel_path = argv[1];

    int num_warmup = 10;
    int num_iters = 500;
    if (argc > 2) num_iters = atoi(argv[2]);

    // --- Initialize OpenCL ---
    cl_int err;
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found!\n");
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    // Find Intel GPU platform
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    for (auto& p : platforms) {
        cl_uint num_devices;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
        for (auto& d : devices) {
            char vendor[256];
            clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
            if (strstr(vendor, "Intel") != nullptr) {
                platform = p;
                device = d;
                break;
            }
        }
        if (device) break;
    }

    if (!device) {
        fprintf(stderr, "No Intel GPU device found!\n");
        return 1;
    }

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    printf("Device: %s\n", dev_name);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    // Enable profiling for accurate kernel timing
    cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queue_props, &err);
    CL_CHECK(err);

    // --- Build kernel ---
    std::string source = read_kernel_source(kernel_path);
    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        fprintf(stderr, "Build error:\n%s\n", log.c_str());
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "mlp_gate_up", &err);
    CL_CHECK(err);
    printf("Kernel built successfully.\n");

    // --- Build L3 cache flush kernel ---
    // PTL 12Xe L3 cache ~18 MB; use 32 MB buffer to ensure full flush
    static constexpr size_t FLUSH_SIZE = 32 * 1024 * 1024;  // 32 MB
    const char* flush_src = R"CL(
        __kernel void cache_flush(__global float* buf, int n) {
            int gid = get_global_id(0);
            if (gid < n) {
                buf[gid] = buf[gid] * 1.0001f + 0.0001f;
            }
        }
    )CL";
    const char* flush_src_ptr = flush_src;
    size_t flush_src_len = strlen(flush_src);
    cl_program flush_program = clCreateProgramWithSource(context, 1, &flush_src_ptr, &flush_src_len, &err);
    CL_CHECK(err);
    CL_CHECK(clBuildProgram(flush_program, 1, &device, nullptr, nullptr, nullptr));
    cl_kernel flush_kernel = clCreateKernel(flush_program, "cache_flush", &err);
    CL_CHECK(err);

    cl_mem buf_flush = clCreateBuffer(context, CL_MEM_READ_WRITE, FLUSH_SIZE, nullptr, &err);
    CL_CHECK(err);
    int flush_n = (int)(FLUSH_SIZE / sizeof(float));
    CL_CHECK(clSetKernelArg(flush_kernel, 0, sizeof(cl_mem), &buf_flush));
    CL_CHECK(clSetKernelArg(flush_kernel, 1, sizeof(int), &flush_n));
    size_t flush_global = (size_t)flush_n;
    size_t flush_local = 256;
    printf("L3 cache flush buffer: %zu MB\n", FLUSH_SIZE / (1024 * 1024));

    // --- Generate random test data ---
    std::mt19937 rng(42);

    // Expert list: 8 unique random indices in [0, 127]
    std::vector<int> expert_list(MAX_TOPK);
    {
        std::vector<int> all_experts(EXPERT_NUM);
        std::iota(all_experts.begin(), all_experts.end(), 0);
        std::shuffle(all_experts.begin(), all_experts.end(), rng);
        std::copy(all_experts.begin(), all_experts.begin() + MAX_TOPK, expert_list.begin());
    }
    printf("Expert IDs: ");
    for (int i = 0; i < MAX_TOPK; i++) printf("%d ", expert_list[i]);
    printf("\n");

    // Generate f16 weights, then compress to u4/scale/zp
    // For testing we generate random u4 values (0-15) directly for weights and zp,
    // and random f16 scales
    std::uniform_int_distribution<int> dist_u4(0, 15);
    std::uniform_real_distribution<float> dist_scale(0.001f, 0.1f);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);

    // Gate weight/scale/zp
    std::vector<uint8_t> gate_weight(TOTAL_WEI_SIZE);
    std::vector<uint16_t> gate_scale(TOTAL_SCALE_SIZE);
    std::vector<uint8_t> gate_zp(TOTAL_ZP_SIZE);

    // Up weight/scale/zp
    std::vector<uint8_t> up_weight(TOTAL_WEI_SIZE);
    std::vector<uint16_t> up_scale(TOTAL_SCALE_SIZE);
    std::vector<uint8_t> up_zp(TOTAL_ZP_SIZE);

    // Only generate data for experts that are actually used (saves time)
    for (int slot = 0; slot < MAX_TOPK; slot++) {
        int eid = expert_list[slot];

        // Gate weight: random u4 packed
        uint8_t* gw = gate_weight.data() + (size_t)eid * EXPERT_WEI_SIZE;
        for (size_t i = 0; i < EXPERT_WEI_SIZE; i++) {
            int lo = dist_u4(rng);
            int hi = dist_u4(rng);
            gw[i] = (uint8_t)((hi << 4) | lo);
        }
        // Gate scale: random f16
        uint16_t* gs = gate_scale.data() + (size_t)eid * EXPERT_SCALE_SIZE;
        for (size_t i = 0; i < EXPERT_SCALE_SIZE; i++) {
            gs[i] = float_to_half(dist_scale(rng));
        }
        // Gate zp: random u4 packed
        uint8_t* gz = gate_zp.data() + (size_t)eid * EXPERT_ZP_SIZE;
        for (size_t i = 0; i < EXPERT_ZP_SIZE; i++) {
            int lo = dist_u4(rng);
            int hi = dist_u4(rng);
            gz[i] = (uint8_t)((hi << 4) | lo);
        }

        // Up weight/scale/zp (same structure)
        uint8_t* uw = up_weight.data() + (size_t)eid * EXPERT_WEI_SIZE;
        for (size_t i = 0; i < EXPERT_WEI_SIZE; i++) {
            int lo = dist_u4(rng);
            int hi = dist_u4(rng);
            uw[i] = (uint8_t)((hi << 4) | lo);
        }
        uint16_t* us = up_scale.data() + (size_t)eid * EXPERT_SCALE_SIZE;
        for (size_t i = 0; i < EXPERT_SCALE_SIZE; i++) {
            us[i] = float_to_half(dist_scale(rng));
        }
        uint8_t* uz = up_zp.data() + (size_t)eid * EXPERT_ZP_SIZE;
        for (size_t i = 0; i < EXPERT_ZP_SIZE; i++) {
            int lo = dist_u4(rng);
            int hi = dist_u4(rng);
            uz[i] = (uint8_t)((hi << 4) | lo);
        }
    }

    // Input x: random f16
    std::vector<uint16_t> x_f16(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        x_f16[i] = float_to_half(dist_x(rng));
    }

    // Output y: zeros
    std::vector<uint16_t> y_gpu(MAX_TOPK * INTERMEDIATE_SIZE, 0);

    // --- Create OpenCL buffers ---
    cl_mem buf_expert_list = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            MAX_TOPK * sizeof(int), expert_list.data(), &err);
    CL_CHECK(err);

    cl_mem buf_gate_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            TOTAL_WEI_SIZE, gate_weight.data(), &err);
    CL_CHECK(err);
    cl_mem buf_gate_scale = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           TOTAL_SCALE_SIZE * sizeof(uint16_t), gate_scale.data(), &err);
    CL_CHECK(err);
    cl_mem buf_gate_zp = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        TOTAL_ZP_SIZE, gate_zp.data(), &err);
    CL_CHECK(err);

    cl_mem buf_up_weight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          TOTAL_WEI_SIZE, up_weight.data(), &err);
    CL_CHECK(err);
    cl_mem buf_up_scale = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         TOTAL_SCALE_SIZE * sizeof(uint16_t), up_scale.data(), &err);
    CL_CHECK(err);
    cl_mem buf_up_zp = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      TOTAL_ZP_SIZE, up_zp.data(), &err);
    CL_CHECK(err);

    cl_mem buf_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  HIDDEN_SIZE * sizeof(uint16_t), x_f16.data(), &err);
    CL_CHECK(err);

    cl_mem buf_y = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  MAX_TOPK * INTERMEDIATE_SIZE * sizeof(uint16_t), nullptr, &err);
    CL_CHECK(err);

    // --- Set kernel arguments ---
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_expert_list));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_gate_weight));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_gate_scale));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_gate_zp));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_up_weight));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_up_scale));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &buf_up_zp));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &buf_x));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &buf_y));

    // --- Dispatch dimensions ---
    // global: [EXPERTS_PER_TOKEN, SUBGROUP_SIZE, INTERMEDIATE_SIZE/N_BLOCK] = [8, 32, 192]
    // local:  [1, SUBGROUP_SIZE, SUBGROUP_NUM] = [1, 32, 8]
    size_t global_work_size[3] = {MAX_TOPK, 32, (size_t)(INTERMEDIATE_SIZE / 4)};
    size_t local_work_size[3] = {1, 32, 8};

    // --- Warmup (with cache flush) ---
    printf("Running %d warmup iterations...\n", num_warmup);
    for (int i = 0; i < num_warmup; i++) {
        CL_CHECK(clEnqueueNDRangeKernel(queue, flush_kernel, 1, nullptr,
                                        &flush_global, &flush_local, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                        global_work_size, local_work_size, 0, nullptr, nullptr));
    }
    CL_CHECK(clFinish(queue));

    // --- Performance measurement with profiling events and cache flush ---
    printf("Running %d iterations for performance (with L3 cache flush)...\n", num_iters);
    std::vector<cl_event> events(num_iters);
    for (int i = 0; i < num_iters; i++) {
        // Flush L3 cache before each kernel execution
        CL_CHECK(clEnqueueNDRangeKernel(queue, flush_kernel, 1, nullptr,
                                        &flush_global, &flush_local, 0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));
        // Run the actual kernel with profiling event
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr,
                                        global_work_size, local_work_size, 0, nullptr, &events[i]));
        CL_CHECK(clFinish(queue));
    }

    // Collect profiling data
    double total_kernel_ns = 0;
    double min_ns = 1e18, max_ns = 0;
    for (int i = 0; i < num_iters; i++) {
        cl_ulong t_start_ns, t_end_ns;
        CL_CHECK(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &t_start_ns, nullptr));
        CL_CHECK(clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &t_end_ns, nullptr));
        double elapsed_ns = (double)(t_end_ns - t_start_ns);
        total_kernel_ns += elapsed_ns;
        if (elapsed_ns < min_ns) min_ns = elapsed_ns;
        if (elapsed_ns > max_ns) max_ns = elapsed_ns;
        clReleaseEvent(events[i]);
    }

    double avg_us = total_kernel_ns / num_iters / 1000.0;
    double min_us = min_ns / 1000.0;
    double max_us = max_ns / 1000.0;
    printf("Performance (profiling API, with cache flush):\n");
    printf("  Average: %.2f us/iter\n", avg_us);
    printf("  Min:     %.2f us\n", min_us);
    printf("  Max:     %.2f us\n", max_us);
    printf("  Total:   %.2f ms for %d iters\n", total_kernel_ns / 1e6, num_iters);

    // Compute bandwidth: read weights for 8 experts + input + write output
    // Weights per expert: gate(N*K/2) + up(N*K/2) + scales(2*N*groups*2) + zps(2*N*groups/2)
    double bytes_per_expert = EXPERT_WEI_SIZE * 2.0 + EXPERT_SCALE_SIZE * 2 * 2.0 + EXPERT_ZP_SIZE * 2.0;
    double total_bytes = bytes_per_expert * MAX_TOPK + HIDDEN_SIZE * 2.0 + MAX_TOPK * INTERMEDIATE_SIZE * 2.0;
    double bandwidth_gbps = (total_bytes / (avg_us * 1e-6)) / 1e9;
    double bandwidth_min = (total_bytes / (max_us * 1e-6)) / 1e9;  // min BW = max time
    double bandwidth_max = (total_bytes / (min_us * 1e-6)) / 1e9;  // max BW = min time
    printf("Effective bandwidth (avg): %.2f GB/s (%.2f MB read per call)\n", bandwidth_gbps, total_bytes / 1e6);
    printf("Effective bandwidth (min/max): %.2f / %.2f GB/s\n", bandwidth_min, bandwidth_max);

    // --- Read back result ---
    CL_CHECK(clEnqueueReadBuffer(queue, buf_y, CL_TRUE, 0,
                                 MAX_TOPK * INTERMEDIATE_SIZE * sizeof(uint16_t),
                                 y_gpu.data(), 0, nullptr, nullptr));

    // --- Reference computation ---
    printf("Computing reference (CPU, f32 accumulation)...\n");
    std::vector<uint16_t> y_ref(MAX_TOPK * INTERMEDIATE_SIZE, 0);
    reference_gate_up(expert_list.data(),
                      gate_weight, gate_scale, gate_zp,
                      up_weight, up_scale, up_zp,
                      x_f16, y_ref);

    // --- Verify correctness ---
    int num_mismatch = 0;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    int max_err_idx = 0;
    for (int i = 0; i < MAX_TOPK * INTERMEDIATE_SIZE; i++) {
        float gpu_val = half_to_float(y_gpu[i]);
        float ref_val = half_to_float(y_ref[i]);
        float abs_err = fabsf(gpu_val - ref_val);
        float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_err_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }

        // Tolerance: u4 quantized GEMV with f16 has significant precision loss
        if (rel_err > 0.1f && abs_err > 0.5f) {
            num_mismatch++;
            if (num_mismatch <= 10) {
                printf("  MISMATCH [%d] (slot=%d, n=%d): gpu=%.4f ref=%.4f abs_err=%.4f rel_err=%.4f\n",
                       i, i / INTERMEDIATE_SIZE, i % INTERMEDIATE_SIZE, gpu_val, ref_val, abs_err, rel_err);
            }
        }
    }

    printf("\n--- Correctness Report ---\n");
    printf("Total outputs: %d\n", MAX_TOPK * INTERMEDIATE_SIZE);
    printf("Mismatches (rel>10%% AND abs>0.5): %d\n", num_mismatch);
    printf("Max absolute error: %.6f (at index %d)\n", max_abs_err, max_err_idx);
    printf("Max relative error: %.6f\n", max_rel_err);

    if (num_mismatch == 0) {
        printf("PASS: All outputs match within tolerance.\n");
    } else {
        printf("FAIL: %d outputs exceed tolerance.\n", num_mismatch);
    }

    // Sample outputs
    printf("\nSample outputs (first 8 of slot 0):\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] gpu=%.4f  ref=%.4f\n", i, half_to_float(y_gpu[i]), half_to_float(y_ref[i]));
    }

    // --- Cleanup ---
    clReleaseMemObject(buf_flush);
    clReleaseKernel(flush_kernel);
    clReleaseProgram(flush_program);
    clReleaseMemObject(buf_expert_list);
    clReleaseMemObject(buf_gate_weight);
    clReleaseMemObject(buf_gate_scale);
    clReleaseMemObject(buf_gate_zp);
    clReleaseMemObject(buf_up_weight);
    clReleaseMemObject(buf_up_scale);
    clReleaseMemObject(buf_up_zp);
    clReleaseMemObject(buf_x);
    clReleaseMemObject(buf_y);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\nDone.\n");
    return (num_mismatch == 0) ? 0 : 1;
}
