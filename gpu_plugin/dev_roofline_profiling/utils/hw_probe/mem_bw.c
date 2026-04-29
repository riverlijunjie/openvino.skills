// mem_bw.c — measure achieved DRAM bandwidth on a GPU via OpenCL (per SKILL §4).
//
// Strategy: allocate one BIG src + dst buffer (>= 1 GiB each so it cannot fit in L2),
// run a streaming copy kernel that touches every byte exactly once per launch,
// average over N iterations, report achieved bandwidth = 2 × bytes / time.
//
// We use float4 loads and a stride pattern that prevents L2 reuse across iters
// by rotating which pair of buffers is used each iteration.
//
// Build: gcc mem_bw.c -lOpenCL -O2 -o mem_bw
// Usage: ./mem_bw [size_MiB=1024] [iters=20]

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHK(x) do { cl_int e=(x); if(e!=CL_SUCCESS){fprintf(stderr,"OCL err %d at %d\n",e,__LINE__); exit(1);} } while(0)

static const char* KSRC =
"__kernel void init_rand(__global float4* buf, uint seed) {\n"
"  size_t i = get_global_id(0);\n"
"  uint x = (uint)i * 2654435761u + seed; uint y = x ^ (x>>16); uint z = y * 2246822519u; uint w = z ^ (z>>13);\n"
"  buf[i] = (float4)(as_float(x|1u), as_float(y|1u), as_float(z|1u), as_float(w|1u));\n"
"}\n"
"__kernel void copy_f4(__global const float4* src, __global float4* dst) {\n"
"  size_t i = get_global_id(0);\n"
"  dst[i] = src[i];\n"
"}\n"
"__kernel void read_f4(__global const float4* src, __global float* sink) {\n"
"  size_t i = get_global_id(0);\n"
"  float4 v = src[i];\n"
"  if ((v.x + v.y + v.z + v.w) == -1e30f) sink[i] = v.x;\n"
"}\n";

int main(int argc, char** argv) {
    size_t MiB = argc > 1 ? (size_t)atoi(argv[1]) : 1024;
    int iters  = argc > 2 ? atoi(argv[2]) : 20;
    size_t bytes = MiB * 1024ULL * 1024ULL;
    size_t nf4   = bytes / sizeof(cl_float4);

    cl_uint np; CHK(clGetPlatformIDs(0, NULL, &np));
    cl_platform_id* ps = calloc(np, sizeof(*ps));
    clGetPlatformIDs(np, ps, NULL);
    cl_platform_id pf = 0; cl_device_id dv = 0;
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
            clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &dv, NULL);
            pf = ps[i]; break;
        }
    }
    if (!dv) { fprintf(stderr, "no GPU\n"); return 1; }
    free(ps);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dv, NULL, NULL, &err); CHK(err);
    cl_command_queue_properties qprops = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue q = clCreateCommandQueue(ctx, dv, qprops, &err); CHK(err);

    // Allocate three buffers: a, b, c. Iteration k copies (k%3) -> ((k+1)%3).
    cl_mem bufs[3];
    for (int i = 0; i < 3; ++i) {
        bufs[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "alloc %zu MiB failed: %d (try smaller --size)\n", MiB, err); return 1; }
    }
    // Fill each buffer with NON-ZERO INCOMPRESSIBLE bytes via a kernel.
    // BMG/Xe2 uses lossless framebuffer compression and short-circuits zero
    // pages and constant-fill patterns from a small metadata cache, so a
    // simple clEnqueueFillBuffer(0) or fill(constant) gives ~2 TB/s and
    // ~900 GB/s respectively — neither hits real GDDR6 BW. Writing pseudo-
    // random per-element values defeats the compressor.
    cl_mem sink = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_float)*nf4, NULL, &err); CHK(err);

    cl_program prg = clCreateProgramWithSource(ctx, 1, &KSRC, NULL, &err); CHK(err);
    if (clBuildProgram(prg, 1, &dv, "-cl-mad-enable", NULL, NULL) != CL_SUCCESS) {
        size_t lg; clGetProgramBuildInfo(prg, dv, CL_PROGRAM_BUILD_LOG, 0, NULL, &lg);
        char* l = malloc(lg); clGetProgramBuildInfo(prg, dv, CL_PROGRAM_BUILD_LOG, lg, l, NULL);
        fprintf(stderr, "build: %s\n", l); return 1;
    }
    cl_kernel ki = clCreateKernel(prg, "init_rand", &err); CHK(err);
    for (int i = 0; i < 3; ++i) {
        cl_uint seed = 0xC0FFEE + i * 17u;
        clSetKernelArg(ki, 0, sizeof(cl_mem), &bufs[i]);
        clSetKernelArg(ki, 1, sizeof(seed), &seed);
        CHK(clEnqueueNDRangeKernel(q, ki, 1, NULL, &nf4, NULL, 0, NULL, NULL));
    }
    clFinish(q);
    clReleaseKernel(ki);
    cl_kernel kc = clCreateKernel(prg, "copy_f4", &err); CHK(err);
    cl_kernel kr = clCreateKernel(prg, "read_f4", &err); CHK(err);

    // warm up
    for (int i = 0; i < 3; ++i) {
        clSetKernelArg(kc, 0, sizeof(cl_mem), &bufs[i%3]);
        clSetKernelArg(kc, 1, sizeof(cl_mem), &bufs[(i+1)%3]);
        clEnqueueNDRangeKernel(q, kc, 1, NULL, &nf4, NULL, 0, NULL, NULL);
    }
    clFinish(q);

    // Measure copy bandwidth (read + write = 2 × bytes per iter)
    cl_event evs[64];
    if (iters > 64) iters = 64;
    for (int i = 0; i < iters; ++i) {
        clSetKernelArg(kc, 0, sizeof(cl_mem), &bufs[i%3]);
        clSetKernelArg(kc, 1, sizeof(cl_mem), &bufs[(i+1)%3]);
        CHK(clEnqueueNDRangeKernel(q, kc, 1, NULL, &nf4, NULL, 0, NULL, &evs[i]));
    }
    clFinish(q);
    double tot_ns = 0;
    for (int i = 0; i < iters; ++i) {
        cl_ulong s, e; clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
        tot_ns += (double)(e - s);
        clReleaseEvent(evs[i]);
    }
    double avg_s = (tot_ns / iters) / 1e9;
    double bw_copy = (2.0 * bytes) / avg_s / 1e9;

    // Measure pure-read bandwidth (1 × bytes)
    for (int i = 0; i < iters; ++i) {
        clSetKernelArg(kr, 0, sizeof(cl_mem), &bufs[i%3]);
        clSetKernelArg(kr, 1, sizeof(cl_mem), &sink);
        CHK(clEnqueueNDRangeKernel(q, kr, 1, NULL, &nf4, NULL, 0, NULL, &evs[i]));
    }
    clFinish(q);
    tot_ns = 0;
    for (int i = 0; i < iters; ++i) {
        cl_ulong s, e; clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
        tot_ns += (double)(e - s); clReleaseEvent(evs[i]);
    }
    avg_s = (tot_ns / iters) / 1e9;
    double bw_read = (double)bytes / avg_s / 1e9;

    char dn[256]; clGetDeviceInfo(dv, CL_DEVICE_NAME, sizeof(dn), dn, NULL);
    printf("Device          : %s\n", dn);
    printf("Buffer per iter : %zu MiB (×3 buffers, rotated to defeat L2 reuse)\n", MiB);
    printf("Iterations      : %d\n", iters);
    printf("Copy  BW (R+W)  : %.2f GB/s\n", bw_copy);
    printf("Read  BW        : %.2f GB/s\n", bw_read);

    clReleaseKernel(kc); clReleaseKernel(kr); clReleaseProgram(prg);
    for (int i = 0; i < 3; ++i) clReleaseMemObject(bufs[i]);
    clReleaseMemObject(sink);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}
