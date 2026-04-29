// gpu_info.c — minimal OpenCL device-info probe (per SKILL §4).
// Prints fields we cross-check against clinfo / Intel arch spec:
//   driver, OpenCL version, max compute units (Xe-cores × EUs), max clock,
//   subgroup sizes, SLM size, max work-group size, global mem, cache size.
//
// Build: gcc gpu_info.c -lOpenCL -O2 -o gpu_info
// Usage: ./gpu_info [device_index]

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHK(x) do { cl_int e=(x); if(e!=CL_SUCCESS){fprintf(stderr,"OCL err %d at %d\n",e,__LINE__); exit(1);} } while(0)

static void str(cl_device_id d, cl_device_info p, const char* lbl) {
    char buf[2048] = {0};
    clGetDeviceInfo(d, p, sizeof(buf), buf, NULL);
    printf("  %-32s : %s\n", lbl, buf);
}
static void u32(cl_device_id d, cl_device_info p, const char* lbl) {
    cl_uint v = 0;
    clGetDeviceInfo(d, p, sizeof(v), &v, NULL);
    printf("  %-32s : %u\n", lbl, v);
}
static void u64(cl_device_id d, cl_device_info p, const char* lbl) {
    cl_ulong v = 0;
    clGetDeviceInfo(d, p, sizeof(v), &v, NULL);
    printf("  %-32s : %llu  (%.2f MiB)\n", lbl, (unsigned long long)v, v/1048576.0);
}
static void sz(cl_device_id d, cl_device_info p, const char* lbl) {
    size_t v = 0;
    clGetDeviceInfo(d, p, sizeof(v), &v, NULL);
    printf("  %-32s : %zu\n", lbl, v);
}

int main(int argc, char** argv) {
    int didx = argc > 1 ? atoi(argv[1]) : 0;
    cl_uint np = 0; CHK(clGetPlatformIDs(0, NULL, &np));
    cl_platform_id* ps = calloc(np, sizeof(*ps));
    CHK(clGetPlatformIDs(np, ps, NULL));
    // pick first GPU platform
    cl_platform_id pf = 0; cl_device_id dv = 0;
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
            cl_device_id* ds = calloc(nd, sizeof(*ds));
            clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, nd, ds, NULL);
            int idx = didx < (int)nd ? didx : 0;
            pf = ps[i]; dv = ds[idx]; free(ds); break;
        }
    }
    if (!dv) { fprintf(stderr, "no GPU device\n"); return 1; }

    char pbuf[256]; clGetPlatformInfo(pf, CL_PLATFORM_NAME, sizeof(pbuf), pbuf, NULL);
    printf("Platform: %s\n", pbuf);
    str(dv, CL_DEVICE_NAME, "Device name");
    str(dv, CL_DEVICE_VERSION, "Device version");
    str(dv, CL_DRIVER_VERSION, "Driver version");
    u32(dv, CL_DEVICE_MAX_COMPUTE_UNITS, "Compute units (Xe×EU)");
    u32(dv, CL_DEVICE_MAX_CLOCK_FREQUENCY, "Max clock (MHz)");
    sz (dv, CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max work-group size");
    u64(dv, CL_DEVICE_LOCAL_MEM_SIZE, "Local mem (SLM)");
    u64(dv, CL_DEVICE_GLOBAL_MEM_SIZE, "Global mem");
    u64(dv, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "Global mem cache (L3)");
    u32(dv, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "Cache line (B)");
    u64(dv, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Max mem alloc");

    // subgroup sizes
    size_t bytes = 0;
    if (clGetDeviceInfo(dv, 0x4108 /*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/, 0, NULL, &bytes) == CL_SUCCESS && bytes) {
        size_t* sgs = malloc(bytes);
        clGetDeviceInfo(dv, 0x4108, bytes, sgs, NULL);
        printf("  %-32s :", "Subgroup sizes (Intel)");
        for (size_t i = 0; i < bytes/sizeof(size_t); ++i) printf(" %zu", sgs[i]);
        printf("\n");
        free(sgs);
    }
    // Intel Xe-core / threads-per-EU (where supported)
    cl_uint v;
    if (clGetDeviceInfo(dv, 0x4205 /*CL_DEVICE_NUM_THREADS_PER_EU_INTEL*/, sizeof(v), &v, NULL) == CL_SUCCESS)
        printf("  %-32s : %u\n", "Threads per EU (Intel)", v);
    if (clGetDeviceInfo(dv, 0x4204 /*CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL*/, sizeof(v), &v, NULL) == CL_SUCCESS)
        printf("  %-32s : %u\n", "EUs per Xe-core (Intel)", v);
    if (clGetDeviceInfo(dv, 0x4203 /*CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL*/, sizeof(v), &v, NULL) == CL_SUCCESS)
        printf("  %-32s : %u\n", "Xe-cores per slice (Intel)", v);

    free(ps);
    return 0;
}
