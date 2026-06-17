// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// Standalone perf + correctness test for the `sdpa_micro__generate` kernel as
// used by OpenVINO PagedAttention during MULTI-SEQUENCE decoding (MIXED stage).
// =============================================================================
//
// This test reproduces, OUTSIDE of OpenVINO, exactly what the GPU plugin's
// SDPAMicroGenerator(/*prefill=*/false, /*gqa_single_token=*/false) does for the
// PagedAttention MIXED stage (IS_PAGED_ATTENTION=1, !IS_PREFILL): the same kernel
// OpenVINO runs when several sequences each decode a small block of query tokens
// (num_tokens != num_seqs and some past_len != 0) against a paged, int8
// (asymmetric, per-token / BY_TOKEN) compressed KV cache.
//
//   1.  Use the PA-decode config xehpc_h128_pa (the config init_microkernels
//       selects for head_size<=128 paged-attention generate on xe2/xe3/xe3p).
//   2.  Build the FOUR oneDNN/gemmstone micro-GEMMs the PA kernel needs, exactly
//       mirroring init_microkernels():
//          KQ  = cached int8 K^T * Q   (Layout::N, scaleA+offsetA, int8 systolic)
//          KcQ = new-token  K^T * Q    (f16, Layout::T)
//          VS  = cached int8 V  * S    (Layout::N, scaleA+offsetA)
//          VcS = new-token  V  * S     (f16, Layout::N)
//       via gemmstone::microkernel::selectGEMM (host-only; ngen emits ISA in mem).
//   3.  Emit the OpenCL-C shims (decorators kq/vs/kcq/vcs, microkernelIDs 0..3)
//       and assemble the full kernel source the way kernels_cache.cpp does:
//          generic_vector_ops.cl + sdpa_utils.cl + tile_ops.cl  (batch headers,
//              hoisted to the front -- tile_ops.cl DEFINES DECLARE_2D_TILE_OPS)
//        + shim_kq + shim_vs + shim_kcq + shim_vcs              (the "jit" section)
//        + all JIT #defines + sdpa_micro.cl body                (the "str" section)
//       The kernel's own #include directives are stripped; the shims (emitted with
//       useTileOps=true) invoke DECLARE_2D_TILE_OPS, so tile_ops.cl MUST precede them.
//   4.  clBuildProgram, then gemmstone::microkernel::fuse() to patch the
//       microkernel machine code into the program binary, then rebuild from the
//       patched binary (clCreateProgramWithBinary).
//   5.  Build the paged int8 KV cache + per-step f16 K/V + the PA index buffers
//       (subsequence_begins / past_lens / block_indices / block_indices_begins /
//       blocked_indexes_start_and_gws_mapping), run, compare against an f32 PA
//       reference, and benchmark.
//
// Configurable parameters (CLI flags, see usage()):
//   --tokens     N   new query tokens per sequence this step (MIXED: > 1)  [6]
//   --seqs       N   number of sequences decoded together (subsequences)   [1]
//   --history    N   past KV length per sequence (past_len)                [4096]
//   --head-dim   N   head size D                                           [128]
//   --kv-heads   N   number of K/V heads (GQA groups)                      [8]
//   --heads      N   number of Q heads (>= kv-heads, multiple of it)       [8]
//   --iters      N   timed iterations (averaged, cache-cold)               [200]
//   --causal     0/1 causal masking                                        [0]
//
// The HOST-side microkernel generation (steps 1-3) is validated on any box.
// Steps 4-5 require a GPU with XMX/systolic support (BMG Xe2 / PTL Xe3); on a
// gen9/UHD machine clBuildProgram of the microkernel shim will fail -- that is
// expected, run this on the remote target (see remote_machine.md).
//
// Build:  ./build_test.sh        (see that script for the exact flags)
// =============================================================================

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
#include <random>
#include <sstream>
#include <string>
#include <vector>

// gemmstone (oneDNN GPU microkernel JIT) host API.
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/package.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/kernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/shim.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/fuser.hpp"

// ngen Core enum -> used only to print/select architecture; the actual value is
// derived at runtime from CL_DEVICE_IP_VERSION_INTEL (gmdid).
//   Unknown=0 Gen9=1 Gen10=2 Gen11=3 XeLP=4 XeHP=5 XeHPG=6 XeHPC=7
//   Xe2=8(BMG) Xe3=9(PTL) Xe3p=10

namespace micro {
using Package = gemmstone::microkernel::Package;
using HWInformation = gemmstone::microkernel::HWInformation;
using GEMMProblem = gemmstone::GEMMProblem;
using ABOffset = gemmstone::ABOffset;
using GEMMStrategy = gemmstone::GEMMStrategy;
using GEMMOptions = gemmstone::microkernel::GEMMOptions;
using MatrixLayout = gemmstone::MatrixLayout;
using Type = gemmstone::Type;
using SizeParams = gemmstone::SizeParams;
using StrategyRequirement = gemmstone::StrategyRequirement;
using ShimOptions = gemmstone::microkernel::ShimOptions;
using HostLanguage = gemmstone::microkernel::HostLanguage;
}  // namespace micro

// ===========================================================================
// Half-float helpers (same as moe_decode_test.cpp).
// ===========================================================================
static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp_val = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;
    if (exp_val > 15) {
        return static_cast<uint16_t>(sign | 0x7C00);  // inf
    } else if (exp_val < -14) {
        if (exp_val < -24)
            return static_cast<uint16_t>(sign);
        mantissa |= 0x800000;
        int shift = -exp_val - 1;
        mantissa >>= shift;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    return static_cast<uint16_t>(sign | ((exp_val + 15) << 10) | (mantissa >> 13));
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp_val = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exp_val == 0) {
        if (mantissa == 0) {
            uint32_t r = sign;
            float f;
            std::memcpy(&f, &r, 4);
            return f;
        }
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exp_val--;
        }
        exp_val++;
        mantissa &= 0x3FF;
    } else if (exp_val == 31) {
        uint32_t r = sign | 0x7F800000 | (mantissa << 13);
        float f;
        std::memcpy(&f, &r, 4);
        return f;
    }
    uint32_t r = sign | ((exp_val + 112) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &r, 4);
    return f;
}

// ===========================================================================
// OpenCL error check.
// ===========================================================================
#define CL_CHECK(err)                                                                  \
    do {                                                                               \
        cl_int _e = (err);                                                             \
        if (_e != CL_SUCCESS) {                                                        \
            std::fprintf(stderr, "OpenCL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

// ===========================================================================
// Architecture handling: mirror OpenVINO's gpu_arch + convert_ngen_arch.
// ===========================================================================
enum class Arch { unknown, gen9, gen11, xe_lp, xe_hp, xe_hpg, xe_hpc, xe2, xe3, xe3p };

static const char* arch_name(Arch a) {
    switch (a) {
    case Arch::gen9: return "gen9";
    case Arch::gen11: return "gen11";
    case Arch::xe_lp: return "xe_lp";
    case Arch::xe_hp: return "xe_hp";
    case Arch::xe_hpg: return "xe_hpg";
    case Arch::xe_hpc: return "xe_hpc";
    case Arch::xe2: return "xe2 (BMG)";
    case Arch::xe3: return "xe3 (PTL)";
    case Arch::xe3p: return "xe3p";
    default: return "unknown";
    }
}

// Map a gmdid (CL_DEVICE_IP_VERSION_INTEL) to an Arch the same way OpenVINO's
// parse_version()+ngen do. The architecture field is bits [22..31] of the
// new-format gmdid.
//   Xe (Gen12LP)=12, XeHPG=12.7x (architecture 12, release 70/71/72/73/74),
//   XeHPC=12.60/12.61, Xe2=20, Xe3=30, Xe3p=31.  These ranges follow ngen's
//   getCore(ProductFamily). We use the architecture major to bucket.
static Arch gmdid_to_arch(uint32_t gmdid) {
    uint32_t architecture = (gmdid >> 22) & 0x3FF;
    uint32_t release = (gmdid >> 14) & 0xFF;
    if (architecture == 0)
        return Arch::unknown;
    if (architecture >= 30) {
        // Xe3 / Xe3p family. release distinguishes p variants in practice; treat
        // 31+ as xe3p, 30 as xe3.
        return (architecture >= 31) ? Arch::xe3p : Arch::xe3;
    }
    if (architecture >= 20)
        return Arch::xe2;  // BMG / LNL
    if (architecture == 12) {
        // Gen12: LP (release < 10), HPG (release 70..), HPC (release 60..69)
        if (release >= 70)
            return Arch::xe_hpg;
        if (release >= 60)
            return Arch::xe_hpc;
        return Arch::xe_lp;
    }
    if (architecture == 11)
        return Arch::gen11;
    if (architecture == 9)
        return Arch::gen9;
    return Arch::unknown;
}

// Inverse of gmdid_to_arch: a representative gmdid for a forced architecture,
// used by --force-arch to validate host-side microkernel generation off-target.
//   BMG Xe2: architecture=20, release=1 -> 0x05004000
//   PTL Xe3: architecture=30           -> 0x07800000
//   Xe3p:    architecture=31           -> 0x07c00000
// NOTE: this OpenVINO oneDNN checkout ships systolic int8 SDPA microkernels for
// BMG (xe2) and PTL (xe3); xe3p depends on the catalog being built with Xe3p
// SDPA configs. xe_hpc is intentionally not offered here because its gmdid
// encoding differs and the catalog does not target it.
static bool arch_from_name(const std::string& name, Arch& arch, uint32_t& gmdid) {
    if (name == "xe2" || name == "bmg") { arch = Arch::xe2; gmdid = (20u << 22) | (1u << 14); return true; }
    if (name == "xe3" || name == "ptl") { arch = Arch::xe3; gmdid = (30u << 22); return true; }
    if (name == "xe3p") { arch = Arch::xe3p; gmdid = (31u << 22); return true; }
    return false;
}

static int get_subgroup_size(Arch arch) {
    switch (arch) {
    case Arch::gen9:
    case Arch::gen11:
    case Arch::xe_lp:
    case Arch::xe_hp:
    case Arch::xe_hpg:
        return 8;
    default:
        return 16;
    }
}

static int get_d_max(int head_size) {
    for (int i = 32; i <= 1024; i *= 2)
        if (i >= head_size)
            return i;
    return 1024;
}

// ===========================================================================
// sdpa_config_t and the pre-tuned table subset we need (head_size <= 128,
// quantized, thin_q == true == generate).  Copied verbatim from
// sdpa_gen_micro.cpp.  {unroll_m_kq, unroll_n_kq, unroll_m_vs, unroll_n_vs,
//  wg_m_kq, wg_n_kq, wg_m_vs, wg_n_vs}
// ===========================================================================
struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq;
    int unroll_m_vs, unroll_n_vs;
    int wg_m_kq, wg_n_kq;
    int wg_m_vs, wg_n_vs;
};

// Paged-attention generate/MIXED config for head_size <= 128.  In
// init_microkernels the PA decode kernel is built for dynamic shapes, so
// nkeys_v == 0; every XMX arch (xe2/xe3/xe3p) therefore takes choose_config_*'s
// `seq <= 0 && is_pa` branch, which (head_size<=128, !is_prefill) returns
// xehpc_h128_pa.  See choose_config_xehpc() in sdpa_gen_micro.cpp.
//   {unroll_m_kq, unroll_n_kq, unroll_m_vs, unroll_n_vs, wg_m_kq, wg_n_kq, wg_m_vs, wg_n_vs}
static sdpa_config_t xehpc_h128_pa = {16, 16, 16, 16, 8, 2, 8, 2};

// PA generate/MIXED, head_size <= 128: the selected config is arch-independent
// (xe2/xe3/xe3p all resolve to xehpc_h128_pa for the dynamic-shape decode kernel).
static sdpa_config_t* choose_config(Arch arch, int head_size, int seq, bool is_integrated) {
    assert(head_size <= 128 && "this test specializes head_size <= 128");
    (void)arch;
    (void)seq;
    (void)is_integrated;
    return &xehpc_h128_pa;
}

// ===========================================================================
// Configurable test parameters.
// ===========================================================================
struct Params {
    int tokens = 6;       // new query tokens per sequence this step (MIXED: q>1)
    int batch = 1;        // sequence (subsequence) count
    int history = 4096;   // past KV length per sequence (past_len)
    int head_dim = 128;   // D
    int kv_heads = 8;     // Hkv
    int heads = 8;        // Hq (>= Hkv, multiple of Hkv)
    int iters = 200;      // timed iterations (averaged; cache-cold per iter)
    int flush_mb = 0;     // cache-flush buffer size in MB (0 = auto: max(4*LLC,128MB))
    bool causal = false;  // causal masking
    bool dump_source = false;
    bool gen_only = false;       // stop after host-side shim generation (no GPU)
    bool gqa_share = false;      // GQA-shared-KV decode: one WG serves all query
                                 // heads of a KV group (KV read/dequant once)
    std::string force_arch;      // override detected arch: xe2|xe3|xe3p
    std::string cfg_override;    // override tiling cfg: "um_kq,un_kq,um_vs,un_vs,wm_kq,wn_kq,wm_vs,wn_vs"
    std::string kernel_dir;  // dir containing sdpa_micro.cl + include/batch_headers
};

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [options]   (PagedAttention MIXED-stage sdpa_micro__generate)\n"
        "  --tokens N      new query tokens per sequence this step (MIXED: >1) (default 6)\n"
        "  --seqs N        number of sequences decoded together (default 1)\n"
        "  --history N     past KV length per sequence (past_len), rounded up to\n"
        "                  a multiple of the KQ wg_tile_m (default 4096)\n"
        "  --head-dim N    head size D (default 128)\n"
        "  --kv-heads N    number of K/V heads (default 8)\n"
        "  --heads N       number of Q heads, >= kv-heads (default 8)\n"
        "  --iters N       timed iterations, averaged cache-cold (default 200)\n"
        "  --flush-mb N    cache-flush buffer size in MB (0=auto max(4*LLC,128); use to\n"
        "                  prove KV eviction by checking cold time is flush-size stable)\n"
        "  --causal 0|1    causal masking (default 0)\n"
        "  --kernel-dir P  path to dir holding sdpa_micro.cl and include/ (default: auto)\n"
        "  --dump-source   write the assembled OpenCL source to sdpa_micro_full.cl\n"
        "  --force-arch A  override detected arch for the host JIT: xe2|xe3|xe3p\n"
        "                  (use to validate microkernel generation on a non-XMX box)\n"
        "  --cfg LIST      override tiling config (8 ints, comma-separated):\n"
        "                  um_kq,un_kq,um_vs,un_vs,wm_kq,wn_kq,wm_vs,wn_vs\n"
        "                  (default 16,16,16,16,8,2,8,2; tunes micro-GEMM tiles + sg_per_wg)\n"
        "  --gqa-share     GQA-shared-KV decode: one work-group serves all query heads\n"
        "                  of a KV group so the cached KV is read+dequantised once\n"
        "                  (requires --causal 0 and a cfg with wg_tile_n == tokens*heads/kv-heads)\n"
        "  --gen-only      stop after host-side shim generation; do not touch the GPU\n"
        "  --help\n",
        prog);
}

static Params parse_args(int argc, char** argv) {
    Params p;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tokens") p.tokens = std::atoi(need(i));
        else if (a == "--seqs") p.batch = std::atoi(need(i));
        else if (a == "--history") p.history = std::atoi(need(i));
        else if (a == "--head-dim") p.head_dim = std::atoi(need(i));
        else if (a == "--kv-heads") p.kv_heads = std::atoi(need(i));
        else if (a == "--heads") p.heads = std::atoi(need(i));
        else if (a == "--iters") p.iters = std::atoi(need(i));
        else if (a == "--flush-mb") p.flush_mb = std::atoi(need(i));
        else if (a == "--causal") p.causal = std::atoi(need(i)) != 0;
        else if (a == "--kernel-dir") p.kernel_dir = need(i);
        else if (a == "--dump-source") p.dump_source = true;
        else if (a == "--force-arch") p.force_arch = need(i);
        else if (a == "--cfg") p.cfg_override = need(i);
        else if (a == "--gqa-share") p.gqa_share = true;
        else if (a == "--gen-only") p.gen_only = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); std::exit(2); }
    }
    if (p.heads % p.kv_heads != 0) {
        std::fprintf(stderr, "heads (%d) must be a multiple of kv-heads (%d)\n", p.heads, p.kv_heads);
        std::exit(2);
    }
    return p;
}

// ===========================================================================
// Source assembly helpers.
// ===========================================================================
static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "Failed to open: %s\n", path.c_str());
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Strip "#include \"...\"" lines (we inline the batch headers manually and they
// have no nested includes).
static std::string strip_includes(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    size_t pos = 0;
    while (pos < src.size()) {
        size_t eol = src.find('\n', pos);
        if (eol == std::string::npos) eol = src.size();
        std::string line = src.substr(pos, eol - pos);
        // Find first non-space.
        size_t s = line.find_first_not_of(" \t");
        bool is_inc = (s != std::string::npos && line.compare(s, 8, "#include") == 0);
        if (!is_inc) {
            out += line;
            out += '\n';
        }
        pos = eol + 1;
    }
    return out;
}

// A single JIT #define.
static void def(std::string& s, const std::string& name, const std::string& value) {
    s += "#define " + name + " " + value + "\n";
}
static void def(std::string& s, const std::string& name, long long value) {
    def(s, name, std::to_string(value));
}

int main(int argc, char** argv) {
    Params P = parse_args(argc, argv);

    // -----------------------------------------------------------------------
    // Resolve kernel source directory.
    // -----------------------------------------------------------------------
    if (P.kernel_dir.empty()) {
        // This file lives in <root>/.github/skills/dev_sdpa_micro_opt; the kernel
        // lives in <root>/src/plugins/intel_gpu/src/graph/impls/ocl_v2.
        P.kernel_dir = "../../../src/plugins/intel_gpu/src/graph/impls/ocl_v2";
    }
    const std::string kernel_cl = P.kernel_dir + "/sdpa_micro.cl";
    const std::string headers_dir =
        P.kernel_dir + "/../../../kernel_selector/cl_kernels/include/batch_headers";

    const int D = P.head_dim;
    const int Hkv = P.kv_heads;
    const int Hq = P.heads;
    const int B = P.batch;
    const int Lq = P.tokens;
    const int Lk = P.history + P.tokens;  // total key/value length
    const int kv_group_size = Hq / Hkv;

    if (P.gqa_share && P.causal) {
        // The GQA-shared kernel stacks (token, query-head) pairs into the KQ
        // N-columns, so the kernel's column->token causal mapping no longer
        // holds.  Restrict to full attention (matches the PA MIXED decode case).
        std::fprintf(stderr, "--gqa-share requires --causal 0\n");
        std::exit(2);
    }

    std::printf("=== sdpa_micro generate test ===\n");
    std::printf("tokens(Lq)=%d  seqs(B)=%d  history=%d  Lk=%d  D=%d  kv_heads=%d  heads=%d  causal=%d\n",
                Lq, B, P.history, Lk, D, Hkv, Hq, (int)P.causal);

    // =======================================================================
    // 1. OpenCL platform/device selection (pick first GPU).
    //    Skipped entirely under --gen-only so the host-side JIT can be exercised
    //    on a box with no usable GPU.
    // =======================================================================
#ifndef CL_DEVICE_IP_VERSION_INTEL
#define CL_DEVICE_IP_VERSION_INTEL 0x4250
#endif
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_int err = CL_SUCCESS;

    cl_uint eu_count = 0;
    cl_uint gmdid = 0;
    bool systolic = false;
    Arch arch = Arch::unknown;

    if (!P.gen_only) {
        cl_uint num_plat = 0;
        CL_CHECK(clGetPlatformIDs(0, nullptr, &num_plat));
        std::vector<cl_platform_id> plats(num_plat);
        CL_CHECK(clGetPlatformIDs(num_plat, plats.data(), nullptr));
        for (auto pl : plats) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
                continue;
            std::vector<cl_device_id> devs(nd);
            CL_CHECK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr));
            device = devs[0];
            break;
        }
        if (!device) {
            std::fprintf(stderr, "No GPU device found (use --gen-only to validate host JIT only).\n");
            return 1;
        }
        char dev_name[256] = {0};
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(eu_count), &eu_count, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_IP_VERSION_INTEL, sizeof(gmdid), &gmdid, nullptr);
        size_t ext_len = 0;
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len);
        std::string extensions(ext_len, '\0');
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_len, &extensions[0], nullptr);
        systolic = extensions.find("cl_intel_subgroup_matrix_multiply_accumulate") != std::string::npos;
        arch = gmdid_to_arch(gmdid);
        std::printf("device: %s\n", dev_name);
        std::printf("  EU/CU count = %u\n", eu_count);
        std::printf("  gmdid (ip_version) = 0x%08x -> arch = %s\n", gmdid, arch_name(arch));
        std::printf("  systolic (XMX) = %d\n", systolic ? 1 : 0);
    } else {
        std::printf("(gen-only: no GPU device opened)\n");
    }

    // --force-arch overrides the detected architecture/gmdid/systolic. Useful to
    // validate microkernel generation on a non-XMX dev box (the generated ISA is
    // for the forced target and cannot be executed locally).
    if (!P.force_arch.empty()) {
        Arch farch;
        uint32_t fgmdid;
        if (!arch_from_name(P.force_arch, farch, fgmdid)) {
            std::fprintf(stderr, "unknown --force-arch '%s' (xe2|xe3|xe3p)\n", P.force_arch.c_str());
            return 2;
        }
        arch = farch;
        gmdid = fgmdid;
        systolic = true;
        if (eu_count == 0)
            eu_count = 160;  // plausible BMG/PTL EU count for HWInformation
        std::printf("forced arch = %s  gmdid = 0x%08x  (systolic=1)\n", arch_name(arch), gmdid);
    }

    if (D > 128) {
        std::fprintf(stderr, "This test specializes head_dim <= 128 (got %d).\n", D);
        return 2;
    }
    if (arch == Arch::unknown) {
        std::fprintf(stderr,
                     "WARNING: unknown arch; defaulting micro config to XeHPC h128.\n"
                     "         Microkernel generation will likely fail on non-XMX hardware.\n"
                     "         Use --force-arch xe2|xe3 to validate the host JIT path.\n");
    }

    if (!P.gen_only) {
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CL_CHECK(err);
        cl_command_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue = clCreateCommandQueueWithProperties(context, device, qprops, &err);
        CL_CHECK(err);
    }

    // =======================================================================
    // 2. Build the KQ and VS micro-GEMMs (mirror init_microkernels for the
    //    non-PA generate, int8 asymmetric per-token path).
    // =======================================================================
    const int d_max = get_d_max(D);
    const int sg = get_subgroup_size(arch);
    const bool is_integrated = false;  // remote targets are discrete BMG/PTL
    const int seq_for_cfg = Lk;        // nkeys_v passed to choose_config

    sdpa_config_t cfg_val = *choose_config(arch, D, seq_for_cfg, is_integrated);
    sdpa_config_t* cfg = &cfg_val;
    // --cfg overrides the tuned tiling table (parameter trust-region search).
    if (!P.cfg_override.empty()) {
        int v[8];
        int n = std::sscanf(P.cfg_override.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d",
                            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
        if (n != 8) {
            std::fprintf(stderr, "--cfg needs 8 comma-separated ints "
                                 "(um_kq,un_kq,um_vs,un_vs,wm_kq,wn_kq,wm_vs,wn_vs), got %d\n", n);
            return 2;
        }
        cfg_val = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
        std::printf("config OVERRIDE via --cfg\n");
    }
    std::printf("config: unroll_kq={%d,%d} unroll_vs={%d,%d} wg_kq={%d,%d} wg_vs={%d,%d}\n",
                cfg->unroll_m_kq, cfg->unroll_n_kq, cfg->unroll_m_vs, cfg->unroll_n_vs,
                cfg->wg_m_kq, cfg->wg_n_kq, cfg->wg_m_vs, cfg->wg_n_vs);

    micro::HWInformation hw_info;
    hw_info.euCount = static_cast<int>(eu_count);
    hw_info.gmdid = gmdid;
    hw_info.systolicAvailable = systolic;

    // Common problem skeleton mirrors init_microkernels (PA generate path):
    //   Ta_ext = s8 (cached int8 K), Tb_ext = f16 (Q), Ta = Tb = f16, Tc = f32.
    // The cached gemms (KQ/VS) carry per-token scale+zp; problem.Ta is flipped to
    // s8 (int8 systolic) AFTER problem_kq is copied, so KQ keeps Ta=f16 while
    // VS / KcQ / VcS inherit Ta=s8 -- this exact ordering matters for byte-for-byte
    // microkernel parity with OpenVINO.
    micro::GEMMProblem problem{};  // value-init: addressing fields have no defaults
    problem.Ta_ext = micro::Type::s8;   // K cache int8
    problem.Tb_ext = micro::Type::f16;  // Q f16
    problem.Ta = problem.Tb = micro::Type::f16;
    problem.Tc = problem.Tc_ext = micro::Type::f32;
    problem.Ts = problem.Tc;

    const int scale_sz = micro::Type(micro::Type::f16).size();  // 2
    const int zp_sz = micro::Type(micro::Type::f16).size();     // 2
    const int pa_block_size = 16;  // paged_attention::block_size

    auto align_for = [](int v) { return gemmstone::microkernel::alignmentForLD(v); };

    micro::Package gemm_kq, gemm_vs, gemm_kcq, gemm_vcs;

    // ---- (1) KQ: cached int8 K^T * Q  (Layout::N, scaleA+offsetA) ----
    micro::GEMMProblem problem_kq = problem;        // Ta = f16 (copied before Ta->s8)
    problem_kq.A.layout = micro::MatrixLayout::N;   // PA cached K is column-major
    micro::GEMMOptions opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;
    problem.Ta = micro::Type::s8;                   // int8 systolic; base now s8 for VS/KcQ/VcS
    problem_kq.Ta_scale = micro::Type::f16;          // per-token scales, layout N, 2D ptr
    problem_kq.A_scale.setAlignment(scale_sz);
    problem_kq.A_scale.layout = micro::MatrixLayout::N;
    problem_kq.asPtrDims = 2;
    problem_kq.Tao = micro::Type::f16;               // per-token zero points (asymmetric)
    problem_kq.AO.setAlignment(zp_sz);
    problem_kq.AO.layout = micro::MatrixLayout::N;
    problem_kq.aoPtrDims = 2;
    problem_kq.aOffset = micro::ABOffset::Calc;
    problem_kq.aqGroupM = 1;
    problem_kq.aqGroupK = D;
    opts_kq.scaleA = true;
    opts_kq.offsetA = true;
    problem_kq.B.layout = micro::MatrixLayout::Pr;
    problem_kq.C.layout = micro::MatrixLayout::T;
    problem_kq.A.setAlignment(align_for(static_cast<int>(D * problem.Ta)));
    problem_kq.A.setAlignment(pa_block_size * problem.Ta);  // PA override: 16 * s8 = 16
    problem_kq.B.setAlignment(64);
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = static_cast<uint16_t>(d_max);
    problem_kq.B.tileC = static_cast<uint16_t>(sg);

    // PA decode kernels are built for dynamic shapes -> m/n/batch = 0.
    micro::SizeParams sizes_kq;
    sizes_kq.m = 0;
    sizes_kq.n = 0;
    sizes_kq.k = D;
    sizes_kq.batch = 0;

    std::vector<micro::StrategyRequirement> reqs_kq;
    reqs_kq.push_back(micro::StrategyRequirement::UnrollM == cfg->unroll_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::UnrollN == cfg->unroll_n_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGM == cfg->wg_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGN == cfg->wg_n_kq);

    try {
        gemm_kq = gemmstone::microkernel::selectGEMM(opts_kq, hw_info, sizes_kq, problem_kq, reqs_kq);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "selectGEMM(KQ) failed: %s\n", ex.what());
        return 3;
    }

    const int wg_tile_m_kq = gemm_kq.getSetting("wg_tile_m");
    const int wg_tile_n_kq = gemm_kq.getSetting("wg_tile_n");
    const int sg_per_wg_m_kq = gemm_kq.getSetting("sg_per_wg_m");
    const int sg_per_wg_n_kq = gemm_kq.getSetting("sg_per_wg_n");
    std::printf("KQ: grfMin=%d barriers=%d systolic=%d wg_tile=(%d,%d) sg_per_wg=(%d,%d)\n",
                gemm_kq.grfMin, gemm_kq.barrierCount, gemm_kq.systolic,
                wg_tile_m_kq, wg_tile_n_kq, sg_per_wg_m_kq, sg_per_wg_n_kq);

    if (P.gqa_share && wg_tile_n_kq != Lq * kv_group_size) {
        // One work-group must cover exactly all (token x query-head-in-group)
        // columns of a KV group, i.e. wg_tile_n == tokens * (heads / kv_heads).
        std::fprintf(stderr,
                     "--gqa-share: wg_tile_n (%d) must equal tokens*heads/kv-heads (%d). "
                     "Use --cfg with un_kq*wn_kq == %d (e.g. 16,16,16,16,8,%d,8,%d).\n",
                     wg_tile_n_kq, Lq * kv_group_size, Lq * kv_group_size,
                     (Lq * kv_group_size) / 16, (Lq * kv_group_size) / 16);
        std::exit(3);
    }

    // ---- (2) KcQ: new-token f16 K^T * Q  (Layout::T, no scale/offset) ----
    opts_kq.scaleA = false;
    opts_kq.offsetA = false;
    micro::GEMMProblem problem_kcq = problem;        // Ta = s8 (post-override)
    problem_kcq.Ta_ext = micro::Type::f16;           // Kc is f16
    problem_kcq.A.layout = micro::MatrixLayout::T;
    problem_kcq.B.layout = micro::MatrixLayout::Pr;
    problem_kcq.C.layout = micro::MatrixLayout::T;
    problem_kcq.A.setAlignment(align_for(static_cast<int>(D * problem.Ta)));
    problem_kcq.B.setAlignment(64);
    problem_kcq.B.crosspack = 2;
    problem_kcq.B.tileR = static_cast<uint16_t>(d_max);
    problem_kcq.B.tileC = static_cast<uint16_t>(sg);
    try {
        gemm_kcq = gemmstone::microkernel::selectGEMM(opts_kq, hw_info, sizes_kq, problem_kcq, reqs_kq);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "selectGEMM(KcQ) failed: %s\n", ex.what());
        return 3;
    }
    std::printf("KcQ: grfMin=%d barriers=%d systolic=%d\n",
                gemm_kcq.grfMin, gemm_kcq.barrierCount, gemm_kcq.systolic);

    // ---- (3) VS: cached int8 V * S  (Layout::N, scaleA+offsetA) ----
    micro::GEMMOptions opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;
    micro::GEMMProblem problem_vs = problem;          // Ta = s8
    problem_vs.Ta_ext = micro::Type::s8;              // V cache int8
    problem_vs.A.layout = micro::MatrixLayout::N;
    problem_vs.Ta_scale = micro::Type::f16;
    problem_vs.A_scale.setAlignment(scale_sz);
    problem_vs.A_scale.layout = micro::MatrixLayout::N;
    problem_vs.asPtrDims = 2;
    problem_vs.Tao = micro::Type::f16;
    problem_vs.AO.setAlignment(zp_sz);
    problem_vs.AO.layout = micro::MatrixLayout::N;
    problem_vs.aoPtrDims = 2;
    problem_vs.aOffset = micro::ABOffset::Calc;
    problem_vs.aqGroupM = D;   // PA: v_head_size (not rnd_up_pow2)
    problem_vs.aqGroupK = 1;
    opts_vs.scaleA = true;
    opts_vs.offsetA = true;
    problem_vs.B.layout = micro::MatrixLayout::Pr;
    problem_vs.C.layout = micro::MatrixLayout::N;
    problem_vs.A.setAlignment(align_for(static_cast<int>(D * problem.Ta)));
    problem_vs.B.setAlignment(64);
    problem_vs.B.crosspack = 16;

    micro::SizeParams sizes_vs;
    sizes_vs.m = D;                  // n_values = v_head_size
    sizes_vs.n = wg_tile_n_kq;       // gemm_kq wg_tile_n
    sizes_vs.k = wg_tile_m_kq;       // gemm_kq wg_tile_m
    sizes_vs.batch = 0;

    std::vector<micro::StrategyRequirement> reqs_vs;
    reqs_vs.push_back(micro::StrategyRequirement::UnrollM == cfg->unroll_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::UnrollN == cfg->unroll_n_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGM == cfg->wg_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGN == cfg->wg_n_vs);

    auto adjust_vs = [](micro::GEMMStrategy& strategy) { strategy.dpasw |= strategy.fused; };
    try {
        gemm_vs = gemmstone::microkernel::selectGEMM(opts_vs, hw_info, sizes_vs, problem_vs, reqs_vs, adjust_vs);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "selectGEMM(VS) failed: %s\n", ex.what());
        return 3;
    }
    const int wg_tile_m_vs = gemm_vs.getSetting("wg_tile_m");
    std::printf("VS: grfMin=%d barriers=%d systolic=%d wg_tile_m=%d\n",
                gemm_vs.grfMin, gemm_vs.barrierCount, gemm_vs.systolic, wg_tile_m_vs);

    // ---- (4) VcS: new-token f16 V * S  (Layout::N, no scale/offset) ----
    opts_vs.scaleA = false;
    opts_vs.offsetA = false;
    micro::GEMMProblem problem_vcs = problem;          // Ta = s8
    problem_vcs.Ta_ext = micro::Type::f16;             // Vc is f16
    problem_vcs.A.layout = micro::MatrixLayout::N;
    problem_vcs.B.layout = micro::MatrixLayout::Pr;
    problem_vcs.C.layout = micro::MatrixLayout::N;
    problem_vcs.A.setAlignment(align_for(static_cast<int>(D * problem.Ta)));
    problem_vcs.B.setAlignment(64);
    problem_vcs.B.crosspack = 16;
    try {
        gemm_vcs = gemmstone::microkernel::selectGEMM(opts_vs, hw_info, sizes_vs, problem_vcs, reqs_vs, adjust_vs);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "selectGEMM(VcS) failed: %s\n", ex.what());
        return 3;
    }
    std::printf("VcS: grfMin=%d barriers=%d systolic=%d\n",
                gemm_vcs.grfMin, gemm_vcs.barrierCount, gemm_vcs.systolic);

    // =======================================================================
    // 3. Generate shims + assemble the OpenCL source.
    // =======================================================================
    // Shim emit order + microkernelIDs MUST match get_kernel_data():
    //   kq=0, vs=1, kcq=2, vcs=3   (decorators name the ugemm_* entry points).
    micro::ShimOptions shim_opts;
    shim_opts.subgroupSize = sg;
    shim_opts.useTileOps = true;
    shim_opts.decorator = "kq";
    std::string shim_kq = gemmstone::microkernel::generateShim(gemm_kq, micro::HostLanguage::OpenCL_C, shim_opts);
    shim_opts.microkernelID++;
    shim_opts.decorator = "vs";
    std::string shim_vs = gemmstone::microkernel::generateShim(gemm_vs, micro::HostLanguage::OpenCL_C, shim_opts);
    shim_opts.microkernelID++;
    shim_opts.decorator = "kcq";
    std::string shim_kcq = gemmstone::microkernel::generateShim(gemm_kcq, micro::HostLanguage::OpenCL_C, shim_opts);
    shim_opts.microkernelID++;
    shim_opts.decorator = "vcs";
    std::string shim_vcs = gemmstone::microkernel::generateShim(gemm_vcs, micro::HostLanguage::OpenCL_C, shim_opts);

    // ---- JIT #defines (mirror get_jit_constants for PA int8 BY_TOKEN, MIXED) ----
    const int head_size = D;       // micro_get_head_size(params, 0)
    const int k_head_size = D;
    const int v_head_size = D;
    const int tile_q = wg_tile_n_kq;  // query-block size (dispatch + gws mapping)

    const long long ldq = (long long)k_head_size * 2;  // Q f16
    const long long ldk = (long long)k_head_size * 1;  // K cache i8
    const long long ldv = (long long)v_head_size * 1;  // V cache i8
    const long long lda = (long long)v_head_size * 2;  // A f16

    auto align_ld = [](long long ld) { return gemmstone::microkernel::alignmentForLD((int)ld); };

    std::string jdefs;
    // Decoration macros that build_code() would emit (we only need FUNC family;
    // sdpa_micro.cl does not use FUNC, but define them defensively).
    def(jdefs, "KERNEL(name)", "__kernel void name");
    def(jdefs, "KERNEL_ID", "micro_sdpa");
    def(jdefs, "OPTIONAL_SHAPE_INFO_ARG", "");
    def(jdefs, "OPTIONAL_SHAPE_INFO_TENSOR", "");
    def(jdefs, "FUNC(name)", "name");
    def(jdefs, "FUNC_CALL(name)", "name");
    def(jdefs, "CONST_ARRAY_DECL(name)", "__constant size_t name []");
    def(jdefs, "CONST_ARRAY_REF(name)", "name");

    def(jdefs, "D_MAX", d_max);
    def(jdefs, "SUBGROUP_SIZE", sg);
    def(jdefs, "INVERT_SCALE", "0");
    def(jdefs, "SCALE_DATA_T", "half");
    def(jdefs, "HEAD_SIZE", k_head_size);
    def(jdefs, "IS_CAUSAL", P.causal ? "1" : "0");
    def(jdefs, "WITH_ATTN_MASK", "0");
    // PA decode passes the scale as a 1-element input (matches get_scale_memory()).
    def(jdefs, "WITH_SCALE", "1");
    def(jdefs, "Q_ALIGN", align_ld(ldq));
    def(jdefs, "K_ALIGN", align_ld(ldk));
    def(jdefs, "V_ALIGN", align_ld(ldv));
    def(jdefs, "A_ALIGN", align_ld(lda));
    def(jdefs, "IS_PREFILL", "0");
    def(jdefs, "IS_GQA_SINGLE_TOKEN", "0");
    def(jdefs, "GQA_SHARED_KV", P.gqa_share ? "1" : "0");
    def(jdefs, "TRANSPOSE_K", "0");

    // ---- Paged attention (MIXED), int8 BY_TOKEN compressed KV cache ----
    // KV_COMPRESSED is intentionally NOT defined: PA carries scale+zp interleaved
    // inside the K/V cache blocks (IS_KV_COMPRESSED_PA), so the kernel takes no
    // separate K_scales/K_zp/V_scales/V_zp arguments.
    def(jdefs, "IS_PAGED_ATTENTION", "1");
    def(jdefs, "PAGED_ATTENTION_BLOCK_SIZE", pa_block_size);
    def(jdefs, "ADJUSTED_K_HEAD_SIZE", k_head_size + 4);  // head_size + scale(2B) + zp(2B)
    def(jdefs, "ADJUSTED_V_HEAD_SIZE", v_head_size + 4);
    def(jdefs, "ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", pa_block_size);
    def(jdefs, "IS_KV_COMPRESSED_PA", "1");
    def(jdefs, "KV_HEADS_NUM", Hkv);
    def(jdefs, "HEADS_NUM", Hq);
    def(jdefs, "KV_GROUP_SIZE", kv_group_size);
    def(jdefs, "QRY_DATA_T", "half");
    def(jdefs, "KEY_DATA_T", "char");  // int8 cache
    def(jdefs, "VAL_DATA_T", "char");
    def(jdefs, "KEY_ELEMENTS_PER_BYTE", "1");
    def(jdefs, "VAL_ELEMENTS_PER_BYTE", "1");
    // Input layout element types + (no) feature padding for Q / Kc / Vc / index.
    def(jdefs, "INPUT0_TYPE", "half");
    def(jdefs, "INPUT1_TYPE", "char");
    def(jdefs, "INPUT2_TYPE", "char");
    def(jdefs, "INPUT3_TYPE", "int");
    def(jdefs, "INPUT0_PAD_BEFORE_FEATURE_NUM", 0);
    def(jdefs, "INPUT0_PAD_AFTER_FEATURE_NUM", 0);
    def(jdefs, "INPUT1_PAD_BEFORE_FEATURE_NUM", 0);
    def(jdefs, "INPUT1_PAD_AFTER_FEATURE_NUM", 0);
    def(jdefs, "INPUT2_PAD_BEFORE_FEATURE_NUM", 0);
    def(jdefs, "INPUT2_PAD_AFTER_FEATURE_NUM", 0);

    // Remainder flags: PA decode kernels are built for dynamic shapes, so n_keys
    // and n_queries are dynamic -> both remainders are enabled, exactly as
    // get_jit_constants emits for paged attention.
    const bool d_full = (head_size == d_max);
    def(jdefs, "REMAINDER_K", "1");
    if (d_full) {
        const int packed_elems_per_uint = 2;  // sizeof(uint)/sizeof(half)
        const int max_block_elems = 16;
        const int q_block_elems = (d_max / packed_elems_per_uint) / sg;
        if (ldq % 4 == 0 && q_block_elems <= max_block_elems)
            def(jdefs, "BLOCK_Q", "1");
        def(jdefs, "REMAINDER_Q", "1");
    }
    // Prefetch (xe_hpc and above). PA non-prefill disables K0/K/V prefetch and
    // keeps PREFETCH_REMAINDER on (n_keys dynamic -> k tiling never exact).
    if (arch == Arch::xe_hpc || arch == Arch::xe2 || arch == Arch::xe3 || arch == Arch::xe3p) {
        def(jdefs, "PREFETCH_MASK", "1");
        def(jdefs, "PREFETCH_K0", "0");
        def(jdefs, "PREFETCH_K", "0");
        def(jdefs, "PREFETCH_V", "0");
        def(jdefs, "PREFETCH_REMAINDER", "1");
        def(jdefs, "PREFETCH_D_MAX", std::min(d_max, 64));
    }

    // ---- Stride/size defines (contiguous [B,H,L,D], identity order) ----
    // QRY[B][Hq][Lq][D], KEY[B][Hkv][Lk][D], VAL[B][Hkv][Lk][D], DST[B][Hq][Lq][D]
    // scales/zp KEY_COMP[B][Hkv][Lk], VAL_COMP[B][Hkv][Lk]
    auto emit_strides = [&](const char* tag, long long s0, long long s1, long long s2, long long s3,
                            long long d0, long long d1, long long d2, long long d3) {
        def(jdefs, std::string(tag) + "_S0", s0);
        def(jdefs, std::string(tag) + "_S1", s1);
        def(jdefs, std::string(tag) + "_S2", s2);
        def(jdefs, std::string(tag) + "_S3", s3);
        def(jdefs, std::string(tag) + "_D0", d0);
        def(jdefs, std::string(tag) + "_D1", d1);
        def(jdefs, std::string(tag) + "_D2", d2);
        def(jdefs, std::string(tag) + "_D3", d3);
        for (int i = 0; i < 4; i++) {
            def(jdefs, std::string(tag) + "_B" + std::to_string(i), 1);
            def(jdefs, std::string(tag) + "_SB" + std::to_string(i), 1);
        }
    };
    // QRY: stride over [B, Hq, Lq, D]
    emit_strides("QRY", (long long)Hq * Lq * D, (long long)Lq * D, D, 1, B, Hq, Lq, D);
    // KEY: [B, Hkv, Lk, D]
    emit_strides("KEY", (long long)Hkv * Lk * D, (long long)Lk * D, D, 1, B, Hkv, Lk, D);
    // VAL: [B, Hkv, Lk, D]
    emit_strides("VAL", (long long)Hkv * Lk * D, (long long)Lk * D, D, 1, B, Hkv, Lk, D);
    // DST: [B, Hq, Lq, D]
    emit_strides("DST", (long long)Hq * Lq * D, (long long)Lq * D, D, 1, B, Hq, Lq, D);
    // KEY_COMP / VAL_COMP: [B, Hkv, Lk] (scales/zp, one per token)
    emit_strides("KEY_COMP", (long long)Hkv * Lk, Lk, 1, 1, B, Hkv, Lk, 1);
    emit_strides("VAL_COMP", (long long)Hkv * Lk, Lk, 1, 1, B, Hkv, Lk, 1);

    // Offsets (contiguous, no padding).
    def(jdefs, "INPUT0_OFFSET", 0);
    def(jdefs, "INPUT1_OFFSET", 0);
    def(jdefs, "INPUT2_OFFSET", 0);

    // ---- Inline the batch headers (no nested includes) ----
    std::string hdr_gvo = strip_includes(read_file(headers_dir + "/generic_vector_ops.cl"));
    std::string hdr_sdpa = strip_includes(read_file(headers_dir + "/sdpa_utils.cl"));
    std::string hdr_tile = strip_includes(read_file(headers_dir + "/tile_ops.cl"));
    std::string body = strip_includes(read_file(kernel_cl));

    // Final source order — must match OpenVINO's kernels_cache::get_program_source:
    //   full batch program = batch_headers + per-kernel(jit + str + undefs)
    // where, in sdpa_gen_micro.cpp, kd.code->jit = generateShim(kq)+generateShim(vs)
    // and kd.code->str = build_code(name, jit_constants, ...) i.e. JIT #defines + body.
    // The kernel's own `#include "include/batch_headers/*.cl"` lines are stripped and
    // their content hoisted to the front as batch headers. Critically, tile_ops.cl
    // (which DEFINES the DECLARE_2D_TILE_OPS macro) must precede the shims, because the
    // shims emitted with ShimOptions::useTileOps=true *invoke* DECLARE_2D_TILE_OPS.
    //   batch headers : generic_vector_ops.cl + sdpa_utils.cl + tile_ops.cl
    //   jit           : shim_kq + shim_vs + shim_kcq + shim_vcs
    //   str           : JIT #defines + sdpa_micro.cl body
    std::string source;
    source += hdr_gvo;
    source += "\n";
    source += hdr_sdpa;
    source += "\n";
    source += hdr_tile;
    source += "\n";
    source += shim_kq;
    source += "\n";
    source += shim_vs;
    source += "\n";
    source += shim_kcq;
    source += "\n";
    source += shim_vcs;
    source += "\n";
    source += jdefs;
    source += "\n";
    source += body;
    source += "\n";

    if (P.dump_source) {
        std::ofstream o("sdpa_micro_full.cl", std::ios::binary);
        o << source;
        std::printf("wrote assembled source -> sdpa_micro_full.cl (%zu bytes)\n", source.size());
    }

    std::printf("host JIT OK: KQ binary=%zuB grfMin=%d barriers=%d systolic=%d "
                "wg_tile_m=%d wg_tile_n=%d sg_per_wg_m=%d sg_per_wg_n=%d\n",
                gemm_kq.binary.size(), gemm_kq.grfMin, gemm_kq.barrierCount,
                (int)gemm_kq.systolic,
                gemm_kq.getSetting("wg_tile_m"), gemm_kq.getSetting("wg_tile_n"),
                gemm_kq.getSetting("sg_per_wg_m"), gemm_kq.getSetting("sg_per_wg_n"));
    std::printf("host JIT OK: VS binary=%zuB grfMin=%d barriers=%d systolic=%d\n",
                gemm_vs.binary.size(), gemm_vs.grfMin, gemm_vs.barrierCount,
                (int)gemm_vs.systolic);
    std::printf("host JIT OK: KcQ binary=%zuB grfMin=%d | VcS binary=%zuB grfMin=%d\n",
                gemm_kcq.binary.size(), gemm_kcq.grfMin,
                gemm_vcs.binary.size(), gemm_vcs.grfMin);

    if (P.gen_only) {
        std::printf("--gen-only: host-side microkernel generation succeeded; "
                    "skipping GPU build/run.\n");
        return 0;
    }

    // =======================================================================
    // 4. Build + fuse the kernel.
    // =======================================================================
    std::string build_opts =
        "-cl-mad-enable -cl-std=CL3.0"
        " -Dcl_intel_dot_accumulate"
        " -Dcl_intel_global_float_atomic"
        " -Dcl_intel_subgroup_matrix_multiply_accumulate"
        " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    if (gemm_kq.grfMin > 128 || gemm_vs.grfMin > 128 ||
        gemm_kcq.grfMin > 128 || gemm_vcs.grfMin > 128)
        build_opts += " -cl-intel-256-GRF-per-thread";

    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);
    err = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        std::fprintf(stderr, "clBuildProgram failed (%d). Build log:\n%s\n", err, log.c_str());
        std::fprintf(stderr,
                     "NOTE: this kernel needs XMX/systolic HW (BMG Xe2 / PTL Xe3).\n"
                     "      On a gen9/UHD GPU the microkernel shim cannot compile.\n");
        return 4;
    }

    // Patch microkernel machine code into the program binary.
    size_t bin_size = 0;
    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr));
    std::vector<uint8_t> binary(bin_size);
    uint8_t* bin_ptr = binary.data();
    CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(bin_ptr), &bin_ptr, nullptr));

    gemmstone::microkernel::fuse(binary, source.c_str());

    clReleaseProgram(program);
    const uint8_t* fused_ptr = binary.data();
    size_t fused_len = binary.size();
    cl_int bin_status = CL_SUCCESS;
    program = clCreateProgramWithBinary(context, 1, &device, &fused_len, &fused_ptr, &bin_status, &err);
    CL_CHECK(err);
    CL_CHECK(bin_status);
    err = clBuildProgram(program, 1, &device, build_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        std::fprintf(stderr, "rebuild from fused binary failed (%d):\n%s\n", err, log.c_str());
        return 4;
    }
    cl_kernel kernel = clCreateKernel(program, "micro_sdpa", &err);
    CL_CHECK(err);
    std::printf("kernel built + fused OK\n");

    // =======================================================================
    // 5. Host buffers (PA MIXED, int8 BY_TOKEN paged KV + f16 new tokens).
    // =======================================================================
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> udist(-1.0f, 1.0f);

    // The kernel splits each k-block as cached (k0 < past_len, int8 ugemm_kq) vs
    // new (k0 >= past_len, f16 ugemm_kcq).  past_len must be a multiple of the KQ
    // wg_tile_m (itself a multiple of PAGED_ATTENTION_BLOCK_SIZE=16) so the split
    // is block-aligned, and the q new tokens must fit a single new k-block.
    auto round_up = [](int v, int m) { return ((v + m - 1) / m) * m; };
    const int past_len = round_up(std::max(0, P.history), wg_tile_m_kq);
    const int q_new = Lq;                  // new query tokens per subsequence
    const int k_total = past_len + q_new;  // total keys per subsequence
    if (past_len != P.history)
        std::printf("note: history rounded %d -> %d (multiple of kq wg_tile_m=%d)\n",
                    P.history, past_len, wg_tile_m_kq);
    if (q_new > wg_tile_m_kq) {
        std::fprintf(stderr, "ERROR: tokens (%d) must be <= kq wg_tile_m (%d).\n",
                     q_new, wg_tile_m_kq);
        return 6;
    }

    const int adj_head = D + 4;                       // head + f16 scale + f16 zp
    const int blocks_per_seq = past_len / pa_block_size;
    const int num_blocks = B * blocks_per_seq;
    const size_t blk_region = (size_t)adj_head * pa_block_size;  // bytes per (block,head)
    const int total_new = B * q_new;

    // Q / A : [total_new, HEADS_NUM, HEAD_SIZE] f16
    const size_t q_elems = (size_t)total_new * Hq * D;
    const size_t a_elems = q_elems;
    std::vector<uint16_t> Q_h(q_elems);
    for (auto& v : Q_h) v = float_to_half(udist(rng) * 0.5f);

    // GQA-shared decode repacks Q so that one work-group's KQ N-columns hold all
    // (token, query-head-in-group) pairs of a KV group contiguously:
    //   Qp[seq][kv_head][token][head_in_group][d]  (token-major, head-minor),
    // i.e. column c = token*(heads/kv_heads) + head_in_group.  The kernel then
    // reads q_new*kv_group_size rows at a HEAD_SIZE stride per (seq, kv_head).
    // Total element count is unchanged (kv_heads*kv_group_size == heads).
    std::vector<uint16_t> Qp_h;
    if (P.gqa_share) {
        Qp_h.resize(q_elems);
        for (int i = 0; i < B; i++)
            for (int g = 0; g < Hkv; g++)
                for (int t = 0; t < q_new; t++)
                    for (int hg = 0; hg < kv_group_size; hg++) {
                        const int hq = g * kv_group_size + hg;
                        const size_t src = ((size_t)(i * q_new + t) * Hq + hq) * D;
                        const size_t dst =
                            ((((size_t)(i * Hkv + g) * q_new + t) * kv_group_size) + hg) * D;
                        std::memcpy(&Qp_h[dst], &Q_h[src], (size_t)D * 2);
                    }
    }

    // New-token Kc / Vc : [total_new, KV_HEADS_NUM, HEAD_SIZE] f16
    const size_t newc_elems = (size_t)total_new * Hkv * D;
    std::vector<uint16_t> Kc_h(newc_elems), Vc_h(newc_elems);
    for (auto& v : Kc_h) v = float_to_half(udist(rng) * 0.5f);
    for (auto& v : Vc_h) v = float_to_half(udist(rng) * 0.5f);

    // Paged int8 KV cache (shared across subsequences):
    //   K cache: [num_blocks, KV_HEADS, D+4, 16]  data K(d,t) at byte d*16 + t
    //   V cache: [num_blocks, KV_HEADS, 16, D+4]  data V(t,d) at byte t*D  + d
    //   both:    f16 scale[t] at byte 16*D + 2t, zp[t] at 16*D + 32 + 2t
    std::vector<int8_t> Kcache((size_t)num_blocks * Hkv * blk_region, 0);
    std::vector<int8_t> Vcache((size_t)num_blocks * Hkv * blk_region, 0);

    // Dequantized f32 "ground truth" for the cached past tokens, so the reference
    // sees exactly the values the kernel reads back.  Indexed [seq, kv_head, t, d].
    std::vector<float> Kpast((size_t)B * Hkv * past_len * D);
    std::vector<float> Vpast((size_t)B * Hkv * past_len * D);

    auto put_half = [](std::vector<int8_t>& buf, size_t byte_off, uint16_t h) {
        buf[byte_off + 0] = (int8_t)(h & 0xFF);
        buf[byte_off + 1] = (int8_t)((h >> 8) & 0xFF);
    };
    // Per-token asymmetric affine quant (q = round(v/scale) + zp); kernel dequant
    // is (q - zp) * scale.  Fills qout[D], returns the f16-rounded scale used.
    auto quantize_token = [&](const std::vector<float>& vals, std::vector<int8_t>& qout,
                              uint16_t& scale_h, uint16_t& zp_h) {
        float vmin = 1e30f, vmax = -1e30f;
        for (int d = 0; d < D; d++) { vmin = std::min(vmin, vals[d]); vmax = std::max(vmax, vals[d]); }
        float scale = (vmax - vmin) / 255.0f;
        if (scale < 1e-8f) scale = 1e-8f;
        float zp_f = -128.0f - vmin / scale;
        zp_h = float_to_half(zp_f);
        float zp_used = half_to_float(zp_h);
        scale_h = float_to_half(scale);
        float scale_used = half_to_float(scale_h);
        qout.resize(D);
        for (int d = 0; d < D; d++) {
            int qi = (int)std::lround(vals[d] / scale + zp_used);
            qi = std::max(-128, std::min(127, qi));
            qout[d] = (int8_t)qi;
        }
        return scale_used;
    };

    // Fill the paged cache + Kpast/Vpast ground truth (one quant per past token).
    std::vector<float> vbuf(D);
    std::vector<int8_t> qbuf(D);
    for (int i = 0; i < B; i++) {
        for (int h = 0; h < Hkv; h++) {
            for (int t = 0; t < past_len; t++) {
                int g = i * blocks_per_seq + (t / pa_block_size);  // global block id
                int tt = t % pa_block_size;                        // token within block
                size_t base = ((size_t)g * Hkv + h) * blk_region;
                // ---- K (dim-major): data byte d*16 + tt ----
                for (int d = 0; d < D; d++) vbuf[d] = udist(rng);
                uint16_t ksc, kzp;
                float ksc_u = quantize_token(vbuf, qbuf, ksc, kzp);
                float kzp_u = half_to_float(kzp);
                for (int d = 0; d < D; d++) {
                    Kcache[base + (size_t)d * pa_block_size + tt] = qbuf[d];
                    Kpast[(((size_t)i * Hkv + h) * past_len + t) * D + d] =
                        ((float)qbuf[d] - kzp_u) * ksc_u;
                }
                put_half(Kcache, base + (size_t)16 * D + 2 * tt, ksc);
                put_half(Kcache, base + (size_t)16 * D + 32 + 2 * tt, kzp);
                // ---- V (token-major): data byte tt*D + d ----
                for (int d = 0; d < D; d++) vbuf[d] = udist(rng);
                uint16_t vsc, vzp;
                float vsc_u = quantize_token(vbuf, qbuf, vsc, vzp);
                float vzp_u = half_to_float(vzp);
                for (int d = 0; d < D; d++) {
                    Vcache[base + (size_t)tt * D + d] = qbuf[d];
                    Vpast[(((size_t)i * Hkv + h) * past_len + t) * D + d] =
                        ((float)qbuf[d] - vzp_u) * vsc_u;
                }
                put_half(Vcache, base + (size_t)16 * D + 2 * tt, vsc);
                put_half(Vcache, base + (size_t)16 * D + 32 + 2 * tt, vzp);
            }
        }
    }

    // ---- PA index buffers (INPUT3_TYPE = int32) ----
    auto ceil_div = [](long long a, long long b) { return (a + b - 1) / b; };
    std::vector<int32_t> subsequence_begins(B + 1);
    for (int i = 0; i <= B; i++) subsequence_begins[i] = i * q_new;
    std::vector<int32_t> past_lens(B, past_len);
    std::vector<int32_t> block_indices(num_blocks);
    for (int g = 0; g < num_blocks; g++) block_indices[g] = g;
    std::vector<int32_t> block_indices_begins(B + 1);
    for (int i = 0; i <= B; i++) block_indices_begins[i] = i * blocks_per_seq;

    // blocked_indexes_start_and_gws_mapping: per query-block pair
    //   [ global_new_token_start, subsequence_index ].
    const int q_blocks_per_seq = (int)ceil_div(q_new, tile_q);
    const int num_q_blocks = B * q_blocks_per_seq;
    std::vector<int32_t> gws_map;
    gws_map.reserve((size_t)num_q_blocks * 2);
    for (int i = 0; i < B; i++)
        for (int j = 0; j < q_new; j += tile_q) {
            gws_map.push_back(subsequence_begins[i] + j);
            gws_map.push_back(i);
        }

    // ---- scale input (1 element, f16): 1/sqrt(D) ----
    uint16_t scale_val = float_to_half(1.0f / std::sqrt((float)D));

    // Device buffers.
    auto mk_ro = [&](size_t bytes, const void* host) {
        cl_int e;
        cl_mem m = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bytes, const_cast<void*>(host), &e);
        CL_CHECK(e);
        return m;
    };
    cl_mem d_K = mk_ro(Kcache.size(), Kcache.data());
    cl_mem d_Q = mk_ro(Q_h.size() * 2, P.gqa_share ? Qp_h.data() : Q_h.data());
    cl_mem d_V = mk_ro(Vcache.size(), Vcache.data());
    cl_mem d_Kc = mk_ro(Kc_h.size() * 2, Kc_h.data());
    cl_mem d_Vc = mk_ro(Vc_h.size() * 2, Vc_h.data());
    cl_mem d_sb = mk_ro(subsequence_begins.size() * 4, subsequence_begins.data());
    cl_mem d_pl = mk_ro(past_lens.size() * 4, past_lens.data());
    cl_mem d_bi = mk_ro(block_indices.size() * 4, block_indices.data());
    cl_mem d_bib = mk_ro(block_indices_begins.size() * 4, block_indices_begins.data());
    cl_mem d_scale = mk_ro(2, &scale_val);
    cl_mem d_gws = mk_ro(gws_map.size() * 4, gws_map.data());
    cl_int e_out;
    cl_mem d_A = clCreateBuffer(context, CL_MEM_WRITE_ONLY, a_elems * 2, nullptr, &e_out);
    CL_CHECK(e_out);

    // Argument order (PA MIXED, int8 BY_TOKEN, WITH_SCALE, no mask/sink/qq_bias):
    //   K, Q, V, Kc, Vc, A, subsequence_begins, past_lens, block_indices,
    //   block_indices_begins, scale_ptr, blocked_indexes_start_and_gws_mapping
    cl_uint ai = 0;
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_K));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_Q));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_V));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_Kc));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_Vc));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_sb));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_pl));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_bi));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_bib));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_scale));
    CL_CHECK(clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_gws));

    // Dispatch (mirror get_dispatch_data_func, PA MIXED):
    //   LWS = {sg, sg_per_wg, 1}; one work-group per (query-block, query-head).
    //   GWS = {sg * num_q_blocks, sg_per_wg * HEADS_NUM, 1}  (sequences flattened).
    //   GQA-shared: the second dim spans KV heads (one WG per KV group), so a
    //   single work-group produces all query heads of that group at once.
    const int sg_per_wg = sg_per_wg_m_kq * sg_per_wg_n_kq;
    const int grid_heads = P.gqa_share ? Hkv : Hq;
    size_t lws[3] = {(size_t)sg, (size_t)sg_per_wg, 1};
    size_t gws[3] = {(size_t)sg * num_q_blocks, (size_t)sg_per_wg * grid_heads, 1};
    std::printf("dispatch: GWS={%zu,%zu,%zu} LWS={%zu,%zu,%zu} "
                "(q_blocks=%d wg_tile_q=%d sg_per_wg=%d past_len=%d)\n",
                gws[0], gws[1], gws[2], lws[0], lws[1], lws[2],
                num_q_blocks, tile_q, sg_per_wg, past_len);

    // Warmup + correctness run.
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));

    std::vector<uint16_t> A_h(a_elems);
    CL_CHECK(clEnqueueReadBuffer(queue, d_A, CL_TRUE, 0, A_h.size() * 2, A_h.data(), 0, nullptr, nullptr));

    // GQA-shared decode writes A in the same repacked [seq][kv_head][token]
    // [head_in_group][d] layout as Qp; undo it so the comparison below can use
    // the canonical [total_new, HEADS_NUM, HEAD_SIZE] order the reference fills.
    if (P.gqa_share) {
        std::vector<uint16_t> A_un(a_elems);
        for (int i = 0; i < B; i++)
            for (int g = 0; g < Hkv; g++)
                for (int t = 0; t < q_new; t++)
                    for (int hg = 0; hg < kv_group_size; hg++) {
                        const int hq = g * kv_group_size + hg;
                        const size_t src =
                            ((((size_t)(i * Hkv + g) * q_new + t) * kv_group_size) + hg) * D;
                        const size_t dst = ((size_t)(i * q_new + t) * Hq + hq) * D;
                        std::memcpy(&A_un[dst], &A_h[src], (size_t)D * 2);
                    }
        A_h.swap(A_un);
    }

    // =======================================================================
    // 6. f32 reference SDPA (PA MIXED): cached int8 past + f16 new tokens.
    // =======================================================================
    // scale = 1/sqrt(D) (matches the WITH_SCALE input we pass).
    const float ref_scale = 1.0f / std::sqrt((float)D);
    std::vector<uint16_t> A_ref(a_elems);

    double max_abs_err = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0;
    for (int i = 0; i < B; i++) {
        for (int hq = 0; hq < Hq; hq++) {
            int hkv = hq / kv_group_size;
            for (int iq = 0; iq < q_new; iq++) {
                int q_pos = past_len + iq;             // absolute query position
                int tok = subsequence_begins[i] + iq;  // global new-token row
                // 1) logits = scale * (Q . K_j) over all keys [0, k_total)
                std::vector<float> logits(k_total);
                float mx = -1e30f;
                for (int jk = 0; jk < k_total; jk++) {
                    if (P.causal && jk > q_pos) {  // kernel masks key_pos > query_pos
                        logits[jk] = -1e30f;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        float qv = half_to_float(Q_h[((size_t)tok * Hq + hq) * D + d]);
                        float kv;
                        if (jk < past_len) {
                            kv = Kpast[(((size_t)i * Hkv + hkv) * past_len + jk) * D + d];
                        } else {
                            int ntok = subsequence_begins[i] + (jk - past_len);
                            kv = half_to_float(Kc_h[((size_t)ntok * Hkv + hkv) * D + d]);
                        }
                        dot += qv * kv;
                    }
                    logits[jk] = dot * ref_scale;
                    mx = std::max(mx, logits[jk]);
                }
                // 2) softmax
                float denom = 0.0f;
                for (int jk = 0; jk < k_total; jk++) {
                    float e = (logits[jk] <= -1e29f) ? 0.0f : std::exp(logits[jk] - mx);
                    logits[jk] = e;
                    denom += e;
                }
                float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
                // 3) out = sum_j P_j * V_j
                for (int d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int jk = 0; jk < k_total; jk++) {
                        if (logits[jk] == 0.0f) continue;
                        float vv;
                        if (jk < past_len) {
                            vv = Vpast[(((size_t)i * Hkv + hkv) * past_len + jk) * D + d];
                        } else {
                            int ntok = subsequence_begins[i] + (jk - past_len);
                            vv = half_to_float(Vc_h[((size_t)ntok * Hkv + hkv) * D + d]);
                        }
                        acc += logits[jk] * vv;
                    }
                    A_ref[((size_t)tok * Hq + hq) * D + d] = float_to_half(acc * inv_denom);
                }
            }
        }
    }

    // Compare.
    for (size_t i = 0; i < a_elems; i++) {
        float got = half_to_float(A_h[i]);
        float ref = half_to_float(A_ref[i]);
        double e = std::fabs((double)got - ref);
        max_abs_err = std::max(max_abs_err, e);
        sum_sq_err += e * e;
        sum_sq_ref += (double)ref * ref;
    }
    double rel_l2 = std::sqrt(sum_sq_err / std::max(1e-30, sum_sq_ref));
    std::printf("correctness: max_abs_err=%.5f  rel_L2=%.5e  (elems=%zu)\n",
                max_abs_err, rel_l2, a_elems);
    bool pass = rel_l2 < 2e-2;  // int8 KV + f16 accum: a few % is expected
    std::printf("correctness: %s\n", pass ? "PASS" : "FAIL");

    // =======================================================================
    // 7. Benchmark.
    //
    //    Each timed iteration runs a cache-flush kernel *before* the SDPA
    //    kernel so the paged KV / Q / V working set is evicted from L2/L3
    //    between iterations.  Without it, the (small, 512-history) KV cache
    //    stays resident in the ~18 MB L3 across iterations and the measured
    //    time reflects L3 bandwidth, not the DRAM-bound steady-state decode
    //    that happens in a real server where many other kernels run in-between.
    //
    //    The SDPA kernel's own profiling event (START..END) is used for the
    //    per-iter time so the flush kernel is excluded from the reported
    //    number; cliloader additionally reports the average device time of
    //    *every* enqueued kernel by name (micro_sdpa and cache_flush).
    // =======================================================================
    const int iters = P.iters;

    // --- Build the cache-flush kernel (read+write a buffer larger than LLC). ---
    cl_ulong llc_bytes = 0;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(llc_bytes), &llc_bytes, nullptr);
    // Thrash 4x the last-level cache, with a 128 MB floor, so every line is evicted.
    // --flush-mb overrides the size (to demonstrate the cold time is flush-size stable).
    size_t flush_bytes = P.flush_mb > 0
                             ? (size_t)P.flush_mb * 1024 * 1024
                             : std::max<size_t>((size_t)llc_bytes * 4, (size_t)128 * 1024 * 1024);
    flush_bytes &= ~((size_t)4096 - 1);  // page-align
    size_t flush_elems = flush_bytes / sizeof(cl_uint);
    const char* flush_src =
        "__kernel void cache_flush(__global uint* buf) {\n"
        "    uint gid = get_global_id(0);\n"
        "    buf[gid] = buf[gid] * 1664525u + 1013904223u;\n"
        "}\n";
    size_t flush_src_len = std::strlen(flush_src);
    cl_program flush_prog = clCreateProgramWithSource(context, 1, &flush_src, &flush_src_len, &err);
    CL_CHECK(err);
    CL_CHECK(clBuildProgram(flush_prog, 1, &device, "", nullptr, nullptr));
    cl_kernel flush_kernel = clCreateKernel(flush_prog, "cache_flush", &err);
    CL_CHECK(err);
    cl_mem d_flush = clCreateBuffer(context, CL_MEM_READ_WRITE, flush_bytes, nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(flush_kernel, 0, sizeof(cl_mem), &d_flush));
    std::printf("cache-flush: buffer=%zu MB (LLC=%llu KB)  evicts L2/L3 before each timed iter\n",
                flush_bytes >> 20, (unsigned long long)(llc_bytes >> 10));

    // --- Read-only DRAM bandwidth probe kernel (streams the flush buffer). ---
    // Used to validate the achievable sustained *read* bandwidth on this device
    // against the 112 GB/s LPDDR5x datasheet figure used for the roofline.
    const char* dread_src =
        "__kernel void dram_read(__global const uint* buf, __global uint* sink) {\n"
        "    uint gid = get_global_id(0);\n"
        "    uint v = buf[gid];\n"
        "    if (v == 0xdeadbeefu) sink[gid & 255u] = v;\n"  // anti-DCE, never taken
        "}\n";
    size_t dread_src_len = std::strlen(dread_src);
    cl_program dread_prog = clCreateProgramWithSource(context, 1, &dread_src, &dread_src_len, &err);
    CL_CHECK(err);
    CL_CHECK(clBuildProgram(dread_prog, 1, &device, "", nullptr, nullptr));
    cl_kernel dread_kernel = clCreateKernel(dread_prog, "dram_read", &err);
    CL_CHECK(err);
    cl_mem d_sink = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(cl_uint), nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(dread_kernel, 0, sizeof(cl_mem), &d_flush));
    CL_CHECK(clSetKernelArg(dread_kernel, 1, sizeof(cl_mem), &d_sink));

    auto enqueue_flush = [&]() {
        size_t fg[1] = {flush_elems};
        CL_CHECK(clEnqueueNDRangeKernel(queue, flush_kernel, 1, nullptr, fg, nullptr, 0, nullptr, nullptr));
    };

    // Warm a few times (flush + SDPA, same pattern as the timed loop).
    for (int i = 0; i < 5; i++) {
        enqueue_flush();
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, nullptr));
    }
    CL_CHECK(clFinish(queue));

    // Timed loop: flush, then SDPA (profiled via its own event).
    std::vector<cl_event> sdpa_evs(iters, nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        enqueue_flush();
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, &sdpa_evs[i]));
    }
    CL_CHECK(clFinish(queue));
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // Per-iter SDPA device time from its profiling events (excludes the flush).
    double dev_ns_sum = 0.0;
    for (int i = 0; i < iters; i++) {
        cl_ulong s = 0, e = 0;
        CL_CHECK(clGetEventProfilingInfo(sdpa_evs[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr));
        CL_CHECK(clGetEventProfilingInfo(sdpa_evs[i], CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr));
        dev_ns_sum += (double)(e - s);
        clReleaseEvent(sdpa_evs[i]);
    }
    double ms = (dev_ns_sum / iters) / 1e6;  // SDPA device ms/iter (cache-cold avg)

    // --- Warm (cache-resident) SDPA time: run SDPA back-to-back WITHOUT the
    //     flush so the KV stays in L3.  If warm << cold, the flush is genuinely
    //     evicting the KV and the cold number is a true from-DRAM read. ---
    double warm_ms = 0.0;
    {
        for (int i = 0; i < 5; i++)
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));
        std::vector<cl_event> warm_evs(iters, nullptr);
        for (int i = 0; i < iters; i++)
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, gws, lws, 0, nullptr, &warm_evs[i]));
        CL_CHECK(clFinish(queue));
        double warm_ns_sum = 0.0;
        for (int i = 0; i < iters; i++) {
            cl_ulong s = 0, e = 0;
            CL_CHECK(clGetEventProfilingInfo(warm_evs[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr));
            CL_CHECK(clGetEventProfilingInfo(warm_evs[i], CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr));
            warm_ns_sum += (double)(e - s);
            clReleaseEvent(warm_evs[i]);
        }
        warm_ms = (warm_ns_sum / iters) / 1e6;
    }

    // --- Sustained DRAM *read* bandwidth probe (validates the 112 GB/s peak). ---
    // Stream the >LLC flush buffer read-only (1x bytes = pure DRAM read traffic)
    // via the kernel's own profiling events.  This sanity-checks the LPDDR5x
    // datasheet figure that the roofline percentages are computed against.
    double measured_read_gbps = 0.0;
    {
        const int bw_iters = 30;
        std::vector<cl_event> bw_evs(bw_iters, nullptr);
        for (int i = 0; i < bw_iters; i++) {
            // Evict between reads so each pass is a true cold DRAM read.
            size_t fg[1] = {flush_elems};
            CL_CHECK(clEnqueueNDRangeKernel(queue, flush_kernel, 1, nullptr, fg, nullptr, 0, nullptr, nullptr));
            CL_CHECK(clEnqueueNDRangeKernel(queue, dread_kernel, 1, nullptr, fg, nullptr, 0, nullptr, &bw_evs[i]));
        }
        CL_CHECK(clFinish(queue));
        double bw_ns_sum = 0.0;
        for (int i = 0; i < bw_iters; i++) {
            cl_ulong s = 0, e = 0;
            CL_CHECK(clGetEventProfilingInfo(bw_evs[i], CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr));
            CL_CHECK(clGetEventProfilingInfo(bw_evs[i], CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr));
            bw_ns_sum += (double)(e - s);
            clReleaseEvent(bw_evs[i]);
        }
        double bw_ms = (bw_ns_sum / bw_iters) / 1e6;
        measured_read_gbps = ((double)flush_bytes) / (bw_ms * 1e6);  // read-only = 1x bytes
    }

    // FLOPs estimate: 2 * (Q.K) + 2 * (P.V) per (subseq, query-head, query, key, d).
    double flops = 2.0 * 2.0 * (double)B * Hq * q_new * (double)k_total * D;
    double gflops = flops / (ms * 1e6);
    // Bytes moved.  Two views:
    //   - per-head: KV re-read once per *query* head (effective, incl. L3 reuse).
    //   - unique:   KV read once per *kv* head = the cold-cache DRAM traffic.
    //   cached K+V int8 incl per-token scale/zp = past_len * 2 * (D+4) per (seq,head)
    //   new    Kc+Vc f16                        = q_new   * 2 * 2 * D  per (seq,head)
    double cache_bytes = (double)B * Hq * past_len * 2.0 * adj_head;
    double new_bytes = (double)B * Hq * q_new * 4.0 * D;
    double gbps_head = (cache_bytes + new_bytes) / (ms * 1e6);
    double dram_bytes = (double)B * Hkv * past_len * 2.0 * adj_head + (double)B * Hkv * q_new * 4.0 * D;
    double gbps_dram = dram_bytes / (ms * 1e6);
    // PTL 12 Xe-core (Arc B390) LPDDR5x datasheet peak used as the roofline ceiling.
    const double PEAK_DRAM_GBPS = 112.0;
    std::printf("perf: %.4f ms/iter (device, cache-cold)  |  %.1f GFLOP/s  |  "
                "KV/head %.1f GB/s  |  DRAM(unique) %.1f GB/s  (iters=%d)\n",
                ms, gflops, gbps_head, gbps_dram, iters);
    std::printf("perf: %.4f ms/iter (host wall, incl. flush)  |  warm(L3-resident) %.4f ms/iter  |  cold/warm %.2fx\n",
                wall_ms, warm_ms, warm_ms > 0 ? ms / warm_ms : 0.0);
    std::printf("roofline: DRAM peak %.0f GB/s (datasheet; read-probe %.1f)  |  "
                "DRAM(unique, GQA-shared %d:1) %.1f%%  |  KV/head %.1f%% of peak\n",
                PEAK_DRAM_GBPS, measured_read_gbps, Hq / Hkv,
                100.0 * gbps_dram / PEAK_DRAM_GBPS,
                100.0 * gbps_head / PEAK_DRAM_GBPS);
    // Cleanup.
    clReleaseMemObject(d_Q);
    clReleaseMemObject(d_K);
    clReleaseMemObject(d_V);
    clReleaseMemObject(d_Kc);
    clReleaseMemObject(d_Vc);
    clReleaseMemObject(d_sb);
    clReleaseMemObject(d_pl);
    clReleaseMemObject(d_bi);
    clReleaseMemObject(d_bib);
    clReleaseMemObject(d_scale);
    clReleaseMemObject(d_gws);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_flush);
    clReleaseKernel(flush_kernel);
    clReleaseProgram(flush_prog);
    clReleaseMemObject(d_sink);
    clReleaseKernel(dread_kernel);
    clReleaseProgram(dread_prog);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return pass ? 0 : 5;
}
