// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Minimal compile/link probe for the gemmstone microkernel JIT API.
//
// Purpose: verify that the gemmstone headers compile standalone and that
// selectGEMM / generateShim / fuse link against libopenvino_onednn_gpu.a.
// The microkernel JIT is pure host code (ngen emits GPU ISA in memory), so
// this runs and produces a real microkernel even on a machine whose local
// GPU has no XMX. See build_probe.sh for the exact build command.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/package.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/kernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/shim.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/fuser.hpp"

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

// BMG (Xe2) gmdid: architecture=20, release=1, revision=0
//   raw = (arch << 22) | (release << 14) | revision
static constexpr uint32_t GMDID_BMG = (20u << 22) | (1u << 14) | 0u;  // 0x05004000

int main() {
    std::cout << "GMDID_BMG = 0x" << std::hex << GMDID_BMG << std::dec << "\n";

    micro::HWInformation hw_info;
    hw_info.euCount = 160;
    hw_info.gmdid = GMDID_BMG;
    hw_info.systolicAvailable = true;

    const int k_head_size = 128;
    const int d_max = 128;
    const int subgroup_size = 16;  // Xe2

    // ---- KQ problem (K^T * Q), asymmetric int8 K cache, thin-Q generate ----
    // Value-initialize: MatrixAddressing::layout/alignment have NO default
    // member initializers, so a plain `GEMMProblem p;` leaves them indeterminate
    // for matrices we don't explicitly configure (BO/CO/B_scale/Ag/Bg/C_scale).
    // selectGEMM()/transpose() read those fields -> garbage -> bad_alloc/SIGSEGV.
    micro::GEMMProblem problem{};
    problem.Ta_ext = micro::Type::s8;   // K int8
    problem.Tb_ext = micro::Type::f16;  // Q f16
    problem.Ta = problem.Tb = micro::Type::f16;
    problem.Tc = problem.Tc_ext = micro::Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.A.layout = micro::MatrixLayout::T;

    micro::GEMMOptions opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;

    // scales
    problem_kq.Ta_scale = micro::Type::f16;
    problem_kq.A_scale.setAlignment(micro::Type(micro::Type::f16).size());
    problem_kq.A_scale.layout = micro::MatrixLayout::N;
    problem_kq.asPtrDims = 2;
    // zero points (asymmetric)
    problem_kq.Tao = micro::Type::f16;
    problem_kq.AO.setAlignment(micro::Type(micro::Type::f16).size());
    problem_kq.AO.layout = micro::MatrixLayout::N;
    problem_kq.aoPtrDims = 2;
    problem_kq.aOffset = micro::ABOffset::Calc;
    problem_kq.aqGroupM = 1;
    problem_kq.aqGroupK = k_head_size;
    opts_kq.scaleA = true;
    opts_kq.offsetA = true;

    problem_kq.B.layout = micro::MatrixLayout::Pr;
    problem_kq.C.layout = micro::MatrixLayout::T;
    problem_kq.A.setAlignment(gemmstone::microkernel::alignmentForLD(static_cast<int>(k_head_size * problem.Ta)));
    problem_kq.B.setAlignment(64);
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = static_cast<uint16_t>(d_max);
    problem_kq.B.tileC = static_cast<uint16_t>(subgroup_size);

    micro::SizeParams sizes;
    sizes.m = 512;   // n_keys (Lk)
    sizes.n = 1;     // n_queries (Lq), generate
    sizes.k = k_head_size;
    sizes.batch = 32;  // B*H

    // xehpc_q_h128_2nd = {16,16,16,16,16,1,16,1}
    std::vector<micro::StrategyRequirement> reqs_kq;
    reqs_kq.push_back(micro::StrategyRequirement::UnrollM == 16);
    reqs_kq.push_back(micro::StrategyRequirement::UnrollN == 16);
    reqs_kq.push_back(micro::StrategyRequirement::WGM == 16);
    reqs_kq.push_back(micro::StrategyRequirement::WGN == 1);

    micro::Package gemm_kq;
    try {
        gemm_kq = gemmstone::microkernel::selectGEMM(opts_kq, hw_info, sizes, problem_kq, reqs_kq);
    } catch (const std::exception& ex) {
        std::cerr << "selectGEMM(KQ) failed: " << ex.what() << "\n";
        return 2;
    }

    std::cout << "KQ microkernel selected:\n";
    std::cout << "  binary size = " << gemm_kq.binary.size() << " bytes\n";
    std::cout << "  grfMin = " << gemm_kq.grfMin << "\n";
    std::cout << "  barrierCount = " << gemm_kq.barrierCount << "\n";
    std::cout << "  systolic = " << gemm_kq.systolic << "\n";
    std::cout << "  wg_tile_m = " << gemm_kq.getSetting("wg_tile_m") << "\n";
    std::cout << "  wg_tile_n = " << gemm_kq.getSetting("wg_tile_n") << "\n";
    std::cout << "  sg_per_wg_m = " << gemm_kq.getSetting("sg_per_wg_m") << "\n";
    std::cout << "  sg_per_wg_n = " << gemm_kq.getSetting("sg_per_wg_n") << "\n";

    // ---- generate a shim ----
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = subgroup_size;
    shim_options.useTileOps = true;
    shim_options.decorator = "kq";
    std::string shim = gemmstone::microkernel::generateShim(gemm_kq, micro::HostLanguage::OpenCL_C, shim_options);
    std::cout << "KQ shim length = " << shim.size() << " chars\n";

    // ---- exercise fuser API (no real binary, just link check) ----
    std::vector<uint8_t> dummy_binary;
    bool has_mk = gemmstone::microkernel::hasMicrokernels(shim.c_str());
    std::cout << "shim hasMicrokernels = " << has_mk << "\n";

    std::cout << "PROBE OK\n";
    return 0;
}
