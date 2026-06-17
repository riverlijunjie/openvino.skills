// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Layout/ABI probe: prints sizeof + member offsets of GEMMProblem so we can
// detect whether the probe TU computes a different struct layout than the
// prebuilt onednn lib (which would explain the segfault in transpose()).

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "gpu/intel/gemm/jit/include/gemmstone/kernel_selector.hpp"

int main() {
    using gemmstone::GEMMProblem;
    GEMMProblem p;
    auto base = reinterpret_cast<char *>(&p);
    std::cout << "__cplusplus           = " << __cplusplus << "\n";
    std::cout << "sizeof(GEMMProblem)   = " << sizeof(GEMMProblem) << "\n";
    std::cout << "sizeof(PostOpsProblem)= " << sizeof(p.postOps) << "\n";
    std::cout << "sizeof(p.postOps.ops) = " << sizeof(p.postOps.ops) << "\n";
    std::cout << "sizeof(product)       = " << sizeof(p.product) << "\n";
    std::cout << "sizeof(binary vec)    = " << sizeof(p.binary) << "\n";
    std::cout << "off(postOps)          = " << (reinterpret_cast<char *>(&p.postOps) - base) << "\n";
    std::cout << "off(product)          = " << (reinterpret_cast<char *>(&p.product) - base) << "\n";
    std::cout << "off(binary)           = " << (reinterpret_cast<char *>(&p.binary) - base) << "\n";
    std::cout << "off(Tbinary)          = " << (reinterpret_cast<char *>(&p.Tbinary) - base) << "\n";
    std::cout << "off(A)                = " << (reinterpret_cast<char *>(&p.A) - base) << "\n";
    std::cout << "off(sroundSeed)       = " << (reinterpret_cast<char *>(&p.sroundSeed) - base) << "\n";
    return 0;
}
