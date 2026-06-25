// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Minimal stand-in for OpenVINO's include/batch_headers/fetch_data.cl.
//
// The GGUF FC kernels only consume the tensor-addressing macros INPUT0_GET_INDEX /
// OUTPUT_GET_INDEX / INPUT0_FEATURE_NUM / OUTPUT_FEATURE_NUM. In OpenVINO those are emitted
// per-build by the JIT layout generator (make_layout_jit_constants), NOT by fetch_data.cl
// itself -- fetch_data.cl supplies generic GET_DATA_*_INDEX helpers that the full layout
// expansion may call. Our harness emits the reduced, self-contained GET_INDEX macros for a
// contiguous bfyx [BM,K]/[BM,N] tensor directly in the JIT preamble (see ov_jit.py), so no
// fetch_data.cl helper is referenced. This empty stub satisfies the kernel's #include while
// keeping addressing byte-identical to OV for this primitive.
