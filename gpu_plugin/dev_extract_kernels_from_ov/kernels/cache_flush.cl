// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// L2/LLC cache-flush kernel for accurate GGUF kernel micro-benchmarks.
//
// The B580 has an 18 MiB read/write L2. A small GGUF weight (e.g. a 512x1024 Q4_K row set
// ~= 288 KiB) stays fully resident in L2 after the first iteration, so back-to-back timed
// runs would measure L2 bandwidth (~TB/s) instead of the DRAM bandwidth the decode GEMV is
// actually bound by -- inflating the achieved-BW / roofline numbers by 5-10x.
//
// To force every timed iteration to re-read the weight from DRAM, between iterations we
// sweep a scratch buffer several times larger than L2 (default 128 MiB) with a read-modify
// pattern and write a reduction back, so the compiler cannot elide the loads and the whole
// L2 is overwritten with flush data. After this the previous kernel's weight/activation
// lines are guaranteed evicted.
//
// global = [FLUSH_THREADS, 1, 1]; each work-item strides through the buffer.
__kernel void cache_flush(__global float* scratch, const uint n_elems, __global float* sink) {
    const uint gid = get_global_id(0);
    const uint nthreads = get_global_size(0);
    float acc = 0.0f;
    // Touch every element (RMW) so the entire buffer transits L2 and evicts prior lines.
    for (uint i = gid; i < n_elems; i += nthreads) {
        const float v = scratch[i] + 1.0f;
        scratch[i] = v;        // write-back keeps the line dirty -> guaranteed L2 residency churn
        acc += v;
    }
    // Defeat dead-code elimination: park the reduction in a sink the host never reads back hot.
    if (acc == -1.0f)
        sink[gid] = acc;
}
