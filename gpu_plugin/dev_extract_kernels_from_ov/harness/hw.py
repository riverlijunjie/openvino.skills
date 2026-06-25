# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Hardware capability + roofline model for the GGUF kernel micro-benchmarks.

Defaults target the detected GPU (Intel Arc B580, "Battlemage" / Xe2). Every field is
overridable from the JSON config ("hw" block) or environment, so the same harness can
profile a different Xe device by editing one config section.

Roofline convention (matches the kernel comments, e.g. fc_gguf_opt.cl Q6_K says
"31.7% of B580 BW roofline"): the GGUF decode GEMV reads the whole weight matrix once and
is bandwidth-bound, so the PRIMARY ceiling is peak DRAM bandwidth. We report:

  * achieved_bw   = bytes_moved / time           -> compare to peak_bw  (BW roofline %)
  * achieved_flops= 2*M*N*K   / time             -> compare to peak compute
  * arithmetic intensity AI = flops / bytes
  * attainable    = min(peak_compute, AI*peak_bw)  (classic roofline)
  * roofline_pct  = achieved_flops / attainable    (how close to the ceiling it actually is)

For the int8 dp4a path the compute ceiling used is the dp4a (4x8 packed) INT8 MAC rate.
"""
import os
import json


# B580 (Battlemage BMG-G21) reference numbers. Sourced from clinfo on this machine plus
# Intel's published B580 spec; all overridable.
#   compute_units (XVE)         : 160     (clinfo "Max compute units"; 5 slices*4 ss*8 EU)
#   max_clock_ghz               : 2.90    (clinfo "Max clock frequency" = boost)
#   simd_fp32_lanes_per_cu      : 16      (Xe2 native SIMD16 FP32)
#   mem_bus / data rate         : 192-bit GDDR6 @ 19 Gbps  -> 456 GB/s
#   l2_cache_bytes              : 18 MiB  (clinfo "Global Memory cache size")
_B580_DEFAULTS = {
    "name": "Intel Arc B580 (Battlemage / Xe2)",
    "compute_units": 160,
    "max_clock_ghz": 2.90,
    "simd_fp32_lanes_per_cu": 16,
    "fma_factor": 2,            # multiply + add
    "peak_bw_gbps": 456.0,      # 192-bit GDDR6 @ 19 Gbps
    "l2_cache_bytes": 18 * 1024 * 1024,
    # dp4a (cl_khr_integer_dot_product, 4x8 packed) does 4 INT8 MACs per FP32-FMA issue slot,
    # i.e. ~4x the FP32 MAC rate. int8_mac_factor scales the FP32 MAC peak to the int8 MAC peak.
    "int8_mac_factor": 4,
    # f16 vector ALU runs at ~2x FP32 on Xe2 (not used as the ceiling here; the kernels
    # accumulate in f32, so the FP32 ceiling governs the float decode path).
    "fp16_factor": 2,
}


class HW:
    def __init__(self, cfg=None):
        d = dict(_B580_DEFAULTS)
        if cfg:
            d.update(cfg)
        # env overrides (handy for quick what-ifs without editing the config)
        if os.environ.get("OV_BENCH_PEAK_BW_GBPS"):
            d["peak_bw_gbps"] = float(os.environ["OV_BENCH_PEAK_BW_GBPS"])
        if os.environ.get("OV_BENCH_MAX_CLOCK_GHZ"):
            d["max_clock_ghz"] = float(os.environ["OV_BENCH_MAX_CLOCK_GHZ"])
        self.__dict__.update(d)

    # ---- peak rates ----
    @property
    def fp32_mac_per_s(self):
        """Peak FP32 multiply-add pairs per second (one MAC = 2 FLOP)."""
        return self.compute_units * self.simd_fp32_lanes_per_cu * self.max_clock_ghz * 1e9

    @property
    def fp32_tflops(self):
        return self.fp32_mac_per_s * self.fma_factor / 1e12

    @property
    def int8_mac_per_s(self):
        return self.fp32_mac_per_s * self.int8_mac_factor

    @property
    def int8_tops(self):
        """Peak INT8 dp4a throughput (TOPS), 1 MAC = 2 ops."""
        return self.int8_mac_per_s * 2 / 1e12

    @property
    def peak_bw_bytes_per_s(self):
        return self.peak_bw_gbps * 1e9

    # ---- roofline ----
    def roofline(self, flops, bytes_moved, seconds, compute_domain="fp32"):
        """Return a dict of achieved/peak/roofline metrics for one measured kernel run."""
        peak_compute = self.int8_mac_per_s * 2 if compute_domain == "int8" else self.fp32_mac_per_s * self.fma_factor
        ach_flops = flops / seconds
        ach_bw = bytes_moved / seconds
        ai = flops / bytes_moved if bytes_moved else 0.0
        attainable = min(peak_compute, ai * self.peak_bw_bytes_per_s)
        return {
            "time_ms": seconds * 1e3,
            "achieved_gflops": ach_flops / 1e9,
            "achieved_bw_gbps": ach_bw / 1e9,
            "arithmetic_intensity": ai,
            "peak_compute_gflops": peak_compute / 1e9,
            "peak_bw_gbps": self.peak_bw_gbps,
            "attainable_gflops": attainable / 1e9,
            "bw_util_pct": 100.0 * ach_bw / self.peak_bw_bytes_per_s,
            "compute_util_pct": 100.0 * ach_flops / peak_compute,
            "roofline_pct": 100.0 * ach_flops / attainable if attainable else 0.0,
            "bound_by": "memory" if ai * self.peak_bw_bytes_per_s < peak_compute else "compute",
        }

    def summary(self):
        return {
            "name": self.name,
            "compute_units": self.compute_units,
            "max_clock_ghz": self.max_clock_ghz,
            "fp32_tflops": round(self.fp32_tflops, 2),
            "int8_tops_dp4a": round(self.int8_tops, 2),
            "peak_bw_gbps": self.peak_bw_gbps,
            "l2_cache_MiB": round(self.l2_cache_bytes / 1024 / 1024, 1),
        }


if __name__ == "__main__":
    hw = HW()
    print(json.dumps(hw.summary(), indent=2))
    # Example: a Q6_K 4096x4096 decode reading ~9 MiB weight in 0.5 ms
    r = hw.roofline(flops=2 * 1 * 4096 * 4096, bytes_moved=4096 * 4096 * 210 // 256, seconds=0.5e-3)
    print(json.dumps(r, indent=2))
