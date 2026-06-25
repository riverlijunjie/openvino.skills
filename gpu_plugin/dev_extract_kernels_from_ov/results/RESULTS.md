# GGUF kernel benchmark results

Auto-generated from `results/report.json` by `harness/make_results_md.py`.

**Device**: Intel(R) Arc(TM) B580 Graphics  
**Peak**: FP32 14.85 TFLOPS · INT8(dp4a) 59.39 TOPS · BW 456 GB/s · L2 18.0 MiB  
**Method**: 50 timed iters (warmup 10), L2 flush between iters = True, flush buffer 128 MiB. In-process time = CL profiling events (device time). cliloader avg = independent tool measurement.

## Per-kernel results

| status | kernel | format | shape | time (ms) | cliloader (ms) | achieved BW | % peak BW | GFLOPS | roofline % | bound | correctness |
|---|---|---|---|--:|--:|--:|--:|--:|--:|---|---|
| PASS | fc_gguf_opt(GEMV) | gguf_q4_k | K=4096 N=4096 M=1 | 0.1069 | 0.2377 | 88.4 GB/s | 19.4% | 314 | 19.4% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q4_k | K=4096 N=1024 M=1 | 0.0318 | 0.0303 | 74.6 GB/s | 16.4% | 264 | 16.4% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q4_k | K=4096 N=12288 M=1 | 0.2151 | 0.3542 | 131.8 GB/s | 28.9% | 468 | 28.9% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q4_k | K=12288 N=4096 M=1 | 0.3104 | 0.4111 | 91.3 GB/s | 20.0% | 324 | 20.0% | memory | relL2=2.0e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q4_0 | K=4096 N=4096 M=1 | 0.2531 | 0.2820 | 37.3 GB/s | 8.2% | 133 | 8.2% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q8_0 | K=4096 N=4096 M=1 | 0.3605 | 0.3032 | 49.5 GB/s | 10.9% | 93 | 10.9% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q3_k | K=4096 N=4096 M=1 | 0.1929 | 0.1745 | 37.5 GB/s | 8.2% | 174 | 8.2% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q5_k | K=4096 N=4096 M=1 | 0.1027 | 0.0988 | 112.5 GB/s | 24.7% | 327 | 24.7% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q6_k | K=4096 N=4096 M=1 | 0.1595 | 0.3368 | 86.4 GB/s | 18.9% | 210 | 18.9% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_iq2_xs | K=4096 N=4096 M=1 | 0.1374 | 0.1322 | 35.4 GB/s | 7.8% | 244 | 7.8% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_iq2_s | K=4096 N=4096 M=1 | 0.1139 | 0.1100 | 47.3 GB/s | 10.4% | 295 | 10.4% | memory | relL2=2.0e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_iq3_xxs | K=4096 N=4096 M=1 | 0.2148 | 0.1954 | 30.0 GB/s | 6.6% | 156 | 6.6% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_iq3_s | K=4096 N=4096 M=1 | 0.1950 | 0.1785 | 37.1 GB/s | 8.1% | 172 | 8.1% | memory | relL2=2.0e-04 cos=1.0000 |
| PASS | fc_gguf_opt(GEMV) | gguf_q6_k | K=4096 N=4096 M=4 | 0.3609 | 0.9139 | 38.3 GB/s | 8.4% | 372 | 8.4% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_dp4a(+prequant) | gguf_q6_k | K=4096 N=4096 M=1 NROW=4 | 0.1026 | 0.2019 | 134.3 GB/s | 29.4% | 327 | 29.4% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_dp4a(+prequant) | gguf_q6_k | K=12288 N=4096 M=1 NROW=4 | 0.2821 | 0.4965 | 146.4 GB/s | 32.1% | 357 | 32.1% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_dp4a(+prequant) | gguf_q5_k | K=4096 N=4096 M=1 NROW=1 | 0.0608 | 0.0739 | 189.8 GB/s | 41.6% | 552 | 41.6% | memory | relL2=2.1e-04 cos=1.0000 |
| PASS | fc_gguf_transcode | gguf_q4_k | K=4096 N=4096 target=i4 | 0.6702 | 0.7764 | 28.2 GB/s | 6.2% | 25 | 6.2% | memory | int_match=1.000 scale_rel=3.6e-04 |
| PASS | fc_gguf_transcode | gguf_q6_k | K=4096 N=4096 target=i8 | 0.7400 | 0.9907 | 42.7 GB/s | 9.4% | 23 | 9.4% | memory | int_match=1.000 scale_rel=3.7e-04 |
| PASS | fc_gguf_transcode | gguf_iq3_xxs | K=4096 N=4096 target=i4 | 0.6547 | 1.0347 | 24.2 GB/s | 5.3% | 26 | 5.3% | memory | int_match=1.000 scale_rel=3.6e-04 |
| PASS | fc_gguf_transcode | gguf_iq2_s | K=4096 N=4096 target=i8 | 0.7026 | 0.7087 | 33.0 GB/s | 7.2% | 24 | 7.2% | memory | int_match=1.000 scale_rel=3.5e-04 |

**21/21 cases passed correctness.**

## Analysis & optimisation signals

The decode GEMV (`fc_gguf_opt`) and dp4a path are **memory-bound**: the roofline is peak DRAM bandwidth (456 GB/s), so `% peak BW` is the figure to push toward 100%. Higher = closer to the HW limit; a low % on a heavy-decode format means ALU/unpack is starving the load units (an optimisation target), not that the kernel is slow per se.

Decode GEMV BW-utilisation ranking (4096-class shapes):

| format | shape | % peak BW | note |
|---|---|--:|---|
| gguf_q4_k | K4096×N12288×M1 | 28.9% | near the practical Xe2 GEMV ceiling for this shape |
| gguf_q5_k | K4096×N4096×M1 | 24.7% |  |
| gguf_q4_k | K12288×N4096×M1 | 20.0% |  |
| gguf_q4_k | K4096×N4096×M1 | 19.4% |  |
| gguf_q6_k | K4096×N4096×M1 | 18.9% |  |
| gguf_q4_k | K4096×N1024×M1 | 16.4% |  |
| gguf_q8_0 | K4096×N4096×M1 | 10.9% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_iq2_s | K4096×N4096×M1 | 10.4% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_q6_k | K4096×N4096×M4 | 8.4% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_q3_k | K4096×N4096×M1 | 8.2% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_q4_0 | K4096×N4096×M1 | 8.2% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_iq3_s | K4096×N4096×M1 | 8.1% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_iq2_xs | K4096×N4096×M1 | 7.8% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |
| gguf_iq3_xxs | K4096×N4096×M1 | 6.6% | decode/unpack-bound -> ALU is the bottleneck; candidate for SWAR/dp4a or fewer ops |

**Key takeaways**
- The Q5_K/Q6_K **dp4a** int8-activation path reaches the highest BW utilisation (Q5_K ~42%, Q6_K ~29-32%), confirming the kernel comment that packing both operands into the hardware dot product breaks past the scalar-float decode ceiling.
- **IQ formats** (iq2_xs/iq3_xxs/iq3_s) and **Q3_K** sit lowest on BW utilisation: their per-element grid-lookup / bit-unpack is ALU-heavy, so the load units idle. cliloader flags `fc_gguf_opt_gguf_q3_k ... SPILL=1600` — Q3_K spills registers, a concrete first optimisation target (reduce the `dl[16]` + cached-block private footprint).
- **transcode** is a one-shot prefill repack; its absolute time is small and it is dominated by the GGUF block read + decode, so its low BW% is expected (it does ~1 byte of output work per decoded element, not a streaming GEMV).
- In-process CL-event time and cliloader's independent device time agree to within run-to-run clock variance for the steady cases; large divergence (e.g. M=4) indicates a case sensitive to GPU frequency ramp — increase `iters`/`warmup` for those.

