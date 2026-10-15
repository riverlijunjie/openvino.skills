# SeedVR2 Roofline Analysis — PTL — 480P→2560×1536 variant (2026-10-15)

**Platform**: Intel PTL 12Xe iGPU (Machine 1, `10.239.132.213`), 2400 MHz, 110 GB/s spec BW (105 GB/s measured read), FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS.

**Model**: SeedVR2 — VAE encoder + windowed-attention DiT + VAE decoder (video super-resolution/restoration).

**Test configuration** (fixed by the task, not swept):
- Input video: 864×496, 121 frames (same source as the 1920×1088 config — `SeedVR2/input/480p-sc-121f-864x496.rgb24`)
- **Output video: 2560×1536, 121 frames** (the new resolution for this report)
- Text condition: 58 tokens
- Activations/weights: F16 throughout, no quantization
- **VAE decoder explicitly modeled UNTILED** (full-frame, no 2×2 spatial tile grid), per explicit request for this profiling exercise — this is the key methodology difference vs. the existing 1920×1088 report, whose decoder uses the real K=4 tile grid.

## 1. Methodology

**No real `OV_VERBOSE` trace exists at this resolution.** The 1920×1088 report (`outputs/seedvr2/SUMMARY_seedvr2_2026-10-14.md`) is grounded in a real `OV_VERBOSE=4` execution trace of the actual compiled IR. No equivalent trace was captured at 2560×1536 — all shapes/call-counts here are a **first-principles re-derivation**, scaling the 1920×1088 config's already-validated architecture using the same tier pattern (full → /2 → /4 → /8-latent) and formulas, **not** an independent real-trace validation. Treat this report's shape grounding as lower-confidence than the 1920×1088 report's.

**Derivation approach**:
1. **Resolution tiers** (both configs follow the same full→/2→/4→/8(latent) pattern):

   | Tier | 1920×1088 | 2560×1536 |
   |---|---|---|
   | full | 1088×1920 | 1536×2560 |
   | /2 | 544×960 | 768×1280 |
   | /4 | 272×480 | 384×640 |
   | /8 (latent) | 136×240 | 192×320 |

2. **Architecture constants** (resolution-independent, copied verbatim): DiT `hidden_size=2560, qkv_size=7680, intermediate_size=6912, num_heads=20, head_dim=128, num_layers=32, txt_tokens=58`. VAE encoder uses `plain` convs (0 `CausalConv3DCarry` nodes); VAE decoder uses `causal` convs (33 `CausalConv3DCarry` nodes) with identical `t_body`/kernel/stride values (temporal structure is resolution-independent).
3. **DiT `video_tokens`** via the empirically-confirmed (at 1920×1088) formula `T_latent × H_latent × W_latent / 4` (patchify factor 4): latent grid 192×320, T=31 (unchanged 121-frame video) → `31 × 192 × 320 / 4 = 476,160` (vs. 252,960 at 1920×1088).
4. **DiT windowed-attention `num_windows`** — **NOT independently derived** (no real log at this resolution): approximated by scaling the 1920×1088 config's window counts proportionally with the `video_tokens` ratio (476160/252960 = 1.8822×): `wattn_window_a: 100→188`, `wattn_window_b: 150→282`. **This is an approximation, flagged explicitly** — the real windowing scheme may not scale linearly with token count.
5. **VAE decoder: explicitly UNTILED**, per request — full-tier dimensions used directly (same convention as the encoder), with BASE call counts only (1 INIT + 29 ACTIVE = 30 segments, no ×4 tile multiplier), unlike the 1920×1088 config's real, log-confirmed K=4 (2×2 grid) tiling.

All 56 benchmarked shapes/call-counts live in `models/seedvr2_2560x1536/ops_mapping.json` (generated/kept in sync via `models/seedvr2_2560x1536/generate_bench_config.py`, mirroring `models/seedvr2/generate_bench_config.py`'s structure).

**Sweep execution note** (methodology caveat, not a scientific finding): the first sweep attempt, launched via a Windows `schtasks /Create ... /SC ONCE` + `/Run` scheduled task (the same pattern used successfully for the 1920×1088 decoder-retile sweep), silently **re-triggered itself multiple times** — confirmed via the `_index.txt` log showing several premature `=== END` markers only minutes apart and at least one spec (`conv_dec_up2_conv1`) re-executed — corrupting the log directory with overlapping/duplicate runs. This was fixed by ending/deleting the scheduled task, killing all stray processes, deleting the contaminated log directory, and **relaunching via a single direct blocking SSH call** instead (no Task Scheduler). The resulting run completed 50/56 specs cleanly; the remaining 6 (lost to a transient recovery-process race, not the duplicate-trigger bug) were individually re-run via a small standalone recovery script using the exact same `cliloader`-wrapped invocation. All 56 final logs were verified non-empty before parsing. **Lesson recorded for future sessions**: prefer a direct blocking SSH call over `schtasks ONCE` for long remote sweeps.

## 2. Hardware roofline (PTL)

| Metric | Value |
|---|---|
| Xe Cores | 12 |
| Frequency | 2400 MHz |
| Memory BW (spec / measured read) | 110 / 105 GB/s |
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.964 TOPS |

## 3. Measured kernel-time totals (one full 121-frame, 864×496→2560×1536 inference)

Per-kernel breakdown (average latency, calls/inference, total ms, GFLOPS, GB/s, efficiency %, memory/compute-bound) for all 56 benchmarked ops is in [kernel_tables.md](kernel_tables.md). The tables below are the **op-level** (kernels summed) bottleneck ranking, reproduced via `outputs/seedvr2_2560x1536/build_op_totals.py`.

### 3.1 VAE Encoder — total ≈ **181.72 s**

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | down0.conv1 (128→128, H1536×W2560) | 61,456 | 33.8% | compute |
| 2 | down1.conv2 (256→256, H768×W1280) | 36,744 | 20.2% | compute |
| 3 | gn_c128_h1536w2560 (GroupNormalizationZSplit) | 17,857 | 9.8% | memory |
| 4 | down2.conv2 (512→512, H384×W640) | 16,436 | 9.0% | compute |
| 5 | mid_attn_sdpa (single-head, 61,440 tokens) | 15,864 | 8.7% | compute |
| 6 | gn_c256_h768w1280 | 7,111 | 3.9% | memory |
| 7 | down1.conv1 | 6,247 | 3.4% | compute |
| — | *(remaining 12 ops)* | 20,006 | 11.0% | mixed |

### 3.2 DiT — total ≈ **156.11 s**

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | vid_qkv (FC, 476160×2560×7680) | 42,379 | 27.1% | compute |
| 2 | vid_mlp_out (FC, 476160×6912×2560) | 36,842 | 23.6% | compute |
| 3 | vid_mlp_up (FC, 476160×2560×6912) | 28,516 | 18.3% | compute |
| 4 | vid_mlp_gate (FC, 476160×2560×6912) | 23,902 | 15.3% | compute |
| 5 | wattn_window_a (VLSDPA, 188 windows, approx.) | 10,519 | 6.7% | compute |
| 6 | wattn_window_b (VLSDPA, 282 windows, approx.) | 7,268 | 4.7% | compute |
| 7 | vid_attn_out (FC, 476160×2560×2560) | 6,530 | 4.2% | compute |
| — | *(txt_* ops, vid_in/out — tiny)* | 150 | 0.1% | compute |

Same pattern as the 1920×1088 config: `txt_*` FC ops (M=58) contribute <0.1% combined — fully dominated by the `vid_*` (M=476,160) ops.

### 3.3 VAE Decoder — total ≈ **721.82 s** (the dominant stage: 3.97× the encoder, 4.62× DiT)

**Modeled UNTILED, full-frame shapes** (§1) — no tile multiplier, unlike the 1920×1088 config's real K=4-tile figures. The totals below are therefore NOT directly comparable to the 1920×1088 decoder table's per-op values (different shape AND tiling convention).

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | up2.post_shuffle_conv (256→256, full H768×W1280) | 193,519 | 26.8% | compute |
| 2 | up1.post_shuffle_conv (512→512, full H384×W640) | 98,585 | 13.7% | compute |
| 3 | up3.conv2 (128→128, full H1536×W2560) | 97,223 | 13.5% | compute |
| 4 | up2.conv2 | 81,673 | 11.3% | compute |
| 5 | up3.conv1 | 46,867 | 6.5% | compute |
| 6 | up1.conv1 | 45,959 | 6.4% | compute |
| 7 | gn_c128_h1536w2560 | 37,084 | 5.1% | memory |
| 8 | up2.conv1 | 34,351 | 4.8% | compute |
| 9 | gn_c256_h768w1280 | 18,662 | 2.6% | memory |
| — | *(remaining 13 ops)* | 67,898 | 9.4% | mixed |

## 4. Overall roofline picture

| Stage | Total kernel ms | % of whole pipeline | Dominant bound |
|---|---:|---:|---|
| VAE Encoder | 181,721 | 17.1% | compute |
| DiT | 156,107 | 14.7% | compute |
| VAE Decoder | 721,820 | **68.1%** | compute (untiled full-frame conv) |
| **Total (kernel-only)** | **1,059,648 ms ≈ 17.66 min** | 100% | — |

**The VAE decoder dominates total latency (68.1%)** — nearly identical to the 1920×1088 config's own decoder share (67.8%), despite the completely different tiling convention (real K=4 tiles there vs. untiled here). This is the expected outcome: per-tile-shape×4-tiles and full-frame×1 cover roughly the same total spatial area, so the dominant-stage *share* of total time is relatively tiling-convention-invariant even though the *absolute* per-op numbers are not comparable across the two reports (§3.3).

### 4.1 Overall hardware utilization (time-weighted roofline efficiency)

Reproduced via `outputs/seedvr2/build_utilization_report.py outputs/seedvr2_2560x1536/kernel_tables.md` (the script is resolution-agnostic, no copy needed).

| Stage | Compute-bound time | Avg % of FP16 XMX peak | Memory-bound time | Avg % of peak BW | Blended utilization |
|---|---:|---:|---:|---:|---:|
| VAE Encoder | 84.9% of stage | 46.1% | 15.1% of stage | 62.2% | **48.5%** |
| DiT | 100.0% of stage | 30.0% | — | — | **30.0%** |
| VAE Decoder | 91.8% of stage | 29.0% | 8.2% of stage | 61.8% | **31.7%** |
| **Overall (all 3 stages)** | 91.8% of total | **31.9%** | 8.2% of total | **62.0%** | **≈ 34.4%** |

**Overall, this sweep achieves ≈34.4% of the hardware's real capability** — notably *lower* than the 1920×1088 config's ≈47.3%. The gap is concentrated in the untiled decoder's own efficiency (31.7% here vs. 45.1% at 1920×1088) and DiT's (30.0% here vs. 46.6% at 1920×1088): both stages now operate at a larger `M`/spatial-extent per call (full-frame convs instead of quarter-area tiles; `video_tokens=476160` instead of `252960`), and the larger GEMM/conv shapes are evidently *less* XMX-efficient at this specific shape class on this hardware, not more — i.e. simply scaling up the problem size does not improve utilization here, it measurably worsens it.

## 5. Peak GPU memory analysis

### 5.1 Per-op isolated bench peaks (what each op needs on its own)

Reproduced via `python3 models/seedvr2/build_memory_report.py outputs/seedvr2_2560x1536/ptl_metrics.json 20` (script is resolution-agnostic).

| Op (top 10 by total GPU memory) | usm_device (GiB) | usm_host (GiB) | Total (GiB) |
|---|---:|---:|---:|
| conv_enc_conv_in | 3.926 | 15.062 | **18.988** |
| conv_enc_down0_conv1 | 9.376 | 3.812 | 13.188 |
| conv_dec_up2_post_shuffle_conv (untiled) | 7.503 | 3.813 | 11.316 |
| conv_dec_up3_shortcut (untiled) | 5.625 | 5.688 | 11.313 |
| conv_enc_down1_conv1 | 3.283 | 7.562 | 10.845 |
| conv_enc_down1_conv2 | 4.691 | 5.688 | 10.378 |
| conv_enc_down1_shortcut | 2.813 | 7.562 | 10.375 |
| conv_dec_up1_upscale_conv (untiled) | 2.113 | 7.562 | 9.676 |
| conv_dec_up1_post_shuffle_conv (untiled) | 3.763 | 5.688 | 9.451 |
| conv_dec_up3_conv1 (untiled) | 7.502 | 1.938 | 9.439 |

**Peak single-op GPU memory observed: 18.988 GiB — within the 20 GiB budget, but with only 1.012 GiB headroom** (vs. 8.98 GiB headroom at 1920×1088). The single largest isolated op is now the VAE encoder's own full-frame `conv_in` (the encoder is untiled at BOTH resolutions, so this scales directly with output resolution), closely followed by several of the now-untiled decoder's own largest ops — this budget is noticeably tighter than at 1920×1088, as expected at ~1.78× the pixel count with no tiling applied anywhere.

### 5.2 Dependency-aware (whole-model) peak, per submodel AND whole pipeline

Reproduced via `python3 models/seedvr2/build_dependency_memory_report.py models/seedvr2_2560x1536/ops_mapping.json` (the script is now dual-config-aware via a `decoder_tiled` flag, added this session).

| Submodel | Resident weight | Peak activation | **Submodel peak** | Peak location |
|---|---:|---:|---:|---|
| VAE encoder (untiled, 1 compiled model, full frame) | 0.057 GiB | 15.000 GiB | **15.057 GiB** | `down0.conv1` |
| **DiT, NO graph split (32 layers' weights resident at once)** | **9.478 GiB** | 14.533 GiB | **24.011 GiB** | during MLP (`hidden` + `gate_out` + `up_out` concurrently live) |
| *(context only)* DiT, WITH graph split (as actually executed — resolution-independent architectural choice) | 0.295 GiB (1-of-32 layers) | 14.533 GiB | **14.828 GiB** structured (no real measured reference exists at this resolution — see §1) | same |
| VAE decoder, **UNTILED** (per this report's config) | 0.091 GiB | 15.000 GiB | **15.091 GiB** | `up3.conv1` |

**Whole-pipeline peak GPU memory** — the three submodels run sequentially (confirmed at 1920×1088 via real `[PERF] module run=` markers; assumed architecturally unchanged here), so the pipeline's peak is the **MAX across stages, not the sum**:

| VAE encoder | DiT (NO graph split) | DiT (WITH graph split) | VAE decoder (untiled) | **Pipeline peak, NO-split scenario** | **Pipeline peak, WITH-split scenario (realistic)** |
|---:|---:|---:|---:|---:|---:|
| 15.057 GiB | 24.011 GiB | 14.828 GiB | 15.091 GiB | **24.011 GiB** (DiT) | **15.091 GiB** (decoder) |

**⚠️ Important finding: the NO-graph-split DiT scenario's pipeline peak (24.011 GiB) EXCEEDS the 20 GiB budget used for the 1920×1088 analysis, by 4.011 GiB.** This scenario was introduced in the 1920×1088 report purely as an analysis convention (comparing a hypothetical "what if DiT weren't split by layer" case) — **DiT's graph-splitting-by-layer is a resolution-independent architectural/runtime choice**, not something this profiling exercise's "no tile splitting" request changes. If the real production pipeline continues to graph-split DiT at this resolution too (as it does, confirmed, at 1920×1088), the **realistic whole-pipeline peak is 15.091 GiB** (driven by the now-untiled VAE decoder, since it's the largest of the three WITH-split figures) — comfortably within the 20 GiB budget, with ≈4.9 GiB headroom.

**Practical reading**: at this resolution, with the decoder run untiled (per this request) but DiT still graph-split (its normal/expected behavior), the pipeline should fit comfortably in a 20 GiB budget (15.09 GiB peak). The 24.01 GiB figure only applies if DiT's layer-by-layer graph splitting were ALSO disabled — which is a different, independent optimization axis from "VAE decoder tiling" and was not part of this request. This distinction is worth flagging explicitly since both the 1920×1088 and 2560×1536 reports present the no-split DiT number as "the pipeline driver" by convention — at 2560×1536 that convention's number actually crosses the budget line, so it deserves attention even though the realistic (graph-split) number does not.

## 6. Comparison vs. the 1920×1088 config

| Metric | 1920×1088 (decoder tiled, K=4) | 2560×1536 (decoder untiled) |
|---|---:|---:|
| Output pixels | 2,088,960 | 3,932,160 (1.883×) |
| `video_tokens` (DiT) | 252,960 | 476,160 (1.883×) |
| Total kernel time | 422.31 s | 1,059.65 s (2.51×) |
| VAE Decoder share of total | 67.8% | 68.1% |
| Overall blended utilization | ≈47.3% | ≈34.4% |
| Isolated-op peak memory (20 GiB budget) | 11.02 GiB (8.98 GiB headroom) | 18.99 GiB (1.01 GiB headroom) |
| Pipeline peak, DiT WITH split (realistic) | 8.94 GiB (4-tile decoder reference) / driven by real-measured DiT at 19.48 GiB | 15.09 GiB (driven by untiled decoder) |
| Pipeline peak, DiT NO split (convention) | 17.20 GiB | 24.01 GiB (**exceeds 20 GiB budget**) |

Total kernel time scales **2.51×** for a 1.88× increase in pixel count — worse than linear, consistent with §4.1's finding that utilization *drops* at the larger shape class (48.5%→34.4% not primarily a tiling-convention artifact, since the decoder's stage share is nearly unchanged — it's a genuine efficiency regression at larger per-call tensor sizes on this hardware).

## 7. Top optimization opportunities

1. **Running the VAE decoder untiled at this resolution measurably reduces hardware utilization** (31.7% vs. 45.1% at 1920×1088's real tiled config) and **tightens GPU memory headroom to ~1–5 GiB** depending on scenario (§5, §6) — tiling (as the real 1920×1088 pipeline already does) is likely to both restore utilization and widen memory headroom at this resolution; this report's untiled numbers should be read as a **worst-case / upper-bound** memory and lower-bound utilization picture, not a recommended operating mode.
2. **DiT's windowed attention and FC layers both show lower XMX efficiency at the larger `video_tokens=476160` shape** (30.0% blended vs. 46.6% at 1920×1088) — worth checking whether larger per-call `M` is hitting a different (less efficient) GEMM/attention kernel tiling path on this hardware, independent of the decoder-tiling question.
3. **If DiT's layer-by-layer graph splitting were ever disabled at this resolution, the pipeline would exceed a 20 GiB budget** (24.01 GiB, §5.2) — confirms graph-splitting remains a hard memory-budget requirement (not just a nice-to-have) as resolution scales up, independent of VAE decoder tiling choices.
4. **`up2.post_shuffle_conv`/`up3.conv2`/`up2.conv2` (full-frame, untiled) are 51.6% of decoder time / 35.2% of total pipeline time** (§3.3) — the single biggest lever if the decoder must remain untiled for some reason; otherwise, re-tiling (as at 1920×1088) would likely reduce both their absolute cost and the overall memory footprint simultaneously.

## 8. Reproduction

```bash
# Regenerate ops_mapping-derived bat script + kernel table config (if ops_mapping.json changes)
python3 models/seedvr2_2560x1536/generate_bench_config.py

# Run the full 56-op sweep under cliloader (remote, Windows) -- direct blocking
# SSH call recommended over schtasks ONCE (see §1 sweep execution note)
ssh Local_Admin@10.239.132.213 "D:\river\ovmx_dev\dev_roofline_profiling\utils\run_seedvr2_2560x1536_ptl.bat"

# Pull logs, parse, build tables (local)
python3 utils/parse_logs.py outputs/seedvr2_2560x1536/logs/ptl outputs/seedvr2_2560x1536/ptl_metrics.json
python3 utils/build_kernel_tables.py --model-dir models/seedvr2_2560x1536 --output-dir outputs/seedvr2_2560x1536 \
    --out outputs/seedvr2_2560x1536/kernel_tables.md
python3 outputs/seedvr2_2560x1536/build_op_totals.py
python3 models/seedvr2/build_memory_report.py outputs/seedvr2_2560x1536/ptl_metrics.json 20

# Dependency-aware (whole-model, liveness-based) peak GPU memory estimate --
# dual-config-aware (decoder_tiled flag), pure static analysis, no remote run:
python3 models/seedvr2/build_dependency_memory_report.py models/seedvr2_2560x1536/ops_mapping.json

# Time-weighted overall hardware utilization -- reads kernel_tables.md, no remote run needed:
python3 outputs/seedvr2/build_utilization_report.py outputs/seedvr2_2560x1536/kernel_tables.md
```

## 9. Caveats

- **No real `OV_VERBOSE` trace exists at this resolution** (§1) — all shapes/call-counts are a first-principles re-derivation from the validated 1920×1088 config's tier pattern and formulas, not independently verified against a real compiled-graph trace. Treat this report's shape grounding as lower-confidence than the 1920×1088 report.
- **DiT windowed-attention `num_windows` (188/282) is an approximation** (§1, point 4) — scaled proportionally from the 1920×1088 config's real values by the `video_tokens` ratio, not independently derived or measured.
- **VAE decoder is deliberately modeled UNTILED** per explicit request for this exercise — this is a real, intentional methodology difference from the 1920×1088 report (which uses the real K=4 tile grid), not an error. Absolute decoder op timings/memory in this report are NOT directly comparable to the 1920×1088 report's decoder figures (§3.3); only stage-level *shares* and overall trends are meaningfully comparable (§6).
- **The dependency-aware memory estimate (§5.2) is a structured estimate, not a measurement**, inheriting the same simplified sequential-chain liveness model and caveats as the 1920×1088 report's own §5.2 (conflated multi-instance resnet/layer specs, no exact compiled-graph buffer-reuse accounting). **No real measured DiT memory reference exists at this resolution** (the 19.475 GiB real-measured figure used at 1920×1088 only applies to `video_tokens=252960`) — the 2560×1536 DiT figures (24.011 / 14.828 GiB) are both structured estimates only, with no real-measurement cross-check available, unlike at 1920×1088.
- A sweep-execution anomaly (Windows Task Scheduler silently re-triggering a `schtasks ONCE` task, corrupting the first attempt's logs) was encountered and resolved this session (§1) — the final 56 logs used for this report were individually verified non-empty and complete against the expected 56-tag list before parsing.
- All other caveats from the 1920×1088 report's §8 apply equally here (unmeasured small ops excluded from totals, GroupNorm plain-kernel-path conservatism, single-run cliloader averages with no variance study).
