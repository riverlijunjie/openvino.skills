# SeedVR2 Roofline Analysis — PTL — 480P→2560×1536, decoder TILED (K=4) variant (2026-10-15)

**Platform**: Intel PTL 12Xe iGPU (Machine 1, `10.239.132.213`), 2400 MHz, 110 GB/s spec BW (105 GB/s measured read), FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS.

**Model**: SeedVR2 — VAE encoder + windowed-attention DiT + VAE decoder (video super-resolution/restoration).

**This report answers**: *"测试一下如果decoder分成4个tile时2K的roofline"* — what is the 2560×1536 roofline if the VAE decoder IS split into 4 tiles (a real 2×2 spatial grid), as opposed to the untiled (full-frame) scenario in [SUMMARY_seedvr2_2560x1536.md](../seedvr2_2560x1536/SUMMARY_seedvr2_2560x1536.md)?

**Test configuration** (fixed by the task, not swept):
- Input video: 864×496, 121 frames (same source as both other configs)
- Output video: 2560×1536, 121 frames
- Text condition: 58 tokens
- Activations/weights: F16 throughout, no quantization
- **VAE decoder modeled TILED: real 2×2 spatial grid, K=4 tiles** (same convention as the 1920×1088 config's real, log-confirmed tiling) — the opposite of the companion untiled report.

## 1. Methodology

**VAE encoder and DiT sections are byte-for-byte identical** to the untiled 2560×1536 config (`models/seedvr2_2560x1536/ops_mapping.json`) — only the VAE decoder's shapes/calls change between the two reports. Their already-measured logs (34 of 56 specs: 19 encoder + 15 DiT) were **reused directly, no re-run needed** — only the 22 `vae_decoder` specs were re-measured on remote PTL at the new tiled shapes (`utils/run_seedvr2_2560x1536_tiled_decoder_only_ptl.bat`, launched via a direct blocking SSH call, completed cleanly with no scheduler issues this time).

**Per-tile shape derivation** uses the same halo-scaling method validated against the real 1920×1088 `OV_VERBOSE` trace's non-constant `carry_in` Parameter layouts (`models/seedvr2/ops_mapping.json`'s own documented derivation): tile core = HALF of the full tier's H and W (2×2 grid: a tile covers the core + a halo border for causal-conv receptive-field overlap); halo = base_halo(4,2 at the latent tier) × tier_scale, where tier_scale is 1/2/4/8 for the latent/tier1/tier2/tier3(full) tiers respectively. This halo-growth pattern is a resolution-independent architectural fact (number of upsample stages between a tile boundary and a given conv), so applying it to the 2560×1536 tiers gives:

| Tier | Full (2560×1536) | Half (tile core) | Halo | **Tile shape** |
|---|---|---|---|---|
| latent (/8) | 192×320 | 96×160 | 4,2 | **100×162** |
| tier1 (/4) | 384×640 | 192×320 | 8,4 | **200×324** |
| tier2 (/2) | 768×1280 | 384×640 | 16,8 | **400×648** |
| tier3 (full) | 1536×2560 | 768×1280 | 32,16 | **800×1296** |

**Call counts**: the untiled config's decoder `calls` values are already the correct "per-tile" (1 tile's worth of) segment counts — `calls` is purely a function of video length/temporal segmentation, not spatial resolution or tile count (confirmed: the untiled 2560×1536 config's calls exactly equal the 1920×1088 tiled config's calls ÷ 4). So this tiled variant's `calls` = untiled 2560×1536's calls × 4 (one spatial-tile multiplier), matching the 1920×1088 config's own convention exactly.

All 56 shapes/calls live in `models/seedvr2_2560x1536_tiled/ops_mapping.json` (`decoder_tiled: true`), generated/kept in sync via `models/seedvr2_2560x1536_tiled/generate_bench_config.py`.

**No real `OV_VERBOSE` trace exists at 2560×1536** (same caveat as the untiled report) — this tiled variant is likewise a first-principles re-derivation, not an independently re-grounded measurement.

## 2. Hardware roofline (PTL)

Unchanged from the companion reports: 12 Xe Cores, 2400 MHz, 110/105 GB/s BW, FP16 XMX 58.982 TFLOPS.

## 3. Measured kernel-time totals (one full 121-frame, 864×496→2560×1536 inference)

VAE Encoder and DiT totals are **identical** to the untiled report (§3.1/§3.2 there) since those submodels are unchanged — reproduced here only for completeness.

### 3.1 VAE Encoder — total ≈ **181.72 s** (unchanged from the untiled report)
### 3.2 DiT — total ≈ **156.11 s** (unchanged from the untiled report)

### 3.3 VAE Decoder (TILED, K=4) — total ≈ **573.14 s** — down from 721.82 s untiled (a **20.6% reduction**)

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | up3.conv2 (128→128, tile 800×1296) | 108,320 | 18.9% | compute |
| 2 | up2.conv2 (256→256, tile 400×648) | 89,676 | 15.6% | compute |
| 3 | up2.post_shuffle_conv (256→256, tile 800×1296) | 69,601 | 12.1% | compute |
| 4 | up1.post_shuffle_conv (512→512, tile 400×648) | 61,222 | 10.7% | compute |
| 5 | up1.conv1 (512→512, tile 200×324) | 52,052 | 9.1% | compute |
| 6 | up3.conv1 | 44,564 | 7.8% | compute |
| 7 | gn_c128_h800w1296 | 38,499 | 6.7% | memory |
| 8 | up2.conv1 | 35,534 | 6.2% | compute |
| 9 | gn_c256_h400w648 | 20,300 | 3.5% | memory |
| — | *(remaining 13 ops)* | 53,377 | 9.3% | mixed |

## 4. Overall roofline picture (TILED decoder)

| Stage | Total kernel ms | % of whole pipeline | Dominant bound |
|---|---:|---:|---|
| VAE Encoder | 181,721 | 19.9% | compute |
| DiT | 156,107 | 17.1% | compute |
| VAE Decoder (tiled, K=4) | 573,144 | **62.9%** | compute |
| **Total (kernel-only)** | **910,972 ms ≈ 15.18 min** | 100% | — |

**Tiling the decoder reduces total pipeline kernel time by 14.0%** (1,059.65 s untiled → 910.97 s tiled) — the decoder itself shrinks by 20.6% (721.82 s → 573.14 s), and since it's the dominant stage, this materially moves the whole-pipeline number. The decoder's stage *share* also drops notably: 68.1% (untiled) → 62.9% (tiled), with the encoder/DiT's own (unchanged) absolute times now making up a larger relative fraction.

### 4.1 Overall hardware utilization (time-weighted roofline efficiency)

| Stage | Compute-bound time | Avg % of FP16 XMX peak | Memory-bound time | Avg % of peak BW | Blended utilization |
|---|---:|---:|---:|---:|---:|
| VAE Encoder | 84.9% of stage | 46.1% | 15.1% of stage | 62.2% | **48.5%** (unchanged) |
| DiT | 100.0% of stage | 30.0% | — | — | **30.0%** (unchanged) |
| VAE Decoder (tiled) | 89.1% of stage | 39.2% | 10.9% of stage | 62.1% | **41.7%** |
| **Overall (all 3 stages)** | 90.1% of total | **38.7%** | 9.9% of total | **62.1%** | **≈ 41.0%** |

**Tiling the decoder measurably improves its own hardware utilization: 31.7% (untiled) → 41.7% (tiled)**, and the overall blended figure improves correspondingly: 34.4% → 41.0%. This is a genuine efficiency gain, not just a memory-saving convenience — smaller per-call tensors (tiles) apparently map more efficiently onto this hardware's conv kernels than the much larger untiled full-frame tensors, consistent with the same pattern already observed at 1920×1088 (where the real production pipeline also tiles, and achieves 45.1% decoder utilization — still higher than this 2560×1536-tiled figure of 41.7%, since 2560×1536's *tiles* are themselves still larger than 1920×1088's tiles).

## 5. Peak GPU memory analysis (TILED decoder)

### 5.1 Per-op isolated bench peaks

| Op (top 5 by total GPU memory) | usm_device (GiB) | usm_host (GiB) | Total (GiB) |
|---|---:|---:|---:|
| conv_enc_conv_in | 3.926 | 15.062 | **18.988** (unchanged — encoder is untiled at both resolutions) |
| conv_enc_down0_conv1 | 9.376 | 3.812 | 13.188 (unchanged) |
| conv_enc_down1_conv1 | 3.283 | 7.562 | 10.845 (unchanged) |
| conv_enc_down1_conv2 | 4.691 | 5.688 | 10.378 (unchanged) |
| conv_dec_up2_post_shuffle_conv (tile) | 1.981 | 4.018 | **5.998** (was 11.316 GiB untiled — tiling cuts this ~1.9×) |

**Peak single-op GPU memory observed: still 18.988 GiB** (unchanged, since the VAE encoder is untiled in both reports and remains the largest isolated op either way) — but the decoder's OWN largest isolated op drops sharply from 11.316 GiB (untiled) to 5.998 GiB (tiled), roughly halving as expected for a 2×2 spatial split.

### 5.2 Dependency-aware (whole-model) peak, per submodel AND whole pipeline

| Submodel | Peak (untiled report) | **Peak (this tiled report)** | Change |
|---|---:|---:|---|
| VAE encoder (unchanged, untiled in both) | 15.057 GiB | 15.057 GiB | — |
| DiT, NO graph split (unchanged, same shapes in both) | 24.011 GiB | 24.011 GiB | — |
| DiT, WITH graph split (unchanged, context only) | 14.828 GiB | 14.828 GiB | — |
| **VAE decoder** | 15.091 GiB (untiled, full-frame) | **4.046 GiB (1 tile, realistic)** | **−73.2%** |

**Whole-pipeline peak**:

| Scenario | Driver | Peak |
|---|---|---:|
| DiT **NO graph split** (analysis convention carried over from earlier reports) | DiT | **24.011 GiB** (unchanged — exceeds the 20 GiB budget either way, since this scenario doesn't depend on decoder tiling at all) |
| DiT **WITH graph split** (realistic — DiT's normal/expected runtime behavior) | **VAE encoder** (was VAE decoder in the untiled report) | **15.057 GiB** (down from 15.091 GiB untiled) |

**Tiling the decoder makes the VAE encoder the new memory-peak driver in the realistic (DiT graph-split) scenario** — previously the untiled decoder (15.091 GiB) narrowly exceeded the encoder (15.057 GiB); with the decoder tiled, its own peak collapses to 4.046 GiB, well below the encoder, which is now unambiguously the pipeline's memory bottleneck at **15.057 GiB** (≈4.9 GiB headroom under a 20 GiB budget, essentially unchanged from the untiled report's own 15.091 GiB figure since both values were already very close). The NO-graph-split DiT scenario's 24.011 GiB figure (which exceeds budget) is **unaffected by decoder tiling at all**, since it's driven entirely by DiT's own weight footprint — tiling the decoder cannot help that specific (hypothetical) scenario.

## 6. Three-way comparison

| Metric | 1920×1088 (tiled, real log) | 2560×1536 (**untiled**) | 2560×1536 (**tiled, K=4**) |
|---|---:|---:|---:|
| Output pixels | 2,088,960 | 3,932,160 | 3,932,160 |
| Total kernel time | 422.31 s | 1,059.65 s | **910.97 s** |
| VAE Decoder share of total | 67.8% | 68.1% | 62.9% |
| VAE Decoder own blended utilization | 45.1% | 31.7% | **41.7%** |
| Overall blended utilization | ≈47.3% | ≈34.4% | **≈41.0%** |
| Isolated-op peak memory (20 GiB budget) | 11.02 GiB | 18.99 GiB | 18.99 GiB (unchanged — encoder-driven) |
| Decoder's own dependency-aware peak | 2.235 GiB (1 tile) | 15.091 GiB (untiled) | **4.046 GiB (1 tile)** |
| Pipeline peak, DiT WITH split (realistic) | 8.94 GiB (structured) / 19.48 GiB (real measured) | 15.091 GiB (decoder-driven) | **15.057 GiB (encoder-driven)** |
| Pipeline peak, DiT NO split (convention) | 17.20 GiB | 24.01 GiB (exceeds budget) | 24.01 GiB (exceeds budget, unaffected by decoder tiling) |

**Bottom line**: tiling the VAE decoder at 2560×1536 (matching the 1920×1088 config's real production behavior) **reduces total pipeline kernel time by 14.0%**, **improves overall hardware utilization from ≈34.4% to ≈41.0%**, and **cuts the decoder's own dependency-aware memory peak by ~73%** (15.09 → 4.05 GiB) — removing the decoder as a memory-budget concern entirely and leaving the VAE encoder (always untiled, 15.06 GiB) as the realistic pipeline's new memory-peak driver, comfortably within a 20 GiB budget. The only scenario where tiling doesn't help is the (hypothetical, convention-only) DiT-no-graph-split case, which remains over budget at 24.01 GiB regardless, since that number is driven entirely by DiT's own weights, independent of decoder tiling.

## 7. Reproduction

```bash
# Regenerate ops_mapping-derived bat script + kernel table config (if ops_mapping.json changes)
python3 models/seedvr2_2560x1536_tiled/generate_bench_config.py

# Re-run ONLY the 22 vae_decoder specs (encoder+DiT reused from the untiled sweep's logs,
# since those submodels are unchanged between the two configs)
ssh Local_Admin@10.239.132.213 "D:\river\ovmx_dev\dev_roofline_profiling\utils\run_seedvr2_2560x1536_tiled_decoder_only_ptl.bat"

# Merge reused encoder/DiT logs + new decoder logs, then parse/build tables (local)
python3 utils/parse_logs.py outputs/seedvr2_2560x1536_tiled/logs/ptl outputs/seedvr2_2560x1536_tiled/ptl_metrics.json
python3 utils/build_kernel_tables.py --model-dir models/seedvr2_2560x1536_tiled --output-dir outputs/seedvr2_2560x1536_tiled \
    --out outputs/seedvr2_2560x1536_tiled/kernel_tables.md
python3 outputs/seedvr2_2560x1536_tiled/build_op_totals.py
python3 models/seedvr2/build_memory_report.py outputs/seedvr2_2560x1536_tiled/ptl_metrics.json 20

# Dependency-aware (whole-model, liveness-based) peak GPU memory estimate --
# dual-config-aware (decoder_tiled flag), pure static analysis, no remote run:
python3 models/seedvr2/build_dependency_memory_report.py models/seedvr2_2560x1536_tiled/ops_mapping.json

# Time-weighted overall hardware utilization -- reads kernel_tables.md, no remote run needed:
python3 outputs/seedvr2/build_utilization_report.py outputs/seedvr2_2560x1536_tiled/kernel_tables.md
```

## 8. Caveats

- All caveats from the untiled 2560×1536 report (`SUMMARY_seedvr2_2560x1536.md` §9) apply equally here: no real `OV_VERBOSE` trace at this resolution, DiT `num_windows` approximation, structured-estimate-only dependency-aware memory figures, no real measured DiT memory reference at this resolution.
- The VAE encoder and DiT measurements in this report are **reused, not re-measured** — they are byte-for-byte the same bench logs as the untiled report (identical shapes/calls for those two submodels), not an independent re-run. Only the decoder's 22 specs were freshly measured at the tiled shapes.
- The tile-shape halo-scaling derivation (§1) is validated at 1920×1088 against a real log, but **applying it to 2560×1536's tiers is itself an unverified extrapolation** (no real log exists at this resolution to confirm the resulting tile shapes are what a real compiled graph would actually use).
