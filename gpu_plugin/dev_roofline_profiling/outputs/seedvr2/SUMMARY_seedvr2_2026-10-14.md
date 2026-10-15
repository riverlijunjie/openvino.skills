# SeedVR2 Roofline Analysis — PTL (2026-10-14)

**Platform**: Intel PTL 12Xe iGPU (Machine 1, `10.239.132.213`), 2400 MHz, 110 GB/s spec BW (105 GB/s measured read), FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS.

**Model**: SeedVR2 — VAE encoder + windowed-attention DiT + VAE decoder (video super-resolution/restoration).

**Test configuration** (fixed by the task, not swept):
- Input video: 864×496, 121 frames (`SeedVR2/input/480p-sc-121f-864x496.rgb24`)
- Output video: 1920×1088, 121 frames
- Text condition: 58 tokens
- Activations/weights: F16 throughout, no quantization

## 1. Methodology

**Correction (2026-10-15)**: an earlier version of this report grounded all shapes/call-counts in `SeedVR2/orbix-seedvr2/mem_report_1080P_full_utf8.log` — a *separate*, non-OpenVINO C++ reference implementation's memory log, not the real OpenVINO pipeline. This has been re-grounded in the **real, authoritative source**:

1. **The exported OpenVINO IR** (`SeedVR2/ir/{vae_encoder,vae_decoder,dit,preprocess}*.xml`) — used to determine which OpenVINO ops the graph *actually* compiles to (`grep type="..."` op-type counts), not a theoretical/assumed decomposition.
2. **A real `OV_VERBOSE=4` execution trace of the actual IR files** (`SeedVR2/ir/seedvr2_verbose_1080P.log`, 163 MB / 1.69M lines, device = "Intel(R) Arc(TM) B390 GPU" == PTL) — this is the genuine GPU-plugin `ProgramBuilder`/`CreateSingleLayerPrimitive` compile trace plus custom `[INFO] SEEDVR2_*` pipeline instrumentation of a real end-to-end run of `dit.xml`, `preprocess.xml`, `vae_encoder_initializing/active.xml`, `vae_decoder_init/active.xml` — i.e. what OpenVINO *literally executes*, not a parallel reference implementation.

**Shape validation**: every channel/kernel/`hidden_size`/`video_tokens`/`txt_tokens` value already in `ops_mapping.json` was cross-checked against this real trace and confirmed correct — weight-constant layouts (e.g. `decoder.conv_in.weight` → `f16:bfzyx:512x16x3x3x3`, `mid_block.resnets.0.conv1.weight` → `512x512x3x3x3`) and activation shapes (`SEEDVR2_DIT_BOUNDARY shape=[252960,2560]` / `[58,2560]`) match exactly. No shape changes were needed.

**Call-count AND shape correction (kept, twice-corrected)**: the VAE decoder uses **K=4 tiles arranged as a 2×2 spatial grid** (NOT 4 pure H-only tiles) at this 1920×1088 output resolution. First evidence: `seedvr2_vae_decoder_init_fused_carry_op` and `seedvr2_vae_decoder_active_fused_carry_op` are each independently compiled **exactly 4 times** (distinct `Build model name` events), vs. the VAE encoder's `seedvr2_vae_encoder_init`/`_active`, each compiled **exactly once** (no tiling there); each of the decoder's 8 compiles independently creates the **full 33-node `CausalConv3DCarry` topology** (264 total creates ÷ 8 compiles = 33/compile, exactly matching `vae_decoder_active.xml`'s own static node count) — genuine 4× replication, not a fraction. **A first attempt (2026-10-15, superseded same day) wrongly multiplied the FULL (untiled) frame shapes by 4** — that overcounts, since each tile only processes a core+halo *fraction* of the full spatial extent, not the whole frame four times over. The REAL per-tile shape was found directly in the log via non-constant `calc_output_layouts` lines for every `*.carry_in` Parameter (these print REAL per-tile activation shapes, not just weight-constant shapes): e.g. `parameter:decoder.mid_block.resnets.0.conv1.carry_in output tensor: f16:bfzyx:1x512x2x72x122:nopad` and `parameter:decoder.conv_out.carry_in output tensor: f16:bfzyx:1x128x2x576x976:nopad`. This proves a genuine **2×2 grid** (core **halved in both H and W**, not just H): base latent tier (full 136×240) → tile 72×122 (core 68×120 + halo_h=4/halo_w=2); halo scales by the tier's own upsample factor at each subsequent tier (confirmed directly for all 4 tiers): tier1 (272×480 full) → 144×244 tile; tier2 (544×960 full) → 288×488 tile; tier3/pixel-res (1088×1920 full) → 576×976 tile (exactly matching `decoder.conv_out.carry_in`'s directly-observed shape). **All vae_decoder conv/groupnorm/fc/attention shapes in `ops_mapping.json` were corrected to these real per-tile values and RE-MEASURED on remote PTL** (the ×4 `calls` multiplier needed no re-run, since it's a pure report-time multiplier, but the shape change is NOT — a kernel's time/efficiency is shape-dependent and cannot be inferred from the old full-frame measurement). GroupNorm entries were renamed to match their new tile shape (e.g. `gn_c512_h136w240` → `gn_c512_h72w122`).

**Call-count correction investigated and rejected**: the same trace also shows **33 distinct DiT model compiles** (`stage0`..`stage32`, with matching `SEEDVR2_DIT_GPU`/`SEEDVR2_DIT_STAGE index=0..32` instrumentation), which superficially looks like "33 diffusion steps" (→ would imply a 33× DiT multiplier). Two pieces of direct evidence disprove this and show it is instead **33 memory-managed per-layer pipeline stages of a single forward pass** (compile one layer's weights → infer → release → next layer):
- `VLSDPA` op creates total only **32** across the *entire* log (== `num_layers`, i.e. one VLSDPA node per stage on average). If each stage recompiled the full 32-layer model, this would be 32×33 = 1056, not 32.
- Each stage's own `compile_ms` is only **~3.5 s** (range 2.1–7.7 s across all 33 stages) — far too short to compile a full 32-layer, `hidden_size=2560` transformer stack from scratch, but consistent with compiling one layer's worth of ops.
- A compute-bound roofline sanity check confirms it independently: at `M=252960` tokens, the `vid_qkv` FC alone needs ≥169 ms/call at 100% of this platform's 58.982 FP16 TFLOPS peak; 33 full-model recompiles (33×32=1056 `vid_qkv` calls) would need ≥178 s for `vid_qkv` *alone* at perfect efficiency — already exceeding the **real measured total DiT `infer_ms` across all 33 stages: 85,086 ms (85.09 s)**.

So DiT's `calls` are **kept unchanged** at their original single-forward-pass counts (32/31/1 per layer-type FC, 16/16 for the two alternating window-attention configs).

**External validation against the real run's own measured wall-clock** (a sanity check, not an exact-match expectation — an isolated single-op microbenchmark vs. a fused/pipelined whole-model compiled graph can legitimately differ):
- DiT: this report's kernel-time-based total (53.39 s) vs. the real run's measured DiT `infer_ms` (85.09 s) → **62.8%** ratio. DiT's layers are strictly sequentially dependent (layer *N+1* needs layer *N*'s output), so little real-world kernel overlap is possible — a near-1:1 serial correspondence is expected, and 62.8% is a plausible bench-vs-real gap in that regime.
- VAE decoder: this report's kernel-time-based total, **now at the corrected real per-tile shapes** (286.34 s) vs. the real run's measured `vae_decoder` module `run=402,113.3 ms` (402.11 s, includes compile+infer for all 8 compiles) → **71.2%** ratio — closely matching DiT's 62.8% ratio, a much more internally-consistent picture than the earlier (wrong, full-frame×4) attempt's 1,107.95 s projection, which *overshot* the real wall-clock by 2.76× and had no plausible mechanism to explain the direction of that gap. The corrected 71.2% ratio needs no special overlap/parallelism explanation at all — it's simply the same order-of-magnitude bench-vs-real gap seen everywhere else in this report.

**Lesson for future re-groundings** (saved to repo memory): "N distinct `ProgramBuilder` compiles for a logically-the-same-named model" does **not** by itself mean "multiply calls by N". Always cross-check whether each compile contains the *full* op topology (→ genuine N-fold replication, e.g. decoder's H-tiles) or only a *fraction* of it (→ memory-managed staging of one pass, e.g. DiT's per-layer stages), via a node-type total-count-vs-per-compile-count comparison plus a compute-bound sanity check against any available real measured wall-clock timing.

Key architectural finding (unchanged, re-confirmed against the real trace): **the whole pipeline runs at the 1920×1088-derived 136×240 latent grid**, not at 864×496. `preprocess.xml`'s `bicubic_pillow` `Interpolate` upsamples the 864×496 source to the 1920×1088 target canvas *before* the VAE encoder ever runs. The 864×496 input only fixes the *source* video's native resolution.

Two SeedVR2-project-specific custom OpenVINO ops are used (not the public opset) and were benchmarked directly, per SKILL.md's "benchmark what OpenVINO really executes" rule:

| Op family | Real op | Where | New bench |
|---|---|---|---|
| GroupNorm | `ov::op::internal::GroupNormalizationZSplit` | both encoder & decoder (22–30 nodes/IR) | **`groupnorm_bench`** (new) |
| Causal 3×3×3 conv | `ov::op::internal::CausalConv3DCarry` | **decoder only** (33 nodes; encoder has 0) | **`conv_bench --causal`** (new) |
| 1×1×1 / stride-2 conv | `ov::op::v1::Convolution` | both (encoder: all convs, incl. 3×3×3 resnet convs via external Concat/carry; decoder: shortcuts/upscale only) | **`conv_bench --plain`** (new) |
| DiT windowed attention | `ov::op::internal::VLSDPA` (`WindowQkvPrep`+`VarlenAttention` in the reference) | DiT, 32 layers, 2 alternating window configs | reused `vlsdpa_bench` |
| FC / MatMul | `ov::op::v0::MatMul` | DiT + VAE mid-block attention, all F16 | reused `fc_bench` (precision=f16) |
| VAE mid-block attention | single-head full self-attention | both | reused `sdpa_bench` |

A notable encoder/decoder asymmetry: **the encoder's exported IR has zero `CausalConv3DCarry` nodes** (`grep type="CausalConv3DCarry" vae_encoder_active.xml` → 0 matches, 27 `Convolution`), so its causal history is composed graph-side (`carry_in` Parameter → Concat with body → plain `Convolution` → Slice `carry_out`). The decoder's IR has 33 `CausalConv3DCarry` nodes and only 5 plain `Convolution` — its 3×3×3 resnet convs genuinely use the fused op. `ops_mapping.json` encodes this distinction precisely (encoder conv `t_body` values already include the pre-concatenated carry frames; decoder entries keep the op's own `carry_in` port separate).

Ops not covered by a dedicated bench (small, bandwidth-bound, or explicitly out of scope per SKILL.md) are listed under `*_estimated_only` in `ops_mapping.json`: `WindowQkvPrep`/`GatherRows` (window gather+RoPE repack), `SwiGLU`/`AdaLNModulation`/`ScaleShiftModulation`/`GatedResidual(AdaLN)` (small broadcast-affine/fused-eltwise, matches SKILL.md's "don't profile swish/multiply separately"), `RmsNorm`, and `PixelShuffle3d` (pure reshape/transpose, memory-bound). These were not measured and are excluded from the totals below — they are a small fraction of total time relative to conv/GEMM/attention at these shapes.

All 56 benchmarked shapes/call-counts live in `models/seedvr2/ops_mapping.json`; the bat sweep script and kernel-table config are generated together (and stay in sync) via `models/seedvr2/generate_bench_config.py`.

## 2. Hardware roofline (PTL)

| Metric | Value |
|---|---|
| Xe Cores | 12 |
| Frequency | 2400 MHz |
| Memory BW (spec / measured read) | 110 / 105 GB/s |
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.964 TOPS |

## 3. Measured kernel-time totals (one full 121-frame, 864×496→1920×1088 inference)

Per-kernel breakdown (average latency, calls/inference, total ms, GFLOPS, GB/s, efficiency %, memory/compute-bound) for **every** benchmarked op is in [kernel_tables.md](kernel_tables.md) (56 ops × their real GPU kernel(s), sorted by total time within each submodel). The tables below are the **op-level** (kernels summed) bottleneck ranking.

### 3.1 VAE Encoder — total ≈ **82.58 s**

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | down0.conv1 (causal via concat, 128→128, H1088×W1920) | 25,818 | 31.3% | compute |
| 2 | down1.conv2 (causal via concat, 256→256, H544×W960) | 18,861 | 22.8% | compute |
| 3 | gn_c128_h1088w1920 (GroupNormalizationZSplit) | 9,456 | 11.4% | memory |
| 4 | down2.conv2 (causal via concat, 512→512, H272×W480) | 8,986 | 10.9% | compute |
| 5 | gn_c256_h544w960 | 3,747 | 4.5% | memory |
| 6 | down1.conv1 | 3,130 | 3.8% | compute |
| 7 | mid_attn_sdpa (single-head, 32640 tokens) | 2,772 | 3.4% | compute |
| — | *(remaining 12 ops)* | 10,812 | 13.1% | mixed |

### 3.2 DiT — total ≈ **53.39 s**

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | vid_mlp_out (FC, 252960×6912×2560) | 13,116 | 24.6% | compute |
| 2 | vid_mlp_gate (FC, 252960×2560×6912) | 11,469 | 21.5% | compute |
| 3 | vid_qkv (FC, 252960×2560×7680) | 10,538 | 19.7% | compute |
| 4 | vid_mlp_up (FC, 252960×2560×6912) | 7,541 | 14.1% | compute |
| 5 | wattn_window_a (VLSDPA, 100 windows) | 4,971 | 9.3% | compute |
| 6 | wattn_window_b (VLSDPA, 150 windows) | 3,516 | 6.6% | compute |
| 7 | vid_attn_out (FC, 252960×2560×2560) | 2,129 | 4.0% | compute |
| — | *(txt_* ops, vid_in/out — tiny)* | 92 | 0.2% | compute |

DiT's 13 `txt_*` FC ops (M=58) contribute **<0.1% combined** — fully dominated by the `vid_*` (M=252960) ops, as expected (text is 58 tokens vs. 252,960 video tokens).

### 3.3 VAE Decoder — total ≈ **286.34 s** (the dominant stage: 3.47× the encoder, 5.36× the DiT)

**Re-measured at the real per-tile shapes** (see §1 — 2×2 grid, core+halo, e.g. up3-tier tile = 576×976 vs. the full 1088×1920 frame), each shape×4 (K=4 tiles). This SUPERSEDES an earlier version of this table that wrongly used the full-frame shape×4.

| # | Op | Total (ms) | % of stage | Bound |
|---|---|---:|---:|---|
| 1 | up3.conv2 (causal, 128→128, tile 576×976) | 56,046 | 19.6% | compute |
| 2 | up2.conv2 (causal, 256→256, tile 288×488) | 43,407 | 15.2% | compute |
| 3 | up2.post_shuffle_conv (causal, 256→256, tile 576×976) | 36,374 | 12.7% | compute |
| 4 | up1.post_shuffle_conv (causal, 512→512, tile 288×488) | 31,144 | 10.9% | compute |
| 5 | up3.conv1 | 24,008 | 8.4% | compute |
| 6 | gn_c128_h576w976 (GroupNormalizationZSplit, tile) | 21,538 | 7.5% | memory |
| 7 | up1.conv1 | 18,942 | 6.6% | compute |
| 8 | up2.conv1 | 17,959 | 6.3% | compute |
| 9 | gn_c256_h288w488 | 10,740 | 3.8% | memory |
| — | *(remaining 13 ops)* | 26,177 | 9.1% | mixed |

## 4. Overall roofline picture

| Stage | Total kernel ms | % of whole pipeline | Dominant bound |
|---|---:|---:|---|
| VAE Encoder | 82,582 | 19.6% | compute (conv, oneDNN causal_conv3d) |
| DiT | 53,387 | 12.6% | compute (FC GEMM + windowed attention) |
| VAE Decoder | 286,337 | **67.8%** | compute (causal conv, full-resolution upsample stages, ×4 real 2×2-grid tiles) |
| **Total (kernel-only)** | **422,306 ms ≈ 7.04 min** | 100% | — |

**The VAE decoder dominates total latency (67.8%)** — specifically its `up2`/`up3` stages (the two *largest-spatial-extent* resnet blocks) plus their `post_shuffle_conv` (the 3×3×3 conv immediately after each `PixelShuffle3d` upsample), at the REAL per-tile spatial extent × the real K=4 tile factor (§1). These 4 ops alone are **58.3%** of decoder time / **39.5%** of total pipeline time. Convolution efficiency at these large-spatial/small-channel shapes (33–53% of PTL's FP16 XMX peak) is noticeably lower than the small-spatial/large-channel shapes earlier in the pipeline (up to 93% for `vid_attn_out`, 60%+ for the H136×W240-tier convs) — consistent with oneDNN's `causal_conv3d` kernel having less XMX-friendly reuse when spatial extent dominates over channel depth. This corrected total (422.31 s, decoder 67.8%) is now very close to the FIRST (pre-any-tile-correction) estimate (412.96 s, decoder 67.1%) — expected, since per-tile-shape×4-tiles has roughly the same total spatial AREA as the full frame×1, plus a small (~3.4%) halo-redundancy overhead from the 2×2 grid's overlapping core+halo regions.

GroupNorm (`GroupNormalizationZSplit`) is consistently memory-bound at **58–68% of measured BW**, not a major bottleneck in absolute terms (≤11.4% of any stage) but a genuine, real cost since this op always runs the **plain (non-FSV16-fused) kernel path** here — our bench feeds a bare `Parameter` with no upstream `Convolution`, so the layout optimizer keeps it plain `bfzyx` (confirmed by the parsed kernel name `group_normalization_zsplit_final_zsplit_plain`). In the real end-to-end graph, GroupNorm is fed directly by a `Convolution`'s `b_fs_zyx_fsv16` output for the large majority of sites (per prior session analysis in repo memory), which selects the **fused** FSV16 kernel path instead and would be measurably faster — the numbers above should be read as a **conservative (slower) bound** for GroupNorm specifically, not the exact in-graph cost.

### 4.1 Overall hardware utilization (time-weighted roofline efficiency)

Each kernel's own `Eff%` column (§3, [kernel_tables.md](kernel_tables.md)) is the % of peak FP16 XMX FLOPS (compute-bound) or peak measured BW (memory-bound) it achieves. Weighting every kernel's `Eff%` by its own `Total ms` (reproducible via `outputs/seedvr2/build_utilization_report.py`, reading `kernel_tables.md`, no remote re-run needed) gives the **overall hardware utilization actually achieved across the whole 121-frame pipeline**:

| Stage | Compute-bound time | Avg % of FP16 XMX peak | Memory-bound time | Avg % of peak BW | Blended utilization |
|---|---:|---:|---:|---:|---:|
| VAE Encoder | 82.4% of stage | 54.1% | 17.6% of stage | 62.4% | **55.6%** |
| DiT | 100.0% of stage | 46.6% | — | — | **46.6%** |
| VAE Decoder | 88.1% of stage | 42.8% | 11.9% of stage | 61.7% | **45.1%** |
| **Overall (all 3 stages)** | 88.5% of total | **45.4%** | 11.5% of total | **62.0%** | **≈ 47.3%** |

**Overall, this roofline sweep achieves ≈47% of the hardware's real capability** — i.e. compute-bound kernels (88.5% of all measured time) run at an average of 45.4% of PTL's 58.982 FP16 XMX TFLOPS peak, and memory-bound kernels (11.5% of time, almost entirely GroupNorm) run at 62.0% of measured BW. Since the VAE decoder dominates total time (67.8%, §4) and has the *lowest* per-stage utilization (45.1%), it pulls the overall blended figure down close to its own number. Practically: if every op could hit 100% of peak, total kernel time would shrink from 422.31 s to roughly 422.31 × 0.473 ≈ **200 s** — i.e. there is still **~2.1× theoretical headroom** at this platform's peak FLOPS/BW, concentrated in the decoder's large-spatial/small-channel conv shapes (§6, point 1). (A handful of tiny GroupNorm entries at the smallest tile tier show `Eff% > 100%` — e.g. `gn_c512_h72w122`'s `calc_sqr_mean` kernel at 116.5% — a normal cache-residency artifact at very small working sets exceeding the nominal DRAM bandwidth figure; these contribute <0.1% of total time and don't materially affect the aggregate.)

## 5. Peak GPU memory analysis

### 5.1 Per-op isolated bench peaks (what each op needs on its own)

Both new benches (`conv_bench`, `groupnorm_bench`) query `ov::intel_gpu::memory_statistics` (`GPU_MEMORY_STATISTICS`) after every run and auto-cap `num_bufs` (down to 1 if needed) to stay under an **18 GiB** internal budget, leaving headroom under the 20 GiB target; the generator additionally caps `num_bufs` for the largest `fc_bench`/`vlsdpa_bench` invocations (which have no built-in guard) the same way. **Re-measured at the corrected per-tile decoder shapes** (§1) — the decoder's own isolated peaks dropped sharply (e.g. `conv_dec_up2_post_shuffle_conv` 10.03 GiB → 3.28 GiB) since a tile is a fraction of the full frame's area.

| Op (top 10 by total GPU memory) | usm_device (GiB) | usm_host (GiB) | Total (GiB) |
|---|---:|---:|---:|
| conv_enc_down0_conv1 | 4.981 | 6.039 | **11.020** |
| conv_enc_conv_in | 2.086 | 8.031 | 10.117 |
| gn_enc_gn_c128_h1088w1920 | 0.000 | 7.969 | 7.969 |
| conv_enc_down1_conv2 | 2.494 | 4.047 | 6.540 |
| conv_enc_down1_conv1 | 1.745 | 4.047 | 5.792 |
| conv_enc_down1_shortcut | 1.494 | 4.047 | 5.541 |
| conv_enc_down0_downsampler | 2.491 | 2.055 | 4.545 |
| gn_enc_gn_c256_h544w960 | 0.000 | 3.984 | 3.984 |
| conv_dec_up2_post_shuffle_conv (tile) | 1.076 | 2.207 | 3.283 |
| conv_dec_up3_conv1 (tile) | 1.074 | 1.135 | 2.209 |

**Peak single-op GPU memory observed across all 56 benchmarked shapes: 11.02 GiB — within the 20 GiB budget (8.98 GiB headroom).** The VAE encoder (untiled, full-frame) is now the single largest isolated op, not the decoder — the decoder's own largest isolated op dropped to 3.28 GiB once measured at its real (smaller) per-tile shape.

Notable platform detail: on this PTL **iGPU**, `RemoteContext::create_tensor()` allocations for `conv_bench`/`groupnorm_bench` report almost entirely under **`usm_host`**, not `usm_device` (e.g. `groupnorm_bench` shows 0 GiB device / up to 8 GiB host) — consistent with PTL's shared (not discrete) memory architecture; "GPU memory" on this device *is* system RAM, so the combined `usm_device + usm_host` figure is the correct one to budget against, not `usm_device` alone. Full per-op table: see `models/seedvr2/build_memory_report.py` output (reproduction command below).

### 5.2 Dependency-aware (whole-model) peak, per submodel AND whole pipeline — why per-op isolation is incomplete

Per-op isolated peaks (§5.1) answer "how much memory does THIS ONE op need in an empty context" — they do **not** answer "what is the real peak while a whole compiled submodel executes", because in a real compiled model **all** weights for that submodel's graph are resident simultaneously for its whole lifetime, and activation tensors have real producer/consumer **lifetimes** — a resnet block's input must stay alive across `conv1`+`norm`+`conv2` for the residual add, a channel-changing block's `shortcut` output must stay alive until that same add, and a DiT layer's residual-stream hidden state must stay alive across its own attention+MLP sub-blocks — none of which an isolated single-op bench can see at all.

`models/seedvr2/build_dependency_memory_report.py` walks each submodel's real architectural sequence using the real shapes in `ops_mapping.json` and tracks concurrently-live weight + activation bytes (full methodology/assumptions documented in the script's own header — these are **structured estimates** built on stated liveness rules, not cliloader-measured numbers, since `ops_mapping.json`'s specs bundle multiple identical-shape resnet/layer instances per entry rather than exposing individual graph edges).

**DiT scope note (per explicit request)**: the real production pipeline actually executes DiT with **graph splitting** (33 memory-managed per-layer compile+infer+release stages, §1 — only 1-of-32 layers' weights resident at any instant). The estimate below **deliberately does NOT follow that as-executed flow** — it instead estimates the **NO-graph-split scenario**: DiT compiled as ONE single model with **all 32 layers' weights resident simultaneously** (plus the `vid_in`/`txt_in`/`vid_out` embed/head weights, which exist once regardless of splitting). Peak activation is assumed to stay the same magnitude as the per-layer figure (a competent memory planner reuses one layer's dead activation buffers for the next layer — no inherent reason activation should grow 32×, since weights/activations are independent axes), so the no-split-vs-split delta is almost entirely in **weight** bytes.

| Submodel | Resident weight | Peak activation | **Submodel peak** | Peak location |
|---|---:|---:|---:|---|
| VAE encoder (untiled, 1 compiled model, full frame) | 0.057 GiB | 7.969 GiB | **8.026 GiB** | `down0.conv1` (full 1088×1920, 128ch, largest T before any downsample) |
| **DiT, NO graph split (32 layers' weights resident at once)** | **9.478 GiB** (32 × 0.295 GiB/layer + 0.025 GiB embed/head) | 7.721 GiB | **17.200 GiB** | during MLP (`hidden` + `gate_out` + `up_out` concurrently live) |
| *(context only)* DiT, WITH graph split (as actually executed) | 0.295 GiB (1-of-32 layers) | 7.721 GiB | 8.017 GiB structured / **19.475 GiB real measured** | same |
| VAE decoder, ONE tile resident (the realistic figure — §5.2 2026-10-15 correction) | 0.091 GiB | 2.145 GiB | **2.235 GiB** | `up3.conv1` (block-input + shortcut-output + conv1-in/out) |
| *(for reference only, NOT used below)* VAE decoder, all 4 tiles resident concurrently | 0.363 GiB | 8.579 GiB | 8.941 GiB (= 2.235 × 4, exactly) | same ×4 |

**Interesting cross-check**: the no-graph-split structured estimate (17.20 GiB) is actually *higher* than the as-executed graph-split structured estimate (8.02 GiB, expected — that's the whole point of splitting) but still *lower* than the as-executed graph-split's own REAL measured value (19.48 GiB). This confirms the earlier finding (§5.2 original) that this simplified liveness model under-counts some real overhead (RoPE-rotated Q/K copies, `SeedVRWindowQKVPrep`'s window gather/pad scratch, GEMM kernel reorder/workspace buffers) **regardless of whether graph splitting is modeled** — the gap is a property of the activation-side modeling, not the weight-side (no-split vs split) distinction.

**Whole-pipeline peak GPU memory** (per this section's no-graph-split DiT estimate): the three submodels run **sequentially** (`preprocess → vae_encoder → denoiser(DiT) → vae_decoder`, confirmed via the real log's own `[PERF] module run=` markers) — they do **not** coexist on-device, so the pipeline's peak is the **MAX across stages, not the sum**:

| VAE encoder | DiT (no graph split) | VAE decoder (1 tile) | **Pipeline peak** |
|---:|---:|---:|---:|
| 8.026 GiB | **17.200 GiB** | 2.235 GiB | **17.200 GiB** (DiT) |

**With DiT estimated as NOT graph-split, the whole-pipeline peak is ≈17.20 GiB — still driven by DiT**, whose weight footprint alone (9.48 GiB, all 32 layers) already exceeds the VAE encoder's entire peak (8.03 GiB). This leaves **≈2.8 GiB of headroom under a 20 GiB budget** — thinner than §5.1's isolated-op view (8.98 GiB headroom) suggested, though not as tight as the as-executed/real-measured scenario (≈0.5 GiB headroom, §5.2 2026-10-15 real-measured note, which reflects the ACTUAL production pipeline's graph-split DiT, not this section's no-split estimate). **Practical reading: graph-splitting DiT by layer is a real, substantial, and apparently necessary memory-saving measure** — removing it (as estimated here) would use roughly 2× the real pipeline's own measured DiT peak just from holding all 32 layers' weights at once.

For completeness, a real process-wide cross-check from the same log (reflects the REAL graph-split pipeline, not the no-split scenario estimated above): `process_peak_working_set = 27,472,494,592 B = 25.580 GiB` (the *same* constant value at every sampled point in the whole log, i.e. already reached before decoder even started and never decreased — standard OS peak-working-set semantics) — ~6.1 GiB *higher* than DiT's own real graph-split GPU-memory peak (19.48 GiB), plausibly the host-RAM-resident cost of the loaded model checkpoints (the DiT safetensors file alone is ~6.78 GiB) that stay resident in process memory independent of what's actively on the GPU at any one instant. This whole-process figure is host-RAM-inclusive, not a pure GPU-memory number, so it is not directly comparable to the 20 GiB *GPU* budget.



## 6. Top optimization opportunities

1. **VAE decoder `up2`/`up3`/post-shuffle stages (58.3% of decoder time, 39.5% of total pipeline time)** are the single biggest lever by far — the whole decoder stage is 67.8% of total pipeline time at the real per-tile shapes (§1, §4). These run at the largest spatial extent with the smallest channel counts (128–256ch) — classically harder to keep XMX-efficient. Worth investigating whether a tuned im2col/winograd variant or different data layout improves the 33–53%-of-peak efficiency seen here, since the small-channel/large-spatial shape class is structurally different from the large-channel/small-spatial shapes earlier in the network (which already hit 50–60%+).
2. **GroupNorm's plain-kernel path is a measured, not just theoretical, cost** (58–68% of measured BW, ≤7.5% of any one stage but present at every ResnetBlock). If the real end-to-end graph is not already getting the fused FSV16 path at every site (per repo memory, some sites still reorder to plain `bfzyx`), closing that gap would directly reduce this.
3. **DiT's FC layers (vid_mlp_gate/up/out, vid_qkv) are 80% of DiT's own time** and run at 37–64% of FP16 XMX peak — standard dense-GEMM optimization territory (precision/tiling), no new architectural finding here, but confirms DiT is genuinely GEMM-bound at this resolution (252,960 video tokens). DiT is only 12.6% of total pipeline time though (§4), so this is a lower-priority lever than the decoder.
4. **Windowed attention (wattn_window_a/b) is only 35–38% of XMX peak** despite being compute-classified — worth checking whether the CM `cm_sdpa_vlen` kernel's window-count/window-size choice (100 vs. 150 windows for the same 252,960 tokens) is itself tunable; window_b (more, smaller windows) is proportionally slower per-token than window_a.
5. **GPU memory is close, not comfortable, once DiT is the reference, whether or not graph-splitting is used** (§5.2): DiT's real as-executed (graph-split) peak is 19.48 GiB measured; even the deliberately-different NO-graph-split structured estimate (all 32 layers' weights resident, 17.20 GiB) still dominates the pipeline, leaving only **~2.8 GiB headroom under the 20 GiB budget** in that scenario (vs. ~0.5 GiB using the real graph-split figure) — both much thinner than the isolated-op view (11.02 GiB, 8.98 GiB headroom) suggested. The VAE encoder (~8.03 GiB) and decoder (~2.24 GiB, realistic 1-tile figure) are both comfortably below either DiT figure and are **not** the memory bottleneck. **Graph-splitting DiT by layer is confirmed as a real, substantial memory-saving measure** (removing it would need ~2× the real measured per-layer-staged peak just for weights) — any further memory-driven optimization should target DiT's per-layer activations (e.g. fusing `SwiGLU`'s gate/up into one pass, or reducing the windowed-attention gather/pad scratch), not further VAE tiling.

## 7. Reproduction

```bash
# Regenerate ops_mapping-derived bat script + kernel table config (if ops_mapping.json changes)
python3 models/seedvr2/generate_bench_config.py

# Build all benches on remote (NEVER build locally)
#   configure: utils/configure_remote_seedvr2.bat
#   build:     utils/build_remote_seedvr2.bat

# Run the full 56-op sweep under cliloader (remote, Windows)
utils/run_seedvr2_ptl.bat
# ...or, for a decoder-shape-only correction (e.g. after a tile-geometry fix),
# just the 22 vae_decoder specs: utils/run_seedvr2_ptl_decoder_retile.bat

# Pull logs, parse, build tables (local)
python3 utils/parse_logs.py outputs/seedvr2/logs/ptl outputs/seedvr2/ptl_metrics.json
python3 utils/build_kernel_tables.py --model-dir models/seedvr2 --output-dir outputs/seedvr2 \
    --out outputs/seedvr2/kernel_tables.md
python3 outputs/seedvr2/build_op_totals.py
python3 models/seedvr2/build_memory_report.py outputs/seedvr2/ptl_metrics.json 20

# Dependency-aware (whole-model, liveness-based) peak GPU memory estimate for
# ALL THREE submodels + whole-pipeline peak (§5.2) -- no remote run needed,
# pure static analysis:
python3 models/seedvr2/build_dependency_memory_report.py

# Time-weighted overall hardware utilization (§4.1) -- reads kernel_tables.md,
# no remote run needed:
python3 outputs/seedvr2/build_utilization_report.py
```

## 8. Caveats

- Call counts (`calls` per op in `ops_mapping.json`) are now grounded in the **real `OV_VERBOSE=4` execution trace** (`SeedVR2/ir/seedvr2_verbose_1080P.log`, §1), not the earlier (incorrect) orbix reference-implementation log. Per-shape counts still take the dominant/most-frequent shape class per logical layer across the whole 121-frame video, with rare INIT/edge-segment variants (<5 calls) folded into the dominant count — a small, intentional approximation, not an exact per-segment replay.
- The decoder's 4× tile multiplier and real per-tile (2×2 grid) shapes, and DiT's rejected 33× "diffusion step" multiplier (§1), were derived from **compile-event counting plus non-constant layout-line shape extraction**, not a direct per-call runtime trace (the real log has no per-segment/per-tile runtime trace for VAE encoder/decoder, only for DiT's own `SEEDVR2_DIT_STAGE` instrumentation) — all conclusions are well-evidenced (node-count-per-compile + real carry_in parameter shapes + compute-bound/wall-clock cross-checks) but are inferences, not a direct count of real `infer()` invocations. Treat the decoder's 67.8%-of-pipeline figure as **high-confidence but not pixel-perfect**.
- The dependency-aware memory estimate (§5.2) is a **structured estimate**, not a measurement: it assumes a simplified sequential-chain liveness model (block-input/shortcut-output held alive across a resnet block's conv1+conv2, or DiT's residual-stream hidden state held alive across its attention+MLP sub-blocks) rather than the exact compiled graph's real scheduling/buffer-reuse decisions, and conflates multiple identical-shape resnet/layer instances per `ops_mapping.json` spec (e.g. `up0.conv1` represents 3 resnets' conv1+conv2 combined) into one representative tensor size rather than per-instance graph edges. **DiT's structured estimate specifically under-shoots its own real measured value by ~11.46 GiB** (8.02 vs. 19.48 GiB, §5.2) — the whole-pipeline peak conclusion (≈19.48 GiB, ~0.5 GiB headroom) relies on DiT's REAL measured figure, not the structured estimate, precisely because that gap is known and real for this submodel. No equivalent real measurement exists in the log for the VAE encoder/decoder's OWN whole-model peaks (only DiT has per-stage `usm_device`/`usm_host` instrumentation), so their 8.03/8.94 GiB figures remain structured estimates, not measurements — treat them as plausible but unverified lower bounds, not an exact ceiling.
- `txt_*` DiT ops and VAE mid-block attention FC/SDPA use `fc_bench`/`sdpa_bench` (plain F16 MatMul / uncompressed SDPA) rather than a model-specific kernel — correct for this model (no weight compression anywhere in SeedVR2), but note these benches are shared infrastructure also used by LLM models in this same toolkit.
- GroupNorm numbers reflect the **plain (unfused) kernel path** only (see §4) — a deliberately conservative/slower bound, not necessarily the exact in-graph cost at every site.
- `PixelShuffle3d`, `WindowQkvPrep`/`GatherRows`, `SwiGLU`, and the DiT modulation ops (`AdaLNModulation`/`ScaleShiftModulation`/`GatedResidual(AdaLN)`) are **not measured** (estimated as small/memory-bound per SKILL.md guidance) and excluded from the totals in §3–4.
- All measurements are single-run cliloader averages (iteration counts chosen so each op's measured region exceeds ~1s of GPU kernel time); no repeated-run variance study was performed.
