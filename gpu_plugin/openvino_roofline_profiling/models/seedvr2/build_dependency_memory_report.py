#!/usr/bin/env python3
"""Dependency-aware (liveness) GPU memory peak estimate for ALL THREE SeedVR2
submodels (VAE encoder, DiT, VAE decoder) -- per-model AND whole-pipeline.

The per-op isolated bench peaks in build_memory_report.py answer "how much
memory does THIS ONE op need in isolation" -- they do NOT answer "what is the
real peak GPU memory while a whole compiled submodel executes", because in a
real compiled model:
  1. ALL weights for that submodel's graph are resident on device
     simultaneously (compiled once, used for every op), not just one op's --
     EXCEPT DiT, which (per ops_mapping.json's notes) recompiles ONE
     transformer layer's weights at a time (33 memory-managed per-layer
     stages), releasing each stage's weights before the next -- so DiT's own
     weight footprint at any instant is ONE layer's worth, not all 32.
  2. Activation tensors have real producer/consumer LIFETIMES: a resnet
     block's input must stay alive across conv1+norm+conv2 for the residual
     add; a shortcut conv's output must stay alive until that same add; a
     DiT layer's residual-stream hidden state must stay alive across its own
     attention+MLP sub-blocks. Isolated per-op benches reset device memory
     between runs and cannot see any of this inter-op overlap at all.

VAE encoder/decoder: walks the real architectural sequence (conv_in ->
mid_block[resnet0+attn+resnet1] -> down/up stages -> conv_out) using the real
per-submodel shapes in ops_mapping.json, tracking concurrently-live weight +
activation bytes. DiT: per explicit request, this estimates the **NO graph
split** scenario (one single compiled model, all 32 layers' weights resident
simultaneously) rather than the as-executed-in-the-log per-layer-staged
scenario -- the latter is still printed for reference/context only, not used
for the whole-pipeline conclusion.

These are STRUCTURED ESTIMATES built on stated liveness assumptions
(documented inline / per section), not cliloader-measured numbers --
ops_mapping.json's specs bundle multiple identical-shape resnet instances per
entry (not individually), so exact per-instance graph edges aren't available.
Where available, this script also prints the REAL measured reference value
from SeedVR2/ir/seedvr2_verbose_1080P.log's own `SEEDVR2_DIT_GPU`/
`SEEDVR2_DIT_STAGE` instrumentation for cross-validation.

Usage: python3 build_dependency_memory_report.py [ops_mapping.json path]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent
GIB = 1024.0 ** 3


def conv_out_hw(h: int, w: int, kh: int, kw: int, sh: int, sw: int) -> tuple[int, int]:
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    h_out = (h + 2 * pad_h - kh) // sh + 1
    w_out = (w + 2 * pad_w - kw) // sw + 1
    return h_out, w_out


def conv_t_out(t_body: int, is_causal: bool, kt: int, st: int) -> tuple[int, int]:
    t_carry = max(kt - st, 0) if is_causal else 0
    t_total = t_body + t_carry
    t_out = (t_total - kt) // st + 1
    return t_out, t_carry


def weight_bytes(c_in: int, c_out: int, k) -> int:
    kt, kh, kw = k
    return c_in * c_out * kt * kh * kw * 2  # f16


def act_bytes(c: int, t: int, h: int, w: int) -> int:
    return c * t * h * w * 2  # f16


# Per-entry "block role" classification driving the VAE encoder/decoder
# liveness model. Roles:
#  standalone      -- simple chain op: only its own input+output need coexist.
#  resnet_same_ch  -- member of a same-channel resnet block (conv1+conv2):
#                     the BLOCK INPUT must stay alive across conv1+conv2 for
#                     the residual add (held alongside conv1's/conv2's own
#                     input/output).
#  shortcut        -- channel-adapting 1x1x1 shortcut conv: its OUTPUT must
#                     stay alive until the matching resnet block's conv2
#                     finishes (for the residual add).
#  resnet_diff_ch  -- member of a channel-changing resnet block paired with a
#                     'shortcut' entry (same liveness as resnet_same_ch, plus
#                     the shortcut's output is concurrently live).
DECODER_ROLES = {
    "conv_in": "standalone",
    "mid.conv1": "resnet_same_ch",
    "up0.conv1": "resnet_same_ch",
    "up0.upscale_conv": "standalone",
    "up0.post_shuffle_conv": "standalone",
    "up1.conv1": "resnet_same_ch",
    "up1.upscale_conv": "standalone",
    "up1.post_shuffle_conv": "standalone",
    "up2.shortcut": "shortcut",
    "up2.conv1": "resnet_diff_ch",
    "up2.conv2": "resnet_diff_ch",
    "up2.post_shuffle_conv": "standalone",
    "up3.shortcut": "shortcut",
    "up3.conv1": "resnet_diff_ch",
    "up3.conv2": "resnet_diff_ch",
    "conv_out": "standalone",
}

# VAE encoder mirrors the decoder's resnet-block pattern but with downsamplers
# instead of upsamplers, and shortcuts appear at down1/down2 (channel growth),
# not down0 (conv_in already emits down0's own channel count) or down3_mid
# (the VAE mid-block, no channel change).
ENCODER_ROLES = {
    "conv_in": "standalone",
    "down0.conv1": "resnet_same_ch",
    "down0.downsampler": "standalone",
    "down1.shortcut": "shortcut",
    "down1.conv1": "resnet_diff_ch",
    "down1.conv2": "resnet_same_ch",
    "down1.downsampler": "standalone",
    "down2.shortcut": "shortcut",
    "down2.conv1": "resnet_diff_ch",
    "down2.conv2": "resnet_same_ch",
    "down2.downsampler": "standalone",
    "down3_mid.conv1": "resnet_same_ch",
    "conv_out": "standalone",
}


def analyze_conv_chain(entries: list[dict[str, Any]], roles: dict[str, str]) -> dict[str, Any]:
    total_weight = 0
    rows = []
    pending_shortcut_out = 0
    peak_activation = 0
    peak_at = None
    block_input_bytes = 0

    for e in entries:
        name = e["name"]
        role = roles.get(name, "standalone")
        kt, kh, kw = e["k"]
        st, sh, sw = e["stride"]
        is_causal = e["kind"] == "causal"
        t_out, t_carry = conv_t_out(e["t_body"], is_causal, kt, st)
        h_out, w_out = conv_out_hw(e["h"], e["w"], kh, kw, sh, sw)
        w_b = weight_bytes(e["c_in"], e["c_out"], e["k"])
        in_b = act_bytes(e["c_in"], e["t_body"] + t_carry, e["h"], e["w"])
        out_b = act_bytes(e["c_out"], t_out, h_out, w_out)
        total_weight += w_b

        if role == "standalone":
            live = in_b + out_b
            block_input_bytes = 0
            pending_shortcut_out = 0
        elif role == "resnet_same_ch":
            if block_input_bytes == 0:
                block_input_bytes = in_b
            live = block_input_bytes + in_b + out_b
        elif role == "shortcut":
            pending_shortcut_out = out_b
            block_input_bytes = in_b
            live = in_b + out_b
        elif role == "resnet_diff_ch":
            live = block_input_bytes + pending_shortcut_out + in_b + out_b
        else:
            live = in_b + out_b

        rows.append({"name": name, "role": role, "in_b": in_b, "out_b": out_b, "w_b": w_b, "live": live})
        if live > peak_activation:
            peak_activation = live
            peak_at = name

    return {"rows": rows, "total_weight": total_weight, "peak_activation": peak_activation, "peak_at": peak_at}


def analyze_vae_submodel(label: str, section: dict[str, Any], roles: dict[str, str]) -> dict[str, Any]:
    result = analyze_conv_chain(section["conv"], roles)

    gn_weight = 0
    gn_extra_peak = 0
    for e in section.get("groupnorm", []):
        gn_weight += e["c"] * 2 * 2  # scale+bias, f16
        gn_act = act_bytes(e["c"], e["z"], e["h"], e["w"])
        gn_extra_peak = max(gn_extra_peak, gn_act * 2)  # in+out coexist briefly

    fc_weight = 0
    attn_peak = 0
    for e in section.get("fc", []):
        fc_weight += e["k"] * e["n"] * 2
    for e in section.get("attention", []):
        # mid-block self-attention: Q+K+V+out, each [s, nh*hd], f16.
        attn_peak = max(attn_peak, e["s"] * e["nh"] * e["hd"] * 2 * 4)

    total_weight = result["total_weight"] + gn_weight + fc_weight
    peak_activation = max(result["peak_activation"], gn_extra_peak, attn_peak)
    peak_bytes = total_weight + peak_activation

    print(f"{'Op':<24} {'role':<16} {'in (MB)':>10} {'out (MB)':>10} {'weight (MB)':>12} {'live now (MB)':>15}")
    print("-" * 92)
    for r in result["rows"]:
        marker = "  <== PEAK" if r["name"] == result["peak_at"] else ""
        print(f"{r['name']:<24} {r['role']:<16} {r['in_b']/1e6:>10.2f} {r['out_b']/1e6:>10.2f} "
              f"{r['w_b']/1e6:>12.3f} {r['live']/1e6:>15.2f}{marker}")
    print("-" * 92)
    print(f"Total resident weight bytes (conv + groupnorm + fc): {total_weight/GIB:.3f} GiB")
    print(f"Peak concurrently-live ACTIVATION bytes (at '{result['peak_at']}'): "
          f"{result['peak_activation']/GIB:.3f} GiB")
    print(f"GroupNorm's own extra in+out buffer peak: {gn_extra_peak/GIB:.3f} GiB")
    if attn_peak:
        print(f"Mid-block attention Q+K+V+out peak: {attn_peak/GIB:.3f} GiB")
    print(f"=> {label} compiled-model peak (weights + dependency-aware peak activation): {peak_bytes/GIB:.3f} GiB")
    print()
    return {"label": label, "total_weight": total_weight, "peak_activation": peak_activation, "peak_bytes": peak_bytes}


def analyze_dit(dit: dict[str, Any]) -> dict[str, Any]:
    """ONE transformer layer's weight + peak activation footprint, consistent
    with the per-layer-staged compile architecture (ops_mapping.json notes:
    33 'stageN' compiles, each holding only ONE layer's weights resident, not
    all 32 at once -- confirmed via VLSDPA create-count staying at 32 total
    across the whole verbose-log compile trace, not 32*33).

    PRIMARY estimate here is deliberately NOT that as-executed (graph-split)
    scenario -- per explicit request, this models DiT WITHOUT any graph
    splitting: one single compiled model holding ALL 32 layers' weights (plus
    the embed/head's own vid_in/txt_in/vid_out weights) resident simultaneously.
    Peak activation is assumed to stay about the same as the per-layer figure
    (a competent memory planner reuses layer N's dead activation buffers for
    layer N+1 -- no inherent reason peak activation should grow 32x just
    because weights aren't released between layers), so the delta between
    this and the graph-split scenario is almost entirely in WEIGHT bytes.
    """
    cfg = dit["config"]
    hidden = cfg["hidden_size"]
    vid_tokens = cfg["video_tokens"]
    txt_tokens = cfg["txt_tokens"]
    num_layers = cfg["num_layers"]

    # Per-layer FC specs (exclude vid_in/txt_in/vid_out -- those are the
    # model's own embed/head, each existing ONCE for the whole model, not
    # once per layer).
    per_layer_names = {"vid_qkv", "txt_qkv", "vid_attn_out", "txt_attn_out",
                        "vid_mlp_gate", "vid_mlp_up", "vid_mlp_out",
                        "txt_mlp_gate", "txt_mlp_up", "txt_mlp_out"}
    embed_head_names = {"vid_in", "txt_in", "vid_out"}
    fc_by_name = {e["name"]: e for e in dit["fc"]}
    weight_per_layer = sum(fc_by_name[n]["k"] * fc_by_name[n]["n"] * 2 for n in per_layer_names if n in fc_by_name)
    weight_embed_head = sum(fc_by_name[n]["k"] * fc_by_name[n]["n"] * 2 for n in embed_head_names if n in fc_by_name)
    weight_all_layers = weight_per_layer * num_layers + weight_embed_head

    def fc_out_bytes(name: str) -> int:
        e = fc_by_name[name]
        return e["m"] * e["n"] * 2

    hidden_vid = vid_tokens * hidden * 2
    hidden_txt = txt_tokens * hidden * 2
    hidden_total = hidden_vid + hidden_txt  # residual stream, held alive all layer

    qkv_out = fc_out_bytes("vid_qkv") + fc_out_bytes("txt_qkv")
    attn_out_out = fc_out_bytes("vid_attn_out") + fc_out_bytes("txt_attn_out")
    mlp_gate_out = fc_out_bytes("vid_mlp_gate") + fc_out_bytes("txt_mlp_gate")
    mlp_up_out = fc_out_bytes("vid_mlp_up") + fc_out_bytes("txt_mlp_up")
    mlp_down_out = fc_out_bytes("vid_mlp_out") + fc_out_bytes("txt_mlp_out")

    # Windowed attention: use the LARGER of the two alternating window
    # configs as the worst-case per-layer estimate (both ultimately produce
    # an output the same size as the video hidden state).
    attn_tmp = max((e.get("total_tokens", 0) * e.get("num_heads", 0) * e.get("head_size", 0) * 2
                    for e in dit.get("attention", [])), default=hidden_vid)

    stages = {
        "after qkv projection (hidden + qkv_out)": hidden_total + qkv_out,
        "during windowed attention (hidden + qkv_out + attn_tmp)": hidden_total + qkv_out + attn_tmp,
        "after attn_out projection, before residual (hidden + attn_out)": hidden_total + attn_out_out,
        "during MLP (hidden + gate_out + up_out concurrently)": hidden_total + mlp_gate_out + mlp_up_out,
        "after mlp_out projection, before residual (hidden + mlp_out)": hidden_total + mlp_down_out,
    }
    peak_stage = max(stages, key=stages.get)
    peak_activation = stages[peak_stage]
    peak_bytes_no_split = weight_all_layers + peak_activation
    peak_bytes_staged = weight_per_layer + peak_activation

    print(f"{'Stage':<58} {'live (GiB)':>12}")
    print("-" * 72)
    for name, live in stages.items():
        marker = "  <== PEAK" if name == peak_stage else ""
        print(f"{name:<58} {live/GIB:>12.3f}{marker}")
    print("-" * 72)
    print(f"Peak concurrently-live activation bytes (same for both scenarios below): {peak_activation/GIB:.3f} GiB")
    print(f"Residual-stream hidden state (vid {vid_tokens}+txt {txt_tokens} tokens x hidden={hidden}): "
          f"{hidden_total/GIB:.3f} GiB (held alive the WHOLE layer)")
    print()
    print(f"PRIMARY scenario (per user request): DiT with NO graph splitting -- one compiled model,")
    print(f"ALL {num_layers} layers' weights resident simultaneously (+ vid_in/txt_in/vid_out embed/head):")
    print(f"  weight bytes = {num_layers} x {weight_per_layer/GIB:.3f} GiB/layer + "
          f"{weight_embed_head/GIB:.4f} GiB embed/head = {weight_all_layers/GIB:.3f} GiB")
    print(f"  => DiT NO-SPLIT structured-estimate peak (weights + activation): {peak_bytes_no_split/GIB:.3f} GiB")
    print()
    print(f"FOR REFERENCE ONLY (as actually executed in the real log, WITH graph splitting into 33")
    print(f"per-layer compile+infer+release stages -- NOT the scenario requested above):")
    print(f"  weight bytes = 1 x {weight_per_layer/GIB:.3f} GiB/layer (only 1-of-{num_layers} resident at a time)")
    print(f"  => DiT GRAPH-SPLIT structured-estimate peak: {peak_bytes_staged/GIB:.3f} GiB")
    if vid_tokens == 252960:
        print(f"  real measured (SEEDVR2_DIT_GPU/SEEDVR2_DIT_STAGE instrumentation, max usm_device "
              f"17.908 GiB + usm_host 1.567 GiB): 19.475 GiB")
    else:
        print(f"  real measured reference (19.475 GiB) exists ONLY for the 1920x1088/video_tokens=252960")
        print(f"  config (SeedVR2/ir/seedvr2_verbose_1080P.log) -- NOT applicable here (video_tokens="
              f"{vid_tokens}, a different resolution with no real log of its own). Not shown as a number")
        print(f"  for this config to avoid implying a direct measurement that doesn't exist.")
    print()
    return {"label": "DiT (no graph split)", "weight_all_layers": weight_all_layers,
            "peak_activation": peak_activation, "peak_bytes": peak_bytes_no_split,
            "peak_bytes_staged": peak_bytes_staged,
            "real_measured_staged_gib": 19.475 if vid_tokens == 252960 else None}


def main() -> int:
    ops_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_DIR / "ops_mapping.json"
    ops = json.loads(ops_path.read_text())
    decoder_tiled = ops["config"].get("decoder_tiled", True)

    print("=" * 92)
    print("1) VAE ENCODER (untiled, one compiled model for the whole frame)")
    print("=" * 92)
    enc = analyze_vae_submodel("VAE encoder", ops["vae_encoder"], ENCODER_ROLES)

    print("=" * 92)
    print("2) DiT -- estimating the NO-GRAPH-SPLIT scenario (all 32 layers' weights resident at once),")
    print("   per explicit request, NOT the as-executed-in-the-log per-layer-staged flow")
    print("=" * 92)
    dit = analyze_dit(ops["dit"])

    print("=" * 92)
    if decoder_tiled:
        print("3) VAE DECODER (ONE of K=4 independently-compiled 2x2-grid tile models)")
    else:
        print("3) VAE DECODER (untiled, one compiled model for the whole frame -- per config)")
    print("=" * 92)
    dec = analyze_vae_submodel("VAE decoder" + (" (one tile)" if decoder_tiled else " (untiled)"),
                               ops["vae_decoder"], DECODER_ROLES)
    dec_4tile = dec["peak_bytes"] * 4

    print("=" * 92)
    print("WHOLE-PIPELINE peak GPU memory")
    print("=" * 92)
    print("The 3 submodels run SEQUENTIALLY (preprocess -> vae_encoder -> denoiser(DiT) -> vae_decoder,")
    print("confirmed via the real log's own [PERF] module run= markers) -- they do NOT coexist, so the")
    print("pipeline's peak is the MAX across stages, not the sum.")
    print()
    if decoder_tiled:
        print("VAE decoder's 4 tiles: '1 tile resident' (2.235 GiB) is the REALISTIC figure to use, NOT")
        print("'4 tiles resident concurrently' (8.941 GiB = exactly 4x, since each tile is an independently-")
        print("compiled model with its own full weight copy and no cross-tile sharing). Evidence tiles most")
        print("likely run SEQUENTIALLY (compile->infer->free->next tile): the decoder's real measured")
        print("wall-clock-vs-serial-kernel-time ratio (71.2%, SUMMARY SS1/SS4) is comparable to DiT's own")
        print("ratio (62.8%) in its as-executed graph-split form -- if the 4 tiles ran with real concurrency,")
        print("the decoder's ratio would be noticeably LOWER, not comparable. '4 tiles' is NOT used below.")
    else:
        print("VAE decoder is modeled UNTILED here (per config decoder_tiled=false) -- no tile multiplier")
        print("applies; the figure below is simply the whole decoder's own dependency-aware peak.")
    print()
    print("DiT: uses the NO-GRAPH-SPLIT estimate (all 32 layers' weights resident) as requested -- this")
    print("is a deliberately DIFFERENT (larger, more conservative) scenario than the real production")
    print("pipeline's own as-executed graph-split behavior, shown below only for context.")
    print()
    print(f"  VAE encoder (untiled, 1 compiled model):                       {enc['peak_bytes']/GIB:>8.3f} GiB")
    print(f"  DiT, NO graph split (all 32 layers resident, structured est.): "
          f"{dit['peak_bytes']/GIB:>8.3f} GiB")
    real_staged_str = f"{dit['real_measured_staged_gib']:.3f} GiB real measured" if dit["real_measured_staged_gib"] is not None else "no real measured reference at this resolution"
    print(f"  DiT, WITH graph split (as executed in the real log, for context only): "
          f"{dit['peak_bytes_staged']/GIB:>8.3f} GiB structured / {real_staged_str}")
    print(f"  VAE decoder{' (1 tile, REALISTIC)' if decoder_tiled else ' (untiled)'}:"
          f"{'':>21}{dec['peak_bytes']/GIB:>8.3f} GiB")
    print()
    pipeline_peak = max(enc["peak_bytes"], dit["peak_bytes"], dec["peak_bytes"])
    driver = "DiT (no graph split)" if dit["peak_bytes"] == pipeline_peak else (
        "VAE decoder" if dec["peak_bytes"] == pipeline_peak else "VAE encoder")
    decoder_desc = "decoder=1 tile" if decoder_tiled else "decoder=untiled"
    print(f"=> PIPELINE PEAK (DiT estimated with NO graph split, {decoder_desc}): "
          f"{pipeline_peak/GIB:.3f} GiB, driven by {driver}")
    print()
    if ops["dit"]["config"]["video_tokens"] == 252960:
        print("Real process-wide cross-check (SEEDVR2_DECODER_MEMORY / SEEDVR2_DIT_GPU instrumentation, the")
        print("SAME constant value appears at every sampled point in the whole log, i.e. it was already")
        print("reached before decoder even started and never decreased -- standard OS peak-working-set")
        print("semantics): process_peak_working_set = 27,472,494,592 B = 25.580 GiB. This reflects the REAL")
        print("(graph-split) production pipeline, not the no-split scenario estimated above, so it is not a")
        print("direct cross-check for this section's headline no-split DiT figure.")
    else:
        print("No real process-wide cross-check available at this resolution (the 25.58 GiB figure found")
        print("for the 1920x1088 config comes from its own real log and does not apply here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

