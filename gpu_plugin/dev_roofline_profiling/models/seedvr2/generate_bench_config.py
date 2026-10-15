#!/usr/bin/env python3
"""One-off generator: reads models/seedvr2/ops_mapping.json and emits
models/seedvr2/kernel_table_config.json plus utils/run_seedvr2_ptl.bat, keeping
bench CLI args, log-file tags, and kernel-table 'log' keys perfectly in sync.
Run locally once; outputs are committed as regular files afterward (this
generator script itself is not part of the SKILL's utils/ engine).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # dev_roofline_profiling/
MODEL_DIR = ROOT / "models" / "seedvr2"
UTILS_DIR = ROOT / "utils"

ops = json.loads((MODEL_DIR / "ops_mapping.json").read_text())

SUBMODEL_TAG = {"vae_encoder": "enc", "dit": "dit", "vae_decoder": "dec"}


def slug(name: str) -> str:
    return name.replace(".", "_")


GPU_MEM_BUDGET_BYTES = 18 * 1024 ** 3  # stay under a 20 GiB target with headroom


def cap_bufs(per_buf_bytes: float, requested: int = 4, budget: int = GPU_MEM_BUDGET_BYTES) -> int:
    """fc_bench/sdpa_bench/vlsdpa_bench have no built-in memory-budget guard
    (unlike conv_bench/groupnorm_bench, which cap num_bufs at runtime) -- cap
    the num_bufs CLI arg generated here instead, so one sweep op never tries
    to allocate requested_bufs x per_buf_bytes of VRAM for large DiT/attention
    shapes."""
    bufs = max(1, requested)
    while bufs > 1 and bufs * per_buf_bytes > budget:
        bufs -= 1
    return bufs


kernel_table = {
    "title": "f'# {model_name} \u2014 Per-submodel Kernel Tables'",
    "variables": {},
    "modes": [],
}

bat_lines = []
bat_lines.append("@echo off")
bat_lines.append("setlocal EnableExtensions EnableDelayedExpansion")
bat_lines.append("")
bat_lines.append("REM ============================================================================")
bat_lines.append("REM SeedVR2 roofline sweep on PTL (Machine 1, D:\\river\\ovmx_dev). Covers the 3")
bat_lines.append("REM submodels (VAE encoder, windowed-attention DiT, VAE decoder) at the real")
bat_lines.append("REM measured resolution (1920x1088-derived 136x240 latent grid, txt=58).")
bat_lines.append("REM Shapes/call-counts are generated from models/seedvr2/ops_mapping.json --")
bat_lines.append("REM regenerate both together via models/seedvr2/generate_bench_config.py, never")
bat_lines.append("REM hand-edit the per-op lines below.")
bat_lines.append("REM ============================================================================")
bat_lines.append("")
bat_lines.append("set OV_BIN=D:\\river\\ovmx_dev\\openvino.pipeline.mx\\bin\\intel64\\Release")
bat_lines.append("set CLI=C:\\Users\\Local_Admin\\Downloads\\clintercept-3.0.6-win64\\Release\\cliloader.exe")
bat_lines.append("set BUILD=D:\\river\\ovmx_dev\\dev_roofline_profiling\\utils\\build\\Release")
bat_lines.append("set LOGS=D:\\river\\ovmx_dev\\roofline\\results\\seedvr2")
bat_lines.append("set PATH=%OV_BIN%;%PATH%")
bat_lines.append("")
bat_lines.append('if not exist "%LOGS%" mkdir "%LOGS%"')
bat_lines.append('echo === START %date% %time% > "%LOGS%\\_index.txt"')
bat_lines.append("")


def add_conv_rows(mode_key, submodel, entries):
    tag = SUBMODEL_TAG[submodel]
    specs = []
    bat_lines.append(f"REM ---------- {submodel}: convolution ----------")
    for e in entries:
        name = slug(e["name"])
        log = f"conv_{tag}_{name}"
        kt, kh, kw = e["k"]
        st, sh, sw = e["stride"]
        is_causal = e["kind"] == "causal"
        t_carry = max(kt - st, 0) if is_causal else 0
        t_total = e["t_body"] + t_carry
        t_out = (t_total - kt) // st + 1
        pad_h = (kh - 1) // 2
        pad_w = (kw - 1) // 2
        h_out = (e["h"] + 2 * pad_h - kh) // sh + 1
        w_out = (e["w"] + 2 * pad_w - kw) // sw + 1
        iters = max(10, min(400, int(2000 / max(1, (e["c_out"] * t_out * h_out * w_out) / 2e7))))
        bat_lines.append(
            f'call :do {log} "%BUILD%\\conv_bench.exe" {e["kind"]} {e["c_in"]} {e["c_out"]} '
            f'{kt} {kh} {kw} {e["t_body"]} {e["h"]} {e["w"]} {st} {sh} {sw} {iters} 5 4 1 0 64'
        )
        specs.append({
            "op": f"'{e['name']}'",
            "log": f"'{log}'",
            "calls": str(e["calls"]),
            "bytes": f"conv3d_bytes({e['c_in']},{e['c_out']},{kt},{kh},{kw},{e['t_body']},{t_carry},{e['h']},{e['w']},{t_out},{h_out},{w_out})",
            "flops": f"conv3d_flops({e['c_in']},{e['c_out']},{kt},{kh},{kw},{t_out},{h_out},{w_out})",
            "bound": "'compute'",
        })
    bat_lines.append("")
    return specs


def add_gn_rows(mode_key, submodel, entries):
    tag = SUBMODEL_TAG[submodel]
    specs = []
    bat_lines.append(f"REM ---------- {submodel}: groupnorm ----------")
    for e in entries:
        name = slug(e["name"])
        log = f"gn_{tag}_{name}"
        n_elems = e["c"] * e["z"] * e["h"] * e["w"]
        iters = max(20, min(2000, int(3e9 / max(1, n_elems))))
        bat_lines.append(
            f'call :do {log} "%BUILD%\\groupnorm_bench.exe" {e["c"]} {e["z"]} {e["h"]} {e["w"]} '
            f'{e["num_groups"]} {iters} 10 4'
        )
        specs.append({
            "op": f"'{e['name']}'",
            "log": f"'{log}'",
            "calls": str(e["calls"]),
            "bytes": f"groupnorm_bytes({e['c']},{e['z']},{e['h']},{e['w']})",
            "flops": f"groupnorm_flops({e['c']},{e['z']},{e['h']},{e['w']})",
            "bound": "'memory'",
        })
    bat_lines.append("")
    return specs


def add_fc_rows(mode_key, submodel, entries):
    tag = SUBMODEL_TAG[submodel]
    specs = []
    bat_lines.append(f"REM ---------- {submodel}: FC (plain f16 matmul) ----------")
    for e in entries:
        name = slug(e["name"])
        log = f"fc_{tag}_{name}"
        m, k, n = e["m"], e["k"], e["n"]
        iters = max(10, min(200, int(2e9 / max(1, m * k * n))))
        # Per-buf device bytes: this request's own input (M*K) + output (M*N);
        # the weight (K*N) is a shared Constant, not per-buf.
        per_buf_bytes = (m * k + m * n) * 2
        bufs = cap_bufs(per_buf_bytes, requested=4)
        bat_lines.append(
            f'call :do {log} "%BUILD%\\fc_bench.exe" {m} {k} {n} 128 {iters} 5 {bufs} f16 64'
        )
        specs.append({
            "op": f"'{e['name']}'",
            "log": f"'{log}'",
            "calls": str(e["calls"]),
            "bytes": f"({m}*{k}+{k}*{n}+{m}*{n})*2",
            "flops": f"fc_flops({m},{k},{n})",
            "bound": "'compute'",
        })
    bat_lines.append("")
    return specs


def add_attn_rows(mode_key, submodel, entries):
    tag = SUBMODEL_TAG[submodel]
    specs = []
    bat_lines.append(f"REM ---------- {submodel}: attention ----------")
    for e in entries:
        name = slug(e["name"])
        log = f"attn_{tag}_{name}"
        if e["family"] == "sdpa":
            nh, nkv, hd, s = e["nh"], e["nkv"], e["hd"], e["s"]
            per_buf_bytes = s * nh * hd * 2 * 4  # Q+K+V+out
            bufs = cap_bufs(per_buf_bytes, requested=4)
            bat_lines.append(
                f'set SDPA_NH={nh}&& set SDPA_NKV={nkv}&& set SDPA_HD={hd}&&'
                f' call :do {log} "%BUILD%\\sdpa_bench.exe" prefill {s} {s} 20 5 {bufs} 0'
            )
            flops = f"(4*{s}*{s}*{hd}*{nh} + 25*{s}*{s}*{nh})"
            byts = f"({s}*{nh}*{hd}*2*3 + {s}*{nh}*{hd}*2)"
        else:  # vlsdpa
            hd, nh, nw, tt = e["head_size"], e["num_heads"], e["num_windows"], e["total_tokens"]
            win = tt // nw
            csv = ",".join(str(i * win) for i in range(nw)) + f",{tt}"
            per_buf_bytes = tt * nh * hd * 2 * 4  # Q+K+V+out
            bufs = cap_bufs(per_buf_bytes, requested=4)
            bat_lines.append(
                f'call :do {log} "%BUILD%\\vlsdpa_bench.exe" {hd} {nh} "{csv}" 50 10 {bufs}'
            )
            flops = f"(4*{win}*{win}*{hd}*{nh}*{nw} + 25*{win}*{win}*{nh}*{nw})"
            byts = f"({tt}*{nh}*{hd}*2*3 + {tt}*{nh}*{hd}*2)"
        specs.append({
            "op": f"'{e['name']}'",
            "log": f"'{log}'",
            "calls": str(e["calls"]),
            "bytes": byts,
            "flops": flops,
            "bound": "'compute'",
        })
    bat_lines.append("")
    return specs


for mode_key, submodel in [("vae_encoder", "vae_encoder"), ("dit", "dit"), ("vae_decoder", "vae_decoder")]:
    section = ops[mode_key]
    specs = []
    if "conv" in section:
        specs += add_conv_rows(mode_key, submodel, section["conv"])
    if "groupnorm" in section:
        specs += add_gn_rows(mode_key, submodel, section["groupnorm"])
    if "fc" in section:
        specs += add_fc_rows(mode_key, submodel, section["fc"])
    if "attention" in section:
        specs += add_attn_rows(mode_key, submodel, section["attention"])

    kernel_table["modes"].append({
        "name": mode_key,
        "heading": f"'### {mode_key.upper()}'",
        "axis_name": "stage",
        "axis_label": "stage",
        "axis_values": [mode_key],
        "peak_compute": "platform.get('fp16_xmx_tflops', 0) * 1000",
        "specs": specs,
    })

(MODEL_DIR / "kernel_table_config.json").write_text(json.dumps(kernel_table, indent=2))

bat_lines.append('echo === END %date% %time% >> "%LOGS%\\_index.txt"')
bat_lines.append('echo Done. Logs in %LOGS%')
bat_lines.append("goto :eof")
bat_lines.append("")
bat_lines.append(":do")
bat_lines.append("set TAG=%~1")
bat_lines.append("shift")
bat_lines.append("set CMDLINE=")
bat_lines.append(":doargs")
bat_lines.append('if "%~1"=="" goto dorun')
bat_lines.append("set CMDLINE=%CMDLINE% %1")
bat_lines.append("shift")
bat_lines.append("goto doargs")
bat_lines.append(":dorun")
bat_lines.append('echo === !TAG! :!CMDLINE! >> "%LOGS%\\_index.txt"')
bat_lines.append('"%CLI%" -d %CMDLINE% > "%LOGS%\\!TAG!.log" 2>&1')
bat_lines.append('if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\\_index.txt"')
bat_lines.append("goto :eof")

(UTILS_DIR / "run_seedvr2_ptl.bat").write_text("\n".join(bat_lines) + "\n")
print("wrote", MODEL_DIR / "kernel_table_config.json")
print("wrote", UTILS_DIR / "run_seedvr2_ptl.bat")
print("total specs:", sum(len(m["specs"]) for m in kernel_table["modes"]))
