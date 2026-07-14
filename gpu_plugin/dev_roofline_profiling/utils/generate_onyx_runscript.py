#!/usr/bin/env python3
"""Generate the Onyx PTL-12Xe cliloader sweep .bat.

Onyx text decoder (measured shapes):
  H=6656, L=52, NH=32, NKV=2, HD=128 -> Q=4096, KV=256
  FC_QKV 6656x4608, FC_OutGate 6656x4096, FC_O 4096x6656
  MLP gate/up 6656x19968, MLP down 19968x6656
  LM_head 6656x202048 (u8)
  sliding_window=2048 (39 sliding + 13 full layers)
  KV cache u4; body FC u4 g=128; LM_head u8 g=128
Token sizes: 1024 2048 4096 8192 16384 32768
"""
SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

# per-S (iters, warmup, bufs, flush_mb) for prefill FC / small-ops
FC_PRE = {1024:(40,8,2,64), 2048:(20,4,2,64), 4096:(10,3,2,64),
          8192:(6,2,1,0),   16384:(3,1,1,0),  32768:(2,1,1,0)}
# per-S for PA prefill (causal, single call grows ~S^2)
PA_PRE = {1024:(60,8,2), 2048:(30,4,2), 4096:(12,2,2),
          8192:(5,1,1),  16384:(3,1,1),  32768:(2,1,1)}
# per-kv for PA full decode
PA_DEC = {1024:(3000,200), 2048:(2000,150), 4096:(1500,120),
          8192:(800,80),   16384:(400,40),  32768:(200,20)}
SO_PRE = {1024:(400,40,4), 2048:(250,25,4), 4096:(120,12,2),
          8192:(60,6,1),   16384:(30,3,1),  32768:(15,2,1)}

L = []
def add(tag, exe, args): L.append(f'call :do {tag:<34} "%BUILD%\\{exe}" {args}')
def sec(t): L.append(""); L.append(f"REM {'='*66}"); L.append(f"REM  {t}"); L.append(f"REM {'='*66}")

L += [
"@echo off",
"setlocal EnableExtensions EnableDelayedExpansion",
"REM ====================================================================",
"REM  Onyx (dense VLM text decoder) roofline sweep - PTL 12Xe (B390 iGPU)",
"REM  Target: Local_Admin@10.239.132.229, 2400 MHz, 110 GB/s",
"REM  Body FC u4 g128, LM_head u8 g128, KV cache u4, PA opencl+micro",
"REM ====================================================================",
r"set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release",
r"set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin",
r"set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe",
r"set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release",
r"set LOGS=D:\river\moe\roofline_results\onyx\ptl_12xe",
"set PATH=%OV_BIN%;%TBB%;%PATH%",
'if not exist "%LOGS%" mkdir "%LOGS%"',
'echo === START %date% %time% > "%LOGS%\\_index.txt"',
]

# ---------------- FC decode (M=1) ----------------
sec("FC decode (M=1) - u4 body, u8 lm_head")
add("fc_qkv_decode_M1",    "fc_bench.exe", "1 6656 4608  128 5000 200 8 u4 64")
add("fc_outgate_decode_M1","fc_bench.exe", "1 6656 4096  128 5000 200 8 u4 64")
add("fc_o_decode_M1",      "fc_bench.exe", "1 4096 6656  128 5000 200 8 u4 64")
add("fc_gate_decode_M1",   "fc_bench.exe", "1 6656 19968 128 2000 100 8 u4 64")
add("fc_up_decode_M1",     "fc_bench.exe", "1 6656 19968 128 2000 100 8 u4 64")
add("fc_down_decode_M1",   "fc_bench.exe", "1 19968 6656 128 2000 100 8 u4 64")
add("lm_head_M1",          "fc_bench.exe", "1 6656 202048 128 100 15 4 u8 64")

# ---------------- FC prefill ----------------
sec("FC prefill (M=S) - u4 body (INT8 XMX path)")
for S in SIZES:
    it,wu,bf,fl = FC_PRE[S]
    for nm,K,N in [("fc_qkv",6656,4608),("fc_outgate",6656,4096),("fc_o",4096,6656),
                   ("fc_gate",6656,19968),("fc_up",6656,19968),("fc_down",19968,6656)]:
        add(f"{nm}_prefill_S{S}", "fc_bench.exe", f"{S} {K} {N} 128 {it} {wu} {bf} u4 {fl}")

# ---------------- PA decode ----------------
sec("PA decode - Onyx uniform heads NH=32 NKV=2 HD=128, u4 KV")
L.append("set PA_NH=32")
L.append("set PA_NKV=2")
L.append("set PA_HD=128")
# sliding: effective kv = min(kv, 2048) -> measure kv=1024 and kv=2048 (cap)
add("pa_sliding_decode_kv1024", "pa_bench.exe", "decode 1 1024 3000 200 4 u4 ocl 64")
add("pa_sliding_decode_kv2048", "pa_bench.exe", "decode 1 2048 2000 150 4 u4 ocl 64")
# full: kv over all sizes
for kv in SIZES:
    it,wu = PA_DEC[kv]
    add(f"pa_full_decode_kv{kv}", "pa_bench.exe", f"decode 1 {kv} {it} {wu} 4 u4 ocl 64")

# ---------------- PA prefill (causal) ----------------
sec("PA prefill (causal) - serves full layers directly; sliding derived by band scaling")
for S in SIZES:
    it,wu,bf = PA_PRE[S]
    fl = 64 if S <= 4096 else 0
    add(f"pa_prefill_S{S}", "pa_bench.exe", f"prefill {S} 0 {it} {wu} {bf} u4 ocl {fl}")

# ---------------- Small ops decode ----------------
sec("Small ops - decode (M=1)")
add("so_rmsnorm_h6656_decode", "small_ops_bench.exe", "rmsnorm   1 6656   --iters 20000 --warmup 300 --bufs 8")
add("so_rmsnorm3d_q_decode",   "small_ops_bench.exe", "rmsnorm3d 1 32 128 --iters 20000 --warmup 300 --bufs 8")
add("so_rmsnorm3d_k_decode",   "small_ops_bench.exe", "rmsnorm3d 1 2  128 --iters 20000 --warmup 300 --bufs 8")
add("so_rope_q_decode",        "small_ops_bench.exe", "rope      1 32 128 --iters 20000 --warmup 300 --bufs 8")
add("so_rope_k_decode",        "small_ops_bench.exe", "rope      1 2  128 --iters 20000 --warmup 300 --bufs 8")
add("so_add_h6656_decode",     "small_ops_bench.exe", "add       1 6656   --iters 20000 --warmup 300 --bufs 8")

# ---------------- Small ops prefill ----------------
sec("Small ops - prefill (M=S)")
for S in SIZES:
    it,wu,bf = SO_PRE[S]
    add(f"so_rmsnorm_h6656_prefill_S{S}", "small_ops_bench.exe", f"rmsnorm   {S} 6656   --iters {it} --warmup {wu} --bufs {bf}")
    add(f"so_rmsnorm3d_q_prefill_S{S}",   "small_ops_bench.exe", f"rmsnorm3d {S} 32 128 --iters {it} --warmup {wu} --bufs {bf}")
    add(f"so_rmsnorm3d_k_prefill_S{S}",   "small_ops_bench.exe", f"rmsnorm3d {S} 2  128 --iters {it} --warmup {wu} --bufs {bf}")
    add(f"so_rope_q_prefill_S{S}",        "small_ops_bench.exe", f"rope      {S} 32 128 --iters {it} --warmup {wu} --bufs {bf}")
    add(f"so_rope_k_prefill_S{S}",        "small_ops_bench.exe", f"rope      {S} 2  128 --iters {it} --warmup {wu} --bufs {bf}")
    add(f"so_add_h6656_prefill_S{S}",     "small_ops_bench.exe", f"add       {S} 6656   --iters {it} --warmup {wu} --bufs {bf}")

# ---------------- footer / :do ----------------
L += [
"",
'echo === END %date% %time% >> "%LOGS%\\_index.txt"',
"echo Done. Logs in %LOGS%",
"goto :eof",
"",
":do",
"set TAG=%~1",
"shift",
"set CMDLINE=",
":doargs",
'if "%~1"=="" goto dorun',
"set CMDLINE=%CMDLINE% %1",
"shift",
"goto doargs",
":dorun",
"echo [%date% %time%] Running !TAG! ...",
'echo === !TAG! :!CMDLINE! >> "%LOGS%\\_index.txt"',
'"%CLI%" -d %CMDLINE% > "%LOGS%\\!TAG!.log" 2>&1',
'if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\\_index.txt"',
"goto :eof",
]

import os
here = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(here, "run_onyx_ptl_12xe.bat")
# CRLF for Windows batch
open(p, "w", newline="\r\n").write("\n".join(L) + "\n")
print("wrote", p, "-", len(L), "lines")
