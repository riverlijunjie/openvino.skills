@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qwen3.5-MoE-35B-A3B roofline sweep on PTL (iGPU, 2400 MHz, 110 GB/s, FP16 XMX 58.98 TFLOPS).
REM Differences vs Qwen3-Coder PTL run:
REM   - MoE: H=2048 I=512 NE=256 TK=8 + shared_I=512 (always-on shared expert)
REM   - LM_Head: vocab=248320
REM   - PA: NH=16 NKV=2 HD=256 (env vars PA_NH/PA_NKV/PA_HD)
REM   - GatedDeltaNet linear-attention bench (HK=H=32, K=V=128)
REM Iter counts smaller than BMG due to ~4x slower throughput.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- MoE fused (with shared expert) ----------
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2048 512 256 8 128 100 10 4 64 512
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 128  20  5 2 64 512
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 128  15  3 2 64 512
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 128  10  2 2 64 512
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 128   8  2 2 64 512

REM ---------- fc_bench: full-attn QKV, O, LM_Head ----------
call :do fc_qkv_decode_M1             "%BUILD%\fc_bench.exe" 1    2048   5120 128 5000 200 8 u4 64
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1    4096   2048 128 5000 200 8 u4 64
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1    2048 248320 128  300  30 4 u8 64

for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S 2048 5120 128 30 5 8 u4 64
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 4096 2048 128 30 5 8 u4 64
)

REM Linear-attention input projections (Q+K+V+gate ≈ 12288 = 2*16*128 + 32*128 + 32*128)
call :do fc_linattn_proj_decode_M1    "%BUILD%\fc_bench.exe" 1    2048  12288 128 1500 100 4 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_linattn_proj_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2048  12288 128 20 5 4 u4 64
)

REM ---------- pa_bench (NH=16 NKV=2 HD=256) ----------
set PA_NH=16
set PA_NKV=2
set PA_HD=256
for %%K in (1024 2048 4096 8192) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 8000 200 4 i8
)
for %%S in (1024 2048 4096 8192) do (
  call :do pa_prefill_S%%S            "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)

REM ---------- gdn_bench: linear attention (GatedDeltaNet) ----------
REM HK=H=32, K=V=128 (Qwen3.5 linear_num_value_heads=32). Re-enabled 2026-04-28
REM after PTL OV runtime DLL rebuilt with PR #34481 (GPU GDN support) +
REM PR #35472 (GQA / 2-output update). Bench wraps both Result outputs.
call :do gdn_decode_T1                "%BUILD%\gdn_bench.exe" 1 1    32 32 128 4000 150 4
for %%S in (1024 2048 4096 8192) do (
  call :do gdn_prefill_S%%S           "%BUILD%\gdn_bench.exe" 1 %%S 32 32 128 20 5 2
)

REM ---------- small ops (Qwen3.5: hidden=2048, full-attn 16/2 heads HD=256) ----------
call :do so_rmsnorm_h2048_decode      "%BUILD%\small_ops_bench.exe" rmsnorm   1 2048 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_qnorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  2 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_decode             "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_decode             "%BUILD%\small_ops_bench.exe" rope      1  2 256 --iters 30000 --warmup 300 --bufs 8
call :do so_add_decode                "%BUILD%\small_ops_bench.exe" add       1 2048 --iters 30000 --warmup 300 --bufs 8

for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2048_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm %%S 2048   --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_prefill_S%%S        "%BUILD%\small_ops_bench.exe" rope    %%S 16 256 --iters 300 --warmup 30 --bufs 4
)

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
goto :eof

:do
set TAG=%~1
shift
set CMDLINE=
:doargs
if "%~1"=="" goto dorun
set CMDLINE=%CMDLINE% %1
shift
goto doargs
:dorun
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
