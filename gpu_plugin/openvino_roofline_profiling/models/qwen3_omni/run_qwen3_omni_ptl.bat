@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM qwen3_omni Thinker text decoder roofline sweep on PTL (Xe2 iGPU, 2400 MHz,
REM 110 GB/s, FP16 XMX 58.98 TFLOPS).
REM
REM Architecture (configuration_qwen3_omni.py defaults / Qwen3VLTextConfig):
REM   hidden=4096, layers=32, MHA NH=NKV=32, head_dim=128,
REM   intermediate=22016, vocab=151936, hidden_act=silu (SwiGLU)
REM
REM Tokens: 1024, 2048, 4096, 8192 (prefill = S, decode = 2nd token at S).
REM PA impl: opencl + micro_kernel.
REM
REM fc_bench positional args: <M> <K> <N> [g] [iters] [warm] [bufs] [u8|i4] [flush_mb]
REM (omit u8 ⇒ INT4 weights; explicit "u8" ⇒ INT8 weights for lm_head)
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_omni\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---- PA: qwen3_omni is MHA (NH=NKV=32, HD=128) ----
set PA_NH=32
set PA_NKV=32
set PA_HD=128

REM =================== PA decode (S_q=1, S_kv = ctx) ===================
call :do pa_decode_kv1024  "%BUILD%\pa_bench.exe" decode 1   1024 300 30 4 i8 ocl
call :do pa_decode_kv2048  "%BUILD%\pa_bench.exe" decode 1   2048 300 30 4 i8 ocl
call :do pa_decode_kv4096  "%BUILD%\pa_bench.exe" decode 1   4096 200 20 4 i8 ocl
call :do pa_decode_kv8192  "%BUILD%\pa_bench.exe" decode 1   8192 150 15 4 i8 ocl

REM =================== PA prefill (S_q=S, S_kv=0) ======================
call :do pa_prefill_S1024  "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl
call :do pa_prefill_S2048  "%BUILD%\pa_bench.exe" prefill 2048 0  60 10 4 i8 ocl
call :do pa_prefill_S4096  "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 i8 ocl
call :do pa_prefill_S8192  "%BUILD%\pa_bench.exe" prefill 8192 0  25  5 2 i8 ocl

REM =================== FC decode (M=1) - INT4 ========================
call :do fc_decode_qkv      "%BUILD%\fc_bench.exe" 1 4096   4096  128 3000 100 8
call :do fc_decode_o        "%BUILD%\fc_bench.exe" 1 4096   4096  128 3000 100 8
call :do fc_decode_gate     "%BUILD%\fc_bench.exe" 1 4096  22016  128  800  50 8
call :do fc_decode_up       "%BUILD%\fc_bench.exe" 1 4096  22016  128  800  50 8
call :do fc_decode_down     "%BUILD%\fc_bench.exe" 1 22016  4096  128  800  50 8
REM lm_head INT8 (u8)
call :do fc_decode_lm_head  "%BUILD%\fc_bench.exe" 1 4096 151936  128  150  10 4 u8

REM =================== FC prefill (M=S) - INT4 =======================
for %%S in (1024 2048 4096 8192) do (
  set IT=200
  if %%S==2048 set IT=120
  if %%S==4096 set IT=60
  if %%S==8192 set IT=30
  call :do fc_prefill_qkv_S%%S   "%BUILD%\fc_bench.exe" %%S 4096   4096 128 !IT! 10 4
  call :do fc_prefill_o_S%%S     "%BUILD%\fc_bench.exe" %%S 4096   4096 128 !IT! 10 4
  call :do fc_prefill_gate_S%%S  "%BUILD%\fc_bench.exe" %%S 4096  22016 128 !IT! 10 4
  call :do fc_prefill_up_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  22016 128 !IT! 10 4
  call :do fc_prefill_down_S%%S  "%BUILD%\fc_bench.exe" %%S 22016  4096 128 !IT! 10 4
)
REM lm_head in prefill is single-token (last token only)
call :do fc_prefill_lm_head     "%BUILD%\fc_bench.exe" 1 4096 151936 128 150 10 4 u8

REM =================== Small ops decode (M=1) ==========================
call :do small_decode_rmsnorm        "%BUILD%\small_ops_bench.exe" rmsnorm   1 4096   --iters 5000 --warmup 200
call :do small_decode_rmsnorm3d_q    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128 --iters 5000 --warmup 200
call :do small_decode_rmsnorm3d_k    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128 --iters 5000 --warmup 200
call :do small_decode_rope           "%BUILD%\small_ops_bench.exe" rope      1 32 128 --iters 5000 --warmup 200
call :do small_decode_swish          "%BUILD%\small_ops_bench.exe" swish     1 22016  --iters 5000 --warmup 200
call :do small_decode_multiply       "%BUILD%\small_ops_bench.exe" multiply  1 22016  --iters 5000 --warmup 200
call :do small_decode_add            "%BUILD%\small_ops_bench.exe" add       1 4096   --iters 5000 --warmup 200

REM =================== Small ops prefill ===============================
for %%S in (1024 2048 4096 8192) do (
  set ITS=1500
  if %%S==2048 set ITS=800
  if %%S==4096 set ITS=400
  if %%S==8192 set ITS=200
  call :do small_prefill_rmsnorm_S%%S      "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 4096    --iters !ITS! --warmup 30
  call :do small_prefill_rmsnorm3d_q_S%%S  "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 128  --iters !ITS! --warmup 30
  call :do small_prefill_rmsnorm3d_k_S%%S  "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 128  --iters !ITS! --warmup 30
  call :do small_prefill_rope_S%%S         "%BUILD%\small_ops_bench.exe" rope      %%S 32 128  --iters !ITS! --warmup 30
  call :do small_prefill_swish_S%%S        "%BUILD%\small_ops_bench.exe" swish     %%S 22016   --iters !ITS! --warmup 30
  call :do small_prefill_multiply_S%%S     "%BUILD%\small_ops_bench.exe" multiply  %%S 22016   --iters !ITS! --warmup 30
  call :do small_prefill_add_S%%S          "%BUILD%\small_ops_bench.exe" add       %%S 4096    --iters !ITS! --warmup 30
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
