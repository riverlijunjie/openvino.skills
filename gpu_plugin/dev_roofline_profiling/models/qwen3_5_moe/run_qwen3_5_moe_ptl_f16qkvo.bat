@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qwen3.5-MoE-35B-A3B roofline sweep on PTL 12Xe (iGPU, 2400 MHz, 110 GB/s, FP16 XMX 58.98 TFLOPS).
REM Key differences from original run_qwen3_5_moe_ptl.bat:
REM   - FC_QKV (2048->5120): uncompressed FP16 weights (precision=f16)
REM   - FC_O   (4096->2048): uncompressed FP16 weights (precision=f16)
REM   - MoE shared expert: FP16 uncompressed weights (shared_quant=f16, last arg)
REM   - Routed MoE experts remain INT4 g=128 as before
REM   - FC linattn_proj (2048->12288): still INT4 g=128 (unchanged)
REM Token sweep: kv/S in {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_f16qkvo\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_run.log"

REM ---------- MoE fused (routed INT4 + shared expert FP16) ----------
REM Args: B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I shared_quant
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2048 512 256 8 128 100 10 4 64 512 f16
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 128  20  5 2 64 512 f16
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 128  15  3 2 64 512 f16
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 128  10  2 2 64 512 f16
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 128   8  2 2 64 512 f16
call :do moe_prefill_S16384           "%BUILD%\moe_bench.exe" 1 16384  2048 512 256 8 128 4 1 1 64 512 f16
call :do moe_prefill_S32768           "%BUILD%\moe_bench.exe" 1 32768  2048 512 256 8 128 2 1 1 64 512 f16
call :do moe_prefill_S65536           "%BUILD%\moe_bench.exe" 1 65536  2048 512 256 8 128 2 1 1 64 512 f16
call :do moe_prefill_S131072          "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 128 1 1 1 64 512 f16

REM ---------- FC_QKV: FP16 uncompressed weights (M=1 decode, M=S prefill) ----------
REM Args: M K N group_size iters warmup num_bufs precision flush_mb
call :do fc_qkv_decode_M1             "%BUILD%\fc_bench.exe" 1    2048   5120 128 5000 200 8 f16 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S 2048 5120 128 30 5 8 f16 64
)
for %%S in (16384 32768 65536 131072) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S 2048 5120 128 10 2 2 f16 64
)

REM ---------- FC_O: FP16 uncompressed weights ----------
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1    4096   2048 128 5000 200 8 f16 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 4096 2048 128 30 5 8 f16 64
)
for %%S in (16384 32768 65536 131072) do (
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 4096 2048 128 10 2 2 f16 64
)

REM ---------- LM_Head: INT8 g=128 (unchanged) ----------
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1    2048 248320 128  300  30 4 u8 64

REM ---------- FC linattn_proj: INT4 g=128 (unchanged) ----------
call :do fc_linattn_proj_decode_M1    "%BUILD%\fc_bench.exe" 1    2048  12288 128 1500 100 4 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_linattn_proj_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2048  12288 128 20 5 4 u4 64
)
for %%S in (16384 32768 65536 131072) do (
  call :do fc_linattn_proj_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2048  12288 128 6 2 2 u4 64
)

REM ---------- PA bench (NH=16 NKV=2 HD=256) ----------
set PA_NH=16
set PA_NKV=2
set PA_HD=256
for %%K in (1024 2048 4096 8192) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 8000 200 4 i8
)
for %%K in (16384 32768 65536 131072) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 2000 50 4 i8
)
for %%S in (1024 2048 4096 8192) do (
  call :do pa_prefill_S%%S            "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)
call :do pa_prefill_S16384            "%BUILD%\pa_bench.exe" prefill 16384  0 5 1 2 i8
call :do pa_prefill_S32768            "%BUILD%\pa_bench.exe" prefill 32768  0 3 1 2 i8
call :do pa_prefill_S65536            "%BUILD%\pa_bench.exe" prefill 65536  0 2 1 1 i8
call :do pa_prefill_S131072           "%BUILD%\pa_bench.exe" prefill 131072 0 1 1 1 i8

REM ---------- GDN bench: linear attention (GatedDeltaNet) ----------
call :do gdn_decode_T1                "%BUILD%\gdn_bench.exe" 1 1    32 32 128 4000 150 4
for %%S in (1024 2048 4096 8192) do (
  call :do gdn_prefill_S%%S           "%BUILD%\gdn_bench.exe" 1 %%S 32 32 128 20 5 2
)
for %%S in (16384 32768 65536 131072) do (
  call :do gdn_prefill_S%%S           "%BUILD%\gdn_bench.exe" 1 %%S 32 32 128 8 2 2
)

REM ---------- Small ops (Qwen3.5: hidden=2048, full-attn 16/2 heads HD=256) ----------
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

echo === END %date% %time% >> "%LOGS%\_run.log"
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
echo === !TAG! :!CMDLINE! >> "%LOGS%\_run.log"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_run.log"
goto :eof
