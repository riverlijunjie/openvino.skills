@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4-12B-it (dense) roofline sweep — PTL 12Xe
REM  Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM ========================================================================
REM  Input token sizes: 713, 912, 1024, 4096
REM  Output tokens (decode): 512 (multiplied in summary)
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM =====================================================================
REM  FC decode (M=1) — Sliding / Full / Dense / LM_Head
REM =====================================================================
call :do fc_qkv_sliding_decode_M1    "%BUILD%\fc_bench.exe" 1 3840  8192 128 5000 200 8 u4 64
call :do fc_o_sliding_decode_M1      "%BUILD%\fc_bench.exe" 1 4096  3840 128 8000 300 8 u4 64
call :do fc_qk_full_decode_M1        "%BUILD%\fc_bench.exe" 1 3840  8704 128 5000 200 8 u4 64
call :do fc_o_full_decode_M1         "%BUILD%\fc_bench.exe" 1 8192  3840 128 5000 200 8 u4 64
call :do fc_gate_dense_decode_M1     "%BUILD%\fc_bench.exe" 1 3840 15360 128 2500 100 8 u4 64
call :do fc_up_dense_decode_M1       "%BUILD%\fc_bench.exe" 1 3840 15360 128 2500 100 8 u4 64
call :do fc_down_dense_decode_M1     "%BUILD%\fc_bench.exe" 1 15360 3840 128 2500 100 8 u4 64
call :do lm_head_decode_M1           "%BUILD%\fc_bench.exe" 1 3840 262144 128  200  20 4 u8 64

REM =====================================================================
REM  FC prefill — Sliding / Full / Dense MLP for S in {713, 912, 1024, 4096}
REM =====================================================================
for %%S in (713 912 1024 4096) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 40 8 4 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 40 8 4 u4 64
)
for %%S in (713 912 1024 4096) do (
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 40 8 4 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 40 8 4 u4 64
)
for %%S in (713 912 1024 4096) do (
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 30 6 4 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 30 6 4 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 30 6 4 u4 64
)

call :do lm_head_prefill             "%BUILD%\fc_bench.exe" 1 3840 262144 128 200 20 4 u8 64

REM =====================================================================
REM  PA decode — Sliding attention (NH=16, NKV=8, HD=256)
REM  Sliding window = 1024 -> effective KV = min(ctx, 1024)
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_decode_kv713     "%BUILD%\pa_bench.exe" decode 1 713   9000 300 4 i8 ocl
call :do pa_sliding_decode_kv912     "%BUILD%\pa_bench.exe" decode 1 912   8000 300 4 i8 ocl
call :do pa_sliding_decode_kv1024    "%BUILD%\pa_bench.exe" decode 1 1024  6000 200 4 i8 ocl

REM =====================================================================
REM  PA decode — Full attention (NH=16, NKV=1, HD=512)
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
for %%K in (713 912 1024 4096) do (
  call :do pa_full_decode_kv%%K      "%BUILD%\pa_bench.exe" decode 1 %%K 3000 150 4 i8 ocl
)

REM =====================================================================
REM  PA prefill — Sliding attention (S<=1024 only; S=4096 derived from S=1024)
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_prefill_S713     "%BUILD%\pa_bench.exe" prefill 713  0 350 25 4 i8 ocl
call :do pa_sliding_prefill_S912     "%BUILD%\pa_bench.exe" prefill 912  0 300 25 4 i8 ocl
call :do pa_sliding_prefill_S1024    "%BUILD%\pa_bench.exe" prefill 1024 0 250 20 4 i8 ocl

REM =====================================================================
REM  PA prefill — Full attention (NH=16, NKV=1, HD=512)
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_prefill_S713        "%BUILD%\pa_bench.exe" prefill 713  0 350 25 4 i8 ocl
call :do pa_full_prefill_S912        "%BUILD%\pa_bench.exe" prefill 912  0 300 20 4 i8 ocl
call :do pa_full_prefill_S1024       "%BUILD%\pa_bench.exe" prefill 1024 0 250 20 4 i8 ocl
call :do pa_full_prefill_S4096       "%BUILD%\pa_bench.exe" prefill 4096 0  60  5 2 i8 ocl

REM =====================================================================
REM  Small ops — Decode (M=1)
REM =====================================================================
call :do so_rmsnorm_h3840_decode         "%BUILD%\small_ops_bench.exe" rmsnorm   1 3840   --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 1  512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 1  512 --iters 30000 --warmup 300 --bufs 8
call :do so_add_h3840_decode             "%BUILD%\small_ops_bench.exe" add       1 3840   --iters 30000 --warmup 300 --bufs 8

REM =====================================================================
REM  Small ops — Prefill (M=S) for S in {713, 912, 1024, 4096}
REM =====================================================================
for %%S in (713 912 1024 4096) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 500 --warmup 40 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 500 --warmup 40 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 500 --warmup 40 --bufs 4
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 500 --warmup 40 --bufs 4
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 500 --warmup 40 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 500 --warmup 40 --bufs 4
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 500 --warmup 40 --bufs 4
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 500 --warmup 40 --bufs 4
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
echo [%date% %time%] Running !TAG! ...
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
