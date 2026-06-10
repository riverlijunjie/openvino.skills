@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4-12B-it (dense) roofline sweep — PTL 12Xe — EXTENDED SIZES
REM  Sizes: 256, 1024, 2048, 4096, 8192, 16384, 32768
REM  Output tokens (decode): 512 (multiplied in summary)
REM  Reuses existing decode FC/lm_head/small_ops measurements (M=1).
REM  Adds PA_full decode at new kv values + FC/PA/small_ops prefill at new S.
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === EXT SIZES START %date% %time% >> "%LOGS%\_index.txt"

REM =====================================================================
REM  PA decode — Sliding (only kv=256 needed; kv>=1024 reuses kv=1024)
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_decode_kv256     "%BUILD%\pa_bench.exe" decode 1 256  20000 600 4 i8 ocl

REM =====================================================================
REM  PA decode — Full (per kv: 256, 2048, 8192, 16384, 32768; 1024/4096 already done)
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_decode_kv256        "%BUILD%\pa_bench.exe" decode 1 256    8000 300 4 i8 ocl
call :do pa_full_decode_kv2048       "%BUILD%\pa_bench.exe" decode 1 2048   3000 150 4 i8 ocl
call :do pa_full_decode_kv8192       "%BUILD%\pa_bench.exe" decode 1 8192   1500  80 4 i8 ocl
call :do pa_full_decode_kv16384      "%BUILD%\pa_bench.exe" decode 1 16384   800  40 4 i8 ocl
call :do pa_full_decode_kv32768      "%BUILD%\pa_bench.exe" decode 1 32768   400  20 4 i8 ocl

REM =====================================================================
REM  FC prefill — S=256 (small, many iters)
REM =====================================================================
for %%S in (256) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 200 30 4 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 200 30 4 u4 64
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 200 30 4 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 200 30 4 u4 64
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 150 25 4 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 150 25 4 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 150 25 4 u4 64
)

REM =====================================================================
REM  FC prefill — S=2048
REM =====================================================================
for %%S in (2048) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 30 6 4 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 30 6 4 u4 64
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 30 6 4 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 30 6 4 u4 64
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 25 5 4 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 25 5 4 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 25 5 4 u4 64
)

REM =====================================================================
REM  FC prefill — S=8192
REM =====================================================================
for %%S in (8192) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 12 3 2 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 12 3 2 u4 64
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 12 3 2 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 12 3 2 u4 64
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 10 2 2 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 10 2 2 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 10 2 2 u4 64
)

REM =====================================================================
REM  FC prefill — S=16384
REM =====================================================================
for %%S in (16384) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 6 2 2 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 6 2 2 u4 64
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 6 2 2 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 6 2 2 u4 64
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 5 2 2 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 5 2 2 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 5 2 2 u4 64
)

REM =====================================================================
REM  FC prefill — S=32768  (memory: gate/up activation = 32K*15360*2 = 960 MB/buf)
REM =====================================================================
for %%S in (32768) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 3840  8192 128 4 1 2 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096  3840 128 4 1 2 u4 64
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 3840  8704 128 4 1 2 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192  3840 128 4 1 2 u4 64
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 3840 15360 128 3 1 1 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 3840 15360 128 3 1 1 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 15360 3840 128 3 1 1 u4 64
)

REM =====================================================================
REM  PA prefill — Sliding (only S=256 needed; S>=1024 derived from S=1024)
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_prefill_S256     "%BUILD%\pa_bench.exe" prefill 256  0 1000 50 4 i8 ocl

REM =====================================================================
REM  PA prefill — Full (S²) — measure all sizes; large S uses tiny iter counts
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_prefill_S256        "%BUILD%\pa_bench.exe" prefill 256   0 1000 50 4 i8 ocl
call :do pa_full_prefill_S2048       "%BUILD%\pa_bench.exe" prefill 2048  0  150 10 4 i8 ocl
call :do pa_full_prefill_S8192       "%BUILD%\pa_bench.exe" prefill 8192  0   20  3 2 i8 ocl
call :do pa_full_prefill_S16384      "%BUILD%\pa_bench.exe" prefill 16384 0    6  1 2 i8 ocl
call :do pa_full_prefill_S32768      "%BUILD%\pa_bench.exe" prefill 32768 0    3  1 1 i8 ocl

REM =====================================================================
REM  Small ops — Prefill (linear in S). Cap iters & bufs for large S.
REM =====================================================================
for %%S in (256) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 2000 --warmup 100 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 2000 --warmup 100 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 2000 --warmup 100 --bufs 4
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 2000 --warmup 100 --bufs 4
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 2000 --warmup 100 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 2000 --warmup 100 --bufs 4
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 2000 --warmup 100 --bufs 4
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 2000 --warmup 100 --bufs 4
)
for %%S in (2048) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 250 --warmup 20 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 250 --warmup 20 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 250 --warmup 20 --bufs 4
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 250 --warmup 20 --bufs 4
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 250 --warmup 20 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 250 --warmup 20 --bufs 4
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 250 --warmup 20 --bufs 4
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 250 --warmup 20 --bufs 4
)
for %%S in (8192) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 80 --warmup 8 --bufs 2
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 80 --warmup 8 --bufs 2
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 80 --warmup 8 --bufs 2
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 80 --warmup 8 --bufs 2
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 80 --warmup 8 --bufs 2
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 80 --warmup 8 --bufs 2
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 80 --warmup 8 --bufs 2
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 80 --warmup 8 --bufs 2
)
for %%S in (16384) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 30 --warmup 4 --bufs 2
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 30 --warmup 4 --bufs 2
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 30 --warmup 4 --bufs 2
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 30 --warmup 4 --bufs 2
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 30 --warmup 4 --bufs 2
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 30 --warmup 4 --bufs 2
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 30 --warmup 4 --bufs 2
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 30 --warmup 4 --bufs 2
)
for %%S in (32768) do (
  call :do so_rmsnorm_h3840_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 3840   --iters 12 --warmup 2 --bufs 1
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 12 --warmup 2 --bufs 1
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 12 --warmup 2 --bufs 1
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 12 --warmup 2 --bufs 1
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 1  512 --iters 12 --warmup 2 --bufs 1
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 12 --warmup 2 --bufs 1
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 12 --warmup 2 --bufs 1
  call :do so_add_h3840_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 3840   --iters 12 --warmup 2 --bufs 1
)

echo === EXT SIZES END %date% %time% >> "%LOGS%\_index.txt"
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
