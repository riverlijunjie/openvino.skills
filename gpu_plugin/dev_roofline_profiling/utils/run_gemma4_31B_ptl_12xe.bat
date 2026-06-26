@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4-31B-it (dense) roofline sweep — PTL 12Xe
REM  Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM ========================================================================
REM  Input token sizes (prefill S / decode kv): 1024, 2048, 4096, 8192
REM  Output tokens (decode): 512 (multiplied in summary)
REM
REM  gemma-4-31B-it text config:
REM    H=5376, NL=60 (50 sliding + 10 full, 5:1)
REM    Sliding: NH=32 NKV=16 HD=256  -> Q=8192 K=V=4096  QKV=16384  O_K=8192
REM    Full:    NH=32 NKV=4  HD=512  -> Q=16384 K=2048 V=K  QK=18432  O_K=16384
REM    Dense GEGLU MLP: I=21504 (gate/up 5376->21504, down 21504->5376)
REM    LM_head: 5376 -> 262144 (INT8 g=128, tied), sliding_window=1024
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM =====================================================================
REM  FC decode (M=1) — Sliding / Full / Dense / LM_Head
REM =====================================================================
call :do fc_qkv_sliding_decode_M1    "%BUILD%\fc_bench.exe" 1  5376 16384 128 3000 150 8 u4 64
call :do fc_o_sliding_decode_M1      "%BUILD%\fc_bench.exe" 1  8192  5376 128 5000 200 8 u4 64
call :do fc_qk_full_decode_M1        "%BUILD%\fc_bench.exe" 1  5376 18432 128 2500 120 8 u4 64
call :do fc_o_full_decode_M1         "%BUILD%\fc_bench.exe" 1 16384  5376 128 3000 150 8 u4 64
call :do fc_gate_dense_decode_M1     "%BUILD%\fc_bench.exe" 1  5376 21504 128 2000 100 8 u4 64
call :do fc_up_dense_decode_M1       "%BUILD%\fc_bench.exe" 1  5376 21504 128 2000 100 8 u4 64
call :do fc_down_dense_decode_M1     "%BUILD%\fc_bench.exe" 1 21504  5376 128 2000 100 8 u4 64
call :do lm_head_decode_M1           "%BUILD%\fc_bench.exe" 1  5376 262144 128 120  15 4 u8 64

REM =====================================================================
REM  FC prefill — Sliding / Full / Dense MLP for S in {1024, 2048, 4096, 8192}
REM  iters/warmup tuned per S (work grows ~linearly with S)
REM =====================================================================
call :fcpre 1024 40 8
call :fcpre 2048 24 5
call :fcpre 4096 12 3
call :fcpre 8192  6 2

call :do lm_head_prefill             "%BUILD%\fc_bench.exe" 1 5376 262144 128 120 15 4 u8 64

REM =====================================================================
REM  PA decode — Sliding attention (NH=32, NKV=16, HD=256)
REM  Sliding window = 1024 -> effective KV = min(ctx, 1024)
REM =====================================================================
set PA_NH=32
set PA_NKV=16
set PA_HD=256
call :do pa_sliding_decode_kv1024    "%BUILD%\pa_bench.exe" decode 1 1024  4000 200 4 i8 ocl

REM =====================================================================
REM  PA decode — Full attention (NH=32, NKV=4, HD=512)
REM =====================================================================
set PA_NH=32
set PA_NKV=4
set PA_HD=512
call :do pa_full_decode_kv1024       "%BUILD%\pa_bench.exe" decode 1 1024 3000 150 4 i8 ocl
call :do pa_full_decode_kv2048       "%BUILD%\pa_bench.exe" decode 1 2048 2500 120 4 i8 ocl
call :do pa_full_decode_kv4096       "%BUILD%\pa_bench.exe" decode 1 4096 2000 100 4 i8 ocl
call :do pa_full_decode_kv8192       "%BUILD%\pa_bench.exe" decode 1 8192 1500  80 4 i8 ocl

REM =====================================================================
REM  PA prefill — Sliding attention (S<=1024 only; S>1024 derived from S=1024)
REM =====================================================================
set PA_NH=32
set PA_NKV=16
set PA_HD=256
call :do pa_sliding_prefill_S1024    "%BUILD%\pa_bench.exe" prefill 1024 0 150 15 4 i8 ocl

REM =====================================================================
REM  PA prefill — Full attention (NH=32, NKV=4, HD=512)
REM =====================================================================
set PA_NH=32
set PA_NKV=4
set PA_HD=512
call :do pa_full_prefill_S1024       "%BUILD%\pa_bench.exe" prefill 1024 0 200 20 4 i8 ocl
call :do pa_full_prefill_S2048       "%BUILD%\pa_bench.exe" prefill 2048 0  80 10 4 i8 ocl
call :do pa_full_prefill_S4096       "%BUILD%\pa_bench.exe" prefill 4096 0  25  5 2 i8 ocl
call :do pa_full_prefill_S8192       "%BUILD%\pa_bench.exe" prefill 8192 0   8  2 2 i8 ocl

REM =====================================================================
REM  Small ops — Decode (M=1)
REM =====================================================================
call :do so_rmsnorm_h5376_decode         "%BUILD%\small_ops_bench.exe" rmsnorm   1 5376   --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 4  512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 32 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 32 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 4  512 --iters 30000 --warmup 300 --bufs 8
call :do so_add_h5376_decode             "%BUILD%\small_ops_bench.exe" add       1 5376   --iters 30000 --warmup 300 --bufs 8

REM =====================================================================
REM  Small ops — Prefill (M=S) for S in {1024, 2048, 4096, 8192}
REM =====================================================================
for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h5376_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 5376   --iters 400 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 256 --iters 400 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 400 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 512 --iters 400 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 4  512 --iters 400 --warmup 30 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 32 256 --iters 400 --warmup 30 --bufs 4
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 400 --warmup 30 --bufs 4
  call :do so_add_h5376_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 5376   --iters 400 --warmup 30 --bufs 4
)

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
goto :eof

REM ---- FC prefill helper: %1=S %2=iters %3=warmup ----------------------
:fcpre
set S=%~1
set IT=%~2
set WU=%~3
call :do fc_qkv_sliding_prefill_S%S%  "%BUILD%\fc_bench.exe" %S% 5376 16384 128 %IT% %WU% 4 u4 64
call :do fc_o_sliding_prefill_S%S%    "%BUILD%\fc_bench.exe" %S% 8192  5376 128 %IT% %WU% 4 u4 64
call :do fc_qk_full_prefill_S%S%      "%BUILD%\fc_bench.exe" %S% 5376 18432 128 %IT% %WU% 4 u4 64
call :do fc_o_full_prefill_S%S%       "%BUILD%\fc_bench.exe" %S% 16384 5376 128 %IT% %WU% 4 u4 64
call :do fc_gate_dense_prefill_S%S%   "%BUILD%\fc_bench.exe" %S% 5376 21504 128 %IT% %WU% 4 u4 64
call :do fc_up_dense_prefill_S%S%     "%BUILD%\fc_bench.exe" %S% 5376 21504 128 %IT% %WU% 4 u4 64
call :do fc_down_dense_prefill_S%S%   "%BUILD%\fc_bench.exe" %S% 21504 5376 128 %IT% %WU% 4 u4 64
goto :eof

REM ---- runner --------------------------------------------------------
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
