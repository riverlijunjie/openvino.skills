@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4-31B-it (dense) roofline — EXTENSION sweep — PTL 12Xe
REM  Adds: (1) new token sizes 16K/32K/48K for prefill FC/MLP/small-ops + PA,
REM        (2) 4-bit (u4) KV-cache variants for all PA decode/prefill sizes.
REM
REM  Naming convention:
REM    - i8 KV (8-bit): UNSUFFIXED tags (extends the original i8 sweep).
REM    - u4 KV (4-bit): tags carry a "_u4" suffix.
REM  Logs land in the SAME results dir so parse_logs.py aggregates everything.
REM ========================================================================
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === EXT START %date% %time% >> "%LOGS%\_index.txt"

REM =====================================================================
REM  (1) FC prefill — new sizes S in {16384, 32768, 49152}
REM      KV-precision independent. bufs shrink as M grows (OOM guard:
REM      gate output M*21504*2B ~= 2.1 GB at S=49152).
REM      :fcpreb  %1=S %2=iters %3=warmup %4=bufs
REM =====================================================================
call :fcpreb 16384 24 4 2
call :fcpreb 32768 14 3 1
call :fcpreb 49152 10 2 1

REM =====================================================================
REM  (2) Small ops prefill — new sizes
REM =====================================================================
for %%S in (16384 32768 49152) do (
  call :sopre %%S
)

REM =====================================================================
REM  (3) PA i8 (8-bit) — NEW sizes only (1024..8192 already collected)
REM      Full attention NH=32 NKV=4 HD=512
REM =====================================================================
set PA_NH=32
set PA_NKV=4
set PA_HD=512
call :do pa_full_decode_kv16384      "%BUILD%\pa_bench.exe" decode 1 16384 1500 80 0 i8 ocl
call :do pa_full_decode_kv32768      "%BUILD%\pa_bench.exe" decode 1 32768 1000 60 0 i8 ocl
call :do pa_full_decode_kv49152      "%BUILD%\pa_bench.exe" decode 1 49152  700 40 0 i8 ocl
call :do pa_full_prefill_S16384      "%BUILD%\pa_bench.exe" prefill 16384 0 3 1 1 i8 ocl
call :do pa_full_prefill_S32768      "%BUILD%\pa_bench.exe" prefill 32768 0 2 1 1 i8 ocl
call :do pa_full_prefill_S49152      "%BUILD%\pa_bench.exe" prefill 49152 0 2 1 1 i8 ocl

REM =====================================================================
REM  (4) PA u4 (4-bit) — ALL sizes
REM      Sliding attention NH=32 NKV=16 HD=256 (window=1024)
REM =====================================================================
set PA_NH=32
set PA_NKV=16
set PA_HD=256
call :do pa_sliding_decode_kv1024_u4 "%BUILD%\pa_bench.exe" decode 1 1024 4000 200 4 u4 ocl
call :do pa_sliding_prefill_S1024_u4 "%BUILD%\pa_bench.exe" prefill 1024 0 150 15 4 u4 ocl

REM      Full attention NH=32 NKV=4 HD=512
set PA_NH=32
set PA_NKV=4
set PA_HD=512
call :do pa_full_decode_kv1024_u4    "%BUILD%\pa_bench.exe" decode 1 1024  3000 150 4 u4 ocl
call :do pa_full_decode_kv2048_u4    "%BUILD%\pa_bench.exe" decode 1 2048  2500 120 4 u4 ocl
call :do pa_full_decode_kv4096_u4    "%BUILD%\pa_bench.exe" decode 1 4096  2000 100 4 u4 ocl
call :do pa_full_decode_kv8192_u4    "%BUILD%\pa_bench.exe" decode 1 8192  1500  80 4 u4 ocl
call :do pa_full_decode_kv16384_u4   "%BUILD%\pa_bench.exe" decode 1 16384 1500  80 0 u4 ocl
call :do pa_full_decode_kv32768_u4   "%BUILD%\pa_bench.exe" decode 1 32768 1000  60 0 u4 ocl
call :do pa_full_decode_kv49152_u4   "%BUILD%\pa_bench.exe" decode 1 49152  700  40 0 u4 ocl

call :do pa_full_prefill_S1024_u4    "%BUILD%\pa_bench.exe" prefill 1024  0 200 20 4 u4 ocl
call :do pa_full_prefill_S2048_u4    "%BUILD%\pa_bench.exe" prefill 2048  0  80 10 4 u4 ocl
call :do pa_full_prefill_S4096_u4    "%BUILD%\pa_bench.exe" prefill 4096  0  25  5 2 u4 ocl
call :do pa_full_prefill_S8192_u4    "%BUILD%\pa_bench.exe" prefill 8192  0   8  2 2 u4 ocl
call :do pa_full_prefill_S16384_u4   "%BUILD%\pa_bench.exe" prefill 16384 0   3  1 1 u4 ocl
call :do pa_full_prefill_S32768_u4   "%BUILD%\pa_bench.exe" prefill 32768 0   2  1 1 u4 ocl
call :do pa_full_prefill_S49152_u4   "%BUILD%\pa_bench.exe" prefill 49152 0   2  1 1 u4 ocl

echo === EXT END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
goto :eof

REM ---- FC prefill helper: %1=S %2=iters %3=warmup %4=bufs ---------------
:fcpreb
set S=%~1
set IT=%~2
set WU=%~3
set BF=%~4
call :do fc_qkv_sliding_prefill_S%S%  "%BUILD%\fc_bench.exe" %S% 5376 16384 128 %IT% %WU% %BF% u4 64
call :do fc_o_sliding_prefill_S%S%    "%BUILD%\fc_bench.exe" %S% 8192  5376 128 %IT% %WU% %BF% u4 64
call :do fc_qk_full_prefill_S%S%      "%BUILD%\fc_bench.exe" %S% 5376 18432 128 %IT% %WU% %BF% u4 64
call :do fc_o_full_prefill_S%S%       "%BUILD%\fc_bench.exe" %S% 16384 5376 128 %IT% %WU% %BF% u4 64
call :do fc_gate_dense_prefill_S%S%   "%BUILD%\fc_bench.exe" %S% 5376 21504 128 %IT% %WU% %BF% u4 64
call :do fc_up_dense_prefill_S%S%     "%BUILD%\fc_bench.exe" %S% 5376 21504 128 %IT% %WU% %BF% u4 64
call :do fc_down_dense_prefill_S%S%   "%BUILD%\fc_bench.exe" %S% 21504 5376 128 %IT% %WU% %BF% u4 64
goto :eof

REM ---- Small ops prefill helper: %1=S ----------------------------------
:sopre
set S=%~1
set SBF=2
if %S% GEQ 32768 set SBF=1
call :do so_rmsnorm_h5376_prefill_S%S%       "%BUILD%\small_ops_bench.exe" rmsnorm   %S% 5376   --iters 150 --warmup 20 --bufs %SBF%
call :do so_rmsnorm3d_q_sliding_prefill_S%S% "%BUILD%\small_ops_bench.exe" rmsnorm3d %S% 32 256 --iters 150 --warmup 20 --bufs %SBF%
call :do so_rmsnorm3d_k_sliding_prefill_S%S% "%BUILD%\small_ops_bench.exe" rmsnorm3d %S% 16 256 --iters 150 --warmup 20 --bufs %SBF%
call :do so_rmsnorm3d_q_full_prefill_S%S%    "%BUILD%\small_ops_bench.exe" rmsnorm3d %S% 32 512 --iters 150 --warmup 20 --bufs %SBF%
call :do so_rmsnorm3d_k_full_prefill_S%S%    "%BUILD%\small_ops_bench.exe" rmsnorm3d %S% 4  512 --iters 150 --warmup 20 --bufs %SBF%
call :do so_rope_q_sliding_prefill_S%S%      "%BUILD%\small_ops_bench.exe" rope      %S% 32 256 --iters 150 --warmup 20 --bufs %SBF%
call :do so_rope_k_sliding_prefill_S%S%      "%BUILD%\small_ops_bench.exe" rope      %S% 16 256 --iters 150 --warmup 20 --bufs %SBF%
call :do so_add_h5376_prefill_S%S%           "%BUILD%\small_ops_bench.exe" add       %S% 5376   --iters 150 --warmup 20 --bufs %SBF%
goto :eof

REM ---- runner (stdin redirected from nul to prevent line-eating) --------
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
"%CLI%" -d %CMDLINE% < nul > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=!errorlevel! >> "%LOGS%\_index.txt"
goto :eof
