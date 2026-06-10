@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4-12B-it (dense) — PA-only sweep with CM backend — PTL 12Xe
REM  Sizes: 256, 1024, 2048, 4096, 8192, 16384, 32768
REM  Re-runs only paged_attention (decode + prefill) with impl=cm.
REM  FC / LM_head / small_ops measurements are reused from the ocl sweep.
REM  Output tags carry _cm suffix to keep ocl logs intact for comparison.
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === CM PA SWEEP START %date% %time% >> "%LOGS%\_index.txt"

REM =====================================================================
REM  PA decode — Sliding (NH=16, NKV=8, HD=256). Window=1024, so kv>=1024
REM  saturates; we measure kv=256 and kv=1024.
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_decode_kv256_cm     "%BUILD%\pa_bench.exe" decode 1 256   20000 600 4 i8 cm
call :do pa_sliding_decode_kv1024_cm    "%BUILD%\pa_bench.exe" decode 1 1024   8000 300 4 i8 cm

REM =====================================================================
REM  PA decode — Full (NH=16, NKV=1, HD=512). All 7 kv values.
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_decode_kv256_cm        "%BUILD%\pa_bench.exe" decode 1 256    8000 300 4 i8 cm
call :do pa_full_decode_kv1024_cm       "%BUILD%\pa_bench.exe" decode 1 1024   4000 200 4 i8 cm
call :do pa_full_decode_kv2048_cm       "%BUILD%\pa_bench.exe" decode 1 2048   3000 150 4 i8 cm
call :do pa_full_decode_kv4096_cm       "%BUILD%\pa_bench.exe" decode 1 4096   2000 100 4 i8 cm
call :do pa_full_decode_kv8192_cm       "%BUILD%\pa_bench.exe" decode 1 8192   1500  80 4 i8 cm
call :do pa_full_decode_kv16384_cm      "%BUILD%\pa_bench.exe" decode 1 16384   800  40 4 i8 cm
call :do pa_full_decode_kv32768_cm      "%BUILD%\pa_bench.exe" decode 1 32768   400  20 4 i8 cm

REM =====================================================================
REM  PA prefill — Sliding (NH=16, NKV=8, HD=256). Only S=256 and S=1024
REM  needed (window saturates at 1024).
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_prefill_S256_cm     "%BUILD%\pa_bench.exe" prefill 256  0 1000 50 4 i8 cm
call :do pa_sliding_prefill_S1024_cm    "%BUILD%\pa_bench.exe" prefill 1024 0  300 30 4 i8 cm

REM =====================================================================
REM  PA prefill — Full (S²). All 7 S values; tiny iter counts for large S.
REM =====================================================================
set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_prefill_S256_cm        "%BUILD%\pa_bench.exe" prefill 256   0 1000 50 4 i8 cm
call :do pa_full_prefill_S1024_cm       "%BUILD%\pa_bench.exe" prefill 1024  0  300 30 4 i8 cm
call :do pa_full_prefill_S2048_cm       "%BUILD%\pa_bench.exe" prefill 2048  0  150 10 4 i8 cm
call :do pa_full_prefill_S4096_cm       "%BUILD%\pa_bench.exe" prefill 4096  0   60  6 4 i8 cm
call :do pa_full_prefill_S8192_cm       "%BUILD%\pa_bench.exe" prefill 8192  0   20  3 2 i8 cm
call :do pa_full_prefill_S16384_cm      "%BUILD%\pa_bench.exe" prefill 16384 0    6  1 2 i8 cm
call :do pa_full_prefill_S32768_cm      "%BUILD%\pa_bench.exe" prefill 32768 0    3  1 1 i8 cm

echo === CM PA SWEEP END %date% %time% >> "%LOGS%\_index.txt"
echo Done. CM PA logs in %LOGS%
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
