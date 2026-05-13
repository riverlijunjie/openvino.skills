@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Qwen3-8B PA bench sweep on PTL (Xe2 iGPU, 2400 MHz, 110 GB/s,
REM FP16 XMX 58.98 TFLOPS). Runs both OCL (default micro-kernel) and CM
REM (XAttention) PA paths so the two implementations can be compared.
REM
REM Qwen3-8B attention config (the pa_bench default): NH=32, NKV=8, HD=128.
REM
REM Input token sizes: 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K (per SKILL.md).
REM Each is run separately (no mixed-shape iteration) so cliloader logs are
REM clean per-shape and easy to diff.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_8b\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- PA decode (S_q=1, S_kv ranges over context lengths) ----------
REM pa_bench args: <mode> <S_q> <S_kv> [iters] [warmup] [num_bufs] [kv_type] [impl]
call :do pa_decode_kv1024_ocl       "%BUILD%\pa_bench.exe" decode 1   1024  300 30 4 i8 ocl
call :do pa_decode_kv2048_ocl       "%BUILD%\pa_bench.exe" decode 1   2048  300 30 4 i8 ocl
call :do pa_decode_kv4096_ocl       "%BUILD%\pa_bench.exe" decode 1   4096  200 20 4 i8 ocl
call :do pa_decode_kv8192_ocl       "%BUILD%\pa_bench.exe" decode 1   8192  150 15 4 i8 ocl
call :do pa_decode_kv16384_ocl      "%BUILD%\pa_bench.exe" decode 1  16384  100 10 4 i8 ocl
call :do pa_decode_kv32768_ocl      "%BUILD%\pa_bench.exe" decode 1  32768   50 10 4 i8 ocl
call :do pa_decode_kv65536_ocl      "%BUILD%\pa_bench.exe" decode 1  65536   30  5 2 i8 ocl
call :do pa_decode_kv131072_ocl     "%BUILD%\pa_bench.exe" decode 1 131072   15  3 2 i8 ocl

call :do pa_decode_kv1024_cm        "%BUILD%\pa_bench.exe" decode 1   1024  300 30 4 i8 cm
call :do pa_decode_kv2048_cm        "%BUILD%\pa_bench.exe" decode 1   2048  300 30 4 i8 cm
call :do pa_decode_kv4096_cm        "%BUILD%\pa_bench.exe" decode 1   4096  200 20 4 i8 cm
call :do pa_decode_kv8192_cm        "%BUILD%\pa_bench.exe" decode 1   8192  150 15 4 i8 cm
call :do pa_decode_kv16384_cm       "%BUILD%\pa_bench.exe" decode 1  16384  100 10 4 i8 cm
call :do pa_decode_kv32768_cm       "%BUILD%\pa_bench.exe" decode 1  32768   50 10 4 i8 cm
call :do pa_decode_kv65536_cm       "%BUILD%\pa_bench.exe" decode 1  65536   30  5 2 i8 cm
call :do pa_decode_kv131072_cm      "%BUILD%\pa_bench.exe" decode 1 131072   15  3 2 i8 cm

REM ---------- PA prefill (S_q=S, S_kv=0) ----------
call :do pa_prefill_S1024_ocl       "%BUILD%\pa_bench.exe" prefill   1024 0  100 10 4 i8 ocl
call :do pa_prefill_S2048_ocl       "%BUILD%\pa_bench.exe" prefill   2048 0   60 10 4 i8 ocl
call :do pa_prefill_S4096_ocl       "%BUILD%\pa_bench.exe" prefill   4096 0   40  8 4 i8 ocl
call :do pa_prefill_S8192_ocl       "%BUILD%\pa_bench.exe" prefill   8192 0   25  5 2 i8 ocl
call :do pa_prefill_S16384_ocl      "%BUILD%\pa_bench.exe" prefill  16384 0   15  3 2 i8 ocl
call :do pa_prefill_S32768_ocl      "%BUILD%\pa_bench.exe" prefill  32768 0    8  2 2 i8 ocl
call :do pa_prefill_S65536_ocl      "%BUILD%\pa_bench.exe" prefill  65536 0    4  1 1 i8 ocl
call :do pa_prefill_S131072_ocl     "%BUILD%\pa_bench.exe" prefill 131072 0    2  1 1 i8 ocl

call :do pa_prefill_S1024_cm        "%BUILD%\pa_bench.exe" prefill   1024 0  100 10 4 i8 cm
call :do pa_prefill_S2048_cm        "%BUILD%\pa_bench.exe" prefill   2048 0   60 10 4 i8 cm
call :do pa_prefill_S4096_cm        "%BUILD%\pa_bench.exe" prefill   4096 0   40  8 4 i8 cm
call :do pa_prefill_S8192_cm        "%BUILD%\pa_bench.exe" prefill   8192 0   25  5 2 i8 cm
call :do pa_prefill_S16384_cm       "%BUILD%\pa_bench.exe" prefill  16384 0   15  3 2 i8 cm
call :do pa_prefill_S32768_cm       "%BUILD%\pa_bench.exe" prefill  32768 0    8  2 2 i8 cm
call :do pa_prefill_S65536_cm       "%BUILD%\pa_bench.exe" prefill  65536 0    4  1 1 i8 cm
call :do pa_prefill_S131072_cm      "%BUILD%\pa_bench.exe" prefill 131072 0    2  1 1 i8 cm

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
