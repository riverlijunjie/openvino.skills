@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Re-run CM decode sweep with auto-sized num_bufs (0 = pa_bench picks bufs so
REM resident KV >= 4x L3 = 72 MB, defeating cache reuse). Validates the
REM cache-hit hypothesis for >100% Eff at small/medium S_kv.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_8b\ptl_nocache
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"

REM args: <mode> <S_q> <S_kv> [iters] [warmup] [num_bufs=0->auto] [kv_type] [impl]
call :do pa_decode_kv1024_cm   decode 1   1024  300 30 0 i8 cm
call :do pa_decode_kv2048_cm   decode 1   2048  300 30 0 i8 cm
call :do pa_decode_kv4096_cm   decode 1   4096  200 20 0 i8 cm
call :do pa_decode_kv8192_cm   decode 1   8192  150 15 0 i8 cm
call :do pa_decode_kv16384_cm  decode 1  16384  100 10 0 i8 cm
call :do pa_decode_kv32768_cm  decode 1  32768   50 10 0 i8 cm
call :do pa_decode_kv65536_cm  decode 1  65536   30  5 0 i8 cm
call :do pa_decode_kv131072_cm decode 1 131072   15  3 0 i8 cm

echo Done. Logs in %LOGS%
goto :eof

:do
set TAG=%~1
shift
set ARGS=
:more
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto more
:run
"%CLI%" -d "%BUILD%\pa_bench.exe" %ARGS% > "%LOGS%\%TAG%.log" 2>&1
echo Finished %TAG%
goto :eof
