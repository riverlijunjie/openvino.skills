@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Re-test PA sliding attention with sliding_window=1024 after pa_bench fix.
REM Only re-runs sliding PA tests (decode + prefill) that were affected by
REM the hardcoded sliding_window=0 bug.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_moe\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === RETEST PA SLIDING %date% %time% >> "%LOGS%\_index.txt"

REM ========== PA: sliding attention (NH=16 NKV=8 HD=256, SW=1024) ==========
set PA_NH=16
set PA_NKV=8
set PA_HD=256
set PA_SW=1024

REM Decode with sliding window (kv=1024 effective)
call :do pa_sliding_decode_kv1024     "%BUILD%\pa_bench.exe" decode 1 1024 8000 200 4 i8

REM Prefill: all S values with sliding window=1024
for %%S in (1024 2048 4096 8192) do (
  call :do pa_sliding_prefill_S%%S    "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)

echo === RETEST DONE %date% %time% >> "%LOGS%\_index.txt"
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
echo Running !TAG! ...
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
