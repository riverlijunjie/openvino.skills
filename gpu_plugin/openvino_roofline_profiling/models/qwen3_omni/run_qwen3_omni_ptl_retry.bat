@echo off
setlocal EnableExtensions EnableDelayedExpansion
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_omni\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

set PA_NH=32
set PA_NKV=32
set PA_HD=128

echo === RETRY %date% %time% >> "%LOGS%\_index.txt"

call :do pa_decode_kv2048   "%BUILD%\pa_bench.exe" decode 1   2048 200 20 4 i8 ocl
call :do pa_decode_kv8192   "%BUILD%\pa_bench.exe" decode 1   8192 100 10 4 i8 ocl
call :do pa_prefill_S2048   "%BUILD%\pa_bench.exe" prefill 2048 0   40  8 4 i8 ocl
call :do fc_decode_o        "%BUILD%\fc_bench.exe" 1 4096   4096  128 3000 100 8
call :do fc_decode_down     "%BUILD%\fc_bench.exe" 1 22016  4096  128  800  50 8

echo === RETRY DONE %date% %time% >> "%LOGS%\_index.txt"
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
