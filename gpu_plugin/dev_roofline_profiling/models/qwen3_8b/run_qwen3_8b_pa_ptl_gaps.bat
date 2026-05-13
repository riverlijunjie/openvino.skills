@echo off
setlocal EnableExtensions EnableDelayedExpansion
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_8b\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

call :do pa_decode_kv4096_cm   "%BUILD%\pa_bench.exe" decode 1   4096 200 20 4 i8 cm
call :do pa_decode_kv8192_cm   "%BUILD%\pa_bench.exe" decode 1   8192 150 15 4 i8 cm
call :do pa_decode_kv16384_cm  "%BUILD%\pa_bench.exe" decode 1  16384 100 10 4 i8 cm
call :do pa_decode_kv32768_cm  "%BUILD%\pa_bench.exe" decode 1  32768  50 10 4 i8 cm
call :do pa_prefill_S1024_ocl  "%BUILD%\pa_bench.exe" prefill   1024 0 100 10 4 i8 ocl
call :do pa_prefill_S2048_ocl  "%BUILD%\pa_bench.exe" prefill   2048 0  60 10 4 i8 ocl
echo === GAPS_DONE %date% %time% >> "%LOGS%\_index.txt"
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
