@echo off
setlocal EnableExtensions EnableDelayedExpansion
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%
if not exist "%LOGS%" mkdir "%LOGS%"

set PA_NH=32
set PA_NKV=16
set PA_HD=256
call :do pa_sliding_decode_kv1024_u4 "%BUILD%\pa_bench.exe" decode 1 1024 4000 200 4 u4 ocl

set PA_NH=32
set PA_NKV=4
set PA_HD=512
call :do pa_full_decode_kv1024_u4    "%BUILD%\pa_bench.exe" decode 1 1024 3000 150 4 u4 ocl
call :do pa_full_decode_kv2048_u4    "%BUILD%\pa_bench.exe" decode 1 2048 2500 120 4 u4 ocl
call :do pa_full_prefill_S49152      "%BUILD%\pa_bench.exe" prefill 49152 0 2 1 1 i8 ocl

echo Done.
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
echo Running !TAG! ...
"%CLI%" -d %CMDLINE% < nul > "%LOGS%\!TAG!.log" 2>&1
echo   rc=!errorlevel!
goto :eof
