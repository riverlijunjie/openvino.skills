@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Re-run the two benches that were skipped in the main 31B sweep.
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"

call :do fc_qk_full_decode_M1   "%BUILD%\fc_bench.exe" 1 5376 18432 128 2500 120 8 u4 64
call :do pa_full_decode_kv8192  "%BUILD%\pa_bench.exe" decode 1 8192 1500 80 4 i8 ocl

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
