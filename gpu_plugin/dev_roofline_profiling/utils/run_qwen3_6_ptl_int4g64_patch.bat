@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM PATCH: re-run only the 4 decode cases missing from the main g64 sweep
REM   moe_decode_M1, fc_o_decode_M1, lm_head_decode_M1, so_rmsnorm_h2048_decode
REM Same env / args as run_qwen3_6_ptl_int4g64.bat. Appends to _index.txt.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_6_int4g64\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === PATCH START %date% %time% >> "%LOGS%\_index.txt"

call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1 2048 512 256 8 64 100 10 4 64 512
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1 4096 2048 64 5000 200 8 u4 64
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1 2048 248320 2048 300 30 4 u8 64
call :do so_rmsnorm_h2048_decode      "%BUILD%\small_ops_bench.exe" rmsnorm 1 2048 --iters 30000 --warmup 300 --bufs 8

echo === PATCH END %date% %time% >> "%LOGS%\_index.txt"
echo Patch done. Logs in %LOGS%
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
