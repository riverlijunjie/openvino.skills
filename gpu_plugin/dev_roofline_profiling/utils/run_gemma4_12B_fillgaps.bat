@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Fill in missing logs from previous gemma4-12B sweep

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

echo === FILLGAPS START %date% %time% >> "%LOGS%\_index.txt"

REM Missing FC decode
call :do fc_o_sliding_decode_M1      "%BUILD%\fc_bench.exe" 1 4096  3840 128 8000 300 8 u4 64

REM Missing FC prefill - sliding (qkv + o) for all S
call :do fc_qkv_sliding_prefill_S512   "%BUILD%\fc_bench.exe" 512  3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S912   "%BUILD%\fc_bench.exe" 912  3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S1024  "%BUILD%\fc_bench.exe" 1024 3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S4096  "%BUILD%\fc_bench.exe" 4096 3840 8192 128 30 6 2 u4 64

call :do fc_o_sliding_prefill_S512     "%BUILD%\fc_bench.exe" 512  4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S912     "%BUILD%\fc_bench.exe" 912  4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S1024    "%BUILD%\fc_bench.exe" 1024 4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S4096    "%BUILD%\fc_bench.exe" 4096 4096 3840 128 30 6 2 u4 64

REM Missing FC prefill - full (qk + o) for all S
call :do fc_qk_full_prefill_S512       "%BUILD%\fc_bench.exe" 512  3840 8704 128 40 8 4 u4 64
call :do fc_qk_full_prefill_S912       "%BUILD%\fc_bench.exe" 912  3840 8704 128 40 8 4 u4 64
call :do fc_qk_full_prefill_S1024      "%BUILD%\fc_bench.exe" 1024 3840 8704 128 40 8 4 u4 64
call :do fc_qk_full_prefill_S4096      "%BUILD%\fc_bench.exe" 4096 3840 8704 128 30 6 2 u4 64

call :do fc_o_full_prefill_S512        "%BUILD%\fc_bench.exe" 512  8192 3840 128 40 8 4 u4 64
call :do fc_o_full_prefill_S912        "%BUILD%\fc_bench.exe" 912  8192 3840 128 40 8 4 u4 64
call :do fc_o_full_prefill_S1024       "%BUILD%\fc_bench.exe" 1024 8192 3840 128 40 8 4 u4 64
call :do fc_o_full_prefill_S4096       "%BUILD%\fc_bench.exe" 4096 8192 3840 128 30 6 2 u4 64

REM Missing small-ops decode
call :do so_rmsnorm3d_q_full_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode         "%BUILD%\small_ops_bench.exe" rope      1 1  512 --iters 30000 --warmup 300 --bufs 8

echo === FILLGAPS END %date% %time% >> "%LOGS%\_index.txt"
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
echo [%date% %time%] Running !TAG! ...
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
