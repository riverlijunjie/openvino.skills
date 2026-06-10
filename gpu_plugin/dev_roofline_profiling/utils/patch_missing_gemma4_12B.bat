@echo off
setlocal EnableExtensions
REM ========================================================================
REM  Patch missing logs from main gemma4-12B sweep (no for-loops; explicit calls)
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

REM FC decode (missing)
call :do fc_qkv_sliding_decode_M1    "%BUILD%\fc_bench.exe" 1 3840  8192 128 5000 200 8 u4 64

REM FC prefill sliding (4 sizes; missing entirely)
call :do fc_qkv_sliding_prefill_S713   "%BUILD%\fc_bench.exe" 713  3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S912   "%BUILD%\fc_bench.exe" 912  3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S1024  "%BUILD%\fc_bench.exe" 1024 3840 8192 128 40 8 4 u4 64
call :do fc_qkv_sliding_prefill_S4096  "%BUILD%\fc_bench.exe" 4096 3840 8192 128 40 8 4 u4 64

call :do fc_o_sliding_prefill_S713     "%BUILD%\fc_bench.exe" 713  4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S912     "%BUILD%\fc_bench.exe" 912  4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S1024    "%BUILD%\fc_bench.exe" 1024 4096 3840 128 40 8 4 u4 64
call :do fc_o_sliding_prefill_S4096    "%BUILD%\fc_bench.exe" 4096 4096 3840 128 40 8 4 u4 64

REM PA sliding decode (missing kv713, kv912)
call :do pa_sliding_decode_kv713     "%BUILD%\pa_bench.exe" decode 1 713  9000 300 4 i8 ocl
call :do pa_sliding_decode_kv912     "%BUILD%\pa_bench.exe" decode 1 912  8000 300 4 i8 ocl

REM Small ops decode (missing q_full, k_full rope)
call :do so_rmsnorm3d_q_full_decode  "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode       "%BUILD%\small_ops_bench.exe" rope      1 1  512 --iters 30000 --warmup 300 --bufs 8

echo Patch done.
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
echo [%date% %time%] Running %TAG% ...
"%CLI%" -d %CMDLINE% > "%LOGS%\%TAG%.log" 2>&1
if errorlevel 1 echo FAIL %TAG% errorlevel=%errorlevel%
goto :eof
