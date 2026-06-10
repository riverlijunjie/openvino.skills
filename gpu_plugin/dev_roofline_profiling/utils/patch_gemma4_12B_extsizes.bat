@echo off
setlocal EnableExtensions EnableDelayedExpansion
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%
echo === GAP PATCH START %date% %time% >> "%LOGS%\_index.txt"

set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_decode_kv256     "%BUILD%\pa_bench.exe" decode 1 256  20000 600 4 i8 ocl

set PA_NH=16
set PA_NKV=1
set PA_HD=512
call :do pa_full_decode_kv256        "%BUILD%\pa_bench.exe" decode 1 256    8000 300 4 i8 ocl
call :do pa_full_decode_kv16384      "%BUILD%\pa_bench.exe" decode 1 16384   800  40 4 i8 ocl

REM FC prefill S=8192 (gap)
call :do fc_qkv_sliding_prefill_S8192  "%BUILD%\fc_bench.exe" 8192 3840  8192 128 12 3 2 u4 64
call :do fc_o_sliding_prefill_S8192    "%BUILD%\fc_bench.exe" 8192 4096  3840 128 12 3 2 u4 64
call :do fc_qk_full_prefill_S8192      "%BUILD%\fc_bench.exe" 8192 3840  8704 128 12 3 2 u4 64
call :do fc_o_full_prefill_S8192       "%BUILD%\fc_bench.exe" 8192 8192  3840 128 12 3 2 u4 64
call :do fc_gate_dense_prefill_S8192   "%BUILD%\fc_bench.exe" 8192 3840 15360 128 10 2 2 u4 64
call :do fc_up_dense_prefill_S8192     "%BUILD%\fc_bench.exe" 8192 3840 15360 128 10 2 2 u4 64
call :do fc_down_dense_prefill_S8192   "%BUILD%\fc_bench.exe" 8192 15360 3840 128 10 2 2 u4 64

REM PA full prefill S=8192 (gap)
call :do pa_full_prefill_S8192       "%BUILD%\pa_bench.exe" prefill 8192 0 20 3 2 i8 ocl

echo === GAP PATCH END %date% %time% >> "%LOGS%\_index.txt"
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
