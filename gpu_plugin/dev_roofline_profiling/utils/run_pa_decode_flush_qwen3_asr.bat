@echo off
REM Re-run only the 4 PA decode sweeps for Qwen3-ASR-0.6B with the new flush-enabled pa_bench.
REM Same iters/warmup/bufs/kv_type/impl as the original sweep. flush_mb defaults to 64 (CLI arg 9 omitted).
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_asr_0_6B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

set PA_NH=16
set PA_NKV=8
set PA_HD=128

"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  512 500 50 4 f16 ocl > "%LOGS%\pa_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 400 40 4 f16 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 4 f16 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 4 f16 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

echo PA_DECODE_DONE
