@echo off
REM Re-run text-decoder PA sweeps (decode + prefill) with INT8 KV cache for Qwen3-ASR-0.6B.
REM Audio-encoder PA prefill (NH=14 NKV=14 HD=64) is intentionally NOT re-run: that
REM model has no persistent KV cache so f16 is the realistic stand-in.
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_asr_0_6B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

set PA_NH=16
set PA_NKV=8
set PA_HD=128

REM PA decode: i8 KV.
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  512 500 50 4 i8 ocl > "%LOGS%\pa_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 400 40 4 i8 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 4 i8 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 4 i8 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM PA prefill: i8 KV.
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill  512 0 200 20 4 i8 ocl > "%LOGS%\pa_prefill_S512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 i8 ocl > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  20  5 2 i8 ocl > "%LOGS%\pa_prefill_S8192.log" 2>&1

echo PA_TEXT_I8_DONE
