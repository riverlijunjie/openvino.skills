@echo off
REM qwen3_omni Thinker text decoder — PA CM-kernel sweep on PTL 12Xe Windows
REM Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM
REM This rerun ONLY re-collects PagedAttention with the CM kernel implementation
REM (pa_bench impl=cm  ->  XAttention path, block_size=256). FC and small-op logs
REM are reused from the existing OCL sweep (run_qwen3_omni_ptl_12xe.bat).

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_omni\ptl_12xe_cm
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

set PA_NH=32
set PA_NKV=8
set PA_HD=128

REM ============ PA decode CM (S_q=1, S_kv = ctx) ============
REM num_bufs=0 -> auto-size from device L3 (target 8x L3) so the rotating-buffer
REM working set genuinely exceeds the LLC. Hard-coding num_bufs=4 with INT8 KV
REM at KV=8192 (per-buf = 16 MiB) on PTL 12Xe / B390 (L3 = 16 MiB) produces a
REM stride-4 access pattern that partially hits the cache and inflates the
REM measured PA BW above DRAM peak.
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 300 30 0 i8 cm > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 2048 300 30 0 i8 cm > "%LOGS%\pa_decode_kv2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 0 i8 cm > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 0 i8 cm > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM ============ PA prefill CM (S_q=S, S_kv=0) ============
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 0 i8 cm > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 2048 0  60 10 0 i8 cm > "%LOGS%\pa_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 0 i8 cm > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  25  5 0 i8 cm > "%LOGS%\pa_prefill_S8192.log" 2>&1

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. PA-CM logs in %LOGS%
