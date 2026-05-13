@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Supplemental benchmarks for Qwen3.5-MoE linear attention breakdown
REM on PTL 12Xe (B390 iGPU, 2400 MHz, 110 GB/s).
REM
REM Breaking down the old linattn_input_proj FC(M, 2048, 12288) into:
REM   - linattn_qkv:  FC(M, 2048, 8192)   INT4 g=128  (Q+K+V fused)
REM   - linattn_z:    FC(M, 2048, 4096)   INT4 g=128  (z/gate for RMSNormGated)
REM   - linattn_a:    FC(M, 2048, 32)     INT4 g=128  (gate-a, tiny)
REM   - linattn_b:    FC(M, 2048, 32)     INT4 g=128  (beta, tiny)
REM
REM Adding missing:
REM   - linattn_out:  FC(M, 4096, 2048)   INT4 g=128  (output projection, 30 layers)
REM
REM fc_bench args: M K N group_size iters warmup num_bufs precision flush_mb

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_f16qkvo\ptl_linattn
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === LINATTN BENCH START %date% %time% >> "%LOGS%\_run.log"

REM ============================================================
REM  DECODE (M=1): all ops are memory-bound, need many iters
REM ============================================================

REM --- linattn_qkv: FC(1, 2048, 8192) INT4 g=128 ---
REM Weight bytes ~ 2048*8192/2 = 8MB, theoretical latency ~8MB/110GB/s ~ 0.073ms
REM Target 1000ms total -> ~14000 iters; use 15000
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 8192 128 15000 500 8 u4 32 > "%LOGS%\linattn_qkv_decode_M1.log" 2>&1
echo === linattn_qkv_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_z: FC(1, 2048, 4096) INT4 g=128 ---
REM Weight bytes ~ 2048*4096/2 = 4MB, ~ 0.036ms -> 28000 iters; use 20000
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 4096 128 20000 500 8 u4 32 > "%LOGS%\linattn_z_decode_M1.log" 2>&1
echo === linattn_z_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_out: FC(1, 4096, 2048) INT4 g=128 ---
REM Weight bytes ~ 4096*2048/2 = 4MB, ~ 0.036ms -> 28000 iters; use 20000
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2048 128 20000 500 8 u4 32 > "%LOGS%\linattn_out_decode_M1.log" 2>&1
echo === linattn_out_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_a: FC(1, 2048, 32) INT4 g=128 ---
REM Weight bytes ~ 2048*32/2 = 32KB, extremely tiny
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 128 200000 1000 16 u4 32 > "%LOGS%\linattn_a_decode_M1.log" 2>&1
echo === linattn_a_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_b: FC(1, 2048, 32) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 128 200000 1000 16 u4 32 > "%LOGS%\linattn_b_decode_M1.log" 2>&1
echo === linattn_b_decode_M1 done %time% >> "%LOGS%\_run.log"

REM ============================================================
REM  PREFILL: S = 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
REM  Only the significant ops: qkv, z, out_proj
REM  (a and b are negligible for prefill too but included for completeness at small S)
REM ============================================================

REM --- S=1024 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 8192 128 100 10 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 4096 128 200 20 4 u4 32 > "%LOGS%\linattn_z_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2048 128 200 20 4 u4 32 > "%LOGS%\linattn_out_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 128 2000 100 4 u4 32 > "%LOGS%\linattn_a_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 128 2000 100 4 u4 32 > "%LOGS%\linattn_b_prefill_S1024.log" 2>&1
echo === linattn_prefill_S1024 done %time% >> "%LOGS%\_run.log"

REM --- S=2048 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 8192 128 50 5 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 4096 128 100 10 4 u4 32 > "%LOGS%\linattn_z_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2048 128 100 10 4 u4 32 > "%LOGS%\linattn_out_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 128 1000 50 4 u4 32 > "%LOGS%\linattn_a_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 128 1000 50 4 u4 32 > "%LOGS%\linattn_b_prefill_S2048.log" 2>&1
echo === linattn_prefill_S2048 done %time% >> "%LOGS%\_run.log"

REM --- S=4096 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 8192 128 25 3 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 4096 128 50 5 4 u4 32 > "%LOGS%\linattn_z_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2048 128 50 5 4 u4 32 > "%LOGS%\linattn_out_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 128 500 25 4 u4 32 > "%LOGS%\linattn_a_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 128 500 25 4 u4 32 > "%LOGS%\linattn_b_prefill_S4096.log" 2>&1
echo === linattn_prefill_S4096 done %time% >> "%LOGS%\_run.log"

REM --- S=8192 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 8192 128 15 2 2 u4 32 > "%LOGS%\linattn_qkv_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 4096 128 25 3 2 u4 32 > "%LOGS%\linattn_z_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2048 128 25 3 2 u4 32 > "%LOGS%\linattn_out_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 128 200 10 2 u4 32 > "%LOGS%\linattn_a_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 128 200 10 2 u4 32 > "%LOGS%\linattn_b_prefill_S8192.log" 2>&1
echo === linattn_prefill_S8192 done %time% >> "%LOGS%\_run.log"

REM --- S=16384 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 8192 128 8 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 4096 128 15 2 1 u4 32 > "%LOGS%\linattn_z_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2048 128 15 2 1 u4 32 > "%LOGS%\linattn_out_prefill_S16384.log" 2>&1
echo === linattn_prefill_S16384 done %time% >> "%LOGS%\_run.log"

REM --- S=32768 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 8192 128 4 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 4096 128 8 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2048 128 8 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S32768.log" 2>&1
echo === linattn_prefill_S32768 done %time% >> "%LOGS%\_run.log"

REM --- S=65536 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 8192 128 2 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 4096 128 4 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2048 128 4 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S65536.log" 2>&1
echo === linattn_prefill_S65536 done %time% >> "%LOGS%\_run.log"

REM --- S=131072 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 8192 128 1 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 4096 128 2 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2048 128 2 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S131072.log" 2>&1
echo === linattn_prefill_S131072 done %time% >> "%LOGS%\_run.log"

echo === ALL DONE %date% %time% >> "%LOGS%\_run.log"
echo All linear attention benchmarks complete.
