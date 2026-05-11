@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ================================================================
REM  Qwen3.5-MoE  INT4 g=128 all-compressed  —  v4
REM  Shared expert FUSED into MoE kernel (shared_I=512, shared_quant=u4)
REM  Only FC/MoE ops (group_size-dependent).
REM  PA/GDN/small_ops reuse from g=64 v4 run (unchanged by group_size).
REM ================================================================
REM fc_bench:  M K N group_size iters warmup num_bufs precision flush_mb
REM moe_bench: B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I [shared_quant]

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_int4g128\ptl_v4
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === INT4g128 v4 START %date% %time% > "%LOGS%\_run.log"

REM ================================================================
REM  PART 1: DECODE (M=1)
REM ================================================================

REM --- FC_QKV: FC(1, 2048, 5120) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 5120 128 15000 500 8 u4 32 > "%LOGS%\fc_qkv_decode_M1.log" 2>&1
echo === fc_qkv_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- FC_O: FC(1, 4096, 2048) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2048 128 20000 500 8 u4 32 > "%LOGS%\fc_o_decode_M1.log" 2>&1
echo === fc_o_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_qkv: FC(1, 2048, 8192) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 8192 128 15000 500 8 u4 32 > "%LOGS%\linattn_qkv_decode_M1.log" 2>&1
echo === linattn_qkv_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_z: FC(1, 2048, 4096) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 4096 128 20000 500 8 u4 32 > "%LOGS%\linattn_z_decode_M1.log" 2>&1
echo === linattn_z_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_out: FC(1, 4096, 2048) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2048 128 20000 500 8 u4 32 > "%LOGS%\linattn_out_decode_M1.log" 2>&1
echo === linattn_out_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_a: FC(1, 2048, 32) INT4 g=128 (tiny) ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 128 200000 1000 16 u4 32 > "%LOGS%\linattn_a_decode_M1.log" 2>&1
echo === linattn_a_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_b: FC(1, 2048, 32) INT4 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 128 200000 1000 16 u4 32 > "%LOGS%\linattn_b_decode_M1.log" 2>&1
echo === linattn_b_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- LM head: FC(1, 2048, 248320) INT8 g=128 (same as g=64 config) ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 248320 128 30 5 2 u8 32 > "%LOGS%\lm_head_decode_M1.log" 2>&1
echo === lm_head_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- MoE fused (routed+shared), INT4 g=128 ---
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1 2048 512 256 8 128 100 10 4 64 512 u4 > "%LOGS%\moe_fused_decode_M1.log" 2>&1
echo === moe_fused_decode_M1 done %time% >> "%LOGS%\_run.log"

REM ================================================================
REM  PART 2: PREFILL S=1024..131072
REM ================================================================

REM --- S=1024 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 5120 128 100 10 4 u4 32 > "%LOGS%\fc_qkv_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2048 128 200 20 4 u4 32 > "%LOGS%\fc_o_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 8192 128 100 10 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 4096 128 200 20 4 u4 32 > "%LOGS%\linattn_z_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2048 128 200 20 4 u4 32 > "%LOGS%\linattn_out_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 128 2000 100 4 u4 32 > "%LOGS%\linattn_a_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 128 2000 100 4 u4 32 > "%LOGS%\linattn_b_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 128 20 5 2 64 512 u4 > "%LOGS%\moe_fused_prefill_S1024.log" 2>&1
echo === prefill_S1024 done %time% >> "%LOGS%\_run.log"

REM --- S=2048 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 5120 128 50 5 4 u4 32 > "%LOGS%\fc_qkv_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2048 128 100 10 4 u4 32 > "%LOGS%\fc_o_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 8192 128 50 5 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 4096 128 100 10 4 u4 32 > "%LOGS%\linattn_z_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2048 128 100 10 4 u4 32 > "%LOGS%\linattn_out_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 128 1000 50 4 u4 32 > "%LOGS%\linattn_a_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 128 1000 50 4 u4 32 > "%LOGS%\linattn_b_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 128 15 3 2 64 512 u4 > "%LOGS%\moe_fused_prefill_S2048.log" 2>&1
echo === prefill_S2048 done %time% >> "%LOGS%\_run.log"

REM --- S=4096 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 5120 128 25 3 4 u4 32 > "%LOGS%\fc_qkv_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2048 128 50 5 4 u4 32 > "%LOGS%\fc_o_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 8192 128 25 3 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 4096 128 50 5 4 u4 32 > "%LOGS%\linattn_z_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2048 128 50 5 4 u4 32 > "%LOGS%\linattn_out_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 128 500 25 4 u4 32 > "%LOGS%\linattn_a_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 128 500 25 4 u4 32 > "%LOGS%\linattn_b_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 128 10 2 2 64 512 u4 > "%LOGS%\moe_fused_prefill_S4096.log" 2>&1
echo === prefill_S4096 done %time% >> "%LOGS%\_run.log"

REM --- S=8192 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 5120 128 15 2 2 u4 32 > "%LOGS%\fc_qkv_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2048 128 25 3 2 u4 32 > "%LOGS%\fc_o_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 8192 128 15 2 2 u4 32 > "%LOGS%\linattn_qkv_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 4096 128 25 3 2 u4 32 > "%LOGS%\linattn_z_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2048 128 25 3 2 u4 32 > "%LOGS%\linattn_out_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 128 200 10 2 u4 32 > "%LOGS%\linattn_a_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 128 200 10 2 u4 32 > "%LOGS%\linattn_b_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 128 8 2 2 64 512 u4 > "%LOGS%\moe_fused_prefill_S8192.log" 2>&1
echo === prefill_S8192 done %time% >> "%LOGS%\_run.log"

REM --- S=16384 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 5120 128 8 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2048 128 15 2 1 u4 32 > "%LOGS%\fc_o_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 8192 128 8 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 4096 128 15 2 1 u4 32 > "%LOGS%\linattn_z_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2048 128 15 2 1 u4 32 > "%LOGS%\linattn_out_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 16384 2048 512 256 8 128 4 1 1 64 512 u4 > "%LOGS%\moe_fused_prefill_S16384.log" 2>&1
echo === prefill_S16384 done %time% >> "%LOGS%\_run.log"

REM --- S=32768 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 5120 128 4 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2048 128 8 1 1 u4 32 > "%LOGS%\fc_o_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 8192 128 4 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 4096 128 8 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2048 128 8 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 32768 2048 512 256 8 128 2 1 1 64 512 u4 > "%LOGS%\moe_fused_prefill_S32768.log" 2>&1
echo === prefill_S32768 done %time% >> "%LOGS%\_run.log"

REM --- S=65536 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 5120 128 2 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2048 128 4 1 1 u4 32 > "%LOGS%\fc_o_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 8192 128 2 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 4096 128 4 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2048 128 4 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 65536 2048 512 256 8 128 2 1 1 64 512 u4 > "%LOGS%\moe_fused_prefill_S65536.log" 2>&1
echo === prefill_S65536 done %time% >> "%LOGS%\_run.log"

REM --- S=131072 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 5120 128 1 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2048 128 2 1 1 u4 32 > "%LOGS%\fc_o_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 8192 128 1 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 4096 128 2 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2048 128 2 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 128 1 1 1 64 512 u4 > "%LOGS%\moe_fused_prefill_S131072.log" 2>&1
echo === prefill_S131072 done %time% >> "%LOGS%\_run.log"

echo === ALL DONE %date% %time% >> "%LOGS%\_run.log"
echo All INT4 g=128 v4 benchmarks complete.
