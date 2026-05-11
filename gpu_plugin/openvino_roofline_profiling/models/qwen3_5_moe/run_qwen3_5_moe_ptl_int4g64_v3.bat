@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ================================================================
REM  Qwen3.5-MoE  INT4 g=64 all-compressed  —  FULL RE-RUN v3
REM  After OpenVINO code update, all ops must be re-benchmarked.
REM ================================================================
REM Config:
REM   Body FC (QKV, O, linattn projs): INT4 asymmetric g=64
REM   Shared expert gate/up/down: INT4 g=64 (3 separate FC, NOT fused)
REM   MoE routed: INT4 g=64 (SI=0, routed only)
REM   LM head: INT8 g=128
REM   KV cache: INT8
REM   PA: OCL implement, NH=16, NKV=2, HD=256
REM   GDN: HK=32, K=V=128
REM
REM fc_bench:  M K N group_size iters warmup num_bufs precision flush_mb
REM moe_bench: B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I
REM pa_bench:  mode Sq Skv iters warmup num_bufs kv_type [impl]
REM gdn_bench: B T HK H K_V iters warmup num_bufs
REM small_ops_bench: op M dim... --iters N --warmup N [--bufs N]

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_int4g64\ptl_v3
set PATH=%OV_BIN%;%TBB%;%PATH%

REM PA head config for Qwen3.5-MoE full attention
set PA_NH=16
set PA_NKV=2
set PA_HD=256

if not exist "%LOGS%" mkdir "%LOGS%"
echo === INT4g64 FULL RE-RUN v3 START %date% %time% > "%LOGS%\_run.log"

REM ================================================================
REM  PART 1: DECODE (M=1)
REM ================================================================

REM --- FC_QKV: FC(1, 2048, 5120) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 5120 64 15000 500 8 u4 32 > "%LOGS%\fc_qkv_decode_M1.log" 2>&1
echo === fc_qkv_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- FC_O: FC(1, 4096, 2048) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2048 64 20000 500 8 u4 32 > "%LOGS%\fc_o_decode_M1.log" 2>&1
echo === fc_o_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_qkv: FC(1, 2048, 8192) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 8192 64 15000 500 8 u4 32 > "%LOGS%\linattn_qkv_decode_M1.log" 2>&1
echo === linattn_qkv_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_z: FC(1, 2048, 4096) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 4096 64 20000 500 8 u4 32 > "%LOGS%\linattn_z_decode_M1.log" 2>&1
echo === linattn_z_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_out: FC(1, 4096, 2048) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2048 64 20000 500 8 u4 32 > "%LOGS%\linattn_out_decode_M1.log" 2>&1
echo === linattn_out_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_a: FC(1, 2048, 32) INT4 g=64 (tiny) ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 64 200000 1000 16 u4 32 > "%LOGS%\linattn_a_decode_M1.log" 2>&1
echo === linattn_a_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- linattn_b: FC(1, 2048, 32) INT4 g=64 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 32 64 200000 1000 16 u4 32 > "%LOGS%\linattn_b_decode_M1.log" 2>&1
echo === linattn_b_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- LM head: FC(1, 2048, 248320) INT8 g=128 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 248320 128 30 5 2 u8 32 > "%LOGS%\lm_head_decode_M1.log" 2>&1
echo === lm_head_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- MoE routed-only (SI=0), INT4 g=64 ---
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1 2048 512 256 8 64 100 10 4 64 0 > "%LOGS%\moe_routed_decode_M1.log" 2>&1
echo === moe_routed_decode_M1 done %time% >> "%LOGS%\_run.log"

REM --- Shared expert: 3 separate INT4 g=64 FC ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 512 64 50000 500 16 u4 32 > "%LOGS%\shared_gate_decode_M1.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 512 64 50000 500 16 u4 32 > "%LOGS%\shared_up_decode_M1.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 512 2048 64 50000 500 16 u4 32 > "%LOGS%\shared_down_decode_M1.log" 2>&1
echo === shared_decode done %time% >> "%LOGS%\_run.log"

REM --- PA decode (INT8 KV, NH=16 NKV=2 HD=256) ---
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 8000 200 4 i8 > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 2048 8000 200 4 i8 > "%LOGS%\pa_decode_kv2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 8000 200 4 i8 > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 4000 100 4 i8 > "%LOGS%\pa_decode_kv8192.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 16384 2000 50 4 i8 > "%LOGS%\pa_decode_kv16384.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 32768 1000 30 4 i8 > "%LOGS%\pa_decode_kv32768.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 65536 500 15 2 i8 > "%LOGS%\pa_decode_kv65536.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 131072 200 5 1 i8 > "%LOGS%\pa_decode_kv131072.log" 2>&1
echo === pa_decode done %time% >> "%LOGS%\_run.log"

REM --- GDN decode (HK=32, K=V=128) ---
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 1 32 32 128 4000 150 4 > "%LOGS%\gdn_decode_T1.log" 2>&1
echo === gdn_decode done %time% >> "%LOGS%\_run.log"

REM --- Small ops decode ---
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   1 2048   --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm_h2048_decode.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_qnorm_decode.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 2 256  --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_knorm_decode.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_q_decode.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 2 256  --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_k_decode.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add       1 2048   --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_add_decode.log" 2>&1
echo === small_ops_decode done %time% >> "%LOGS%\_run.log"

REM ================================================================
REM  PART 2: PREFILL S=1024..131072
REM ================================================================

REM --- S=1024 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 5120 64 100 10 4 u4 32 > "%LOGS%\fc_qkv_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2048 64 200 20 4 u4 32 > "%LOGS%\fc_o_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 8192 64 100 10 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 4096 64 200 20 4 u4 32 > "%LOGS%\linattn_z_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2048 64 200 20 4 u4 32 > "%LOGS%\linattn_out_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 64 2000 100 4 u4 32 > "%LOGS%\linattn_a_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 32 64 2000 100 4 u4 32 > "%LOGS%\linattn_b_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 64 20 5 2 64 0 > "%LOGS%\moe_routed_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_gate_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_up_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 512 2048 64 100 10 4 u4 32 > "%LOGS%\shared_down_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 25 5 4 i8 > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 1024 32 32 128 20 5 2 > "%LOGS%\gdn_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 1024 2048 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2048_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1024 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_prefill_S1024.log" 2>&1
echo === prefill_S1024 done %time% >> "%LOGS%\_run.log"

REM --- S=2048 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 5120 64 50 5 4 u4 32 > "%LOGS%\fc_qkv_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2048 64 100 10 4 u4 32 > "%LOGS%\fc_o_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 8192 64 50 5 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 4096 64 100 10 4 u4 32 > "%LOGS%\linattn_z_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2048 64 100 10 4 u4 32 > "%LOGS%\linattn_out_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 64 1000 50 4 u4 32 > "%LOGS%\linattn_a_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 32 64 1000 50 4 u4 32 > "%LOGS%\linattn_b_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 64 15 3 2 64 0 > "%LOGS%\moe_routed_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_gate_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_up_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 512 2048 64 100 10 4 u4 32 > "%LOGS%\shared_down_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 2048 0 25 5 4 i8 > "%LOGS%\pa_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 2048 32 32 128 20 5 2 > "%LOGS%\gdn_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 2048 2048 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2048_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 2048 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_prefill_S2048.log" 2>&1
echo === prefill_S2048 done %time% >> "%LOGS%\_run.log"

REM --- S=4096 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 5120 64 25 3 4 u4 32 > "%LOGS%\fc_qkv_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2048 64 50 5 4 u4 32 > "%LOGS%\fc_o_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 8192 64 25 3 4 u4 32 > "%LOGS%\linattn_qkv_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 4096 64 50 5 4 u4 32 > "%LOGS%\linattn_z_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2048 64 50 5 4 u4 32 > "%LOGS%\linattn_out_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 64 500 25 4 u4 32 > "%LOGS%\linattn_a_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 32 64 500 25 4 u4 32 > "%LOGS%\linattn_b_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 64 10 2 2 64 0 > "%LOGS%\moe_routed_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_gate_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 512 64 100 10 4 u4 32 > "%LOGS%\shared_up_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 512 2048 64 100 10 4 u4 32 > "%LOGS%\shared_down_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0 25 5 4 i8 > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 4096 32 32 128 20 5 2 > "%LOGS%\gdn_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 4096 2048 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2048_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 4096 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_prefill_S4096.log" 2>&1
echo === prefill_S4096 done %time% >> "%LOGS%\_run.log"

REM --- S=8192 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 5120 64 15 2 2 u4 32 > "%LOGS%\fc_qkv_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2048 64 25 3 2 u4 32 > "%LOGS%\fc_o_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 8192 64 15 2 2 u4 32 > "%LOGS%\linattn_qkv_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 4096 64 25 3 2 u4 32 > "%LOGS%\linattn_z_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2048 64 25 3 2 u4 32 > "%LOGS%\linattn_out_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 64 200 10 2 u4 32 > "%LOGS%\linattn_a_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 32 64 200 10 2 u4 32 > "%LOGS%\linattn_b_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 64 8 2 2 64 0 > "%LOGS%\moe_routed_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 512 64 50 5 2 u4 32 > "%LOGS%\shared_gate_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 512 64 50 5 2 u4 32 > "%LOGS%\shared_up_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 512 2048 64 50 5 2 u4 32 > "%LOGS%\shared_down_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0 25 5 2 i8 > "%LOGS%\pa_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 8192 32 32 128 20 5 2 > "%LOGS%\gdn_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 8192 2048 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2048_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 8192 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_prefill_S8192.log" 2>&1
echo === prefill_S8192 done %time% >> "%LOGS%\_run.log"

REM --- S=16384 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 5120 64 8 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2048 64 15 2 1 u4 32 > "%LOGS%\fc_o_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 8192 64 8 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 4096 64 15 2 1 u4 32 > "%LOGS%\linattn_z_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2048 64 15 2 1 u4 32 > "%LOGS%\linattn_out_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 16384 2048 512 256 8 64 4 1 1 64 0 > "%LOGS%\moe_routed_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 512 64 15 2 1 u4 32 > "%LOGS%\shared_gate_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048 512 64 15 2 1 u4 32 > "%LOGS%\shared_up_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 512 2048 64 15 2 1 u4 32 > "%LOGS%\shared_down_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 16384 0 5 1 2 i8 > "%LOGS%\pa_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 16384 32 32 128 8 2 2 > "%LOGS%\gdn_prefill_S16384.log" 2>&1
echo === prefill_S16384 done %time% >> "%LOGS%\_run.log"

REM --- S=32768 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 5120 64 4 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2048 64 8 1 1 u4 32 > "%LOGS%\fc_o_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 8192 64 4 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 4096 64 8 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2048 64 8 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 32768 2048 512 256 8 64 2 1 1 64 0 > "%LOGS%\moe_routed_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 512 64 8 1 1 u4 32 > "%LOGS%\shared_gate_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048 512 64 8 1 1 u4 32 > "%LOGS%\shared_up_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 512 2048 64 8 1 1 u4 32 > "%LOGS%\shared_down_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 32768 0 3 1 2 i8 > "%LOGS%\pa_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 32768 32 32 128 8 2 2 > "%LOGS%\gdn_prefill_S32768.log" 2>&1
echo === prefill_S32768 done %time% >> "%LOGS%\_run.log"

REM --- S=65536 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 5120 64 2 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2048 64 4 1 1 u4 32 > "%LOGS%\fc_o_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 8192 64 2 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 4096 64 4 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2048 64 4 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 65536 2048 512 256 8 64 2 1 1 64 0 > "%LOGS%\moe_routed_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 512 64 4 1 1 u4 32 > "%LOGS%\shared_gate_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048 512 64 4 1 1 u4 32 > "%LOGS%\shared_up_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 512 2048 64 4 1 1 u4 32 > "%LOGS%\shared_down_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 65536 0 2 1 1 i8 > "%LOGS%\pa_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 65536 32 32 128 8 2 2 > "%LOGS%\gdn_prefill_S65536.log" 2>&1
echo === prefill_S65536 done %time% >> "%LOGS%\_run.log"

REM --- S=131072 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 5120 64 1 1 1 u4 32 > "%LOGS%\fc_qkv_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2048 64 2 1 1 u4 32 > "%LOGS%\fc_o_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 8192 64 1 1 1 u4 32 > "%LOGS%\linattn_qkv_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 4096 64 2 1 1 u4 32 > "%LOGS%\linattn_z_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2048 64 2 1 1 u4 32 > "%LOGS%\linattn_out_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 64 1 1 1 64 0 > "%LOGS%\moe_routed_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 512 64 2 1 1 u4 32 > "%LOGS%\shared_gate_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048 512 64 2 1 1 u4 32 > "%LOGS%\shared_up_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 512 2048 64 2 1 1 u4 32 > "%LOGS%\shared_down_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 131072 0 1 1 1 i8 > "%LOGS%\pa_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\gdn_bench.exe" 1 131072 32 32 128 8 2 2 > "%LOGS%\gdn_prefill_S131072.log" 2>&1
echo === prefill_S131072 done %time% >> "%LOGS%\_run.log"

echo === ALL DONE %date% %time% >> "%LOGS%\_run.log"
echo All INT4 g=64 v3 full benchmarks complete.
