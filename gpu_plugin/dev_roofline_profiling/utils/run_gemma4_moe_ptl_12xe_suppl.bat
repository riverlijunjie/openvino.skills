@echo off
REM ========================================================================
REM  Gemma4 MoE — Supplementary benchmarks (FC, small ops, missing MoE)
REM  PTL 12Xe: Local_Admin@10.239.132.229
REM ========================================================================
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_moe\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"

REM =====================================================================
REM  MoE ALL sizes — re-run with g=64 (704 % 128 != 0, 704 % 64 == 0)
REM =====================================================================
echo [%date% %time%] moe_decode_M1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1 2816 704 128 8 64 500 50 4 64 > "%LOGS%\moe_decode_M1.log" 2>&1
echo [%date% %time%] moe_prefill_S1024
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1024 2816 704 128 8 64 20 3 2 64 > "%LOGS%\moe_prefill_S1024.log" 2>&1
echo [%date% %time%] moe_prefill_S2048
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 2048 2816 704 128 8 64 15 3 2 64 > "%LOGS%\moe_prefill_S2048.log" 2>&1
echo [%date% %time%] moe_prefill_S4096
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 4096 2816 704 128 8 64 10 2 2 64 > "%LOGS%\moe_prefill_S4096.log" 2>&1
echo [%date% %time%] moe_prefill_S8192
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 8192 2816 704 128 8 64 5 1 1 64 > "%LOGS%\moe_prefill_S8192.log" 2>&1
echo [%date% %time%] moe_prefill_S16384
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 16384 2816 704 128 8 64 5 1 1 64 > "%LOGS%\moe_prefill_S16384.log" 2>&1
echo [%date% %time%] moe_prefill_S32768
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 32768 2816 704 128 8 64 3 1 1 64 > "%LOGS%\moe_prefill_S32768.log" 2>&1
echo [%date% %time%] moe_prefill_S65536
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 65536 2816 704 128 8 64 2 1 1 64 > "%LOGS%\moe_prefill_S65536.log" 2>&1
echo [%date% %time%] moe_prefill_S131072
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 131072 2816 704 128 8 64 2 1 1 64 > "%LOGS%\moe_prefill_S131072.log" 2>&1

REM =====================================================================
REM  FC decode (M=1) — All projections
REM =====================================================================
echo [%date% %time%] fc_qkv_sliding_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 8192 128 5000 200 8 u4 64 > "%LOGS%\fc_qkv_sliding_decode_M1.log" 2>&1
echo [%date% %time%] fc_o_sliding_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096 2816 128 8000 300 8 u4 64 > "%LOGS%\fc_o_sliding_decode_M1.log" 2>&1
echo [%date% %time%] fc_qk_full_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 9216 128 5000 200 8 u4 64 > "%LOGS%\fc_qk_full_decode_M1.log" 2>&1
echo [%date% %time%] fc_o_full_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 8192 2816 128 5000 200 8 u4 64 > "%LOGS%\fc_o_full_decode_M1.log" 2>&1
echo [%date% %time%] fc_gate_dense_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 2112 128 15000 500 8 u4 64 > "%LOGS%\fc_gate_dense_decode_M1.log" 2>&1
echo [%date% %time%] fc_up_dense_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 2112 128 15000 500 8 u4 64 > "%LOGS%\fc_up_dense_decode_M1.log" 2>&1
echo [%date% %time%] fc_down_dense_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2112 2816 128 15000 500 8 u4 64 > "%LOGS%\fc_down_dense_decode_M1.log" 2>&1
echo [%date% %time%] lm_head_decode_M1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 262144 128 200 20 4 u8 64 > "%LOGS%\lm_head_decode_M1.log" 2>&1

REM =====================================================================
REM  FC prefill (M=S) — Sliding attention
REM =====================================================================
echo [%date% %time%] fc_qkv_sliding_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2816 8192 128 20 5 4 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2816 8192 128 20 5 4 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2816 8192 128 20 5 4 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2816 8192 128 15 5 2 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2816 8192 128 10 3 2 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2816 8192 128 5 2 1 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2816 8192 128 3 1 1 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_qkv_sliding_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2816 8192 128 2 1 1 u4 64 > "%LOGS%\fc_qkv_sliding_prefill_S131072.log" 2>&1

echo [%date% %time%] fc_o_sliding_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 4096 2816 128 20 5 4 u4 64 > "%LOGS%\fc_o_sliding_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 4096 2816 128 20 5 4 u4 64 > "%LOGS%\fc_o_sliding_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 4096 2816 128 20 5 4 u4 64 > "%LOGS%\fc_o_sliding_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 2816 128 15 5 2 u4 64 > "%LOGS%\fc_o_sliding_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 4096 2816 128 10 3 2 u4 64 > "%LOGS%\fc_o_sliding_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 4096 2816 128 5 2 1 u4 64 > "%LOGS%\fc_o_sliding_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 4096 2816 128 3 1 1 u4 64 > "%LOGS%\fc_o_sliding_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_o_sliding_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 4096 2816 128 2 1 1 u4 64 > "%LOGS%\fc_o_sliding_prefill_S131072.log" 2>&1

REM =====================================================================
REM  FC prefill (M=S) — Full attention
REM =====================================================================
echo [%date% %time%] fc_qk_full_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2816 9216 128 20 5 4 u4 64 > "%LOGS%\fc_qk_full_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2816 9216 128 20 5 4 u4 64 > "%LOGS%\fc_qk_full_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2816 9216 128 15 5 2 u4 64 > "%LOGS%\fc_qk_full_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2816 9216 128 10 3 2 u4 64 > "%LOGS%\fc_qk_full_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2816 9216 128 5 2 1 u4 64 > "%LOGS%\fc_qk_full_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2816 9216 128 3 1 1 u4 64 > "%LOGS%\fc_qk_full_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2816 9216 128 2 1 1 u4 64 > "%LOGS%\fc_qk_full_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_qk_full_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2816 9216 128 2 1 1 u4 64 > "%LOGS%\fc_qk_full_prefill_S131072.log" 2>&1

echo [%date% %time%] fc_o_full_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 8192 2816 128 20 5 4 u4 64 > "%LOGS%\fc_o_full_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 8192 2816 128 20 5 4 u4 64 > "%LOGS%\fc_o_full_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 8192 2816 128 15 5 2 u4 64 > "%LOGS%\fc_o_full_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 8192 2816 128 10 3 2 u4 64 > "%LOGS%\fc_o_full_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 8192 2816 128 5 2 1 u4 64 > "%LOGS%\fc_o_full_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 8192 2816 128 3 1 1 u4 64 > "%LOGS%\fc_o_full_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 8192 2816 128 2 1 1 u4 64 > "%LOGS%\fc_o_full_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_o_full_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 8192 2816 128 2 1 1 u4 64 > "%LOGS%\fc_o_full_prefill_S131072.log" 2>&1

REM =====================================================================
REM  FC prefill (M=S) — Dense MLP (gate/up/down)
REM =====================================================================
echo [%date% %time%] fc_gate_dense_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2816 2112 128 30 5 4 u4 64 > "%LOGS%\fc_gate_dense_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2816 2112 128 20 5 4 u4 64 > "%LOGS%\fc_gate_dense_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2816 2112 128 15 5 2 u4 64 > "%LOGS%\fc_gate_dense_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2816 2112 128 10 3 2 u4 64 > "%LOGS%\fc_gate_dense_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2816 2112 128 5 2 1 u4 64 > "%LOGS%\fc_gate_dense_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2816 2112 128 3 1 1 u4 64 > "%LOGS%\fc_gate_dense_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2816 2112 128 2 1 1 u4 64 > "%LOGS%\fc_gate_dense_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_gate_dense_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2816 2112 128 2 1 1 u4 64 > "%LOGS%\fc_gate_dense_prefill_S131072.log" 2>&1

echo [%date% %time%] fc_up_dense_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2816 2112 128 30 5 4 u4 64 > "%LOGS%\fc_up_dense_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2816 2112 128 20 5 4 u4 64 > "%LOGS%\fc_up_dense_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2816 2112 128 15 5 2 u4 64 > "%LOGS%\fc_up_dense_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2816 2112 128 10 3 2 u4 64 > "%LOGS%\fc_up_dense_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2816 2112 128 5 2 1 u4 64 > "%LOGS%\fc_up_dense_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2816 2112 128 3 1 1 u4 64 > "%LOGS%\fc_up_dense_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2816 2112 128 2 1 1 u4 64 > "%LOGS%\fc_up_dense_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_up_dense_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2816 2112 128 2 1 1 u4 64 > "%LOGS%\fc_up_dense_prefill_S131072.log" 2>&1

echo [%date% %time%] fc_down_dense_prefill_S1024
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2112 2816 128 30 5 4 u4 64 > "%LOGS%\fc_down_dense_prefill_S1024.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S2048
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2112 2816 128 20 5 4 u4 64 > "%LOGS%\fc_down_dense_prefill_S2048.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S4096
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2112 2816 128 15 5 2 u4 64 > "%LOGS%\fc_down_dense_prefill_S4096.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S8192
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2112 2816 128 10 3 2 u4 64 > "%LOGS%\fc_down_dense_prefill_S8192.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S16384
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2112 2816 128 5 2 1 u4 64 > "%LOGS%\fc_down_dense_prefill_S16384.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S32768
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2112 2816 128 3 1 1 u4 64 > "%LOGS%\fc_down_dense_prefill_S32768.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S65536
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2112 2816 128 2 1 1 u4 64 > "%LOGS%\fc_down_dense_prefill_S65536.log" 2>&1
echo [%date% %time%] fc_down_dense_prefill_S131072
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2112 2816 128 2 1 1 u4 64 > "%LOGS%\fc_down_dense_prefill_S131072.log" 2>&1

REM  LM_Head prefill (single token)
echo [%date% %time%] lm_head_prefill
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2816 262144 128 200 20 4 u8 64 > "%LOGS%\lm_head_prefill.log" 2>&1

REM =====================================================================
REM  Small ops — Decode (M=1)
REM =====================================================================
echo [%date% %time%] so_rmsnorm_h2816_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 1 2816 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm_h2816_decode.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_q_sliding_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_q_sliding_decode.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_k_sliding_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 8 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_k_sliding_decode.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_q_full_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_q_full_decode.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_k_full_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 2 512 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rmsnorm3d_k_full_decode.log" 2>&1
echo [%date% %time%] so_rope_q_sliding_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1 16 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_q_sliding_decode.log" 2>&1
echo [%date% %time%] so_rope_k_sliding_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1 8 256 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_k_sliding_decode.log" 2>&1
echo [%date% %time%] so_rope_q_full_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1 16 512 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_q_full_decode.log" 2>&1
echo [%date% %time%] so_rope_k_full_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1 2 512 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_rope_k_full_decode.log" 2>&1
echo [%date% %time%] so_add_h2816_decode
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add 1 2816 --iters 30000 --warmup 300 --bufs 8 > "%LOGS%\so_add_h2816_decode.log" 2>&1

REM =====================================================================
REM  Small ops — Prefill (M=S)
REM =====================================================================
echo [%date% %time%] so_rmsnorm_h2816_prefill_S1024
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 1024 2816 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2816_prefill_S1024.log" 2>&1
echo [%date% %time%] so_rmsnorm_h2816_prefill_S2048
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 2048 2816 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2816_prefill_S2048.log" 2>&1
echo [%date% %time%] so_rmsnorm_h2816_prefill_S4096
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 4096 2816 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm_h2816_prefill_S4096.log" 2>&1
echo [%date% %time%] so_rmsnorm_h2816_prefill_S8192
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm 8192 2816 --iters 200 --warmup 20 --bufs 4 > "%LOGS%\so_rmsnorm_h2816_prefill_S8192.log" 2>&1

echo [%date% %time%] so_rmsnorm3d_q_sliding_prefill_S1024
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1024 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm3d_q_sliding_prefill_S1024.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_q_sliding_prefill_S2048
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 2048 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rmsnorm3d_q_sliding_prefill_S2048.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_q_sliding_prefill_S4096
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 4096 16 256 --iters 200 --warmup 20 --bufs 4 > "%LOGS%\so_rmsnorm3d_q_sliding_prefill_S4096.log" 2>&1
echo [%date% %time%] so_rmsnorm3d_q_sliding_prefill_S8192
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 8192 16 256 --iters 100 --warmup 10 --bufs 4 > "%LOGS%\so_rmsnorm3d_q_sliding_prefill_S8192.log" 2>&1

echo [%date% %time%] so_rope_q_sliding_prefill_S1024
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 1024 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_sliding_prefill_S1024.log" 2>&1
echo [%date% %time%] so_rope_q_sliding_prefill_S2048
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 2048 16 256 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_rope_q_sliding_prefill_S2048.log" 2>&1
echo [%date% %time%] so_rope_q_sliding_prefill_S4096
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 4096 16 256 --iters 200 --warmup 20 --bufs 4 > "%LOGS%\so_rope_q_sliding_prefill_S4096.log" 2>&1
echo [%date% %time%] so_rope_q_sliding_prefill_S8192
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope 8192 16 256 --iters 100 --warmup 10 --bufs 4 > "%LOGS%\so_rope_q_sliding_prefill_S8192.log" 2>&1

echo [%date% %time%] so_add_h2816_prefill_S1024
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add 1024 2816 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_add_h2816_prefill_S1024.log" 2>&1
echo [%date% %time%] so_add_h2816_prefill_S2048
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add 2048 2816 --iters 300 --warmup 30 --bufs 4 > "%LOGS%\so_add_h2816_prefill_S2048.log" 2>&1
echo [%date% %time%] so_add_h2816_prefill_S4096
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add 4096 2816 --iters 200 --warmup 20 --bufs 4 > "%LOGS%\so_add_h2816_prefill_S4096.log" 2>&1
echo [%date% %time%] so_add_h2816_prefill_S8192
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add 8192 2816 --iters 100 --warmup 10 --bufs 4 > "%LOGS%\so_add_h2816_prefill_S8192.log" 2>&1

echo [%date% %time%] ALL DONE
