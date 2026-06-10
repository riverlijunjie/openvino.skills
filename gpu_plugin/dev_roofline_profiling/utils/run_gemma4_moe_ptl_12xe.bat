@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  Gemma4 MoE (google/gemma-4-26B-A4B-it) roofline sweep — PTL 12Xe
REM  Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM ========================================================================
REM Architecture (config.json text_config):
REM   hidden=2816, layers=30, hidden_act=gelu_pytorch_tanh (GEGLU)
REM   vocab=262144, tie_word_embeddings=true, enable_moe_block=true
REM
REM   Sliding attention (25 layers): NH=16 NKV=8 HD=256
REM     QKV fused: 2816 -> 8192 (Q=4096 + K=2048 + V=2048)
REM     O:         4096 -> 2816
REM
REM   Full attention (5 layers): NH=16 NKV=2 HD=512, attention_k_eq_v=true
REM     QK fused:  2816 -> 9216 (Q=8192 + K=1024, V reuses K)
REM     O:         8192 -> 2816
REM
REM   Dense MLP (30 layers, GEGLU):
REM     gate: 2816 -> 2112,  up: 2816 -> 2112,  down: 2112 -> 2816
REM
REM   MoE (30 layers): NE=128, TK=8, I=704
REM     gate_up fused: 2816 -> 1408 per expert, down: 704 -> 2816 per expert
REM     NOTE: moe_bench uses Swish as proxy for GEGLU (GPU MoE fusion only supports Swish)
REM
REM   LM_Head: 2816 -> 262144 (INT8 g=128, single-token only)
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_moe\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM =====================================================================
REM  MoE decode (M=1, H=2816, I=704, NE=128, TK=8, g=128)
REM  NOTE: Using Swish activation as proxy (GPU MoE fusion doesn't support GELU yet)
REM =====================================================================
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2816 704 128 8 128  500 50 4 64

REM =====================================================================
REM  MoE prefill (M=S, H=2816, I=704, NE=128, TK=8, g=128)
REM =====================================================================
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2816 704 128 8 128   20  5 2 64
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2816 704 128 8 128   15  3 2 64
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2816 704 128 8 128   10  2 2 64
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2816 704 128 8 128    8  2 2 64
call :do moe_prefill_S16384           "%BUILD%\moe_bench.exe" 1 16384 2816 704 128 8 128   5  1 1 64
call :do moe_prefill_S32768           "%BUILD%\moe_bench.exe" 1 32768 2816 704 128 8 128   3  1 1 64
call :do moe_prefill_S65536           "%BUILD%\moe_bench.exe" 1 65536 2816 704 128 8 128   2  1 1 64
call :do moe_prefill_S131072          "%BUILD%\moe_bench.exe" 1 131072 2816 704 128 8 128  2  1 1 64

REM =====================================================================
REM  FC decode (M=1) — Sliding attention (25 layers)
REM  QKV fused: 2816 -> 8192 (INT4 g=128)
REM  O:         4096 -> 2816 (INT4 g=128)
REM =====================================================================
call :do fc_qkv_sliding_decode_M1    "%BUILD%\fc_bench.exe" 1 2816  8192 128 5000 200 8 u4 64
call :do fc_o_sliding_decode_M1      "%BUILD%\fc_bench.exe" 1 4096  2816 128 8000 300 8 u4 64

REM =====================================================================
REM  FC decode (M=1) — Full attention (5 layers)
REM  QK fused:  2816 -> 9216 (INT4 g=128)  (Q=8192 + K=1024, V=K)
REM  O:         8192 -> 2816 (INT4 g=128)
REM =====================================================================
call :do fc_qk_full_decode_M1        "%BUILD%\fc_bench.exe" 1 2816  9216 128 5000 200 8 u4 64
call :do fc_o_full_decode_M1         "%BUILD%\fc_bench.exe" 1 8192  2816 128 5000 200 8 u4 64

REM =====================================================================
REM  FC decode (M=1) — Dense MLP (30 layers, GEGLU)
REM  gate: 2816 -> 2112,  up: 2816 -> 2112,  down: 2112 -> 2816  (INT4 g=128)
REM =====================================================================
call :do fc_gate_dense_decode_M1     "%BUILD%\fc_bench.exe" 1 2816  2112 128 15000 500 8 u4 64
call :do fc_up_dense_decode_M1       "%BUILD%\fc_bench.exe" 1 2816  2112 128 15000 500 8 u4 64
call :do fc_down_dense_decode_M1     "%BUILD%\fc_bench.exe" 1 2112  2816 128 15000 500 8 u4 64

REM =====================================================================
REM  LM_Head decode (M=1): 2816 -> 262144 (INT8 g=128)
REM =====================================================================
call :do lm_head_decode_M1           "%BUILD%\fc_bench.exe" 1 2816 262144 128  200  20 4 u8 64

REM =====================================================================
REM  FC prefill — Sliding attention (M=S)
REM =====================================================================
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do fc_qkv_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 2816 8192 128 20 5 4 u4 64
  call :do fc_o_sliding_prefill_S%%S    "%BUILD%\fc_bench.exe" %%S 4096 2816 128 20 5 4 u4 64
)

REM =====================================================================
REM  FC prefill — Full attention (M=S)
REM =====================================================================
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do fc_qk_full_prefill_S%%S      "%BUILD%\fc_bench.exe" %%S 2816 9216 128 20 5 4 u4 64
  call :do fc_o_full_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 8192 2816 128 20 5 4 u4 64
)

REM =====================================================================
REM  FC prefill — Dense MLP (M=S)
REM =====================================================================
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do fc_gate_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2816 2112 128 20 5 4 u4 64
  call :do fc_up_dense_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 2816 2112 128 20 5 4 u4 64
  call :do fc_down_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2112 2816 128 20 5 4 u4 64
)

REM  LM_Head prefill (always single-token)
call :do lm_head_prefill             "%BUILD%\fc_bench.exe" 1 2816 262144 128 200 20 4 u8 64

REM =====================================================================
REM  PA decode — Sliding attention (NH=16, NKV=8, HD=256)
REM  Sliding window = 1024, so effective KV = min(ctx, 1024)
REM  Only need KV=1024 (capped by sliding window for all ctx >= 1024)
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_decode_kv1024    "%BUILD%\pa_bench.exe" decode 1 1024 5000 200 4 i8 ocl

REM =====================================================================
REM  PA decode — Full attention (NH=16, NKV=2, HD=512)
REM  Full context, test at all KV lengths
REM =====================================================================
set PA_NH=16
set PA_NKV=2
set PA_HD=512
for %%K in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_full_decode_kv%%K      "%BUILD%\pa_bench.exe" decode 1 %%K 2000 100 4 i8 ocl
)

REM =====================================================================
REM  PA prefill — Sliding attention (NH=16, NKV=8, HD=256)
REM  Note: pa_bench uses full causal mask; for S>1024, overestimates sliding window work
REM  We profile at S=1024 (exact match with window), then scale linearly for larger S
REM =====================================================================
set PA_NH=16
set PA_NKV=8
set PA_HD=256
call :do pa_sliding_prefill_S1024    "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl

REM =====================================================================
REM  PA prefill — Full attention (NH=16, NKV=2, HD=512)
REM  Full causal attention, test at all S values
REM =====================================================================
set PA_NH=16
set PA_NKV=2
set PA_HD=512
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_full_prefill_S%%S      "%BUILD%\pa_bench.exe" prefill %%S 0 20 3 2 i8 ocl
)

REM =====================================================================
REM  Small ops — Decode (M=1)
REM =====================================================================
call :do so_rmsnorm_h2816_decode         "%BUILD%\small_ops_bench.exe" rmsnorm   1 2816   --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_sliding_decode   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_full_decode      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 2  512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_sliding_decode        "%BUILD%\small_ops_bench.exe" rope      1 8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode           "%BUILD%\small_ops_bench.exe" rope      1 2  512 --iters 30000 --warmup 300 --bufs 8
call :do so_add_h2816_decode             "%BUILD%\small_ops_bench.exe" add       1 2816   --iters 30000 --warmup 300 --bufs 8

REM =====================================================================
REM  Small ops — Prefill (M=S)
REM =====================================================================
for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2816_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 2816   --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_q_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 512 --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_k_full_prefill_S%%S    "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 2  512 --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 300 --warmup 30 --bufs 4
  call :do so_rope_k_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 8  256 --iters 300 --warmup 30 --bufs 4
  call :do so_add_h2816_prefill_S%%S           "%BUILD%\small_ops_bench.exe" add       %%S 2816   --iters 300 --warmup 30 --bufs 4
)

echo === END %date% %time% >> "%LOGS%\_index.txt"
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
