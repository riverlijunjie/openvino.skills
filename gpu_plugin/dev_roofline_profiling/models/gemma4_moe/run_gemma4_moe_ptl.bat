@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Gemma4-26B-A4B MoE roofline sweep on PTL 12Xe (iGPU, 2400 MHz, 110 GB/s, FP16 XMX 58.98 TFLOPS).
REM Architecture:
REM   - 30 layers: 25 sliding_attention + 5 full_attention (5:1 pattern)
REM   - Each layer: dense MLP (H=2816, I=2112) + MoE (H=2816, I=704, NE=128, TK=8) in parallel
REM   - Sliding attn: NH=16 NKV=8 HD=256, sliding_window=1024
REM   - Full attn: NH=16 NKV=2 HD=512, k_eq_v (V=K, no V proj)
REM   - Dense MLP: gate(2112)+up(2112)+down(2816), GEGLU activation
REM   - MoE: 128 experts, top_k=8, I=704, GEGLU activation, no shared expert
REM   - LM_Head: vocab=262144, INT8 g=128
REM   - MoE uses g=64 (I=704 not divisible by 128)
REM   - Dense MLP down uses g=64 (K=2112 not divisible by 128)

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_moe\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ========== MoE fused (NO shared expert) ==========
REM moe_bench <B> <S> <H> <I> <NE> <TK> [g] [iters] [warmup] [num_bufs] [flush_mb] [shared_I]
REM Note: g=64 because I=704 not divisible by 128 (704/64=11)
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2816 704 128 8 64 500 50 4 64 0
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2816 704 128 8 64  20  5 2 64 0
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2816 704 128 8 64  15  3 2 64 0
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2816 704 128 8 64  10  2 2 64 0
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2816 704 128 8 64   8  2 2 64 0

REM ========== Dense MLP: gate + up (same shape, test once) ==========
REM fc_bench <M> <K> <N> [g] [iters] [warmup] [num_bufs] [precision] [flush_mb]
REM Note: N=2112 not divisible by 128 -> use g=64 (2112/64=33)
call :do fc_mlp_gate_decode_M1        "%BUILD%\fc_bench.exe" 1    2816   2112  64 15000 300 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_mlp_gate_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2816   2112  64 30 5 4 u4 64
)

REM ========== Dense MLP: down (K=2112, g=64) ==========
call :do fc_mlp_down_decode_M1        "%BUILD%\fc_bench.exe" 1    2112   2816  64 15000 300 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_mlp_down_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2112   2816  64 30 5 4 u4 64
)

REM ========== Sliding attention FC: QKV fused (Q=4096 + K=2048 + V=2048 = 8192) ==========
call :do fc_qkv_sliding_decode_M1     "%BUILD%\fc_bench.exe" 1    2816   8192 128 5000 200 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_sliding_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2816   8192 128 30 5 4 u4 64
)

REM ========== Sliding attention FC: O proj (4096 -> 2816) ==========
call :do fc_o_sliding_decode_M1       "%BUILD%\fc_bench.exe" 1    4096   2816 128 8000 200 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_o_sliding_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 4096   2816 128 30 5 4 u4 64
)

REM ========== Full attention FC: QKV fused (Q=8192 + K=1024, no V; k_eq_v) = 9216 ==========
call :do fc_qkv_full_decode_M1        "%BUILD%\fc_bench.exe" 1    2816   9216 128 5000 200 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_full_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2816   9216 128 30 5 4 u4 64
)

REM ========== Full attention FC: O proj (8192 -> 2816) ==========
call :do fc_o_full_decode_M1          "%BUILD%\fc_bench.exe" 1    8192   2816 128 5000 200 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_o_full_prefill_S%%S     "%BUILD%\fc_bench.exe" %%S 8192   2816 128 30 5 4 u4 64
)

REM ========== LM_Head (INT8 g=128) ==========
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1    2816 262144 128  200  20 4 u8 64

REM ========== PA: sliding attention (NH=16 NKV=8 HD=256, SW=1024) ==========
set PA_NH=16
set PA_NKV=8
set PA_HD=256
set PA_SW=1024
REM Sliding window = 1024: decode always has effective kv=1024
call :do pa_sliding_decode_kv1024     "%BUILD%\pa_bench.exe" decode 1 1024 8000 200 4 i8
REM Prefill: test various S values (sliding attention uses window internally)
for %%S in (1024 2048 4096 8192) do (
  call :do pa_sliding_prefill_S%%S    "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)

REM ========== PA: full attention (NH=16 NKV=2 HD=512, no sliding window) ==========
set PA_NH=16
set PA_NKV=2
set PA_HD=512
set PA_SW=0
REM Decode: test various kv lengths (full attention, no sliding window)
for %%K in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_full_decode_kv%%K       "%BUILD%\pa_bench.exe" decode 1 %%K 8000 200 4 i8
)
REM Prefill: test various S values
for %%S in (1024 2048 4096 8192) do (
  call :do pa_full_prefill_S%%S       "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)

REM ========== Small ops (H=2816) ==========
REM Gemma4 MoE layer has 7 RMSNorms: input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln,
REM   post_ffn_ln_1 (dense MLP), post_ffn_ln_2 (MoE), pre_ffn_ln_2 (MoE)
call :do so_rmsnorm_h2816_decode      "%BUILD%\small_ops_bench.exe" rmsnorm   1 2816 --iters 30000 --warmup 300 --bufs 8
call :do so_add_h2816_decode          "%BUILD%\small_ops_bench.exe" add       1 2816 --iters 30000 --warmup 300 --bufs 8

REM Sliding attention norms/rope (NH=16, NKV=8, HD=256)
call :do so_rmsnorm3d_qnorm_sliding   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_sliding   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  8 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_vnorm_sliding   "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  8 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_sliding_decode     "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_sliding_decode     "%BUILD%\small_ops_bench.exe" rope      1  8 256 --iters 30000 --warmup 300 --bufs 8

REM Full attention norms/rope (NH=16, NKV=2, HD=512)
call :do so_rmsnorm3d_qnorm_full      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_full      "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  2 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_full_decode        "%BUILD%\small_ops_bench.exe" rope      1 16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode        "%BUILD%\small_ops_bench.exe" rope      1  2 512 --iters 30000 --warmup 300 --bufs 8

REM Prefill small ops
for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2816_prefill_S%%S   "%BUILD%\small_ops_bench.exe" rmsnorm %%S 2816   --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S  "%BUILD%\small_ops_bench.exe" rope    %%S 16 256 --iters 300 --warmup 30 --bufs 4
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
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
echo Running !TAG! ...
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
