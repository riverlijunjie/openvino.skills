@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  diffusion_gemma (google/diffusiongemma-26B-A4B-it) roofline sweep — PTL 12Xe
REM  Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM ========================================================================
REM  Block-diffusion model. text_config is architecturally identical to
REM  gemma4-26B-A4B-it, but the runtime workload differs:
REM    * ENCODER  = autoregressive over the prompt (causal/sliding) -> builds a
REM                 read-only KV cache.  This is the "prefill" phase.
REM    * DECODER  = block diffusion over a fixed canvas (canvas_length=256),
REM                 BIDIRECTIONAL self-attention + cross-attention to the encoder
REM                 KV cache. Every denoising step is M=256 (NOT M=1).
REM
REM  Architecture (config.json text_config):
REM    hidden=2816, layers=30, hidden_act=gelu_pytorch_tanh (GEGLU)
REM    vocab=262144, tie_word_embeddings=true
REM    Per layer: dense GEGLU MLP (gate/up 2816->2112, down 2112->2816)
REM               + MoE (NE=128, TK=8, I=704; gate_up 2816->1408, down 704->2816)
REM               outputs summed.
REM
REM    Sliding attention (25 layers): NH=16 NKV=8 HD=256
REM      QKV fused: 2816 -> 8192 (Q=4096 + K=2048 + V=2048)
REM      O:         4096 -> 2816 ; sliding_window=1024
REM    Full attention (5 layers): NH=16 NKV=2 HD=512, V reuses K (attention_k_eq_v)
REM      QK fused:  2816 -> 9216 (Q=8192 + K=1024, V=K)
REM      O:         8192 -> 2816
REM
REM    LM_Head: 2816 -> 262144 (INT8 g=128) applied over decoder canvas (M=256)
REM
REM  User decisions: KV=int8, SDPA implementation (sdpa_bench), canvas M=256
REM  decode + M=1 reference. Prompt/context sweep: 1024 2048 4096 8192.
REM ========================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\diffusion_gemma\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM =====================================================================
REM  MoE  (H=2816, I=704, NE=128, TK=8, g=64)
REM    NOTE: g=64 (not 128) because the down-proj reduction dim I=704 is not
REM    divisible by 128 (704/128=5.5). 704/64=11. Matches gemma4-26B-A4B ref.
REM    decode M1 reference, canvas M256, prefill S-sweep
REM =====================================================================
call :do moe_decode_M1          "%BUILD%\moe_bench.exe" 1 1     2816 704 128 8 64  500 50 4 64
call :do moe_canvas_M256        "%BUILD%\moe_bench.exe" 1 256   2816 704 128 8 64   80 10 4 64
call :do moe_prefill_S1024      "%BUILD%\moe_bench.exe" 1 1024  2816 704 128 8 64   20  5 2 64
call :do moe_prefill_S2048      "%BUILD%\moe_bench.exe" 1 2048  2816 704 128 8 64   15  3 2 64
call :do moe_prefill_S4096      "%BUILD%\moe_bench.exe" 1 4096  2816 704 128 8 64   10  2 2 64
call :do moe_prefill_S8192      "%BUILD%\moe_bench.exe" 1 8192  2816 704 128 8 64    8  2 2 64

REM =====================================================================
REM  Dense MLP FC (GEGLU): gate 2816->2112, up 2816->2112, down 2112->2816
REM =====================================================================
call :do fc_gate_dense_decode_M1   "%BUILD%\fc_bench.exe" 1   2816 2112 128 15000 500 8 u4 64
call :do fc_up_dense_decode_M1     "%BUILD%\fc_bench.exe" 1   2816 2112 128 15000 500 8 u4 64
call :do fc_down_dense_decode_M1   "%BUILD%\fc_bench.exe" 1   2112 2816 64  15000 500 8 u4 64
call :do fc_gate_dense_canvas_M256 "%BUILD%\fc_bench.exe" 256 2816 2112 128  1500 100 8 u4 64
call :do fc_up_dense_canvas_M256   "%BUILD%\fc_bench.exe" 256 2816 2112 128  1500 100 8 u4 64
call :do fc_down_dense_canvas_M256 "%BUILD%\fc_bench.exe" 256 2112 2816 64   1500 100 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_gate_dense_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2816 2112 128 20 5 4 u4 64
  call :do fc_up_dense_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 2816 2112 128 20 5 4 u4 64
  call :do fc_down_dense_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2112 2816 64  20 5 4 u4 64
)

REM =====================================================================
REM  Attention FC — Sliding (25 layers): QKV 2816->8192, O 4096->2816
REM =====================================================================
call :do fc_qkv_sliding_decode_M1    "%BUILD%\fc_bench.exe" 1   2816 8192 128 5000 200 8 u4 64
call :do fc_o_sliding_decode_M1      "%BUILD%\fc_bench.exe" 1   4096 2816 128 8000 300 8 u4 64
call :do fc_qkv_sliding_canvas_M256  "%BUILD%\fc_bench.exe" 256 2816 8192 128  800 100 8 u4 64
call :do fc_o_sliding_canvas_M256    "%BUILD%\fc_bench.exe" 256 4096 2816 128 1000 100 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_sliding_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2816 8192 128 20 5 4 u4 64
  call :do fc_o_sliding_prefill_S%%S   "%BUILD%\fc_bench.exe" %%S 4096 2816 128 20 5 4 u4 64
)

REM =====================================================================
REM  Attention FC — Full (5 layers): QK 2816->9216, O 8192->2816
REM =====================================================================
call :do fc_qk_full_decode_M1     "%BUILD%\fc_bench.exe" 1   2816 9216 128 5000 200 8 u4 64
call :do fc_o_full_decode_M1      "%BUILD%\fc_bench.exe" 1   8192 2816 128 5000 200 8 u4 64
call :do fc_qk_full_canvas_M256   "%BUILD%\fc_bench.exe" 256 2816 9216 128  800 100 8 u4 64
call :do fc_o_full_canvas_M256    "%BUILD%\fc_bench.exe" 256 8192 2816 128  800 100 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_qk_full_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2816 9216 128 20 5 4 u4 64
  call :do fc_o_full_prefill_S%%S  "%BUILD%\fc_bench.exe" %%S 8192 2816 128 20 5 4 u4 64
)

REM =====================================================================
REM  LM_Head: 2816 -> 262144 (INT8 g=128)
REM    decoder applies it over the whole canvas -> M=256 (primary)
REM    M=1 reference also captured
REM =====================================================================
call :do lm_head_decode_M1      "%BUILD%\fc_bench.exe" 1   2816 262144 128 200 20 4 u8 64
call :do lm_head_canvas_M256    "%BUILD%\fc_bench.exe" 256 2816 262144 128 100 10 4 u8 64

REM =====================================================================
REM  SDPA — Sliding (NH=16, NKV=8, HD=256), KV capped at sliding_window=1024
REM    encoder prefill: causal=1, S=1024 (window match)
REM    canvas decode:   bidirectional (causal=0), Sq=256, Skv=1024+256=1280
REM    M=1 reference:   bidirectional (causal=0), Sq=1, Skv=1024
REM =====================================================================
set SDPA_NH=16
set SDPA_NKV=8
set SDPA_HD=256
call :do sdpa_sliding_prefill_S1024 "%BUILD%\sdpa_bench.exe" prefill 1024 1024 100 10 4 1
call :do sdpa_sliding_canvas_M256   "%BUILD%\sdpa_bench.exe" prefill 256  1280 400 40 4 0
call :do sdpa_sliding_decode_M1     "%BUILD%\sdpa_bench.exe" decode  1    1024 3000 200 4 0

REM =====================================================================
REM  SDPA — Full (NH=16, NKV=2, HD=512)
REM    encoder prefill: causal=1, S-sweep
REM    canvas decode:   bidirectional (causal=0), Sq=256, Skv=context+256
REM    M=1 reference:   bidirectional (causal=0), Sq=1, Skv=context
REM =====================================================================
set SDPA_NH=16
set SDPA_NKV=2
set SDPA_HD=512
for %%S in (1024 2048 4096 8192) do (
  call :do sdpa_full_prefill_S%%S "%BUILD%\sdpa_bench.exe" prefill %%S %%S 30 5 2 1
)
call :do sdpa_full_canvas_ctx1024 "%BUILD%\sdpa_bench.exe" prefill 256 1280 400 40 4 0
call :do sdpa_full_canvas_ctx2048 "%BUILD%\sdpa_bench.exe" prefill 256 2304 300 30 4 0
call :do sdpa_full_canvas_ctx4096 "%BUILD%\sdpa_bench.exe" prefill 256 4352 200 20 4 0
call :do sdpa_full_canvas_ctx8192 "%BUILD%\sdpa_bench.exe" prefill 256 8448 150 15 4 0
call :do sdpa_full_decode_ctx1024 "%BUILD%\sdpa_bench.exe" decode 1 1024 3000 200 4 0
call :do sdpa_full_decode_ctx2048 "%BUILD%\sdpa_bench.exe" decode 1 2048 2500 150 4 0
call :do sdpa_full_decode_ctx4096 "%BUILD%\sdpa_bench.exe" decode 1 4096 2000 100 4 0
call :do sdpa_full_decode_ctx8192 "%BUILD%\sdpa_bench.exe" decode 1 8192 1500 100 4 0

REM =====================================================================
REM  Small ops — RMSNorm / RMSNorm3d (q/k norm) / RoPE / Add
REM    decode M1, canvas M256, prefill S-sweep
REM =====================================================================
call :do so_rmsnorm_h2816_decode        "%BUILD%\small_ops_bench.exe" rmsnorm   1   2816   --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_sliding_decode  "%BUILD%\small_ops_bench.exe" rmsnorm3d 1   16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_sliding_decode  "%BUILD%\small_ops_bench.exe" rmsnorm3d 1   8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_q_full_decode     "%BUILD%\small_ops_bench.exe" rmsnorm3d 1   16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_k_full_decode     "%BUILD%\small_ops_bench.exe" rmsnorm3d 1   2  512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_sliding_decode       "%BUILD%\small_ops_bench.exe" rope      1   16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_sliding_decode       "%BUILD%\small_ops_bench.exe" rope      1   8  256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_full_decode          "%BUILD%\small_ops_bench.exe" rope      1   16 512 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_full_decode          "%BUILD%\small_ops_bench.exe" rope      1   2  512 --iters 30000 --warmup 300 --bufs 8
call :do so_add_h2816_decode            "%BUILD%\small_ops_bench.exe" add       1   2816   --iters 30000 --warmup 300 --bufs 8

call :do so_rmsnorm_h2816_canvas        "%BUILD%\small_ops_bench.exe" rmsnorm   256 2816   --iters 4000 --warmup 100 --bufs 8
call :do so_rmsnorm3d_q_sliding_canvas  "%BUILD%\small_ops_bench.exe" rmsnorm3d 256 16 256 --iters 4000 --warmup 100 --bufs 8
call :do so_rmsnorm3d_k_sliding_canvas  "%BUILD%\small_ops_bench.exe" rmsnorm3d 256 8  256 --iters 4000 --warmup 100 --bufs 8
call :do so_rmsnorm3d_q_full_canvas     "%BUILD%\small_ops_bench.exe" rmsnorm3d 256 16 512 --iters 4000 --warmup 100 --bufs 8
call :do so_rmsnorm3d_k_full_canvas     "%BUILD%\small_ops_bench.exe" rmsnorm3d 256 2  512 --iters 4000 --warmup 100 --bufs 8
call :do so_rope_q_sliding_canvas       "%BUILD%\small_ops_bench.exe" rope      256 16 256 --iters 4000 --warmup 100 --bufs 8
call :do so_rope_k_sliding_canvas       "%BUILD%\small_ops_bench.exe" rope      256 8  256 --iters 4000 --warmup 100 --bufs 8
call :do so_add_h2816_canvas            "%BUILD%\small_ops_bench.exe" add       256 2816   --iters 4000 --warmup 100 --bufs 8

for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2816_prefill_S%%S       "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 2816   --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_q_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 256 --iters 300 --warmup 30 --bufs 4
  call :do so_rmsnorm3d_k_sliding_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  256 --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_sliding_prefill_S%%S      "%BUILD%\small_ops_bench.exe" rope      %%S 16 256 --iters 300 --warmup 30 --bufs 4
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
