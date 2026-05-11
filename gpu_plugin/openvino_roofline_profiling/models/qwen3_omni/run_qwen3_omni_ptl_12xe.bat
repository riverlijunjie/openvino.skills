@echo off
REM qwen3_omni Thinker text decoder roofline sweep on PTL 12Xe Windows
REM Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU (B390), 2400 MHz, 110 GB/s
REM
REM Architecture (config.json thinker_config.text_config):
REM   hidden=2560, layers=36, GQA NH=32 NKV=8, head_dim=128,
REM   intermediate=9728, vocab=151936, tie_word_embeddings=true,
REM   hidden_act=silu (SwiGLU), rope_theta=1e6
REM
REM GQA shapes:
REM   QKV (fused) : 2560 -> 6144  (Q=4096 + K=1024 + V=1024)
REM   O           : 4096 -> 2560
REM   gate/up     : 2560 -> 9728
REM   down        : 9728 -> 2560
REM   lm_head     : 2560 -> 151936  (INT8, single-token)

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_omni\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

set PA_NH=32
set PA_NKV=8
set PA_HD=128

REM ============ PA decode (S_q=1, S_kv = ctx) ============
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 300 30 4 i8 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 2048 300 30 4 i8 ocl > "%LOGS%\pa_decode_kv2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 4 i8 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 4 i8 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM ============ PA prefill (S_q=S, S_kv=0) ============
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 2048 0  60 10 4 i8 ocl > "%LOGS%\pa_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 i8 ocl > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  25  5 2 i8 ocl > "%LOGS%\pa_prefill_S8192.log" 2>&1

REM ============ FC decode (M=1) - INT4 ============
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   6144 128 3000 150 8 > "%LOGS%\fc_decode_qkv.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096   2560 128 5000 200 8 > "%LOGS%\fc_decode_o.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   9728 128 2000 100 8 > "%LOGS%\fc_decode_gate.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   9728 128 2000 100 8 > "%LOGS%\fc_decode_up.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 9728   2560 128 2000 100 8 > "%LOGS%\fc_decode_down.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 151936 128  150  10 4 u8 > "%LOGS%\fc_decode_lm_head.log" 2>&1

REM ============ FC prefill (M=S) - INT4 ============
for %%S in (1024 2048 4096 8192) do (
  set IT=200
  if %%S==2048 set IT=120
  if %%S==4096 set IT=60
  if %%S==8192 set IT=30
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   6144 128 !IT! 10 4 > "%LOGS%\fc_prefill_qkv_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 4096   2560 128 !IT! 10 4 > "%LOGS%\fc_prefill_o_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   9728 128 !IT! 10 4 > "%LOGS%\fc_prefill_gate_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   9728 128 !IT! 10 4 > "%LOGS%\fc_prefill_up_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 9728   2560 128 !IT! 10 4 > "%LOGS%\fc_prefill_down_S%%S.log" 2>&1
  endlocal
)
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 151936 128 150 10 4 u8 > "%LOGS%\fc_prefill_lm_head.log" 2>&1

REM ============ Small ops decode (M=1) ============
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   1 2560   --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128 --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_q.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 8  128 --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_k.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 32 128 --iters 5000 --warmup 200 > "%LOGS%\small_decode_rope_q.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 8  128 --iters 5000 --warmup 200 > "%LOGS%\small_decode_rope_k.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" multiply  1 9728   --iters 5000 --warmup 200 > "%LOGS%\small_decode_multiply.log" 2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add       1 2560   --iters 5000 --warmup 200 > "%LOGS%\small_decode_add.log" 2>&1

REM ============ Small ops prefill ============
for %%S in (1024 2048 4096 8192) do (
  set ITS=1500
  if %%S==2048 set ITS=800
  if %%S==4096 set ITS=400
  if %%S==8192 set ITS=200
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 2560    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_q_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 8  128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_k_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S 32 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_q_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S 8  128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_k_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" multiply  %%S 9728    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_multiply_S%%S.log" 2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" add       %%S 2560    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_add_S%%S.log" 2>&1
  endlocal
)

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
