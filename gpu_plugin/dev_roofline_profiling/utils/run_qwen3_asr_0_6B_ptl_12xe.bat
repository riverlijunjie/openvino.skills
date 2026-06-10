@echo off
REM ============================================================================
REM Qwen3-ASR-0.6B roofline sweep on PTL 12Xe Windows  (FP16 weights / FP16 KV)
REM Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU, 2400 MHz, 110 GB/s
REM
REM Text decoder (Qwen3, dense):
REM   hidden=1024, layers=28, GQA NH=16 NKV=8 HD=128, intermediate=3072,
REM   vocab=151936, tie_word_embeddings=true, hidden_act=silu (SwiGLU)
REM
REM Audio encoder (Whisper-style, MHA):
REM   d_model=896, layers=18, NH=14 NKV=14 HD=64, FFN=3584, S=1500 (fixed)
REM   The encoder runs ONCE per inference; profiled below at M=1500.
REM
REM All FC: precision=f16 (no compression). LM_Head also f16.
REM PA: f16 KV cache, opencl + micro_kernel.
REM
REM Input token sweep (text decoder prefill / decode KV): 512, 1024, 4096, 8192.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_asr_0_6B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ===================== Text decoder PA (GQA 16:8, HD=128) =====================
set PA_NH=16
set PA_NKV=8
set PA_HD=128

REM PA decode (S_q=1, S_kv = ctx). f16 KV, ocl impl.
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  512 500 50 4 f16 ocl > "%LOGS%\pa_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 400 40 4 f16 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 4 f16 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 4 f16 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM PA prefill (S_q=S, S_kv=0).
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill  512 0 200 20 4 f16 ocl > "%LOGS%\pa_prefill_S512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 f16 ocl > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 f16 ocl > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  20  5 2 f16 ocl > "%LOGS%\pa_prefill_S8192.log" 2>&1

REM ===================== Text decoder FC (decode, M=1, FP16) ====================
REM fc_bench: M K N gs iters warmup bufs precision flush_mb
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   4096 128 6000 200 8 f16 > "%LOGS%\fc_decode_qkv.log"     2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048   1024 128 8000 200 8 f16 > "%LOGS%\fc_decode_o.log"       2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   3072 128 6000 200 8 f16 > "%LOGS%\fc_decode_gate.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   3072 128 6000 200 8 f16 > "%LOGS%\fc_decode_up.log"      2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 3072   1024 128 6000 200 8 f16 > "%LOGS%\fc_decode_down.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024 151936 128  300  20 4 f16 > "%LOGS%\fc_decode_lm_head.log" 2>&1

REM ===================== Text decoder FC (prefill, M=S, FP16) ===================
for %%S in (512 1024 4096 8192) do (
  set IT=300
  if %%S==1024 set IT=200
  if %%S==4096 set IT=80
  if %%S==8192 set IT=40
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   4096 128 !IT! 10 4 f16 > "%LOGS%\fc_prefill_qkv_S%%S.log"     2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2048   1024 128 !IT! 10 4 f16 > "%LOGS%\fc_prefill_o_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   3072 128 !IT! 10 4 f16 > "%LOGS%\fc_prefill_gate_S%%S.log"    2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   3072 128 !IT! 10 4 f16 > "%LOGS%\fc_prefill_up_S%%S.log"      2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 3072   1024 128 !IT! 10 4 f16 > "%LOGS%\fc_prefill_down_S%%S.log"    2>&1
  endlocal
)
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024 151936 128 300 20 4 f16 > "%LOGS%\fc_prefill_lm_head.log" 2>&1

REM ===================== Text decoder small ops (decode, M=1) ===================
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   1 1024    --iters 8000 --warmup 200 > "%LOGS%\small_decode_rmsnorm.log"      2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 128  --iters 8000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_q.log"  2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  8 128  --iters 8000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_k.log"  2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 16 128  --iters 8000 --warmup 200 > "%LOGS%\small_decode_rope_q.log"       2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1  8 128  --iters 8000 --warmup 200 > "%LOGS%\small_decode_rope_k.log"       2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add       1 1024    --iters 8000 --warmup 200 > "%LOGS%\small_decode_add.log"          2>&1

REM ===================== Text decoder small ops (prefill, M=S) ==================
for %%S in (512 1024 4096 8192) do (
  set ITS=2000
  if %%S==1024 set ITS=1500
  if %%S==4096 set ITS=400
  if %%S==8192 set ITS=200
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 1024    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm_S%%S.log"      2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 16 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_q_S%%S.log"  2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S  8 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_k_S%%S.log"  2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S 16 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_q_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S  8 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_k_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" add       %%S 1024    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_add_S%%S.log"          2>&1
  endlocal
)

REM ===================== Audio encoder FC (M=1500 fixed, FP16) ==================
REM Encoder runs once per inference. Numbers below feed the fixed-overhead column
REM in the SUMMARY. K=896 is not divisible by 128, but precision=f16 bypasses
REM the group_size constraint (plain FP16 MatMul).
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 2688 128 200 10 4 f16 > "%LOGS%\fc_enc_qkv_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896  896 128 200 10 4 f16 > "%LOGS%\fc_enc_o_S1500.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 3584 128 200 10 4 f16 > "%LOGS%\fc_enc_fc1_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 3584  896 128 200 10 4 f16 > "%LOGS%\fc_enc_fc2_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 1024 128 200 10 4 f16 > "%LOGS%\fc_enc_outproj_S1500.log" 2>&1

REM Audio encoder SDPA — bidirectional non-causal MHA. The PA-prefill kernel
REM uses a causal mask and (Sq*(Sq+1))/2 attention pairs; encoder needs Sq*Skv.
REM We bench the PA-prefill path as a *lower bound* on encoder SDPA time and
REM scale it x2 in the analysis to approximate full-square attention.
set PA_NH=14
set PA_NKV=14
set PA_HD=64
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1500 0 60 10 4 f16 ocl > "%LOGS%\pa_prefill_encS1500.log" 2>&1

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
