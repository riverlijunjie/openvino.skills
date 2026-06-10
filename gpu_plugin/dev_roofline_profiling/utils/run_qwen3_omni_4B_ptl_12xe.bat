@echo off
REM ============================================================================
REM Qwen3-Omni-4B roofline sweep on PTL 12Xe Windows  (INT4 g128 / INT8 LM_head)
REM Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU, 2400 MHz, 110 GB/s
REM
REM Thinker text decoder (Qwen3VL-style dense, 36-layer):
REM   hidden=2560, NH=32 NKV=8 HD=128, intermediate=9728, vocab=151936
REM   tie_word_embeddings=true, hidden_act=silu (SwiGLU), rope_theta=1e6
REM
REM Audio encoder (Whisper-style MHA, 32-layer, bidir):
REM   d_model=1280, NH=NKV=20, HD=64, FFN=5120, S=1500, output_dim=2560
REM
REM Vision encoder (SigLIP-style ViT, 27-layer, bidir):
REM   hidden=1152, NH=16, HD=72, FFN=4304, patch=16, image=768, S=2304 patches
REM   spatial_merge=2 -> 576 vision tokens; out_hidden=2560
REM
REM Talker (audio-output AR decoder, 28-layer, GQA 16:8):
REM   hidden=2560, NH=16 NKV=8 HD=128, intermediate=3072, vocab=3072
REM
REM All matmul: INT4 g128 (precision=u4 default). LM_head: INT8 g128 (u8).
REM Talker LM head: FP16 (small vocab=3072).
REM PA: INT8 KV cache, opencl + micro_kernel.
REM
REM Input token sweep (thinker prefill / decode KV): 512, 1024, 4096, 8192.
REM Output: 512 tokens.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_omni_4B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ==========================================================================
REM Thinker text decoder — PA (GQA 32:8, HD=128, INT8 KV, OCL micro_kernel)
REM ==========================================================================
set PA_NH=32
set PA_NKV=8
set PA_HD=128

REM pa_bench positional: <mode> <S_q> <S_kv> <iters> <warmup> <bufs> <kv_dtype:f16|i8> <impl:ocl|cm>
REM PA decode (Sq=1, Skv=ctx). INT8 KV, ocl.
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1   512 500 50 4 i8 ocl > "%LOGS%\pa_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  1024 400 40 4 i8 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  4096 200 20 4 i8 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  8192 150 15 4 i8 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM PA prefill (Sq=S, Skv=0). Causal mask, S*(S+1)/2 effective pairs.
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill  512 0 200 20 4 i8 ocl > "%LOGS%\pa_prefill_S512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 i8 ocl > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  20  5 2 i8 ocl > "%LOGS%\pa_prefill_S8192.log" 2>&1

REM ==========================================================================
REM Thinker text decoder — FC decode (M=1, INT4 g128)
REM ==========================================================================
REM fc_bench: M K N gs iters warmup bufs precision flush_mb
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   6144 128 3000 150 8 > "%LOGS%\fc_decode_qkv.log"      2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 4096   2560 128 5000 200 8 > "%LOGS%\fc_decode_o.log"        2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   9728 128 2000 100 8 > "%LOGS%\fc_decode_gate.log"     2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560   9728 128 2000 100 8 > "%LOGS%\fc_decode_up.log"       2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 9728   2560 128 2000 100 8 > "%LOGS%\fc_decode_down.log"     2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 151936 128  150  10 4 u8 > "%LOGS%\fc_decode_lm_head.log" 2>&1

REM ==========================================================================
REM Thinker text decoder — FC prefill (M=S, INT4 g128)
REM ==========================================================================
for %%S in (512 1024 4096 8192) do (
  set IT=200
  if %%S==1024 set IT=150
  if %%S==4096 set IT=60
  if %%S==8192 set IT=30
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   6144 128 !IT! 10 4 > "%LOGS%\fc_prefill_qkv_S%%S.log"     2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 4096   2560 128 !IT! 10 4 > "%LOGS%\fc_prefill_o_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   9728 128 !IT! 10 4 > "%LOGS%\fc_prefill_gate_S%%S.log"    2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2560   9728 128 !IT! 10 4 > "%LOGS%\fc_prefill_up_S%%S.log"      2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 9728   2560 128 !IT! 10 4 > "%LOGS%\fc_prefill_down_S%%S.log"    2>&1
  endlocal
)
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 151936 128 150 10 4 u8 > "%LOGS%\fc_prefill_lm_head.log" 2>&1

REM ==========================================================================
REM Thinker text decoder — small ops (decode, M=1)
REM ==========================================================================
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   1 2560    --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm.log"      2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128  --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_q.log"  2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  8 128  --iters 5000 --warmup 200 > "%LOGS%\small_decode_rmsnorm3d_k.log"  2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1 32 128  --iters 5000 --warmup 200 > "%LOGS%\small_decode_rope_q.log"       2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      1  8 128  --iters 5000 --warmup 200 > "%LOGS%\small_decode_rope_k.log"       2>&1
"%CLI%" -d "%BUILD%\small_ops_bench.exe" add       1 2560    --iters 5000 --warmup 200 > "%LOGS%\small_decode_add.log"          2>&1

REM ==========================================================================
REM Thinker text decoder — small ops (prefill, M=S)
REM ==========================================================================
for %%S in (512 1024 4096 8192) do (
  set ITS=2000
  if %%S==1024 set ITS=1500
  if %%S==4096 set ITS=400
  if %%S==8192 set ITS=200
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm   %%S 2560    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm_S%%S.log"      2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S 32 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_q_S%%S.log"  2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rmsnorm3d %%S  8 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rmsnorm3d_k_S%%S.log"  2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S 32 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_q_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" rope      %%S  8 128  --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_rope_k_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\small_ops_bench.exe" add       %%S 2560    --iters !ITS! --warmup 30 > "%LOGS%\small_prefill_add_S%%S.log"          2>&1
  endlocal
)

REM ==========================================================================
REM Audio encoder — fixed S=1500 (Whisper-style MHA, INT4 g128, bidirectional)
REM ==========================================================================
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 1280 3840 128 200 10 4 > "%LOGS%\fc_enc_qkv_S1500.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 1280 1280 128 200 10 4 > "%LOGS%\fc_enc_o_S1500.log"     2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 1280 5120 128 200 10 4 > "%LOGS%\fc_enc_fc1_S1500.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 5120 1280 128 200 10 4 > "%LOGS%\fc_enc_fc2_S1500.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 1280 2560 128 200 10 4 > "%LOGS%\fc_enc_outproj_S1500.log" 2>&1

REM Audio encoder SDPA — bidirectional MHA NH=20 HD=64. Bench via PA prefill (causal)
REM as a lower bound; build_report scales × 2 for full bidir attention.
set PA_NH=20
set PA_NKV=20
set PA_HD=64
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1500 0 60 10 4 f16 ocl > "%LOGS%\pa_prefill_encS1500.log" 2>&1

REM ==========================================================================
REM Vision encoder — fixed S=2304 patches (SigLIP ViT, INT4 g128, bidirectional)
REM ==========================================================================
"%CLI%" -d "%BUILD%\fc_bench.exe" 2304 1152 3456 128 60 10 4 > "%LOGS%\fc_vis_qkv_S2304.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2304 1152 1152 128 80 10 4 > "%LOGS%\fc_vis_o_S2304.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2304 1152 4304 128 50 10 4 > "%LOGS%\fc_vis_fc1_S2304.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2304 4304 1152 128 50 10 4 > "%LOGS%\fc_vis_fc2_S2304.log" 2>&1
REM Merge projection runs once on S=576 (after spatial_merge=2 collapse 4 patches),
REM K = 1152 * 4 = 4608.
"%CLI%" -d "%BUILD%\fc_bench.exe"  576 4608 2560 128 100 10 4 > "%LOGS%\fc_vis_merge_S576.log" 2>&1

REM Vision encoder SDPA — bidirectional NH=16 HD=72. HD is non-standard; some kernels
REM pad to HD=80. Bench as causal PA prefill, × 2 in report.
set PA_NH=16
set PA_NKV=16
set PA_HD=72
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 2304 0 40 8 4 f16 ocl > "%LOGS%\pa_prefill_visS2304.log" 2>&1

REM ==========================================================================
REM Talker (audio-output AR decoder) — decode-only profile (M=1, GQA 16:8)
REM ==========================================================================
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 4096 128 5000 200 8 > "%LOGS%\fc_talker_decode_qkv.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 2560 128 6000 200 8 > "%LOGS%\fc_talker_decode_o.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 3072 128 5000 200 8 > "%LOGS%\fc_talker_decode_gate.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 3072 128 5000 200 8 > "%LOGS%\fc_talker_decode_up.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 3072 2560 128 5000 200 8 > "%LOGS%\fc_talker_decode_down.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2560 3072 128 5000 200 8 f16 > "%LOGS%\fc_talker_decode_lmhead.log" 2>&1

set PA_NH=16
set PA_NKV=8
set PA_HD=128
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1   512 500 50 4 i8 ocl > "%LOGS%\pa_talker_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  1024 400 40 4 i8 ocl > "%LOGS%\pa_talker_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  4096 200 20 4 i8 ocl > "%LOGS%\pa_talker_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  8192 150 15 4 i8 ocl > "%LOGS%\pa_talker_decode_kv8192.log" 2>&1

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
