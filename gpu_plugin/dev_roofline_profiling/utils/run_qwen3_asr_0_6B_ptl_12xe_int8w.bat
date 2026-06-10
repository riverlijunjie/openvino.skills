@echo off
REM ============================================================================
REM Qwen3-ASR-0.6B roofline sweep on PTL 12Xe Windows  (INT8 weights / INT8 KV)
REM Target: Local_Admin@10.239.132.229 — Intel PTL 12Xe iGPU, 2400 MHz, 110 GB/s
REM
REM Same coverage as run_qwen3_asr_0_6B_ptl_12xe.bat but with:
REM   - FC weights: u8 (group_size=128, f16 scale per OC per group)
REM     -> prefill enables INT8 XMX via dynamic_quantization_group_size
REM   - PA KV cache: i8 (matches Intel GPU plugin default for PA models)
REM   - small_ops_bench unchanged (activation-only, not affected by weight precision)
REM
REM K=2048 (fc_o) / K=896 (audio enc) / K=3584 (encoder fc2) are all divisible
REM by group_size=128, so u8 grouping is valid for every FC sweep below.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_asr_0_6B\ptl_12xe_int8w
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ===================== Text decoder PA (GQA 16:8, HD=128) =====================
set PA_NH=16
set PA_NKV=8
set PA_HD=128

REM PA decode (S_q=1, S_kv = ctx). i8 KV, ocl impl.
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1  512 500 50 4 i8 ocl > "%LOGS%\pa_decode_kv512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 1024 400 40 4 i8 ocl > "%LOGS%\pa_decode_kv1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 4096 200 20 4 i8 ocl > "%LOGS%\pa_decode_kv4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" decode 1 8192 150 15 4 i8 ocl > "%LOGS%\pa_decode_kv8192.log" 2>&1

REM PA prefill (S_q=S, S_kv=0).
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill  512 0 200 20 4 i8 ocl > "%LOGS%\pa_prefill_S512.log"  2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1024 0 100 10 4 i8 ocl > "%LOGS%\pa_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 4096 0  40  8 4 i8 ocl > "%LOGS%\pa_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 8192 0  20  5 2 i8 ocl > "%LOGS%\pa_prefill_S8192.log" 2>&1

REM ===================== Text decoder FC (decode, M=1, INT8 u8) =================
REM fc_bench: M K N gs iters warmup bufs precision flush_mb
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   4096 128 6000 200 8 u8 > "%LOGS%\fc_decode_qkv.log"     2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048   1024 128 8000 200 8 u8 > "%LOGS%\fc_decode_o.log"       2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   3072 128 6000 200 8 u8 > "%LOGS%\fc_decode_gate.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024   3072 128 6000 200 8 u8 > "%LOGS%\fc_decode_up.log"      2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 3072   1024 128 6000 200 8 u8 > "%LOGS%\fc_decode_down.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024 151936 128  300  20 4 u8 > "%LOGS%\fc_decode_lm_head.log" 2>&1

REM ===================== Text decoder FC (prefill, M=S, INT8 u8) ================
for %%S in (512 1024 4096 8192) do (
  set IT=300
  if %%S==1024 set IT=200
  if %%S==4096 set IT=80
  if %%S==8192 set IT=40
  setlocal EnableDelayedExpansion
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   4096 128 !IT! 10 4 u8 > "%LOGS%\fc_prefill_qkv_S%%S.log"     2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2048   1024 128 !IT! 10 4 u8 > "%LOGS%\fc_prefill_o_S%%S.log"       2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   3072 128 !IT! 10 4 u8 > "%LOGS%\fc_prefill_gate_S%%S.log"    2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 1024   3072 128 !IT! 10 4 u8 > "%LOGS%\fc_prefill_up_S%%S.log"      2>&1
  "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 3072   1024 128 !IT! 10 4 u8 > "%LOGS%\fc_prefill_down_S%%S.log"    2>&1
  endlocal
)
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 1024 151936 128 300 20 4 u8 > "%LOGS%\fc_prefill_lm_head.log" 2>&1

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

REM ===================== Audio encoder FC (M=1500 fixed, INT8 u8) ===============
REM K=896 (896/128=7), K=3584 (3584/128=28): both divisible by group_size=128.
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 2688 128 200 10 4 u8 > "%LOGS%\fc_enc_qkv_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896  896 128 200 10 4 u8 > "%LOGS%\fc_enc_o_S1500.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 3584 128 200 10 4 u8 > "%LOGS%\fc_enc_fc1_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500 3584  896 128 200 10 4 u8 > "%LOGS%\fc_enc_fc2_S1500.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1500  896 1024 128 200 10 4 u8 > "%LOGS%\fc_enc_outproj_S1500.log" 2>&1

REM Audio encoder SDPA — bench keeps PA causal as a lower-bound proxy, KV in i8
REM for consistency with the rest of the run (encoder has no real persistent KV).
set PA_NH=14
set PA_NKV=14
set PA_HD=64
"%CLI%" -d "%BUILD%\pa_bench.exe" prefill 1500 0 60 10 4 i8 ocl > "%LOGS%\pa_prefill_encS1500.log" 2>&1

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
echo INT8W_SWEEP_DONE
