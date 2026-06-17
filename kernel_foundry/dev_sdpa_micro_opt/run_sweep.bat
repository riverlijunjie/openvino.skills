@echo off
REM PagedAttention MIXED-stage sweep for the standalone sdpa_micro__generate test.
REM MIXED requires num_tokens != num_seqs AND past_len > 0, so we sweep
REM tokens=16 (new query tokens per sequence) x seqs {1..10} with a decode
REM history of 512.  head-dim=128, kv-heads=8, heads=32 (GQA 4:1).
REM history is rounded up to a multiple of the KQ wg_tile_m by the test.
REM Each config is timed over 200 iterations; a cache-flush kernel evicts
REM L2/L3 before every timed iteration so each measurement is cache-cold.
setlocal enabledelayedexpansion

set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set ITERS=200
set HISTORY=512

for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ==== CONFIG tokens=16 seqs=%%S history=%HISTORY% head-dim=128 kv-heads=8 heads=32 ====
  %EXE% --tokens 16 --seqs %%S --history %HISTORY% --head-dim 128 --kv-heads 8 --heads 32 --iters %ITERS% --kernel-dir "%KDIR%"
  echo.
)
echo ==== SWEEP DONE ====
