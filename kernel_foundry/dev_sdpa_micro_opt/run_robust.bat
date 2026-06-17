@echo off
setlocal enabledelayedexpansion
cd /d D:\river\workspace\dev_sdpa_micro_opt
set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set COMMON=--tokens 16 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --iters 40 --kernel-dir %KDIR%
del robust_results.txt 2>nul

REM ---- baseline (no gqa-share), config-search winner ----
for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ### BASE cfg=16,16,16,16,8,1,8,1 seqs=%%S ### >> robust_results.txt
  %EXE% %COMMON% --seqs %%S --cfg 16,16,16,16,8,1,8,1 >> robust_results.txt 2>&1
)

REM ---- gqa-share SAFE candidates (all KQ/VS unroll_n=16 -> low GRF) ----
REM   sg32: 16,16,16,16,8,4,8,4  (KQ tile_m=128, VS tile_m=128, all unroll=16)
REM   sg16: 16,16,32,16,4,4,4,4  (KQ tile_m=64,  VS tile_m=128, unroll_m_vs=32)
REM   sg8 : 16,16,64,16,2,4,2,4  (KQ tile_m=32,  VS tile_m=128, unroll_m_vs=64 - boundary)
for %%C in ("16,16,16,16,8,4,8,4" "16,16,32,16,4,4,4,4" "16,16,64,16,2,4,2,4") do (
  for %%S in (1 2 3 4 5 6 7 8 9 10) do (
    echo ### GQA cfg=%%~C seqs=%%S ### >> robust_results.txt
    %EXE% %COMMON% --gqa-share --seqs %%S --cfg %%~C >> robust_results.txt 2>&1
  )
)

echo ### ALL_DONE ### >> robust_results.txt
