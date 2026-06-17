@echo off
setlocal enabledelayedexpansion
cd /d D:\river\workspace\dev_sdpa_micro_opt
set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set COMMON=--tokens 16 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --iters 60 --kernel-dir %KDIR%
del confirm_results.txt 2>nul

for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ### BASE seqs=%%S ### >> confirm_results.txt
  %EXE% %COMMON% --seqs %%S --cfg 16,16,16,16,8,1,8,1 >> confirm_results.txt 2>&1
  echo ### SG32 seqs=%%S ### >> confirm_results.txt
  %EXE% %COMMON% --gqa-share --seqs %%S --cfg 16,16,16,16,8,4,8,4 >> confirm_results.txt 2>&1
)
echo ### ALL_DONE ### >> confirm_results.txt
