@echo off
setlocal enabledelayedexpansion
cd /d D:\river\workspace\dev_sdpa_micro_opt
set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set OUT=qnew_sweep.txt
echo === q_new sweep: NEW wg_n=1 (..8,1,8,1) vs OLD wg_n=2 (..8,2,8,2), seqs=8 history=512 head-dim=128 GQA32:8 > %OUT%
for %%T in (1 4 8 16 32 64) do (
  echo. >> %OUT%
  echo ====== tokens=%%T  NEW 16,16,16,16,8,1,8,1 ====== >> %OUT%
  %EXE% --tokens %%T --seqs 8 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,1,8,1 --iters 50 --kernel-dir %KDIR% >> %OUT% 2>&1
  echo ====== tokens=%%T  OLD 16,16,16,16,8,2,8,2 ====== >> %OUT%
  %EXE% --tokens %%T --seqs 8 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,2,8,2 --iters 50 --kernel-dir %KDIR% >> %OUT% 2>&1
)
echo FINISHED >> %OUT%
echo DONE
