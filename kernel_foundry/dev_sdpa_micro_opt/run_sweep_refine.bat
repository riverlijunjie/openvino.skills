@echo off
REM Refinement around elite A (wg_tile_n=16, wg_n=1). Vary sg_per_wg/occupancy.
REM   A = 16,16,16,16,8,1,8,1   (sg_per_wg=8,  wg_tile_m_kq=128, unroll_m_vs=16)  [elite anchor]
REM   E = 16,16,32,16,4,1,4,1   (sg_per_wg=4,  wg_tile_m_kq=64,  unroll_m_vs=32)
REM   F = 16,16,8,16,16,1,16,1  (sg_per_wg=16, wg_tile_m_kq=256, unroll_m_vs=8)
REM   G = 16,16,64,16,2,1,2,1   (sg_per_wg=2,  wg_tile_m_kq=32,  unroll_m_vs=64)
cd /d D:\river\workspace\dev_sdpa_micro_opt
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set OUT=sweep_refine.txt
set COMMON=--tokens 16 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --iters 30 --kernel-dir %KDIR%
echo SWEEP REFINE START > %OUT%
for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ==== seq=%%S cfg=A 16,16,16,16,8,1,8,1 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,16,16,8,1,8,1 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=E 16,16,32,16,4,1,4,1 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,32,16,4,1,4,1 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=F 16,16,8,16,16,1,16,1 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,8,16,16,1,16,1 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=G 16,16,64,16,2,1,2,1 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,64,16,2,1,2,1 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
)
echo SWEEP REFINE DONE >> %OUT%
