@echo off
REM Layout-preserving config sweep (no GQA-share, no layout change).
REM Candidates (all unroll_n=16 -> grf~76, crash-free; structurally valid):
REM   A = 16,16,16,16,8,1,8,1   (wg_n=1, sg_per_wg=8,  wg_tile_n=16, wg_tile_m_kq=128)
REM   B = 16,16,16,16,8,2,8,2   (PRODUCTION BASELINE; sg_per_wg=16, wg_tile_n=32)
REM   C = 16,16,16,16,8,4,8,4   (wg_n=4, sg_per_wg=32, wg_tile_n=64)
REM   D = 16,16,32,16,4,2,4,2   (unroll_m_vs=32, wg_tile_m_kq=64, sg_per_wg=8)
cd /d D:\river\workspace\dev_sdpa_micro_opt
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set OUT=sweep_nolayout.txt
set COMMON=--tokens 16 --history 512 --head-dim 128 --kv-heads 8 --heads 32 --iters 30 --kernel-dir %KDIR%
echo SWEEP START > %OUT%
for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ==== seq=%%S cfg=A 16,16,16,16,8,1,8,1 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,16,16,8,1,8,1 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=B 16,16,16,16,8,2,8,2 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,16,16,8,2,8,2 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=C 16,16,16,16,8,4,8,4 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,16,16,8,4,8,4 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
  echo ==== seq=%%S cfg=D 16,16,32,16,4,2,4,2 ==== >> %OUT%
  sdpa_micro_generate_test.exe %COMMON% --seqs %%S --cfg 16,16,32,16,4,2,4,2 2>&1 | findstr /C:"roofline" /C:"perf:" /C:"correctness: PASS" /C:"correctness: FAIL" /C:"OpenCL error" /C:"CL_OUT" >> %OUT%
)
echo SWEEP DONE >> %OUT%
