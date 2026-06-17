@echo off
REM Full 10-seq sweep + cliloader profile of the GEN1 config-search WINNER
REM (16,16,16,16,8,1,8,1 => sg_per_wg=8).  Establishes the post-config-tuning
REM curve and tells us whether seqs=10 (occupancy-saturated) is occupancy-bound
REM or gather-pattern-bound before deciding on structural split-K.
setlocal enabledelayedexpansion

set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set ITERS=200
set HISTORY=512
set BESTCFG=16,16,16,16,8,1,8,1
set COMMON=--tokens 16 --history %HISTORY% --head-dim 128 --kv-heads 8 --heads 32 --iters %ITERS% --kernel-dir "%KDIR%" --cfg %BESTCFG%

echo ==== WINNER full sweep (cfg=%BESTCFG%) ====
for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo -- seqs=%%S --
  %EXE% %COMMON% --seqs %%S 2>&1 | findstr /C:"correctness: PASS" /C:"correctness: FAIL" /C:"DRAM(unique)"
)
echo.

echo ==== cliloader profile (cfg=%BESTCFG%) seqs=1 and seqs=10 ====
set CLI_DevicePerformanceTiming=1
set CLI_DevicePerformanceTimeKernelInfoTracking=1
for %%S in (1 10) do (
  echo -- profile seqs=%%S --
  %CLI% %EXE% %COMMON% --seqs %%S > best_profile_seqs%%S.txt 2>&1
  findstr /C:"Function Name" /C:"micro_sdpa" /C:"cache_flush" /C:"perf:" /C:"correctness: " best_profile_seqs%%S.txt
  echo.
)
echo ==== RUN_BEST DONE ====
