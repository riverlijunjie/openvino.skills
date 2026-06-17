@echo off
REM cliloader device-timing profile for the 10 PA MIXED configs.
REM Reports the per-kernel device time (name + Calls + Average ns) for BOTH the
REM micro_sdpa kernel and the cache_flush kernel, plus the in-test perf line.
REM A full per-config cliloader report is also written to profile_seqsNN.txt.
setlocal enabledelayedexpansion

set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set CLI_DevicePerformanceTiming=1
set CLI_DevicePerformanceTimeKernelInfoTracking=1
set ITERS=200
set HISTORY=512

for %%S in (1 2 3 4 5 6 7 8 9 10) do (
  echo ==== PROFILE tokens=16 seqs=%%S history=%HISTORY% head-dim=128 kv-heads=8 heads=32 ====
  %CLI% %EXE% --tokens 16 --seqs %%S --history %HISTORY% --head-dim 128 --kv-heads 8 --heads 32 --iters %ITERS% --kernel-dir "%KDIR%" > profile_seqs%%S.txt 2>&1
  findstr /C:"Function Name" /C:"micro_sdpa" /C:"cache_flush" /C:"perf:" /C:"correctness:" profile_seqs%%S.txt
  echo.
)
echo ==== PROFILE DONE ====
