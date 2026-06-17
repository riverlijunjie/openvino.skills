@echo off
REM Parameter trust-region search driver for the sdpa_micro__generate config.
REM Builds ONCE (config is a --cfg CLI override); reads candidate configs from
REM cfgs.txt (one per line) and runs each at the two triage points seqs=1
REM (most launch-bound) and seqs=10 (most occupancy-rich).  Prints "CFG <list>"
REM then the device perf line so the search can rank candidates by DRAM(unique)
REM GB/s on the cold KV read.
REM   cfg = um_kq,un_kq,um_vs,un_vs,wm_kq,wn_kq,wm_vs,wn_vs
setlocal enabledelayedexpansion

set EXE=sdpa_micro_generate_test.exe
set KDIR=D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2
set ITERS=200
set HISTORY=512
set COMMON=--tokens 16 --history %HISTORY% --head-dim 128 --kv-heads 8 --heads 32 --iters %ITERS% --kernel-dir "%KDIR%"

if "%~1"=="" ( set CFGFILE=cfgs.txt ) else ( set CFGFILE=%~1 )

for /F "usebackq tokens=* delims=" %%C in ("%CFGFILE%") do (
  echo ==== CFG %%C ====
  for %%S in (1 10) do (
    echo -- seqs=%%S --
    %EXE% %COMMON% --seqs %%S --cfg %%C 2>&1 | findstr /C:"correctness:" /C:"perf:" /C:"selectGEMM" /C:"ERROR" /C:"GRF"
  )
  echo.
)
echo ==== CFGSEARCH DONE ====
