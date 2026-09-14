@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem PTL QSA long-context benchmark sweep + roofline summary.
rem
rem Runs benchmark_long_context.py for the full prefill/decode x size grid (same
rem shapes/flags as test_cmd.txt: --topk-finalizer fast-wg16-cached, --q3-head-major
rem for prefill only, --warmup 20 --samples 10 --cache cold96MiB no-flush), then
rem runs analyze_ptl_roofline.py once to write the consolidated markdown/JSON
rem report AND print every shape's per-kernel measured/compute/memory/roofline
rem table + Q0-Q3 total (plus a grand total across all shapes) to the console.
rem
rem Usage (run from the remote PTL D:\river\qsa\cm_kernel checkout):
rem   run_ptl_sweep.bat [LOGDIR]
rem LOGDIR defaults to ptl_sweep_<YYYYMMDD_HHMMSS> under the current directory.
rem ============================================================================

set "PYTHON=D:\river\py312\Scripts\python.exe"
set "OPENBLAS_NUM_THREADS=1"

@REM set "SIZES=1024 2048 4096 8192 16384 32768 65536 131072"
set "SIZES=1024 2048 4096 8192 16384 32768 "
set "COMMON=--topk-finalizer fast-wg16-cached --warmup 20 --samples 100 --cache cold96MiB no-flush"

set "LOGDIR=%~1"
if "%LOGDIR%"=="" (
    rem wmic is removed on newer Windows builds; use PowerShell's Get-Date instead.
    for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "LOGDIR=ptl_sweep_%%I"
)
if "%LOGDIR%"=="" set "LOGDIR=ptl_sweep_%RANDOM%"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if not exist "%PYTHON%" (
    echo ERROR: python not found at %PYTHON%
    exit /b 1
)

set "FAILED="
set /a TOTAL=0
set /a OKCOUNT=0

echo ===== prefill sweep =====
for %%S in (%SIZES%) do (
    set /a TOTAL+=1
    echo [prefill %%S] running...
    "%PYTHON%" benchmark_long_context.py --phase prefill --size %%S --q3-head-major %COMMON% > "%LOGDIR%\prefill_%%S.log" 2>&1
    findstr /c:"DONE " "%LOGDIR%\prefill_%%S.log" >nul 2>&1
    if errorlevel 1 (
        echo [prefill %%S] FAILED - see %LOGDIR%\prefill_%%S.log
        set "FAILED=!FAILED! prefill_%%S"
    ) else (
        echo [prefill %%S] OK
        set /a OKCOUNT+=1
    )
)

echo ===== decode sweep =====
for %%S in (%SIZES%) do (
    set /a TOTAL+=1
    echo [decode %%S] running...
    "%PYTHON%" benchmark_long_context.py --phase decode --size %%S %COMMON% > "%LOGDIR%\decode_%%S.log" 2>&1
    findstr /c:"DONE " "%LOGDIR%\decode_%%S.log" >nul 2>&1
    if errorlevel 1 (
        echo [decode %%S] FAILED - see %LOGDIR%\decode_%%S.log
        set "FAILED=!FAILED! decode_%%S"
    ) else (
        echo [decode %%S] OK
        set /a OKCOUNT+=1
    )
)

echo.
echo ===== sweep done: !OKCOUNT!/!TOTAL! shapes passed =====
if not "%FAILED%"=="" (
    echo FAILED shapes:%FAILED%
)

echo.
echo ===== roofline analysis: %LOGDIR% =====
"%PYTHON%" analyze_ptl_roofline.py --log-dir "%LOGDIR%" --report "%LOGDIR%_ROOFLINE_CN.md" --json-out "%LOGDIR%_roofline.json" --print-shapes

echo.
echo Report:  %LOGDIR%_ROOFLINE_CN.md
echo JSON:    %LOGDIR%_roofline.json
echo Logs:    %LOGDIR%\

endlocal
