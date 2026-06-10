@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Sweep cache_interval for the PagedGatedDeltaNet opt kernel to quantify the
REM per-interval recurrent-state snapshot (paging) overhead in prefill.
REM Larger interval => fewer state snapshots; interval>=T => single final
REM snapshot (matches the non-paged ref's single state store).

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_6\ptl\gdn_sweep
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === GDN INTERVAL SWEEP %date% %time% > "%LOGS%\_index.txt"

REM Prefill sizes x intervals. interval 999999 => single snapshot (ref-like).
for %%S in (1024 4096 8192) do (
  for %%I in (64 256 1024 999999) do (
    call :do gdn_S%%S_iv%%I "%BUILD%\gdn_bench.exe" 1 %%S 16 32 128 20 5 2 %%I
  )
)
echo === GDN INTERVAL SWEEP END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
goto :eof

:do
set TAG=%~1
shift
set CMDLINE=
:doargs
if "%~1"=="" goto dorun
set CMDLINE=%CMDLINE% %1
shift
goto doargs
:dorun
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
