@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Re-run ONLY the GDN (PagedGatedDeltaNet opt) cases on PTL 12Xe.
REM qk_heads=16, v_heads=32 (GQA group=2), head_dim=128. Writes into the same log dir.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_6\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === GDN RERUN %date% %time% >> "%LOGS%\_index.txt"

call :do gdn_decode_T1                "%BUILD%\gdn_bench.exe" 1 1    16 32 128 4000 150 4 0
for %%S in (1024 2048 4096 8192) do (
  call :do gdn_prefill_S%%S           "%BUILD%\gdn_bench.exe" 1 %%S 16 32 128 20 5 2 0
)
echo === GDN RERUN END %date% %time% >> "%LOGS%\_index.txt"
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
