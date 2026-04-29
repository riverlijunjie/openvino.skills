@echo off
setlocal enabledelayedexpansion

rem Template runner for Windows hosts.
rem Fill in the placeholders below for your environment and model.

set MODEL_NAME=replace_me
set TOKEN_POINTS=1024 2048 4096 8192

if "%OPENVINO_ROOT%"=="" (
  echo Please set OPENVINO_ROOT
  exit /b 1
)
if "%CLILOADER_BIN%"=="" (
  echo Please set CLILOADER_BIN
  exit /b 1
)
if "%BENCH_BUILD%"=="" (
  echo Please set BENCH_BUILD
  exit /b 1
)
if "%OUTPUT_DIR%"=="" (
  echo Please set OUTPUT_DIR
  exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

for %%S in (%TOKEN_POINTS%) do (
  echo [template] Run prefill/decode measurements for %MODEL_NAME% at S=%%S
  echo [template] Invoke the appropriate bench binaries here with cliloader.
  echo [template] Save each log to: %OUTPUT_DIR%\%MODEL_NAME%_^<mode^>_%%S.log
)

echo Template complete.
echo   1. Replace echo statements with actual bench invocations.
echo   2. Copy logs into outputs\%MODEL_NAME%\logs*.
echo   3. Parse with utils\parse_logs.py.
echo   4. Rebuild db\metrics.db with utils\build_db.py.
endlocal
