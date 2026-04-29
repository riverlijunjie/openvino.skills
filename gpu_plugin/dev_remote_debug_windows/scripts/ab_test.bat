@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Generic A/B Test Runner for OpenVINO on Windows
REM
REM  Compares two source variants (A vs B) by swapping source files,
REM  rebuilding, and running a benchmark multiple times per scenario.
REM
REM  Usage:
REM    ab_test.bat [a|b|all]        (default: all)
REM
REM  Configuration:
REM    1. Copy ab_test_config.bat.template to ab_test_config.bat
REM    2. Edit ab_test_config.bat to set paths, file mappings, scenarios
REM    3. Run ab_test.bat
REM ============================================================

REM ---- Load user configuration ----
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%ab_test_config.bat" (
    call "%SCRIPT_DIR%ab_test_config.bat"
) else (
    echo ERROR: ab_test_config.bat not found in %SCRIPT_DIR%
    echo Copy ab_test_config.bat.template and edit it for your test.
    exit /b 1
)

REM ---- Validate required variables ----
if not defined OV_DIR       ( echo ERROR: OV_DIR not set      & exit /b 1 )
if not defined BUILD_DIR    ( echo ERROR: BUILD_DIR not set   & exit /b 1 )
if not defined INSTALL_DIR  ( echo ERROR: INSTALL_DIR not set & exit /b 1 )
if not defined AB_DIR       ( echo ERROR: AB_DIR not set      & exit /b 1 )
if not defined LOG_DIR      ( echo ERROR: LOG_DIR not set     & exit /b 1 )
if not defined PY           ( echo ERROR: PY not set          & exit /b 1 )
if not defined BENCH_CMD    ( echo ERROR: BENCH_CMD not set   & exit /b 1 )
if not defined FILE_COUNT   ( echo ERROR: FILE_COUNT not set  & exit /b 1 )

if not defined RUNS_PER_SCENARIO set RUNS_PER_SCENARIO=3
if not defined LOG_FILTER        set LOG_FILTER=Pipeline initialization
if not defined LABEL_A           set LABEL_A=A
if not defined LABEL_B           set LABEL_B=B
if not defined SCENARIO_COUNT    set SCENARIO_COUNT=0
if not defined BUILD_PARALLEL    set BUILD_PARALLEL=16

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ---- Parse mode argument ----
set MODE=%1
if "%MODE%"=="" set MODE=all

if /i "%MODE%"=="a"   goto :run_a
if /i "%MODE%"=="b"   goto :run_b
if /i "%MODE%"=="all" goto :run_all
echo Unknown mode: %MODE%
echo Usage: ab_test.bat [a|b|all]
goto :eof

:run_all
call :run_variant a "%LABEL_A%"
if errorlevel 1 goto :eof
call :run_variant b "%LABEL_B%"
goto :eof

:run_a
call :run_variant a "%LABEL_A%"
goto :eof

:run_b
call :run_variant b "%LABEL_B%"
goto :eof

REM ============================================================
REM  :run_variant <a|b> <label>
REM  Copies source files, rebuilds, installs, and runs all scenarios
REM ============================================================
:run_variant
set VARIANT=%~1
set VLABEL=%~2
echo.
echo ########################################################
echo  VARIANT %VLABEL% - Copying source files
echo ########################################################

REM Copy source files from AB_DIR/<variant>/ to their target locations
for /L %%f in (1,1,%FILE_COUNT%) do (
    set SRC=!FILE_%%f_SRC_%VARIANT%!
    set DST=!FILE_%%f_DST!
    if not defined SRC (
        echo ERROR: FILE_%%f_SRC_%VARIANT% not defined
        exit /b 1
    )
    copy /Y "!SRC!" "!DST!"
)

REM Force MSBuild to detect changed files by updating timestamps
echo Touching source files to force recompilation...
for /L %%f in (1,1,%FILE_COUNT%) do (
    set DST=!FILE_%%f_DST!
    powershell -Command "(Get-Item '!DST!').LastWriteTime = Get-Date"
)

echo.
echo Building %VLABEL%...
cd /d "%BUILD_DIR%"
cmake --build . --config Release --parallel %BUILD_PARALLEL%
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )

echo Installing %VLABEL%...
cmake --install . --prefix "%INSTALL_DIR%" --config Release
if errorlevel 1 ( echo INSTALL FAILED & exit /b 1 )

REM Copy DLLs to Python site-packages if DLL_DEPLOY_DIR is set
if defined DLL_DEPLOY_DIR (
    copy /Y "%INSTALL_DIR%\runtime\bin\intel64\Release\*.dll" "%DLL_DEPLOY_DIR%"
)

REM Run scenarios
if %SCENARIO_COUNT% EQU 0 (
    REM No scenarios defined - just run BENCH_CMD RUNS_PER_SCENARIO times
    call :run_scenario "%VLABEL%" "default" "" %RUNS_PER_SCENARIO%
) else (
    for /L %%s in (1,1,%SCENARIO_COUNT%) do (
        set S_NAME=!SCENARIO_%%s_NAME!
        set S_SETUP=!SCENARIO_%%s_SETUP!
        call :run_scenario "%VLABEL%" "!S_NAME!" "!S_SETUP!" %RUNS_PER_SCENARIO%
    )
)

echo.
echo %VLABEL% PHASE COMPLETE
exit /b 0

REM ============================================================
REM  :run_scenario <label> <scenario_name> <setup_cmd> <runs>
REM  Runs per-scenario setup then executes the benchmark N times
REM ============================================================
:run_scenario
set S_LABEL=%~1
set S_NAME=%~2
set S_SETUP_CMD=%~3
set S_RUNS=%~4

echo.
echo ======== %S_LABEL% %S_NAME% ========

REM Execute scenario setup command (e.g. write config file)
if not "%S_SETUP_CMD%"=="" (
    %S_SETUP_CMD%
)

cd /d "%BENCH_WORK_DIR%"

for /L %%i in (1,1,%S_RUNS%) do (
    echo.
    echo --- %S_LABEL%_%S_NAME% run %%i / %S_RUNS% ---
    set LOG_FILE=%LOG_DIR%\%S_LABEL%_%S_NAME%_%%i.log
    %BENCH_CMD% 2> "!LOG_FILE!"
    type "!LOG_FILE!" | findstr /C:"%LOG_FILTER%"
)
exit /b 0
