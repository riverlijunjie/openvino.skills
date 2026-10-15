@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Recovery script: re-run the 6 specs missing from the main 2560x1536 sweep
REM (lost due to the earlier scheduled-task duplicate-trigger incident).

set OV_BIN=D:\river\ovmx_dev\openvino.pipeline.mx\bin\intel64\Release
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\ovmx_dev\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536
set PATH=%OV_BIN%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === RECOVERY START %date% %time% >> "%LOGS%\_index.txt"

call :do conv_enc_conv_in "%BUILD%\conv_bench.exe" plain 3 128 3 3 3 6 1536 2560 1 1 1 19 5 4 1 0 64
call :do conv_enc_down1_shortcut "%BUILD%\conv_bench.exe" plain 128 256 1 1 1 4 768 1280 1 1 1 39 5 4 1 0 64
call :do conv_enc_down2_conv2 "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 4 384 640 1 1 1 158 5 4 1 0 64
call :do gn_enc_gn_c512_h384w640 "%BUILD%\groupnorm_bench.exe" 512 2 384 640 32 20 10 4
call :do fc_dit_vid_qkv "%BUILD%\fc_bench.exe" 476160 2560 7680 128 10 5 1 f16 64
call :do fc_dit_vid_mlp_up "%BUILD%\fc_bench.exe" 476160 2560 6912 128 10 5 2 f16 64

echo === RECOVERY END %date% %time% >> "%LOGS%\_index.txt"
echo RecoveryDone
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
