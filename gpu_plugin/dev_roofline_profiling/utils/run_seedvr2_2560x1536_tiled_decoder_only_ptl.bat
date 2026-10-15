@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Decoder-only sweep for the 2560x1536 TILED (K=4, 2x2 grid) variant.
REM Encoder+DiT are IDENTICAL to models/seedvr2_2560x1536 (untiled) and are
REM reused from that sweep's already-measured logs -- only these 22 decoder
REM specs need fresh remote measurement at the tiled per-tile shapes.

set OV_BIN=D:\river\ovmx_dev\openvino.pipeline.mx\bin\intel64\Release
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\ovmx_dev\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536_tiled
set PATH=%OV_BIN%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% >> "%LOGS%\_index.txt"

REM ---------- vae_decoder: convolution ----------
call :do conv_dec_conv_in "%BUILD%\conv_bench.exe" causal 16 512 3 3 3 1 100 162 1 1 1 400 5 4 1 0 64
call :do conv_dec_mid_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 100 162 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 100 162 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 100 162 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 200 324 1 1 1 400 5 4 1 0 64
call :do conv_dec_up1_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 200 324 1 1 1 400 5 4 1 0 64
call :do conv_dec_up1_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 200 324 1 1 1 150 5 4 1 0 64
call :do conv_dec_up1_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 400 648 1 1 1 150 5 4 1 0 64
call :do conv_dec_up2_shortcut "%BUILD%\conv_bench.exe" plain 512 256 1 1 1 2 400 648 1 1 1 301 5 4 1 0 64
call :do conv_dec_up2_conv1 "%BUILD%\conv_bench.exe" causal 512 256 3 3 3 2 400 648 1 1 1 301 5 4 1 0 64
call :do conv_dec_up2_conv2 "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 400 648 1 1 1 301 5 4 1 0 64
call :do conv_dec_up2_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 800 1296 1 1 1 75 5 4 1 0 64
call :do conv_dec_up3_shortcut "%BUILD%\conv_bench.exe" plain 256 128 1 1 1 2 800 1296 1 1 1 150 5 4 1 0 64
call :do conv_dec_up3_conv1 "%BUILD%\conv_bench.exe" causal 256 128 3 3 3 2 800 1296 1 1 1 150 5 4 1 0 64
call :do conv_dec_up3_conv2 "%BUILD%\conv_bench.exe" causal 128 128 3 3 3 2 800 1296 1 1 1 150 5 4 1 0 64
call :do conv_dec_conv_out "%BUILD%\conv_bench.exe" causal 128 3 3 3 3 2 800 1296 1 1 1 400 5 4 1 0 64

REM ---------- vae_decoder: groupnorm ----------
call :do gn_dec_gn_c512_h100w162 "%BUILD%\groupnorm_bench.exe" 512 1 100 162 32 361 10 4
call :do gn_dec_gn_c512_h200w324 "%BUILD%\groupnorm_bench.exe" 512 2 200 324 32 45 10 4
call :do gn_dec_gn_c256_h400w648 "%BUILD%\groupnorm_bench.exe" 256 2 400 648 32 22 10 4
call :do gn_dec_gn_c128_h800w1296 "%BUILD%\groupnorm_bench.exe" 128 2 800 1296 32 20 10 4

REM ---------- vae_decoder: FC (plain f16 matmul) ----------
call :do fc_dec_mid_attn_qkvo "%BUILD%\fc_bench.exe" 16200 512 512 128 10 5 4 f16 64

REM ---------- vae_decoder: attention ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_dec_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 16200 16200 20 5 4 0

echo === END %date% %time% >> "%LOGS%\_index.txt"
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
