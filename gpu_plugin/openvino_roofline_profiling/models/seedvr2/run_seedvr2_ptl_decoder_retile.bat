@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM SeedVR2 decoder RE-TILE re-measurement on PTL (Machine 1, D:\river\ovmx_dev).
REM The real pipeline tiles the VAE decoder into a 2x2 spatial grid (core+halo
REM per tile, NOT the full 1920x1088-derived frame x4) -- see
REM models/seedvr2/ops_mapping.json's 2026-10-15 notes. This script re-measures
REM ONLY the 22 vae_decoder conv/groupnorm/fc/attention specs at their corrected
REM per-tile shapes (encoder/DiT specs are unchanged, not re-run). Logs reuse
REM the SAME tags as the full sweep so parse_logs.py output overwrites the
REM previously-wrong (full-frame) decoder entries in place.
REM ============================================================================

set OV_BIN=D:\river\ovmx_dev\openvino.pipeline.mx\bin\intel64\Release
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\ovmx_dev\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\ovmx_dev\roofline\results\seedvr2
set PATH=%OV_BIN%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === RETILE START %date% %time% >> "%LOGS%\_index.txt"

REM ---------- vae_decoder: convolution (real per-tile shapes) ----------
call :do conv_dec_conv_in "%BUILD%\conv_bench.exe" causal 16 512 3 3 3 1 72 122 1 1 1 400 5 4 1 0 64
call :do conv_dec_mid_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 72 122 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 72 122 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 72 122 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 144 244 1 1 1 400 5 4 1 0 64
call :do conv_dec_up1_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 144 244 1 1 1 400 5 4 1 0 64
call :do conv_dec_up1_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 144 244 1 1 1 277 5 4 1 0 64
call :do conv_dec_up1_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 288 488 1 1 1 277 5 4 1 0 64
call :do conv_dec_up2_shortcut "%BUILD%\conv_bench.exe" plain 512 256 1 1 1 2 288 488 1 1 1 400 5 4 1 0 64
call :do conv_dec_up2_conv1 "%BUILD%\conv_bench.exe" causal 512 256 3 3 3 2 288 488 1 1 1 400 5 4 1 0 64
call :do conv_dec_up2_conv2 "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 288 488 1 1 1 400 5 4 1 0 64
call :do conv_dec_up2_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 576 976 1 1 1 138 5 4 1 0 64
call :do conv_dec_up3_shortcut "%BUILD%\conv_bench.exe" plain 256 128 1 1 1 2 576 976 1 1 1 277 5 4 1 0 64
call :do conv_dec_up3_conv1 "%BUILD%\conv_bench.exe" causal 256 128 3 3 3 2 576 976 1 1 1 277 5 4 1 0 64
call :do conv_dec_up3_conv2 "%BUILD%\conv_bench.exe" causal 128 128 3 3 3 2 576 976 1 1 1 277 5 4 1 0 64
call :do conv_dec_conv_out "%BUILD%\conv_bench.exe" causal 128 3 3 3 3 2 576 976 1 1 1 400 5 4 1 0 64

REM ---------- vae_decoder: groupnorm (real per-tile shapes, renamed tags) ----------
call :do gn_dec_gn_c512_h72w122 "%BUILD%\groupnorm_bench.exe" 512 1 72 122 32 667 10 4
call :do gn_dec_gn_c512_h144w244 "%BUILD%\groupnorm_bench.exe" 512 2 144 244 32 83 10 4
call :do gn_dec_gn_c256_h288w488 "%BUILD%\groupnorm_bench.exe" 256 2 288 488 32 41 10 4
call :do gn_dec_gn_c128_h576w976 "%BUILD%\groupnorm_bench.exe" 128 2 576 976 32 20 10 4

REM ---------- vae_decoder: FC (plain f16 matmul, real per-tile token count) ----------
call :do fc_dec_mid_attn_qkvo "%BUILD%\fc_bench.exe" 8784 512 512 128 10 5 4 f16 64

REM ---------- vae_decoder: attention (real per-tile token count) ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_dec_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 8784 8784 20 5 4 0

echo === RETILE END %date% %time% >> "%LOGS%\_index.txt"
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
