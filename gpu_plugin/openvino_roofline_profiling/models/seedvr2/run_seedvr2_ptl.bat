@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM SeedVR2 roofline sweep on PTL (Machine 1, D:\river\ovmx_dev). Covers the 3
REM submodels (VAE encoder, windowed-attention DiT, VAE decoder) at the real
REM measured resolution (1920x1088-derived 136x240 latent grid, txt=58).
REM Shapes/call-counts are generated from models/seedvr2/ops_mapping.json --
REM regenerate both together via models/seedvr2/generate_bench_config.py, never
REM hand-edit the per-op lines below.
REM ============================================================================

set OV_BIN=D:\river\ovmx_dev\openvino.pipeline.mx\bin\intel64\Release
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\ovmx_dev\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\ovmx_dev\roofline\results\seedvr2
set PATH=%OV_BIN%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- vae_encoder: convolution ----------
call :do conv_enc_conv_in "%BUILD%\conv_bench.exe" plain 3 128 3 3 3 6 1088 1920 1 1 1 37 5 4 1 0 64
call :do conv_enc_down0_conv1 "%BUILD%\conv_bench.exe" plain 128 128 3 3 3 6 1088 1920 1 1 1 37 5 4 1 0 64
call :do conv_enc_down0_downsampler "%BUILD%\conv_bench.exe" plain 128 128 1 3 3 4 1088 1920 1 2 2 149 5 4 1 0 64
call :do conv_enc_down1_shortcut "%BUILD%\conv_bench.exe" plain 128 256 1 1 1 4 544 960 1 1 1 74 5 4 1 0 64
call :do conv_enc_down1_conv1 "%BUILD%\conv_bench.exe" plain 128 256 3 3 3 6 544 960 1 1 1 74 5 4 1 0 64
call :do conv_enc_down1_conv2 "%BUILD%\conv_bench.exe" plain 256 256 3 3 3 6 544 960 1 1 1 74 5 4 1 0 64
call :do conv_enc_down1_downsampler "%BUILD%\conv_bench.exe" plain 256 256 3 3 3 5 544 960 2 2 2 400 5 4 1 0 64
call :do conv_enc_down2_shortcut "%BUILD%\conv_bench.exe" plain 256 512 1 1 1 2 272 480 1 1 1 299 5 4 1 0 64
call :do conv_enc_down2_conv1 "%BUILD%\conv_bench.exe" plain 256 512 3 3 3 4 272 480 1 1 1 299 5 4 1 0 64
call :do conv_enc_down2_conv2 "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 4 272 480 1 1 1 299 5 4 1 0 64
call :do conv_enc_down2_downsampler "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 3 272 480 2 2 2 400 5 4 1 0 64
call :do conv_enc_down3_mid_conv1 "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 3 136 240 1 1 1 400 5 4 1 0 64
call :do conv_enc_conv_out "%BUILD%\conv_bench.exe" plain 512 32 3 3 3 3 136 240 1 1 1 400 5 4 1 0 64

REM ---------- vae_encoder: groupnorm ----------
call :do gn_enc_gn_c128_h1088w1920 "%BUILD%\groupnorm_bench.exe" 128 4 1088 1920 32 20 10 4
call :do gn_enc_gn_c256_h544w960 "%BUILD%\groupnorm_bench.exe" 256 4 544 960 32 20 10 4
call :do gn_enc_gn_c512_h272w480 "%BUILD%\groupnorm_bench.exe" 512 2 272 480 32 22 10 4
call :do gn_enc_gn_c512_h136w240 "%BUILD%\groupnorm_bench.exe" 512 1 136 240 32 179 10 4

REM ---------- vae_encoder: FC (plain f16 matmul) ----------
call :do fc_enc_mid_attn_qkvo "%BUILD%\fc_bench.exe" 32640 512 512 128 10 5 4 f16 64

REM ---------- vae_encoder: attention ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_enc_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 32640 32640 20 5 4 0

REM ---------- dit: FC (plain f16 matmul) ----------
call :do fc_dit_vid_in "%BUILD%\fc_bench.exe" 252960 132 2560 128 10 5 4 f16 64
call :do fc_dit_txt_in "%BUILD%\fc_bench.exe" 58 5120 2560 128 10 5 4 f16 64
call :do fc_dit_vid_qkv "%BUILD%\fc_bench.exe" 252960 2560 7680 128 10 5 3 f16 64
call :do fc_dit_txt_qkv "%BUILD%\fc_bench.exe" 58 2560 7680 128 10 5 4 f16 64
call :do fc_dit_vid_attn_out "%BUILD%\fc_bench.exe" 252960 2560 2560 128 10 5 4 f16 64
call :do fc_dit_txt_attn_out "%BUILD%\fc_bench.exe" 58 2560 2560 128 10 5 4 f16 64
call :do fc_dit_vid_mlp_gate "%BUILD%\fc_bench.exe" 252960 2560 6912 128 10 5 4 f16 64
call :do fc_dit_vid_mlp_up "%BUILD%\fc_bench.exe" 252960 2560 6912 128 10 5 4 f16 64
call :do fc_dit_vid_mlp_out "%BUILD%\fc_bench.exe" 252960 6912 2560 128 10 5 4 f16 64
call :do fc_dit_txt_mlp_gate "%BUILD%\fc_bench.exe" 58 2560 6912 128 10 5 4 f16 64
call :do fc_dit_txt_mlp_up "%BUILD%\fc_bench.exe" 58 2560 6912 128 10 5 4 f16 64
call :do fc_dit_txt_mlp_out "%BUILD%\fc_bench.exe" 58 6912 2560 128 10 5 4 f16 64
call :do fc_dit_vid_out "%BUILD%\fc_bench.exe" 252960 2560 64 128 10 5 4 f16 64

REM ---------- dit: attention ----------
call :do attn_dit_wattn_window_a "%BUILD%\vlsdpa_bench.exe" 128 20 "0,2529,5058,7587,10116,12645,15174,17703,20232,22761,25290,27819,30348,32877,35406,37935,40464,42993,45522,48051,50580,53109,55638,58167,60696,63225,65754,68283,70812,73341,75870,78399,80928,83457,85986,88515,91044,93573,96102,98631,101160,103689,106218,108747,111276,113805,116334,118863,121392,123921,126450,128979,131508,134037,136566,139095,141624,144153,146682,149211,151740,154269,156798,159327,161856,164385,166914,169443,171972,174501,177030,179559,182088,184617,187146,189675,192204,194733,197262,199791,202320,204849,207378,209907,212436,214965,217494,220023,222552,225081,227610,230139,232668,235197,237726,240255,242784,245313,247842,250371,252960" 50 10 3
call :do attn_dit_wattn_window_b "%BUILD%\vlsdpa_bench.exe" 128 20 "0,1686,3372,5058,6744,8430,10116,11802,13488,15174,16860,18546,20232,21918,23604,25290,26976,28662,30348,32034,33720,35406,37092,38778,40464,42150,43836,45522,47208,48894,50580,52266,53952,55638,57324,59010,60696,62382,64068,65754,67440,69126,70812,72498,74184,75870,77556,79242,80928,82614,84300,85986,87672,89358,91044,92730,94416,96102,97788,99474,101160,102846,104532,106218,107904,109590,111276,112962,114648,116334,118020,119706,121392,123078,124764,126450,128136,129822,131508,133194,134880,136566,138252,139938,141624,143310,144996,146682,148368,150054,151740,153426,155112,156798,158484,160170,161856,163542,165228,166914,168600,170286,171972,173658,175344,177030,178716,180402,182088,183774,185460,187146,188832,190518,192204,193890,195576,197262,198948,200634,202320,204006,205692,207378,209064,210750,212436,214122,215808,217494,219180,220866,222552,224238,225924,227610,229296,230982,232668,234354,236040,237726,239412,241098,242784,244470,246156,247842,249528,251214,252960" 50 10 3

REM ---------- vae_decoder: convolution ----------
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

REM ---------- vae_decoder: groupnorm ----------
call :do gn_dec_gn_c512_h72w122 "%BUILD%\groupnorm_bench.exe" 512 1 72 122 32 667 10 4
call :do gn_dec_gn_c512_h144w244 "%BUILD%\groupnorm_bench.exe" 512 2 144 244 32 83 10 4
call :do gn_dec_gn_c256_h288w488 "%BUILD%\groupnorm_bench.exe" 256 2 288 488 32 41 10 4
call :do gn_dec_gn_c128_h576w976 "%BUILD%\groupnorm_bench.exe" 128 2 576 976 32 20 10 4

REM ---------- vae_decoder: FC (plain f16 matmul) ----------
call :do fc_dec_mid_attn_qkvo "%BUILD%\fc_bench.exe" 8784 512 512 128 10 5 4 f16 64

REM ---------- vae_decoder: attention ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_dec_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 8784 8784 20 5 4 0

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
