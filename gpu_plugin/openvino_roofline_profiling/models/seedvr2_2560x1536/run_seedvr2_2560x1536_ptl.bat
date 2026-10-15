@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM SeedVR2 roofline sweep on PTL (Machine 1, D:\river\ovmx_dev) -- 2560x1536
REM variant, VAE decoder UNTILED (full-frame, no 2x2 tile grid), per explicit
REM request. Shapes/call-counts are generated from
REM models/seedvr2_2560x1536/ops_mapping.json -- regenerate both together via
REM models/seedvr2_2560x1536/generate_bench_config.py, never hand-edit below.
REM ============================================================================

set OV_BIN=D:\river\ovmx_dev\openvino.pipeline.mx\bin\intel64\Release
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\ovmx_dev\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\ovmx_dev\roofline\results\seedvr2_2560x1536
set PATH=%OV_BIN%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- vae_encoder: convolution ----------
call :do conv_enc_conv_in "%BUILD%\conv_bench.exe" plain 3 128 3 3 3 6 1536 2560 1 1 1 19 5 4 1 0 64
call :do conv_enc_down0_conv1 "%BUILD%\conv_bench.exe" plain 128 128 3 3 3 6 1536 2560 1 1 1 19 5 4 1 0 64
call :do conv_enc_down0_downsampler "%BUILD%\conv_bench.exe" plain 128 128 1 3 3 4 1536 2560 1 2 2 79 5 4 1 0 64
call :do conv_enc_down1_shortcut "%BUILD%\conv_bench.exe" plain 128 256 1 1 1 4 768 1280 1 1 1 39 5 4 1 0 64
call :do conv_enc_down1_conv1 "%BUILD%\conv_bench.exe" plain 128 256 3 3 3 6 768 1280 1 1 1 39 5 4 1 0 64
call :do conv_enc_down1_conv2 "%BUILD%\conv_bench.exe" plain 256 256 3 3 3 6 768 1280 1 1 1 39 5 4 1 0 64
call :do conv_enc_down1_downsampler "%BUILD%\conv_bench.exe" plain 256 256 3 3 3 5 768 1280 2 2 2 317 5 4 1 0 64
call :do conv_enc_down2_shortcut "%BUILD%\conv_bench.exe" plain 256 512 1 1 1 2 384 640 1 1 1 158 5 4 1 0 64
call :do conv_enc_down2_conv1 "%BUILD%\conv_bench.exe" plain 256 512 3 3 3 4 384 640 1 1 1 158 5 4 1 0 64
call :do conv_enc_down2_conv2 "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 4 384 640 1 1 1 158 5 4 1 0 64
call :do conv_enc_down2_downsampler "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 3 384 640 2 2 2 400 5 4 1 0 64
call :do conv_enc_down3_mid_conv1 "%BUILD%\conv_bench.exe" plain 512 512 3 3 3 3 192 320 1 1 1 400 5 4 1 0 64
call :do conv_enc_conv_out "%BUILD%\conv_bench.exe" plain 512 32 3 3 3 3 192 320 1 1 1 400 5 4 1 0 64

REM ---------- vae_encoder: groupnorm ----------
call :do gn_enc_gn_c128_h1536w2560 "%BUILD%\groupnorm_bench.exe" 128 4 1536 2560 32 20 10 4
call :do gn_enc_gn_c256_h768w1280 "%BUILD%\groupnorm_bench.exe" 256 4 768 1280 32 20 10 4
call :do gn_enc_gn_c512_h384w640 "%BUILD%\groupnorm_bench.exe" 512 2 384 640 32 20 10 4
call :do gn_enc_gn_c512_h192w320 "%BUILD%\groupnorm_bench.exe" 512 1 192 320 32 95 10 4

REM ---------- vae_encoder: FC (plain f16 matmul) ----------
call :do fc_enc_mid_attn_qkvo "%BUILD%\fc_bench.exe" 61440 512 512 128 10 5 4 f16 64

REM ---------- vae_encoder: attention ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_enc_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 61440 61440 20 5 4 0

REM ---------- dit: FC (plain f16 matmul) ----------
call :do fc_dit_vid_in "%BUILD%\fc_bench.exe" 476160 132 2560 128 10 5 4 f16 64
call :do fc_dit_txt_in "%BUILD%\fc_bench.exe" 58 5120 2560 128 10 5 4 f16 64
call :do fc_dit_vid_qkv "%BUILD%\fc_bench.exe" 476160 2560 7680 128 10 5 1 f16 64
call :do fc_dit_txt_qkv "%BUILD%\fc_bench.exe" 58 2560 7680 128 10 5 4 f16 64
call :do fc_dit_vid_attn_out "%BUILD%\fc_bench.exe" 476160 2560 2560 128 10 5 3 f16 64
call :do fc_dit_txt_attn_out "%BUILD%\fc_bench.exe" 58 2560 2560 128 10 5 4 f16 64
call :do fc_dit_vid_mlp_gate "%BUILD%\fc_bench.exe" 476160 2560 6912 128 10 5 2 f16 64
call :do fc_dit_vid_mlp_up "%BUILD%\fc_bench.exe" 476160 2560 6912 128 10 5 2 f16 64
call :do fc_dit_vid_mlp_out "%BUILD%\fc_bench.exe" 476160 6912 2560 128 10 5 2 f16 64
call :do fc_dit_txt_mlp_gate "%BUILD%\fc_bench.exe" 58 2560 6912 128 10 5 4 f16 64
call :do fc_dit_txt_mlp_up "%BUILD%\fc_bench.exe" 58 2560 6912 128 10 5 4 f16 64
call :do fc_dit_txt_mlp_out "%BUILD%\fc_bench.exe" 58 6912 2560 128 10 5 4 f16 64
call :do fc_dit_vid_out "%BUILD%\fc_bench.exe" 476160 2560 64 128 10 5 4 f16 64

REM ---------- dit: attention ----------
call :do attn_dit_wattn_window_a "%BUILD%\vlsdpa_bench.exe" 128 20 "0,2532,5064,7596,10128,12660,15192,17724,20256,22788,25320,27852,30384,32916,35448,37980,40512,43044,45576,48108,50640,53172,55704,58236,60768,63300,65832,68364,70896,73428,75960,78492,81024,83556,86088,88620,91152,93684,96216,98748,101280,103812,106344,108876,111408,113940,116472,119004,121536,124068,126600,129132,131664,134196,136728,139260,141792,144324,146856,149388,151920,154452,156984,159516,162048,164580,167112,169644,172176,174708,177240,179772,182304,184836,187368,189900,192432,194964,197496,200028,202560,205092,207624,210156,212688,215220,217752,220284,222816,225348,227880,230412,232944,235476,238008,240540,243072,245604,248136,250668,253200,255732,258264,260796,263328,265860,268392,270924,273456,275988,278520,281052,283584,286116,288648,291180,293712,296244,298776,301308,303840,306372,308904,311436,313968,316500,319032,321564,324096,326628,329160,331692,334224,336756,339288,341820,344352,346884,349416,351948,354480,357012,359544,362076,364608,367140,369672,372204,374736,377268,379800,382332,384864,387396,389928,392460,394992,397524,400056,402588,405120,407652,410184,412716,415248,417780,420312,422844,425376,427908,430440,432972,435504,438036,440568,443100,445632,448164,450696,453228,455760,458292,460824,463356,465888,468420,470952,473484,476160" 50 10 1
call :do attn_dit_wattn_window_b "%BUILD%\vlsdpa_bench.exe" 128 20 "0,1688,3376,5064,6752,8440,10128,11816,13504,15192,16880,18568,20256,21944,23632,25320,27008,28696,30384,32072,33760,35448,37136,38824,40512,42200,43888,45576,47264,48952,50640,52328,54016,55704,57392,59080,60768,62456,64144,65832,67520,69208,70896,72584,74272,75960,77648,79336,81024,82712,84400,86088,87776,89464,91152,92840,94528,96216,97904,99592,101280,102968,104656,106344,108032,109720,111408,113096,114784,116472,118160,119848,121536,123224,124912,126600,128288,129976,131664,133352,135040,136728,138416,140104,141792,143480,145168,146856,148544,150232,151920,153608,155296,156984,158672,160360,162048,163736,165424,167112,168800,170488,172176,173864,175552,177240,178928,180616,182304,183992,185680,187368,189056,190744,192432,194120,195808,197496,199184,200872,202560,204248,205936,207624,209312,211000,212688,214376,216064,217752,219440,221128,222816,224504,226192,227880,229568,231256,232944,234632,236320,238008,239696,241384,243072,244760,246448,248136,249824,251512,253200,254888,256576,258264,259952,261640,263328,265016,266704,268392,270080,271768,273456,275144,276832,278520,280208,281896,283584,285272,286960,288648,290336,292024,293712,295400,297088,298776,300464,302152,303840,305528,307216,308904,310592,312280,313968,315656,317344,319032,320720,322408,324096,325784,327472,329160,330848,332536,334224,335912,337600,339288,340976,342664,344352,346040,347728,349416,351104,352792,354480,356168,357856,359544,361232,362920,364608,366296,367984,369672,371360,373048,374736,376424,378112,379800,381488,383176,384864,386552,388240,389928,391616,393304,394992,396680,398368,400056,401744,403432,405120,406808,408496,410184,411872,413560,415248,416936,418624,420312,422000,423688,425376,427064,428752,430440,432128,433816,435504,437192,438880,440568,442256,443944,445632,447320,449008,450696,452384,454072,455760,457448,459136,460824,462512,464200,465888,467576,469264,470952,472640,474328,476160" 50 10 1

REM ---------- vae_decoder: convolution ----------
call :do conv_dec_conv_in "%BUILD%\conv_bench.exe" causal 16 512 3 3 3 1 192 320 1 1 1 400 5 4 1 0 64
call :do conv_dec_mid_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 192 320 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 1 192 320 1 1 1 400 5 4 1 0 64
call :do conv_dec_up0_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 192 320 1 1 1 158 5 4 1 0 64
call :do conv_dec_up0_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 384 640 1 1 1 158 5 4 1 0 64
call :do conv_dec_up1_conv1 "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 384 640 1 1 1 158 5 4 1 0 64
call :do conv_dec_up1_upscale_conv "%BUILD%\conv_bench.exe" plain 512 4096 1 1 1 1 384 640 1 1 1 39 5 4 1 0 64
call :do conv_dec_up1_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 512 512 3 3 3 2 768 1280 1 1 1 39 5 4 1 0 64
call :do conv_dec_up2_shortcut "%BUILD%\conv_bench.exe" plain 512 256 1 1 1 2 768 1280 1 1 1 79 5 4 1 0 64
call :do conv_dec_up2_conv1 "%BUILD%\conv_bench.exe" causal 512 256 3 3 3 2 768 1280 1 1 1 79 5 4 1 0 64
call :do conv_dec_up2_conv2 "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 768 1280 1 1 1 79 5 4 1 0 64
call :do conv_dec_up2_post_shuffle_conv "%BUILD%\conv_bench.exe" causal 256 256 3 3 3 2 1536 2560 1 1 1 19 5 4 1 0 64
call :do conv_dec_up3_shortcut "%BUILD%\conv_bench.exe" plain 256 128 1 1 1 2 1536 2560 1 1 1 39 5 4 1 0 64
call :do conv_dec_up3_conv1 "%BUILD%\conv_bench.exe" causal 256 128 3 3 3 2 1536 2560 1 1 1 39 5 4 1 0 64
call :do conv_dec_up3_conv2 "%BUILD%\conv_bench.exe" causal 128 128 3 3 3 2 1536 2560 1 1 1 39 5 4 1 0 64
call :do conv_dec_conv_out "%BUILD%\conv_bench.exe" causal 128 3 3 3 3 2 1536 2560 1 1 1 400 5 4 1 0 64

REM ---------- vae_decoder: groupnorm ----------
call :do gn_dec_gn_c512_h192w320 "%BUILD%\groupnorm_bench.exe" 512 1 192 320 32 95 10 4
call :do gn_dec_gn_c512_h384w640 "%BUILD%\groupnorm_bench.exe" 512 2 384 640 32 20 10 4
call :do gn_dec_gn_c256_h768w1280 "%BUILD%\groupnorm_bench.exe" 256 2 768 1280 32 20 10 4
call :do gn_dec_gn_c128_h1536w2560 "%BUILD%\groupnorm_bench.exe" 128 2 1536 2560 32 20 10 4

REM ---------- vae_decoder: FC (plain f16 matmul) ----------
call :do fc_dec_mid_attn_qkvo "%BUILD%\fc_bench.exe" 61440 512 512 128 10 5 4 f16 64

REM ---------- vae_decoder: attention ----------
set SDPA_NH=1&& set SDPA_NKV=1&& set SDPA_HD=512&& call :do attn_dec_mid_attn_sdpa "%BUILD%\sdpa_bench.exe" prefill 61440 61440 20 5 4 0

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
