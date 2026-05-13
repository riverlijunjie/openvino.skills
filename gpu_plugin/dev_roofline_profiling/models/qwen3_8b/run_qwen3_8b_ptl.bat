@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qwen3-8B (dense) full roofline sweep on PTL (Arc B390 iGPU, 2400 MHz, ~94 GB/s, 58.98 FP16 TFLOPS).
REM NL=36 H=4096 I=12288 NH=32 NKV=8 HD=128 vocab=151936
REM QKV concat = (NH+2*NKV)*HD = 6144;  O = NH*HD x H = 4096x4096
REM pa_bench/small_ops_bench defaults already match Qwen3-8B (NH=32 NKV=8 HD=128, H=4096, I=12288).

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_8b\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- fc_bench: QKV / O / MLP / LM_Head (decode M=1, INT4 g=128) ----------
call :do fc_qkv_decode_M1             "%BUILD%\fc_bench.exe" 1     4096   6144  128 5000 200 8 u4 64
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1     4096   4096  128 5000 200 8 u4 64
call :do fc_gate_decode_M1            "%BUILD%\fc_bench.exe" 1     4096  12288  128 2500 150 8 u4 64
call :do fc_up_decode_M1              "%BUILD%\fc_bench.exe" 1     4096  12288  128 2500 150 8 u4 64
call :do fc_down_decode_M1            "%BUILD%\fc_bench.exe" 1    12288   4096  128 2500 150 8 u4 64
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1     4096 151936  128  500  50 8 u8 64

REM ---------- fc_bench: prefill (M=S, INT4 g=128) ----------
for %%S in (1024 2048 4096 8192 16384 32768) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S  4096   6144  128 20 5 4 u4 64
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S  4096   4096  128 20 5 4 u4 64
  call :do fc_gate_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S  4096  12288  128 20 5 4 u4 64
  call :do fc_up_prefill_S%%S         "%BUILD%\fc_bench.exe" %%S  4096  12288  128 20 5 4 u4 64
  call :do fc_down_prefill_S%%S       "%BUILD%\fc_bench.exe" %%S 12288   4096  128 20 5 4 u4 64
)

REM ---------- pa_bench (NH=32 NKV=8 HD=128 INT8 KV) ----------
for %%K in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 5000 200 4 i8
)
for %%S in (1024 2048 4096 8192 16384 32768) do (
  call :do pa_prefill_S%%S            "%BUILD%\pa_bench.exe" prefill %%S 0 20 3 2 i8
)

REM ---------- small ops (Qwen3-8B shapes) ----------
call :do so_rmsnorm_h4096_decode      "%BUILD%\small_ops_bench.exe" rmsnorm   1 4096    --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_qnorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128  --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  8 128  --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_decode             "%BUILD%\small_ops_bench.exe" rope      1 32 128  --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_decode             "%BUILD%\small_ops_bench.exe" rope      1  8 128  --iters 30000 --warmup 300 --bufs 8
call :do so_add_decode                "%BUILD%\small_ops_bench.exe" add       1 4096    --iters 30000 --warmup 300 --bufs 8
call :do so_swish_decode              "%BUILD%\small_ops_bench.exe" swish     1 12288   --iters 30000 --warmup 300 --bufs 8
call :do so_multiply_decode           "%BUILD%\small_ops_bench.exe" multiply  1 12288   --iters 30000 --warmup 300 --bufs 8

for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h4096_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm %%S 4096    --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_prefill_S%%S        "%BUILD%\small_ops_bench.exe" rope    %%S 32 128 --iters 300 --warmup 30 --bufs 4
)

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
