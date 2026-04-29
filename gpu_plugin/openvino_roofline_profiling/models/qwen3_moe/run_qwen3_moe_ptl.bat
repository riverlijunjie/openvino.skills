@echo off
setlocal EnableExtensions EnableDelayedExpansion

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_moe\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2048 768 128 8 128  100 10 4 64
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2048 768 128 8 128   20  5 2 64
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2048 768 128 8 128   15  3 2 64
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2048 768 128 8 128   10  2 2 64
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2048 768 128 8 128    8  2 2 64
call :do moe_prefill_S16384           "%BUILD%\moe_bench.exe" 1 16384 2048 768 128 8 128   5  1 1 64
call :do moe_prefill_S32768           "%BUILD%\moe_bench.exe" 1 32768 2048 768 128 8 128   3  1 1 64
call :do moe_prefill_S65536           "%BUILD%\moe_bench.exe" 1 65536 2048 768 128 8 128   2  1 1 64
call :do moe_prefill_S131072          "%BUILD%\moe_bench.exe" 1 131072 2048 768 128 8 128  2  1 1 64

call :do fc_qkv_decode_M1             "%BUILD%\fc_bench.exe" 1    2048   5120 128 5000 200 8 u4 64
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1    4096   2048 128 5000 200 8 u4 64
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1    2048 151936 128  500  50 8 u8 64

for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S 2048 5120 128 20 5 4 u4 64
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 4096 2048 128 20 5 4 u4 64
)

for %%K in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 5000 200 4 i8
)
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
  call :do pa_prefill_S%%S            "%BUILD%\pa_bench.exe" prefill %%S 0 20 3 2 i8
)

call :do so_rmsnorm_h2048_decode      "%BUILD%\small_ops_bench.exe" rmsnorm   1 2048 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_qnorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 32 128 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  4 128 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_decode             "%BUILD%\small_ops_bench.exe" rope      1 32 128 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_decode             "%BUILD%\small_ops_bench.exe" rope      1  4 128 --iters 30000 --warmup 300 --bufs 8
call :do so_add_decode                "%BUILD%\small_ops_bench.exe" add       1 2048 --iters 30000 --warmup 300 --bufs 8

for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2048_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm %%S 2048   --iters 300 --warmup 30 --bufs 4
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
