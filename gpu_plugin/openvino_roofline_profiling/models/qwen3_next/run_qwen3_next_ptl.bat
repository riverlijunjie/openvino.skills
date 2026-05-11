@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qwen3-Next-80B-A3B roofline sweep on PTL (iGPU, 2400 MHz, 110 GB/s, FP16 XMX 58.98 TFLOPS).
REM Only the parts that differ from qwen3_5_moe are re-run here:
REM   - MoE: NE=512, TK=10 (vs qwen3_5_moe NE=256, TK=8). Same H=2048, I=512, SI=512.
REM   - LM_Head: vocab=151936 (vs qwen3_5_moe vocab=248320).
REM Everything else (fc_qkv 5120, fc_o 4096->2048, linattn 12288, PA NH=16/NKV=2/HD=256,
REM gdn HK=32/K=V=128, small_ops H=2048 NH/NKV=16/2 HD=256) has identical shapes
REM and is reused from qwen3_5_moe ptl logs.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_next\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- MoE fused (with shared expert): H=2048 I=512 NE=512 TK=10 ----------
REM moe_bench args: B M H I NE TK GROUP iters warmup bufs MoE_iters_inner shared_I
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1     2048 512 512 10 128 100 10 4 64 512
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024  2048 512 512 10 128  20  5 2 64 512
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048  2048 512 512 10 128  15  3 2 64 512
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096  2048 512 512 10 128  10  2 2 64 512
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192  2048 512 512 10 128   8  2 2 64 512
call :do moe_prefill_S16384           "%BUILD%\moe_bench.exe" 1 16384 2048 512 512 10 128   4  1 1 64 512
call :do moe_prefill_S32768           "%BUILD%\moe_bench.exe" 1 32768 2048 512 512 10 128   2  1 1 64 512
call :do moe_prefill_S65536           "%BUILD%\moe_bench.exe" 1 65536 2048 512 512 10 128   2  1 1 64 512
call :do moe_prefill_S131072          "%BUILD%\moe_bench.exe" 1 131072 2048 512 512 10 128  1  1 1 64 512

REM ---------- LM head (INT8 g=128, vocab=151936) ----------
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1 2048 151936 128 300 30 4 u8 64

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
