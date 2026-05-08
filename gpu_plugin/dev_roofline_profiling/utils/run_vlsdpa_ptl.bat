@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM VLSDPA bench sweep on PTL (Xe2 iGPU). VLSDPA = variable-length SDPA used
REM in Qwen2-VL / Qwen2.5-VL ViT encoders. The GPU plugin only ships a CM
REM implementation (cm_sdpa_vlen.cm), so this exclusively exercises the CM
REM kernel path.
REM
REM Args: <head_size> <num_heads> <cu_seqlens_csv>
REM   - Qwen2.5-VL ViT default: head_size=80, num_heads=16
REM   - cu_seqlens models per-image attention windows; the kernel runs one
REM     block-diagonal attention per consecutive cu_seqlens pair.
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\vlsdpa\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- Qwen2.5-VL ViT defaults: 80x16, single-image windows ----------
call :do vlsdpa_h80_n16_w256        "%BUILD%\vlsdpa_bench.exe" 80 16 "0,256"          200 20 4
call :do vlsdpa_h80_n16_w512        "%BUILD%\vlsdpa_bench.exe" 80 16 "0,512"          200 20 4
call :do vlsdpa_h80_n16_w1024       "%BUILD%\vlsdpa_bench.exe" 80 16 "0,1024"         150 15 4
call :do vlsdpa_h80_n16_w2048       "%BUILD%\vlsdpa_bench.exe" 80 16 "0,2048"         100 10 4
call :do vlsdpa_h80_n16_w4096       "%BUILD%\vlsdpa_bench.exe" 80 16 "0,4096"          50  5 2
call :do vlsdpa_h80_n16_w8192       "%BUILD%\vlsdpa_bench.exe" 80 16 "0,8192"          25  5 2

REM ---------- Multi-window (multi-image) cases ----------
call :do vlsdpa_h80_n16_2x1024      "%BUILD%\vlsdpa_bench.exe" 80 16 "0,1024,2048"    100 10 4
call :do vlsdpa_h80_n16_4x512       "%BUILD%\vlsdpa_bench.exe" 80 16 "0,512,1024,1536,2048" 100 10 4

REM ---------- Reference test sizes for kernel sanity (matches vlsdpa_gpu_test) ---
call :do vlsdpa_h64_n1_w16          "%BUILD%\vlsdpa_bench.exe" 64  1 "0,16"           500 50 4
call :do vlsdpa_h72_n2_w32          "%BUILD%\vlsdpa_bench.exe" 72  2 "0,16,32"        500 50 4
call :do vlsdpa_h128_n1_w16         "%BUILD%\vlsdpa_bench.exe" 128 1 "0,16"           500 50 4

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
