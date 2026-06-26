@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  diffusion_gemma SDPA re-run — after rebuilding sdpa_bench with GQA
REM  repeat_kv KV-head expansion (NH=16, NKV=8 sliding / NKV=2 full).
REM  Overwrites the previously FATAL sdpa_*.log files.
REM ========================================================================
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\diffusion_gemma\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%
echo === SDPA RERUN START %date% %time% >> "%LOGS%\_index.txt"

REM ---- Sliding (NH=16, NKV=8, HD=256) ----
set SDPA_NH=16
set SDPA_NKV=8
set SDPA_HD=256
call :do sdpa_sliding_prefill_S1024 "%BUILD%\sdpa_bench.exe" prefill 1024 1024 100 10 4 1
call :do sdpa_sliding_canvas_M256   "%BUILD%\sdpa_bench.exe" prefill 256  1280 400 40 4 0
call :do sdpa_sliding_decode_M1     "%BUILD%\sdpa_bench.exe" decode  1    1024 3000 200 4 0

REM ---- Full (NH=16, NKV=2, HD=512) ----
set SDPA_NH=16
set SDPA_NKV=2
set SDPA_HD=512
for %%S in (1024 2048 4096 8192) do (
  call :do sdpa_full_prefill_S%%S "%BUILD%\sdpa_bench.exe" prefill %%S %%S 30 5 2 1
)
call :do sdpa_full_canvas_ctx1024 "%BUILD%\sdpa_bench.exe" prefill 256 1280 400 40 4 0
call :do sdpa_full_canvas_ctx2048 "%BUILD%\sdpa_bench.exe" prefill 256 2304 300 30 4 0
call :do sdpa_full_canvas_ctx4096 "%BUILD%\sdpa_bench.exe" prefill 256 4352 200 20 4 0
call :do sdpa_full_canvas_ctx8192 "%BUILD%\sdpa_bench.exe" prefill 256 8448 150 15 4 0
call :do sdpa_full_decode_ctx1024 "%BUILD%\sdpa_bench.exe" decode 1 1024 3000 200 4 0
call :do sdpa_full_decode_ctx2048 "%BUILD%\sdpa_bench.exe" decode 1 2048 2500 150 4 0
call :do sdpa_full_decode_ctx4096 "%BUILD%\sdpa_bench.exe" decode 1 4096 2000 100 4 0
call :do sdpa_full_decode_ctx8192 "%BUILD%\sdpa_bench.exe" decode 1 8192 1500 100 4 0

echo === SDPA RERUN END %date% %time% >> "%LOGS%\_index.txt"
echo Done.
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
echo [%date% %time%] Running !TAG! ...
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
