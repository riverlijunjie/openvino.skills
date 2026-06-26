@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ========================================================================
REM  diffusion_gemma SUPPLEMENT — re-run group-size-constrained ops at g=64
REM    MoE down-proj K=I=704 (not divisible by 128)  -> g=64
REM    Dense MLP down-proj K=2112 (not divisible by 128) -> g=64
REM  (matches the gemma4-26B-A4B reference which used MoE g=64)
REM ========================================================================
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\diffusion_gemma\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%
if not exist "%LOGS%" mkdir "%LOGS%"
echo === SUPPLEMENT START %date% %time% >> "%LOGS%\_index.txt"

REM ---- MoE (g=64) ----
call :do moe_decode_M1     "%BUILD%\moe_bench.exe" 1 1    2816 704 128 8 64  500 50 4 64
call :do moe_canvas_M256   "%BUILD%\moe_bench.exe" 1 256  2816 704 128 8 64   80 10 4 64
call :do moe_prefill_S1024 "%BUILD%\moe_bench.exe" 1 1024 2816 704 128 8 64   20  5 2 64
call :do moe_prefill_S2048 "%BUILD%\moe_bench.exe" 1 2048 2816 704 128 8 64   15  3 2 64
call :do moe_prefill_S4096 "%BUILD%\moe_bench.exe" 1 4096 2816 704 128 8 64   10  2 2 64
call :do moe_prefill_S8192 "%BUILD%\moe_bench.exe" 1 8192 2816 704 128 8 64    8  2 2 64

REM ---- Dense MLP down (K=2112, g=64) ----
call :do fc_down_dense_decode_M1   "%BUILD%\fc_bench.exe" 1   2112 2816 64 15000 500 8 u4 64
call :do fc_down_dense_canvas_M256 "%BUILD%\fc_bench.exe" 256 2112 2816 64  1500 100 8 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_down_dense_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2112 2816 64 20 5 4 u4 64
)

echo === SUPPLEMENT END %date% %time% >> "%LOGS%\_index.txt"
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
