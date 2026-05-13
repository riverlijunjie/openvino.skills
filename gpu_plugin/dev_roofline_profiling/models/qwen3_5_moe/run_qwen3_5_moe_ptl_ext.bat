@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qwen3.5-MoE extension sweep on PTL: S/kv = 16K, 32K, 64K, 128K.
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === EXT START %date% %time% >> "%LOGS%\_index.txt"

REM ---------- MoE prefill (NE=256, TK=8, shared_I=512) ----------
call :do moe_prefill_S16384  "%BUILD%\moe_bench.exe" 1 16384  2048 512 256 8 128 4 1 1 64 512
call :do moe_prefill_S32768  "%BUILD%\moe_bench.exe" 1 32768  2048 512 256 8 128 2 1 1 64 512
call :do moe_prefill_S65536  "%BUILD%\moe_bench.exe" 1 65536  2048 512 256 8 128 2 1 1 64 512
call :do moe_prefill_S131072 "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 128 1 1 1 64 512

REM ---------- FC prefill ----------
for %%S in (16384 32768 65536 131072) do (
  call :do fc_qkv_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 2048   5120 128 10 2 2 u4 64
  call :do fc_o_prefill_S%%S            "%BUILD%\fc_bench.exe" %%S 4096   2048 128 10 2 2 u4 64
  call :do fc_linattn_proj_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2048  12288 128  6 2 2 u4 64
)

REM ---------- PA decode (NH=16 NKV=2 HD=256) ----------
set PA_NH=16
set PA_NKV=2
set PA_HD=256
for %%K in (16384 32768 65536 131072) do (
  call :do pa_decode_kv%%K "%BUILD%\pa_bench.exe" decode 1 %%K 2000 50 4 i8
)

REM ---------- PA prefill ----------
call :do pa_prefill_S16384  "%BUILD%\pa_bench.exe" prefill 16384  0 5 1 2 i8
call :do pa_prefill_S32768  "%BUILD%\pa_bench.exe" prefill 32768  0 3 1 2 i8
call :do pa_prefill_S65536  "%BUILD%\pa_bench.exe" prefill 65536  0 2 1 1 i8
call :do pa_prefill_S131072 "%BUILD%\pa_bench.exe" prefill 131072 0 1 1 1 i8

REM ---------- GDN prefill ----------
for %%S in (16384 32768 65536 131072) do (
  call :do gdn_prefill_S%%S "%BUILD%\gdn_bench.exe" 1 %%S 32 32 128 8 2 2
)

echo === EXT END %date% %time% >> "%LOGS%\_index.txt"
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
