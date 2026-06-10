@echo off
setlocal EnableExtensions
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%

REM Missing decode FC: up + down. iters=2500 same as gate (already validated).
echo Running fc_up_dense_decode_M1 ...
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 3840 15360 128 2500 100 8 u4 64 > "%LOGS%\fc_up_dense_decode_M1.log" 2>&1
if errorlevel 1 echo FAIL fc_up_dense_decode_M1

echo Running fc_down_dense_decode_M1 ...
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 15360 3840 128 2500 100 8 u4 64 > "%LOGS%\fc_down_dense_decode_M1.log" 2>&1
if errorlevel 1 echo FAIL fc_down_dense_decode_M1

echo Done.
