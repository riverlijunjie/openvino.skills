@echo off
setlocal EnableExtensions EnableDelayedExpansion

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_8b\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

"%CLI%" -d "%BUILD%\fc_bench.exe" 1  4096  4096 128 5000 200 8 u4 64 > "%LOGS%\fc_o_decode_M1.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1  4096 12288 128 2500 150 8 u4 64 > "%LOGS%\fc_up_decode_M1.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1  4096 12288 128 2500 150 8 u4 64 > "%LOGS%\fc_gate_decode_M1.log" 2>&1
echo DONE
