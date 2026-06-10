@echo off
setlocal EnableExtensions
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set PATH=%OV_BIN%;%TBB%;%PATH%
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 4096 3840 128 12 3 2 u4 64 > "%LOGS%\fc_o_sliding_prefill_S8192.log" 2>&1
echo Done
