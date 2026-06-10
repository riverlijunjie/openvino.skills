@echo off
setlocal
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
set LOGS=D:\river\moe\roofline_results\gemma4_12B\ptl_12xe
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set FC=D:\river\moe\dev_roofline_profiling\utils\build\Release\fc_bench.exe
"%CLI%" -d "%FC%" 912 4096 3840 128 40 8 4 u4 64 > "%LOGS%\fc_o_sliding_prefill_S912.log" 2>&1
echo EXIT=%errorlevel% >> "%LOGS%\fc_o_sliding_prefill_S912.log"
