@echo off
setlocal
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
D:\river\moe\dev_roofline_profiling\utils\build\Release\sdpa_bench.exe > C:\Users\Local_Admin\Downloads\sdpa_test.txt 2>&1
echo EXIT=%errorlevel% >> C:\Users\Local_Admin\Downloads\sdpa_test.txt
