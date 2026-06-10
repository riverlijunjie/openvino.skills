@echo off
setlocal
set PATH=D:\river\moe\openvino\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
set PA_NH=16
set PA_NKV=8
set PA_HD=256
D:\river\moe\dev_roofline_profiling\utils\build\Release\pa_bench.exe decode 1 1024 100 10 4 i8 ocl > C:\Users\Local_Admin\Downloads\pa_test.txt 2>&1
echo EXIT=%errorlevel% >> C:\Users\Local_Admin\Downloads\pa_test.txt
