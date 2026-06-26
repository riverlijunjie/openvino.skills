@echo off
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
echo --- run with no args (expect usage, rc=1) ---
D:\river\moe\dev_roofline_profiling\utils\build\Release\fc_bench.exe
echo NOARG_RC=%errorlevel%
echo --- run tiny GPU fc ---
D:\river\moe\dev_roofline_profiling\utils\build\Release\fc_bench.exe 1 1024 1024 128 5 2 2 u4 0
echo BENCH_RC=%errorlevel%
