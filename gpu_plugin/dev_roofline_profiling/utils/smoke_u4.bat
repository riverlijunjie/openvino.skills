@echo off
setlocal EnableExtensions EnableDelayedExpansion
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set PATH=%OV_BIN%;%TBB%;%PATH%
set PA_NH=32
set PA_NKV=4
set PA_HD=512
echo ===== u4 DECODE smoke =====
"%BUILD%\pa_bench.exe" decode 1 2048 30 5 2 u4 ocl < nul
echo u4_decode_rc=!errorlevel!
echo ===== u4 PREFILL smoke =====
"%BUILD%\pa_bench.exe" prefill 1024 0 5 2 2 u4 ocl < nul
echo u4_prefill_rc=!errorlevel!
