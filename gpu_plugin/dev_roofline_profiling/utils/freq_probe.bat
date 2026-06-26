@echo off
REM GPU frequency-stability probe: fixed compute-bound INT8-XMX prefill GEMM.
REM Kept at group_size=128 so the mean kernel time is directly comparable to the
REM prior g128 baseline reference (~2.83 ms). Arg %1 = output log path.
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=C:\Users\Local_Admin\river_roofline\utils\build\Release
set PATH=%OV_BIN%;%TBB%;%PATH%
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 9216 128 30 5 8 u4 64 > "%~1" 2>&1
