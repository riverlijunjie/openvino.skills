@echo off
REM Build & run the custom OpenCL HW probes on the PTL Windows target (per SKILL §4).
REM Run this script ON the remote PTL machine (it expects vcvars to set up cl.exe).

set SDK=C:\Users\Local_Admin\ywang2\OpenCL-SDK
set DST=D:\river\moe\dev_roofline_profiling\utils\hw_probe
if not exist %DST% mkdir %DST%

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

cd /d %DST%
cl /nologo /O2 /DCL_TARGET_OPENCL_VERSION=300 /I %SDK%\include gpu_info.c /Fe:gpu_info.exe /link %SDK%\lib\OpenCL.lib
if errorlevel 1 goto :err
cl /nologo /O2 /DCL_TARGET_OPENCL_VERSION=300 /I %SDK%\include mem_bw.c   /Fe:mem_bw.exe   /link %SDK%\lib\OpenCL.lib
if errorlevel 1 goto :err

echo === gpu_info ===
gpu_info.exe
echo === mem_bw 1 GiB ===
mem_bw.exe 1024 20
echo === mem_bw 2 GiB ===
mem_bw.exe 2048 10
goto :eof
:err
echo BUILD FAILED
exit /b 1
