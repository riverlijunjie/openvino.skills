@echo off
rem ============================================================================
rem Build the standalone sdpa_micro__generate test on the remote PTL (Xe3) box
rem with MSVC, against the consistent OpenVINO build at D:\river\moe\openvino.
rem
rem Because the oneDNN static lib and the gemmstone headers come from the SAME
rem commit (build is consistent), we link openvino_onednn_gpu.lib DIRECTLY -- no
rem stale-library fresh-archive workaround is needed (that is only for the Linux
rem dev box where the prebuilt .a drifted from the headers).
rem ============================================================================
setlocal enabledelayedexpansion

call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"

set ROOT=D:\river\moe\openvino
set O=%ROOT%\src\plugins\intel_gpu\thirdparty\onednn_gpu
set B=%ROOT%\build\src\plugins\intel_gpu\thirdparty\onednn_gpu_build
set OCL=C:\Users\Local_Admin\OpenCL-SDK-v2025.07.23-Win-x64
set LIB_ONEDNN=%ROOT%\build\src\plugins\intel_gpu\thirdparty\onednn_gpu_install\lib\openvino_onednn_gpu.lib

cd /d D:\river\workspace\dev_sdpa_micro_opt

echo === Compiling sdpa_micro_generate_test.cpp (MSVC) ===
cl /nologo /EHsc /O2 /std:c++17 /MD /bigobj /Zc:__cplusplus ^
   /DNDEBUG /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS ^
   /DCL_TARGET_OPENCL_VERSION=300 ^
   /DDNNL_ENABLE_CONCURRENT_EXEC /DDNNL_ENABLE_CPU_ISA_HINTS /DDNNL_ENABLE_MAX_CPU_ISA ^
   /DDNNL_GPU_ISA_XE2 /DDNNL_X64=1 ^
   /DGEMMSTONE_BUILD_12HP /DGEMMSTONE_BUILD_12LP /DGEMMSTONE_BUILD_12P7 /DGEMMSTONE_BUILD_12P8 ^
   /DGEMMSTONE_BUILD_XE2 /DGEMMSTONE_BUILD_XE3 /DGEMMSTONE_BUILD_XE3P ^
   /DGEMMSTONE_CONFIG /DNGEN_CONFIG /D__STDC_CONSTANT_MACROS /D__STDC_LIMIT_MACROS ^
   /I"%O%\src\gpu\intel\gemm\jit" ^
   /I"%B%\include" ^
   /I"%O%\include" ^
   /I"%ROOT%\thirdparty\ocl\cl_headers" ^
   /I"%O%\third_party" ^
   /I"%O%\src" ^
   /I"%O%\src\gpu\intel\jit\config" ^
   /I"%O%\third_party\ngen" ^
   /I"%O%\src\gpu\intel\gemm\jit\include" ^
   sdpa_micro_generate_test.cpp ^
   /Fe:sdpa_micro_generate_test.exe ^
   /link /LIBPATH:"%OCL%\lib" OpenCL.lib "%LIB_ONEDNN%"

if errorlevel 1 (
  echo.
  echo *** BUILD FAILED ***
  exit /b 1
)
echo.
echo BUILD OK -^> D:\river\workspace\dev_sdpa_micro_opt\sdpa_micro_generate_test.exe
endlocal
