@echo off
setlocal
cd /d D:\river\moe\dev_roofline_profiling\utils
if exist build rmdir /s /q build
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DOpenVINO_DIR=D:\river\moe\openvino\release_install\runtime\cmake ^
  -DOV_SRC_DIR=D:\river\moe\openvino ..
echo CONFIGURE_EXIT=%ERRORLEVEL%
endlocal
