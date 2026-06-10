@echo off
setlocal
cd /d D:\river\moe\dev_roofline_profiling\utils\build
cmake --build . --config Release -j 8
echo BUILD_EXIT=%ERRORLEVEL%
dir /b Release
endlocal
