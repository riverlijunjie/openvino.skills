@echo off
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set OUT=D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
if not exist "%OUT%" mkdir "%OUT%"
"%CLI%" -d D:\river\moe\dev_roofline_profiling\utils\build\Release\fc_bench.exe 1 8192 5376 128 300 30 8 u4 64 > "%OUT%\cli_smoke.log" 2>&1
echo CLI_RC=%errorlevel%
echo ---- tail of log ----
powershell -NoProfile -Command "Get-Content -Tail 25 '%OUT%\cli_smoke.log'"
