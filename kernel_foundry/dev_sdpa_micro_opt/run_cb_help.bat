@echo off
set "PATH=D:\river\moe\openvino\bin\intel64\Release;D:\river\moe\openvino.genai\build\openvino_genai;D:\river\moe\openvino.genai\build\bin\Release;%PATH%"
cd /d D:\river\moe\openvino.genai\build\bin\Release
echo ===== continuous_batching_accuracy --help =====
continuous_batching_accuracy.exe --help > cb_help.txt 2>&1
type cb_help.txt
echo ===== benchmark_genai --help =====
benchmark_genai.exe --help > bg_help.txt 2>&1
type bg_help.txt
echo DONE
