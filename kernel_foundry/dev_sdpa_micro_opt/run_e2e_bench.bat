@echo off
set "PATH=D:\river\moe\openvino\bin\intel64\Release;D:\river\moe\openvino.genai\build\openvino_genai;D:\river\moe\openvino.genai\build\bin\Release;%PATH%"
set "MODEL=D:\river\models\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "PF=D:\river\workspace\dev_sdpa_micro_opt\long_prompt.txt"
set "OUT=%1"
if "%OUT%"=="" set "OUT=e2e_out.txt"
cd /d D:\river\moe\openvino.genai\build\bin\Release
echo Running benchmark_genai GPU long-context (~7091 tok in), mt=128, nw=2, n=5 -> %OUT%
benchmark_genai.exe -m "%MODEL%" -d GPU --pf "%PF%" --nw 2 -n 5 --mt 128 > D:\river\workspace\dev_sdpa_micro_opt\%OUT% 2>&1
echo exit=%ERRORLEVEL%
type D:\river\workspace\dev_sdpa_micro_opt\%OUT%
echo DONE
