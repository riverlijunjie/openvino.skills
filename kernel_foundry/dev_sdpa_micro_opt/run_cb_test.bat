@echo off
set "PATH=D:\river\moe\openvino\bin\intel64\Release;D:\river\moe\openvino.genai\build\openvino_genai;D:\river\moe\openvino.genai\build\bin\Release;%PATH%"
set "MODEL=D:\river\models\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
cd /d D:\river\moe\openvino.genai\build\bin\Release

echo ===== [1] continuous_batching_accuracy (PagedAttention) on GPU, n=2 =====
continuous_batching_accuracy.exe -m "%MODEL%" -d GPU -n 2 > D:\river\workspace\dev_sdpa_micro_opt\cb_acc_gpu.txt 2>&1
echo accuracy exit=%ERRORLEVEL%

echo ===== [2] benchmark_genai on GPU (decode tok/s), mt=128 =====
benchmark_genai.exe -m "%MODEL%" -d GPU -n 3 --nw 1 --mt 128 -p "Explain what a neural network is in two sentences." > D:\river\workspace\dev_sdpa_micro_opt\cb_bench_gpu.txt 2>&1
echo benchmark exit=%ERRORLEVEL%
echo DONE
