@echo off
REM run_cb_bench.bat <plugin_dll_to_use> <outfile>
REM Copies the chosen plugin DLL into the runtime bin, then runs the
REM continuous_batching_benchmark on GPU with the synthetic dataset.

set "PLUGIN=%~1"
set "OUT=%~2"
if "%PLUGIN%"=="" set "PLUGIN=plugin_NEW.dll"
if "%OUT%"=="" set "OUT=cb_out.txt"

set "OVBIN=D:\river\moe\openvino\bin\intel64\Release"
set "PATH=%OVBIN%;D:\river\moe\openvino.genai\build\openvino_genai;D:\river\moe\openvino.genai\build\bin\Release;%PATH%"

REM swap in the requested plugin DLL
copy /Y "D:\river\workspace\dev_sdpa_micro_opt\%PLUGIN%" "%OVBIN%\openvino_intel_gpu_plugin.dll" >nul

set "MODEL=D:\river\models\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "DS=D:\river\workspace\dev_sdpa_micro_opt\cb_dataset.json"

continuous_batching_benchmark.exe -m "%MODEL%" --device GPU --dataset "%DS%" -n 64 -b 256 --max_input_len 1024 --max_output_len 128 --cache_size 8 > "%OUT%" 2>&1
echo CB_DONE_%PLUGIN%
