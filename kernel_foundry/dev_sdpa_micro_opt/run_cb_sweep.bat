@echo off
REM run_cb_sweep.bat <plugin_dll> <outfile>
REM Sweeps number of sequences n = 1..10, each generating 16 output tokens,
REM using the continuous_batching_benchmark on GPU. Appends a one-line summary
REM per n so NEW vs OLD can be compared across concurrency.

set "PLUGIN=%~1"
set "OUT=%~2"
if "%PLUGIN%"=="" set "PLUGIN=plugin_NEW.dll"
if "%OUT%"=="" set "OUT=cb_sweep_out.txt"

set "OVBIN=D:\river\moe\openvino\bin\intel64\Release"
set "PATH=%OVBIN%;D:\river\moe\openvino.genai\build\openvino_genai;D:\river\moe\openvino.genai\build\bin\Release;%PATH%"

REM swap in the requested plugin DLL once
copy /Y "D:\river\workspace\dev_sdpa_micro_opt\%PLUGIN%" "%OVBIN%\openvino_intel_gpu_plugin.dll" >nul

set "MODEL=D:\river\models\qwen3-8b\pytorch\ov\OV_FP16-4BIT_DEFAULT"
set "DS=D:\river\workspace\dev_sdpa_micro_opt\cb_dataset.json"

echo ===== SWEEP %PLUGIN% (n=1..10, max_output_len=16) ===== > "%OUT%"
for %%N in (1 2 3 4 5 6 7 8 9 10) do (
    echo --- n=%%N --- >> "%OUT%"
    continuous_batching_benchmark.exe -m "%MODEL%" --device GPU --dataset "%DS%" -n %%N -b 256 --max_input_len 1024 --max_output_len 16 --cache_size 8 > "%TEMP%\cb_run.txt" 2>&1
    findstr /C:"Benchmark duration" /C:"output tokens" /C:"Output throughput" /C:"Mean TTFT" /C:"Mean TPOT" "%TEMP%\cb_run.txt" >> "%OUT%"
)
echo SWEEP_DONE_%PLUGIN%
