@echo off
REM Run all FC, PA, and small-op benchmarks with cliloader on Windows PTL iGPU.
REM Shapes match the real Qwen3-8B verbose dump.
REM
REM Per-category iteration counts are tuned so each decode op's total kernel
REM execution time is >= ~1 s (where possible), which is required to get
REM stable avg-per-iter for microsecond-scale decode kernels. Prefill ops
REM already have multi-ms kernels so small iter counts are sufficient.

call D:\river\moe\openvino\release_install\setupvars.bat
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%

set CLILOADER=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BENCH_DIR=D:\river\moe\dev_roofline_profiling\utils\build\Release
set RESULTS_DIR=D:\river\moe\roofline_results
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

set CLI_DevicePerformanceTiming=1
set CLI_DevicePerformanceTimingSkipUnmap=1

REM -------- iteration budgets --------
set FC_DEC_ITERS=8000
set FC_DEC_WARM=500
set FC_DEC_LM_ITERS=1000
set FC_DEC_LM_WARM=100
set FC_PRE_ITERS=50
set FC_PRE_WARM=10
set PA_DEC_ITERS=20000
set PA_DEC_WARM=500
set PA_PRE_ITERS=50
set PA_PRE_WARM=10
set SM_DEC_ITERS=50000
set SM_DEC_WARM=500
set SM_PRE_ITERS=500
set SM_PRE_WARM=50

set BUFS=8
set FLUSH_MB=64
REM L2/L3 cache-flush kernel between every FC infer (see run_all.sh notes).
REM 64 MB exceeds BMG L2 (18 MB) and PTL CPU LLC (~30 MB).

echo ===== FC DECODE (QKV fused; Gate/Up SEPARATE) =====
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 4096 6144   128 %FC_DEC_ITERS%    %FC_DEC_WARM%    %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_QKV.log"     2>&1
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 4096 4096   128 %FC_DEC_ITERS%    %FC_DEC_WARM%    %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_O.log"       2>&1
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 4096 12288  128 %FC_DEC_ITERS%    %FC_DEC_WARM%    %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_Gate.log"    2>&1
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 4096 12288  128 %FC_DEC_ITERS%    %FC_DEC_WARM%    %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_Up.log"      2>&1
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 12288 4096  128 %FC_DEC_ITERS%    %FC_DEC_WARM%    %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_Down.log"    2>&1
%CLILOADER% %BENCH_DIR%\fc_bench.exe 1 4096 151936 128 %FC_DEC_LM_ITERS% %FC_DEC_LM_WARM% %BUFS% u8 %FLUSH_MB% > "%RESULTS_DIR%\fc_decode_LMHead.log"  2>&1

echo ===== FC PREFILL (body FCs only; LM_Head is decode-only per SKILL) =====
for %%S in (1024 2048 4096 8192) do (
    %CLILOADER% %BENCH_DIR%\fc_bench.exe %%S 4096 6144  128 %FC_PRE_ITERS% %FC_PRE_WARM% %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_prefill_QKV_S%%S.log"  2>&1
    %CLILOADER% %BENCH_DIR%\fc_bench.exe %%S 4096 4096  128 %FC_PRE_ITERS% %FC_PRE_WARM% %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_prefill_O_S%%S.log"    2>&1
    %CLILOADER% %BENCH_DIR%\fc_bench.exe %%S 4096 12288 128 %FC_PRE_ITERS% %FC_PRE_WARM% %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_prefill_Gate_S%%S.log" 2>&1
    %CLILOADER% %BENCH_DIR%\fc_bench.exe %%S 4096 12288 128 %FC_PRE_ITERS% %FC_PRE_WARM% %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_prefill_Up_S%%S.log"   2>&1
    %CLILOADER% %BENCH_DIR%\fc_bench.exe %%S 12288 4096 128 %FC_PRE_ITERS% %FC_PRE_WARM% %BUFS% u4 %FLUSH_MB% > "%RESULTS_DIR%\fc_prefill_Down_S%%S.log" 2>&1
)

set BUFS=4
echo ===== PA DECODE (i8 KV) =====
for %%K in (1024 2048 4096 8192) do (
    %CLILOADER% %BENCH_DIR%\pa_bench.exe decode 1 %%K %PA_DEC_ITERS% %PA_DEC_WARM% %BUFS% i8 > "%RESULTS_DIR%\pa_decode_kv%%K_i8.log" 2>&1
)

echo ===== PA PREFILL (i8 KV) =====
for %%S in (1024 2048 4096 8192) do (
    %CLILOADER% %BENCH_DIR%\pa_bench.exe prefill %%S 0 %PA_PRE_ITERS% %PA_PRE_WARM% %BUFS% i8 > "%RESULTS_DIR%\pa_prefill_S%%S_i8.log" 2>&1
)

REM Optional SDPA sweep -- set RUN_SDPA=1 to enable. Used when attention is NOT
REM converted to PagedAttention, or to compare SDPA vs PA on the same shapes.
REM KV-cache compression is NOT modelled here.
if "%RUN_SDPA%"=="1" (
    echo ===== SDPA DECODE =====
    for %%K in (1024 2048 4096 8192) do (
        %CLILOADER% %BENCH_DIR%\sdpa_bench.exe decode 1 %%K %PA_DEC_ITERS% %PA_DEC_WARM% %BUFS% 0 > "%RESULTS_DIR%\sdpa_decode_kv%%K.log" 2>&1
    )
    echo ===== SDPA PREFILL (causal) =====
    for %%S in (1024 2048 4096 8192) do (
        %CLILOADER% %BENCH_DIR%\sdpa_bench.exe prefill %%S %%S %PA_PRE_ITERS% %PA_PRE_WARM% %BUFS% 1 > "%RESULTS_DIR%\sdpa_prefill_S%%S.log" 2>&1
    )
)

set BUFS=8

echo ===== SMALL OPS DECODE =====
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm   1 4096     --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_RMSNorm_hidden.log"  2>&1
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm3d 1 32 128   --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_QNorm.log"           2>&1
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm3d 1 8 128    --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_KNorm.log"           2>&1
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe rope      1 32 128   --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_RoPE_Q.log"          2>&1
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe rope      1 8 128    --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_RoPE_K.log"          2>&1
%CLILOADER% %BENCH_DIR%\small_ops_bench.exe add       1 4096     --iters %SM_DEC_ITERS% --warmup %SM_DEC_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_decode_Add_residual.log"    2>&1

echo ===== SMALL OPS PREFILL =====
for %%M in (1024 2048 4096 8192) do (
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm   %%M 4096    --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_RMSNorm_hidden_S%%M.log" 2>&1
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm3d %%M 32 128  --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_QNorm_S%%M.log"          2>&1
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe rmsnorm3d %%M 8 128   --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_KNorm_S%%M.log"          2>&1
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe rope      %%M 32 128  --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_RoPE_Q_S%%M.log"         2>&1
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe rope      %%M 8 128   --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_RoPE_K_S%%M.log"         2>&1
    %CLILOADER% %BENCH_DIR%\small_ops_bench.exe add       %%M 4096    --iters %SM_PRE_ITERS% --warmup %SM_PRE_WARM% --bufs %BUFS% > "%RESULTS_DIR%\sm_prefill_Add_residual_S%%M.log"   2>&1
)

echo ===== ALL DONE =====
dir "%RESULTS_DIR%"
