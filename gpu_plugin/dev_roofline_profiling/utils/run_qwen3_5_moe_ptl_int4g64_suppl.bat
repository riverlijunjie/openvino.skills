@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ================================================================
REM  Supplemental: MoE routed-only (SI=0) + 3 shared expert FC
REM  INT4 g=64 variant
REM ================================================================
REM The full moe_bench with shared_I=512 shared_quant=u4 g=64 failed
REM (shared expert fusion rejected, no compute kernels executed).
REM Solution: benchmark routed MoE separately with SI=0, and shared
REM expert as 3 standalone INT4 g=64 FC ops.

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_int4g64\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

echo === SUPPL START %date% %time% >> "%LOGS%\_run_suppl.log"

REM ================================================================
REM  MoE routed-only (SI=0), INT4 g=64
REM ================================================================
REM moe_bench B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I

REM --- Decode M=1 ---
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1 2048 512 256 8 64 100 10 4 64 0 > "%LOGS%\moe_routed_decode_M1.log" 2>&1
echo === moe_routed_decode_M1 done %time% >> "%LOGS%\_run_suppl.log"

REM --- Prefill sweep ---
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1024  2048 512 256 8 64  20 5 2 64 0 > "%LOGS%\moe_routed_prefill_S1024.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 2048  2048 512 256 8 64  15 3 2 64 0 > "%LOGS%\moe_routed_prefill_S2048.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 4096  2048 512 256 8 64  10 2 2 64 0 > "%LOGS%\moe_routed_prefill_S4096.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 8192  2048 512 256 8 64   8 2 2 64 0 > "%LOGS%\moe_routed_prefill_S8192.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 16384 2048 512 256 8 64   4 1 1 64 0 > "%LOGS%\moe_routed_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 32768 2048 512 256 8 64   2 1 1 64 0 > "%LOGS%\moe_routed_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 65536 2048 512 256 8 64   2 1 1 64 0 > "%LOGS%\moe_routed_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 64  1 1 1 64 0 > "%LOGS%\moe_routed_prefill_S131072.log" 2>&1
echo === moe_routed_prefill done %time% >> "%LOGS%\_run_suppl.log"

REM ================================================================
REM  Shared expert: 3 separate INT4 g=64 FC ops
REM  gate: FC(M, 2048, 512)  up: FC(M, 2048, 512)  down: FC(M, 512, 2048)
REM ================================================================

REM --- Decode M=1 ---
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 512 64 50000 500 16 u4 32 > "%LOGS%\shared_gate_decode_M1.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 512 64 50000 500 16 u4 32 > "%LOGS%\shared_up_decode_M1.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 512 2048 64 50000 500 16 u4 32 > "%LOGS%\shared_down_decode_M1.log" 2>&1
echo === shared_decode done %time% >> "%LOGS%\_run_suppl.log"

REM --- Prefill sweep ---
for %%S in (1024 2048 4096 8192 16384 32768 65536 131072) do (
    if %%S LEQ 4096 (
        set IT=100
        set WM=10
        set NB=4
    ) else if %%S LEQ 16384 (
        set IT=15
        set WM=2
        set NB=1
    ) else if %%S LEQ 65536 (
        set IT=4
        set WM=1
        set NB=1
    ) else (
        set IT=2
        set WM=1
        set NB=1
    )
    "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2048 512 64 !IT! !WM! !NB! u4 32 > "%LOGS%\shared_gate_prefill_S%%S.log" 2>&1
    "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 2048 512 64 !IT! !WM! !NB! u4 32 > "%LOGS%\shared_up_prefill_S%%S.log" 2>&1
    "%CLI%" -d "%BUILD%\fc_bench.exe" %%S 512 2048 64 !IT! !WM! !NB! u4 32 > "%LOGS%\shared_down_prefill_S%%S.log" 2>&1
    echo === shared_prefill_S%%S done %time% >> "%LOGS%\_run_suppl.log"
)

echo === ALL SUPPL DONE %date% %time% >> "%LOGS%\_run_suppl.log"
echo Supplemental benchmarks complete.
