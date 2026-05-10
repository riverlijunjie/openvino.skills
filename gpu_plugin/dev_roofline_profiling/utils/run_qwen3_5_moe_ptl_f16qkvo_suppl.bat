@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Supplemental benchmarks for Qwen3.5-MoE on PTL 12Xe with F16 QKV/O.
REM
REM Key findings from first run:
REM   - F16 shared expert can NOT be fused into MoE3GemmFusedCompressed
REM     (ConvertMOEToMOECompressed requires compressed INT4/INT8 shared weights)
REM   - Instead: benchmark MoE with SI=0 (routed experts only) for fused timing,
REM     and add 3 separate f16 FC benches for shared expert (gate/up/down)
REM
REM This script collects:
REM   - moe_bench SI=0 (routed experts only, fused primitive)
REM   - Shared expert as 3 x fc_bench f16: gate(2048->512), up(2048->512), down(512->2048)
REM   - Missing fc_qkv decode_M1 and prefill S1024-S8192
REM   - Missing fc_linattn_proj decode_M1

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_5_moe_f16qkvo\ptl
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === SUPPL START %date% %time% >> "%LOGS%\_run.log"

REM ---------- MoE routed experts only (SI=0, no shared expert) ----------
REM These replace the failed SI=512 f16 runs.
REM The fused MoE primitive covers routed experts; shared expert is separate below.
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1    2048 512 256 8 128 100 10 4 64 0   > "%LOGS%\moe_routed_decode_M1.log"    2>&1
echo === moe_routed_decode_M1 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 128  20  5 2 64 0   > "%LOGS%\moe_routed_prefill_S1024.log"  2>&1
echo === moe_routed_prefill_S1024 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 128  15  3 2 64 0   > "%LOGS%\moe_routed_prefill_S2048.log"  2>&1
echo === moe_routed_prefill_S2048 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 128  10  2 2 64 0   > "%LOGS%\moe_routed_prefill_S4096.log"  2>&1
echo === moe_routed_prefill_S4096 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 128   8  2 2 64 0   > "%LOGS%\moe_routed_prefill_S8192.log"  2>&1
echo === moe_routed_prefill_S8192 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 16384  2048 512 256 8 128 4 1 1 64 0   > "%LOGS%\moe_routed_prefill_S16384.log" 2>&1
echo === moe_routed_prefill_S16384 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 32768  2048 512 256 8 128 2 1 1 64 0   > "%LOGS%\moe_routed_prefill_S32768.log" 2>&1
echo === moe_routed_prefill_S32768 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 65536  2048 512 256 8 128 2 1 1 64 0   > "%LOGS%\moe_routed_prefill_S65536.log" 2>&1
echo === moe_routed_prefill_S65536 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\moe_bench.exe" 1 131072 2048 512 256 8 128 1 1 1 64 0   > "%LOGS%\moe_routed_prefill_S131072.log" 2>&1
echo === moe_routed_prefill_S131072 done >> "%LOGS%\_run.log"

REM ---------- Shared expert: 3 separate f16 FC ops per layer ----------
REM Shared expert uses uncompressed FP16 weights:
REM   gate:  MatMul(M, 2048) x W(512, 2048)^T  -> (M, 512)
REM   up:    MatMul(M, 2048) x W(512, 2048)^T  -> (M, 512)
REM   down:  MatMul(M, 512)  x W(2048, 512)^T  -> (M, 2048)
REM Args: M K N group_size iters warmup bufs precision flush_mb
"%CLI%" -d "%BUILD%\fc_bench.exe" 1    2048  512 128 10000 500 8 f16 32 > "%LOGS%\shared_gate_decode_M1.log"    2>&1
echo === shared_gate_decode_M1 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 1    2048  512 128 10000 500 8 f16 32 > "%LOGS%\shared_up_decode_M1.log"      2>&1
echo === shared_up_decode_M1 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 1     512 2048 128 10000 500 8 f16 32 > "%LOGS%\shared_down_decode_M1.log"    2>&1
echo === shared_down_decode_M1 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048  512 128 200 20 4 f16 32 > "%LOGS%\shared_gate_prefill_S1024.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048  512 128 200 20 4 f16 32 > "%LOGS%\shared_up_prefill_S1024.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024  512 2048 128 200 20 4 f16 32 > "%LOGS%\shared_down_prefill_S1024.log"  2>&1
echo === shared_expert_prefill_S1024 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048  512 128 100 10 4 f16 32 > "%LOGS%\shared_gate_prefill_S2048.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048  512 128 100 10 4 f16 32 > "%LOGS%\shared_up_prefill_S2048.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048  512 2048 128 100 10 4 f16 32 > "%LOGS%\shared_down_prefill_S2048.log"  2>&1
echo === shared_expert_prefill_S2048 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048  512 128  50  5 4 f16 32 > "%LOGS%\shared_gate_prefill_S4096.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048  512 128  50  5 4 f16 32 > "%LOGS%\shared_up_prefill_S4096.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096  512 2048 128  50  5 4 f16 32 > "%LOGS%\shared_down_prefill_S4096.log"  2>&1
echo === shared_expert_prefill_S4096 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048  512 128  30  3 2 f16 32 > "%LOGS%\shared_gate_prefill_S8192.log"  2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048  512 128  30  3 2 f16 32 > "%LOGS%\shared_up_prefill_S8192.log"    2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192  512 2048 128  30  3 2 f16 32 > "%LOGS%\shared_down_prefill_S8192.log"  2>&1
echo === shared_expert_prefill_S8192 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048  512 128  10  2 2 f16 32 > "%LOGS%\shared_gate_prefill_S16384.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384 2048  512 128  10  2 2 f16 32 > "%LOGS%\shared_up_prefill_S16384.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 16384  512 2048 128  10  2 2 f16 32 > "%LOGS%\shared_down_prefill_S16384.log" 2>&1
echo === shared_expert_prefill_S16384 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048  512 128   5  1 1 f16 32 > "%LOGS%\shared_gate_prefill_S32768.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768 2048  512 128   5  1 1 f16 32 > "%LOGS%\shared_up_prefill_S32768.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 32768  512 2048 128   5  1 1 f16 32 > "%LOGS%\shared_down_prefill_S32768.log" 2>&1
echo === shared_expert_prefill_S32768 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048  512 128   3  1 1 f16 32 > "%LOGS%\shared_gate_prefill_S65536.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536 2048  512 128   3  1 1 f16 32 > "%LOGS%\shared_up_prefill_S65536.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 65536  512 2048 128   3  1 1 f16 32 > "%LOGS%\shared_down_prefill_S65536.log" 2>&1
echo === shared_expert_prefill_S65536 done >> "%LOGS%\_run.log"

"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048  512 128  2  1 1 f16 32 > "%LOGS%\shared_gate_prefill_S131072.log" 2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072 2048  512 128  2  1 1 f16 32 > "%LOGS%\shared_up_prefill_S131072.log"   2>&1
"%CLI%" -d "%BUILD%\fc_bench.exe" 131072  512 2048 128  2  1 1 f16 32 > "%LOGS%\shared_down_prefill_S131072.log" 2>&1
echo === shared_expert_prefill_S131072 done >> "%LOGS%\_run.log"

REM ---------- Missing FC_QKV decode M1 and prefill S1024-S8192 ----------
"%CLI%" -d "%BUILD%\fc_bench.exe" 1    2048 5120 128 5000 200 8 f16 64 > "%LOGS%\fc_qkv_decode_M1.log"       2>&1
echo === fc_qkv_decode_M1 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 1024 2048 5120 128   30   5 8 f16 64 > "%LOGS%\fc_qkv_prefill_S1024.log"   2>&1
echo === fc_qkv_prefill_S1024 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 2048 2048 5120 128   30   5 8 f16 64 > "%LOGS%\fc_qkv_prefill_S2048.log"   2>&1
echo === fc_qkv_prefill_S2048 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 4096 2048 5120 128   30   5 8 f16 64 > "%LOGS%\fc_qkv_prefill_S4096.log"   2>&1
echo === fc_qkv_prefill_S4096 done >> "%LOGS%\_run.log"
"%CLI%" -d "%BUILD%\fc_bench.exe" 8192 2048 5120 128   30   5 8 f16 64 > "%LOGS%\fc_qkv_prefill_S8192.log"   2>&1
echo === fc_qkv_prefill_S8192 done >> "%LOGS%\_run.log"

REM ---------- Missing fc_linattn_proj decode_M1 ----------
"%CLI%" -d "%BUILD%\fc_bench.exe" 1 2048 12288 128 1500 100 4 u4 64 > "%LOGS%\fc_linattn_proj_decode_M1.log" 2>&1
echo === fc_linattn_proj_decode_M1 done >> "%LOGS%\_run.log"

echo === SUPPL END %date% %time% >> "%LOGS%\_run.log"
echo Done. Logs in %LOGS%
