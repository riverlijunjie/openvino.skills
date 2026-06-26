@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM Qwen3.6-35B-A3B roofline sweep on PTL (iGPU 12 Xe, 2400 MHz, 110 GB/s,
REM FP16 XMX 58.98 TFLOPS, INT8 XMX 117.96 TOPS).
REM
REM *** WEIGHT COMPRESSION group_size = 64 VARIANT ***
REM   All compressed weights use g64 (vs g128 baseline): INT4 body FC + MoE
REM   experts AND INT8 LM_Head/Embedding. g64 doubles the per-weight scale/zp
REM   metadata (INT4 weight bytes +3.8%, INT8 +1.5%) -> heavier memory-bound
REM   decode. fc_bench/moe_bench take group_size as a CLI arg, so no rebuild is
REM   needed; only the gs argument changes 128 -> 64. GatedDeltaNet head_dim=128
REM   and PagedAttention KV lengths are NOT group sizes and stay unchanged.
REM
REM Architecture = qwen3_5_moe + attention output gate (attn_output_gate=true):
REM   - full-attn QKV+gate fused FC width = 2*16*256 + 2*2*256 = 9216 (was 5120)
REM   - extra attn output gate elementwise (attn_output * sigmoid(gate)) H=4096
REM   - MoE: H=2048 I=512 NE=256 TK=8 + shared_I=512 (always-on shared expert)
REM   - LM_Head: vocab=248320 (INT8 g=64); body FC INT4 g=64; KV cache INT8
REM   - PA: NH=16 NKV=2 HD=256 (GQA=8)
REM   - GatedDeltaNet linear-attention bench (PagedGatedDeltaNet opt; qk_heads=16, v_heads=32, K=V=128)
REM ============================================================================

set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release
set LOGS=D:\river\moe\roofline_results\qwen3_6\ptl_g64
set PATH=%OV_BIN%;%TBB%;%PATH%

if not exist "%LOGS%" mkdir "%LOGS%"
echo === START %date% %time% > "%LOGS%\_index.txt"

REM ---------- MoE fused (with shared expert), group_size=64 ----------
call :do moe_decode_M1                "%BUILD%\moe_bench.exe" 1 1    2048 512 256 8 64 100 10 4 64 512
call :do moe_prefill_S1024            "%BUILD%\moe_bench.exe" 1 1024 2048 512 256 8 64  20  5 2 64 512
call :do moe_prefill_S2048            "%BUILD%\moe_bench.exe" 1 2048 2048 512 256 8 64  15  3 2 64 512
call :do moe_prefill_S4096            "%BUILD%\moe_bench.exe" 1 4096 2048 512 256 8 64  10  2 2 64 512
call :do moe_prefill_S8192            "%BUILD%\moe_bench.exe" 1 8192 2048 512 256 8 64   8  2 2 64 512

REM ---------- fc_bench: full-attn QKV+gate (9216), O, LM_Head; group_size=64 ----------
call :do fc_qkv_decode_M1             "%BUILD%\fc_bench.exe" 1    2048   9216 64 5000 200 8 u4 64
call :do fc_o_decode_M1               "%BUILD%\fc_bench.exe" 1    4096   2048 64 5000 200 8 u4 64
call :do lm_head_decode_M1            "%BUILD%\fc_bench.exe" 1    2048 248320 64  300  30 4 u8 64

for %%S in (1024 2048 4096 8192) do (
  call :do fc_qkv_prefill_S%%S        "%BUILD%\fc_bench.exe" %%S 2048 9216 64 30 5 8 u4 64
  call :do fc_o_prefill_S%%S          "%BUILD%\fc_bench.exe" %%S 4096 2048 64 30 5 8 u4 64
)

REM ---------- Linear-attention input projection (in_proj_qkv 8192 + in_proj_z 4096 = 12288); group_size=64 ----------
call :do fc_linattn_proj_decode_M1    "%BUILD%\fc_bench.exe" 1    2048  12288 64 1500 100 4 u4 64
for %%S in (1024 2048 4096 8192) do (
  call :do fc_linattn_proj_prefill_S%%S "%BUILD%\fc_bench.exe" %%S 2048  12288 64 20 5 4 u4 64
)

REM ---------- pa_bench (NH=16 NKV=2 HD=256, i8 KV) ----------
set PA_NH=16
set PA_NKV=2
set PA_HD=256
REM Decode KV sweep: window start (P), average (P+256), end (P+512) for each prompt P.
for %%K in (1024 1280 1536 2048 2304 2560 4096 4352 4608 8192 8448 8704) do (
  call :do pa_decode_kv%%K            "%BUILD%\pa_bench.exe" decode 1 %%K 8000 200 4 i8
)
REM Prefill self-attention (causal) for each prompt length.
for %%S in (1024 2048 4096 8192) do (
  call :do pa_prefill_S%%S            "%BUILD%\pa_bench.exe" prefill %%S 0 25 5 4 i8
)

REM ---------- gdn_bench: linear attention (PagedGatedDeltaNet opt kernel) ----------
REM qk_heads=16, v_heads=32 (GQA group=2), head_dim=128. Paged op -> paged_gated_delta_net_opt.
REM cache_interval=0 -> single final state snapshot (interval=T), no intermediate paging snapshots.
REM NOTE: the 128 here is head_dim, NOT a weight group size -> stays 128.
call :do gdn_decode_T1                "%BUILD%\gdn_bench.exe" 1 1    16 32 128 4000 150 4 0
for %%S in (1024 2048 4096 8192) do (
  call :do gdn_prefill_S%%S           "%BUILD%\gdn_bench.exe" 1 %%S 16 32 128 20 5 2 0
)

REM ---------- small ops (hidden=2048; full-attn 16/2 heads HD=256; gate H=4096) ----------
call :do so_rmsnorm_h2048_decode      "%BUILD%\small_ops_bench.exe" rmsnorm   1 2048 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_qnorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rmsnorm3d_knorm_decode    "%BUILD%\small_ops_bench.exe" rmsnorm3d 1  2 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_q_decode             "%BUILD%\small_ops_bench.exe" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
call :do so_rope_k_decode             "%BUILD%\small_ops_bench.exe" rope      1  2 256 --iters 30000 --warmup 300 --bufs 8
call :do so_add_decode                "%BUILD%\small_ops_bench.exe" add       1 2048 --iters 30000 --warmup 300 --bufs 8
call :do so_gate_decode               "%BUILD%\small_ops_bench.exe" gate      1 4096 --iters 30000 --warmup 300 --bufs 8

for %%S in (1024 2048 4096 8192) do (
  call :do so_rmsnorm_h2048_prefill_S%%S "%BUILD%\small_ops_bench.exe" rmsnorm %%S 2048   --iters 300 --warmup 30 --bufs 4
  call :do so_rope_q_prefill_S%%S        "%BUILD%\small_ops_bench.exe" rope    %%S 16 256 --iters 300 --warmup 30 --bufs 4
  call :do so_gate_prefill_S%%S          "%BUILD%\small_ops_bench.exe" gate    %%S 4096   --iters 300 --warmup 30 --bufs 4
)

echo === END %date% %time% >> "%LOGS%\_index.txt"
echo Done. Logs in %LOGS%
goto :eof

:do
set TAG=%~1
shift
set CMDLINE=
:doargs
if "%~1"=="" goto dorun
set CMDLINE=%CMDLINE% %1
shift
goto doargs
:dorun
echo === !TAG! :!CMDLINE! >> "%LOGS%\_index.txt"
"%CLI%" -d %CMDLINE% > "%LOGS%\!TAG!.log" 2>&1
if errorlevel 1 echo FAIL !TAG! errorlevel=%errorlevel% >> "%LOGS%\_index.txt"
goto :eof
