# QSA sparse_attention_dpas performance analysis (session 2026-09-14)

Empirical optimization-space analysis of `qsa_sparse_attention_dpas` (Q3 prefill,
systolic) using the standalone `cm_kernel/` harness + a dedicated profiler
`cm_kernel/profile_q3_dpas.py`, run on the remote BMG (Arc B580).

## Method
- `profile_q3_dpas.py` sweeps the two JIT tunables `QSA_ATT_WORKERS` (W, head-dim
  split) x `QSA_HEADS_PER_TILE` (HT, heads sharing a dpas tile), times each build
  (mean over N iters, 96MB cache-flush between measurements), and computes a KV
  traffic model (union blocks per sub-tile, head-group redundancy, union-compute amp).
- IGC assembly: `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igc`, then grep
  `load.ugm` / `dpas` / spill in `/tmp/igc/VC_asm*_qsa_sparse_attention_dpas.asm`.
- Config from config.json: Hq=24, Hkv=2, n_rep=12, Dh=Dv=256, r=4, block_topk=512.

## Findings
1. Tunables ALREADY OPTIMAL (plugin auto-picks W=4, HT=4):
   | HT/W       | q=2048  | q=4096  |
   |------------|---------|---------|
   | HT4 W4 *   | 3.80 ms | 14.05 ms|
   | HT4 W8     | 6.04    | 22.6    |
   | HT2 W4     | 4.84    | 18.3    |
   | HT2 W8     | 8.13    | 31.3    |
   W=8 ~1.6x slower (extra SLM St-reduction + 2x threads); HT=2 ~1.27x slower.
   Only W in {4,8} fit the 12KB reg budget; HT in {2,4} valid (HT<=4 forced by dpas
   N=16 and n_rep=12). No tuning win left.

2. NO GRF spill: IGC `numGRF=256`, only 13-17 predicate-flag spills (cheap). Healthy.

3. BOTTLENECK = LSC load-message ISSUE RATE (scattered KV gather), not bandwidth:
   - asm: ~273 `load.ugm` vs 181 `dpas` per worker (W=4, 512 tok) - load-message dominated.
   - measured effective BW ~850 GB/s > 456 GB/s DRAM peak => most loads hit L2 =>
     message-issue-bound, NOT DRAM-BW-bound. (Also why cm_prefetch was a regression:
     it adds messages that compete for the same issue slots.)

4. MAIN structural inefficiency = 3x head-group KV redundancy:
   - Hq/HT = 24/4 = 6 head-groups, but n_rep/HT = 12/4 = 3 of them share each kv_head
     and EACH independently re-gathers the same KV. HT=4 is already the max.
   - HT sensitivity confirms it: redundancy 6x->3x (HT 2->4) gave 1.27x, so eliminating
     3x->1x has a ~1.5-2x ceiling (bounded by dpas/softmax/Q/O/barrier costs).

5. Union COMPUTE amplification negligible (1.0-1.3x) at block_topk=512 (dense selection):
   the N-axis-over-heads tiling design is sound at this config.

## Verdict: ~1.5-2x optimization space, but hard + risky
- No low-risk free wins: tunables optimal, no spill, union waste minimal.
- Only real lever: share a kv_head's staged KV across its 3 head-groups (SLM-resident
  per-step KV, or one WG per kv_head computing all n_rep=12 heads). BUT rO for 12 heads
  won't fit the register file -> needs partial/SLM restructure. HIGH correctness risk
  (this kernel has hung the GPU before).
- Ruled out: cm_prefetch (2.3x regression, message-rate bound); wider messages (64B row
  cap; 32 halfs already max, 64 corrupts silently); coalescing K/V (separate caches +
  scattered pages).

## Reprofile recipe
```
cd /mnt/river/qsa/cm_kernel && export CM_FE_DIR=/mnt/river
./.venv/bin/python profile_q3_dpas.py --sizes 2048 4096 --iters 15
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igc ./.venv/bin/python profile_q3_dpas.py --sizes 512 --iters 1 --no-flush
```

## Related capped-work behaviour (analysis in same session)
Q3 sparse attention caps at 512 blocks = 2048 tokens PER QUERY (+ <=3 tail). So:
- DECODE (1 query): FLOP constant beyond past=2048 (0.050 GFLOP flat at 2048/4096/8192);
  small time creep is scattered-gather locality over a bigger KV cache, not compute.
- PREFILL: total grows because it is N queries, each capped; the cap shows as the FLOP
  growth ratio per doubling dropping 4.0x -> 3.0x -> 2.33x (converging to 2x/linear).
  Verified: measured GFLOP 12.897/51.565/154.719 match the capped-selection model exactly.
