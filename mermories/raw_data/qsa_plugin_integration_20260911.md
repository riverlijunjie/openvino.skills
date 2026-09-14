# QSA standalone -> OpenVINO GPU plugin integration (2026-09-11)

Integrated standalone cm_kernel optimizations into
openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/qsa/{qsa.cpp,qsa.hpp}.
Branch: river/qsa_gr_impl. NOT build/GPU tested here (no local CM compiler/GPU;
must validate on remote BMG/PTL machines per remote_machine.txt before merge).

## What changed
- Copied newer (backward-compatible superset, JIT-flag-gated) kernels from
  cm_kernel/kernels/ over the plugin's qsa_prepare.cm, qsa_score_partition.cm,
  qsa_score_tile_dpas.cm, qsa_score_topk_fused.cm, qsa_sparse_attention_dpas.cm,
  qsa_topk_finalization.cm, and qsa_common.hpp (all old plugin behavior preserved
  when new JIT constants are left at their defaults).
- Added new kernel files (previously missing from the plugin): qsa_project_dpas.cm,
  qsa_sparse_attention_gqa_dpas.cm, qsa_sparse_attention_split.cm,
  qsa_sparse_attention_split_reduce.cm, qsa_topk_radix_wg.cm, plus headers
  qsa_topk_fast.hpp, qsa_split_span.hpp. CMakeLists globs *.cm/*.hpp automatically.
- qsa.hpp: QSAInternalBufIdx grew to 10 entries (added PROJECTED, TOKEN_MAP,
  SPLIT_ACC, SPLIT_ML). QSARuntimeParams gained token_map and use_split_attention.
- qsa.cpp additions:
  - make_qsa_jit(): now sets QSA_DENSE_BYPASS=1, QSA_DENSE_SCORE_BYPASS=1 globally
    (free, correctness-safe dense-prefix bypass optimization).
  - can_use_dpas_projection(), can_use_gqa_dpas_attention(): new capability predicates
    (mirror qsa_project_dpas.cm's static_asserts / Q3GQAAttention.unsupported_reason).
  - QSAProjectDpasGenerator: new Q1 systolic projection stage; QSAPrepareGenerator
    conditionally adds QSA_PREPROJECTED=1 + a leading PROJECTED buffer arg when
    can_use_dpas_projection() holds (else identical to before).
  - QSATopkRadixWgGenerator: replaces QSATopkFinalizationGenerator everywhere
    (decode + prefill-dpas paths). QSA_TOPK_WG=16, QSA_TOPK_CACHE_CAPACITY=2048
    (kernel's own max; falls back to uncached global scan above it, so fixed
    compile-time 2048 is always safe regardless of context length -- no need to
    recompile per shape like the standalone harness does).
  - QSASparseAttentionGqaDpasGenerator: new prefill Q3, QSA_GQA_HEAD_MAJOR=1 +
    QSA_GQA_VECTOR_META=1 (the validated PTL winner). Preferred over the old
    union-based QSASparseAttentionDpasGenerator when can_use_gqa_dpas_attention()
    holds (true for the current Qwen3.8-Flash-Next config: Hq=24,Hkv=2,Dh=Dv=256,r=4).
    Needs a NEW per-token (not per-16-tile) TOKEN_MAP internal buffer.
  - QSASparseAttentionSplitGenerator/...SplitReduceGenerator: decode split-selected-KV
    Q3 (Stage4 P16). Always add_stage'd (capability is runtime past_len, not static),
    gated at execute() time via rtp->use_split_attention = GENERATE stage && min
    past_len (over sequences with q_len>0) >= 2048. Generalized standalone's
    "batch in (1,4)" restriction to any batch size (kernel dispatch grid [HQ,tokens,P]
    doesn't care about batch count; that restriction was just the standalone harness's
    validated test matrix, not a kernel limit).
  - SPLIT_ACC/SPLIT_ML buffers sized by `seqs` (not `tokens`): safe because split is
    ONLY dispatched when num_tokens==num_seqs (GENERATE stage), so seqs bounds it
    even though the SAME primitive's prefill-shaped specializations have huge `tokens`.
  - execute(): Q1 chain is now project_dpas(if supported)->prepare (was: prepare
    independent). Q2 finalizer swapped to topk_radix_wg. Q3 dispatch order:
    GENERATE -> split (if use_split_attention) else scalar; else GQA dpas (if
    supported) else old union dpas (if supported) else scalar.
  - update_rt_params(): added token_map construction (one (seq, flat_token) entry
    per token) and use_split_attention computation.

## Known deliberately-NOT-fixed limitation
BLOCK_SCORES internal buffer is still allocated unbounded as [tokens, max_blocks]
f32 (get_internal_buffer_descs, unchanged). For very long prefill (128K tokens,
max_blocks~32K) this is ~17 GB -- a real scalability problem the standalone fixed
via qsa_score_chunked.py's query-row ChunkedQ2 (bounded ~128 MiB scratch + multiple
score/topk kernel launches per shape). Replicating that here requires a much more
invasive host-driven multi-launch loop (QSA_SCORE_ROW_CHUNK kernel variants already
copied in and JIT-guarded, ready to use, but the C++ dispatch loop was NOT
implemented in this pass due to time/risk). Flagged to user as follow-up.

## Validation update (2026-09-11, remote BMG 10.239.140.245)
- Built via `cd /mnt/river/qsa/openvino.mx && CM_FE_DIR=/mnt/river bash build.sh`
  (incremental, ~40s once build-x86_64-release/ already configured).
- `bin/intel64/Release/ov_gpu_unit_tests --gtest_filter='*qsa*:-DISABLED_*'`:
  all 38 tests PASS (qsa_gpu_test, qsa_gr_fusion, smoke/qsa_long_{prefill,decode,
  chunked_prefill,gqa_prefill}_test). Note: qsa_gpu_test.cpp's own "chunked_prefill"
  naming means prefill-with-past (vLLM-style), NOT the new BLOCK_SCORES row-chunking
  feature below -- unrelated terminology, do not confuse the two.
- Found + fixed a real CM compile bug during integration: qsa_split_span.hpp (private
  copy of qsa_sparse_attention.cm's qsa_softmax_step/qsa_attend_span helpers) redefines
  those exact function names. Standalone compiles one kernel per program (harmless);
  production batches ALL QSA stages (incl. the always-present scalar
  qsa_sparse_attention.cm) into ONE clBuildProgram call -> "redefinition" compile error.
  Fixed by renaming to qsa_split_softmax_step/qsa_split_attend_span in the PLUGIN's
  copy only (bodies otherwise identical); standalone repo's copy is unaffected.
  Diagnosed by temporarily replacing GPU_DEBUG_INFO with std::cerr in
  ocl_kernel_builder.hpp's cl::BuildError catch (reverted after); the normal
  OV_GPU_Verbose env var did NOT surface the CM build log in this build (ExecutionConfig
  verbose gating mechanism not fully traced, low priority).
- Implemented the query-row chunked BLOCK_SCORES feature (previously flagged as NOT
  done): bounded to 128 MiB (score_budget_bytes, matches qsa_score_chunked.py's
  DEFAULT_SCORE_BYTES) whenever can_use_dpas_score() holds (decode score_partition +
  prefill score_tile_dpas, both now always QSA_SCORE_ROW_CHUNK=1); legacy
  qsa_score_topk_fused (Xe1 prefill fallback) keeps the old unbounded sizing since it
  has no row-chunk variant. New QSAScoreChunk plan in QSARuntimeParams, built in
  update_rt_params() mirroring plan_score_chunks() exactly (chunk-and-sequence
  re-anchored 16-tiles); execute() loops per chunk, chaining score->topk->next-chunk
  strictly sequentially (shared bounded scratch reused every chunk).
- CRITICAL GOTCHA (cost significant debugging): `execute_stage()` in
  primitive_ocl_base.hpp only calls a stage's DispatchDataFunc / rebinds kernel args
  when `kd.need_dispatch_data_update` / `need_args_update` are true, then clears them
  -- it assumes each Stage is dispatched AT MOST ONCE per execute() call. Calling
  execute_stage() on the SAME Stage::Ptr multiple times per execute() (as the chunk
  loop does for score_partition/score_tile_dpas/topk_radix_wg) silently redispatches
  with the FIRST call's stale scalars/workgroup sizes for every subsequent call unless
  you force both flags back to true immediately before each call:
  `stage->kd.need_dispatch_data_update = stage->kd.need_args_update = true;`
  (see `force_stage_refresh()` in qsa.cpp). Without this, results were silently WRONG
  (not a crash) -- verified by temporarily shrinking score_budget_bytes to 4096 bytes
  to force many chunks per test call; all 38 tests failed with plausible-looking but
  incorrect numeric output before the fix, all 38 passed after. ANY future code that
  calls execute_stage() on the same Stage::Ptr more than once per execute() must apply
  this same force-refresh pattern.
- Remaining follow-ups: none blocking; the memory-bounding gap from the previous note
  is now closed. Recommend: also exercise this on the PTL Windows machine and re-run
  the standalone cm_kernel PTL sweep for a performance sanity check (not yet done).


## Cleanup pass (2026-09-11, after "Optimize QSA kernels" commit 6ea132508d)
Removed dead code and standalone/design-doc references from the QSA plugin code per
user request ("清理code，删掉不需要的kernel代码和注释以及设计文档章节引用"):
- Deleted qsa_topk_finalization.cm (dead: superseded by qsa_topk_radix_wg.cm) and its
  unused QSATopkFinalizationGenerator C++ class (was defined but never make_stage'd).
  Verified via grep that every remaining `class QSA*Generator` has exactly one
  `make_stage<...>` use, and every remaining `.cm` file is referenced by name in qsa.cpp.
- Removed all "(design section N.N)" references across qsa.cpp, qsa.hpp,
  qsa_kv_cache_update.cm, qsa_prepare.cm, qsa_score_partition.cm, qsa_score_topk_fused.cm,
  qsa_sparse_attention.cm, qsa_common.hpp, qsa_score_common.hpp -- replaced with
  self-contained rationale (the design doc these pointed at is not part of this repo).
- Removed standalone-harness references (Stage4/Stage5 labels, `.py` file names like
  qsa_score_chunked.py/qsa_attention_gqa.py/qsa_attention_split.py/test_qsa_pipeline.py,
  SHA256 hash pointer, validate_q3_split.py) from qsa.cpp, qsa_sparse_attention_split.cm,
  qsa_sparse_attention_gqa_dpas.cm, qsa_split_span.hpp -- replaced with descriptions of
  the actual in-repo mechanism (e.g. why qsa_split_span.hpp renames its helpers to avoid
  a symbol clash, described directly instead of citing the standalone harness's SHA256).
- Left cm/gated_residual/ and cm/paged_attention.* untouched (out of scope: user asked
  for QSA-related code specifically; GR's own "design section 4.3" comment in
  gated_residual.cpp was not touched).
- Re-synced to remote BMG, rebuilt clean, reran the full `*qsa*` gtest filter: still
  38/38 PASS. local/remote qsa.cpp and qsa.hpp confirmed byte-identical after sync.
- qsa_split_span.hpp / qsa_topk_fast.hpp remain untracked (`??` in git status) -- new
  headers added during the "Optimize QSA kernels" work, never committed. Not committed
  by me either; flagged to user since a future commit should include them or they'll
  look like local-only stray files.

## GR (gated_residual) cleanup + whole-cm/ unused-kernel audit (2026-09-11, same day)
- Removed 2 remaining design-doc references: gr_read.cm's
  "OV_SUPPORT_QSA_GR_DESIGN_CN.md section 8.2" and gated_residual.cpp's
  "design section 4.3 转换前必须验证" (the embedded Chinese quote was a stray doc
  citation, not real code). No logic changes; GR math (branch-wise RMSNorm, low-rank
  read gate, per-branch write gate in (0,2), gr_write's paired-half load trick) was
  reviewed and found correct -- did not find an actual bug, only the two comments above.
- Unused-kernel audit across the ENTIRE cm/ implementation dir (not just qsa/gated_residual):
  cross-referenced every *.cm file's basename against KernelGenerator("name", ...)
  string literals in all .cpp/.hpp under cm/. Every single .cm file (qsa/*, gated_residual/*,
  pa_*.cm, xattn_*.cm, xetla_lstm_*.cm, cm_sdpa_vlen.cm) is referenced exactly once --
  no other dead kernel files exist beyond the qsa_topk_finalization.cm already removed
  in the prior QSA cleanup pass. Did not need to touch pa_*/xattn_*/xetla_*/cm_sdpa_vlen
  (unrelated pre-existing paged-attention/xattention/lstm/sdpa features, all still used).
- Rebuilt clean on remote BMG; `--gtest_filter='*qsa*:*gated_residual*:-DISABLED_*'`:
  45/45 PASS (38 qsa + 7 gated_residual). Local/remote gated_residual/* confirmed
  byte-identical after sync.

## Continuous-batching / multi-chunk test gap analysis + fix (2026-09-11, later same day)
User asked whether QSA supports continue_batch/multi-chunk. Findings, then closed the two
real test-coverage gaps found (dispatch code itself was already generic):
- `QSAStage::MIXED` (any_multi && any_single in update_rt_params) and the whole
  Q0/Q1/Q2/Q3 per-sequence dispatch (prepare_map/score_tile_map/token_map, all built by
  looping `num_seqs = past_lens.size()`) were ALREADY fully generic for arbitrary
  num_seqs -- but every existing qsa_gpu_test.cpp helper (`run_qsa`, `make_reference*`)
  hard-codes `past_lens={ref.past_len}` / `subsequence_begins={0,ref.q_len}`, i.e.
  num_seqs==1 always. MIXED had never actually been exercised on real GPU.
- `use_split_attention` (decode split-selected-KV Q3) is deliberately GENERATE-stage-only
  (SPLIT_ACC/SPLIT_ML sized by `seqs`, assumes num_tokens==num_seqs) -- a MIXED batch's
  long-decode sequence falls back to gqa-dpas/scalar Q3 instead of the faster split path.
  Correct, just not optimal; left as-is (documented, not a bug).
- Q2's BLOCK_SCORES row-chunking (`QSAScoreChunk`, added earlier the same day) has a
  hardcoded `score_budget_bytes = 128 MiB` in qsa.cpp; no existing test's BLOCK_SCORES
  ([tokens, max_blocks] f32) gets anywhere near that at up to 4096 tokens, so score_chunks
  always planned exactly 1 chunk in every prior test -- the chunk-boundary logic itself
  had zero permanent regression coverage (was only manually verified once by temporarily
  editing the constant, per the entry above).

Fixes applied (both committed together, see qsa.cpp + qsa_gpu_test.cpp diffs):
1. `qsa.cpp`: `score_budget_bytes` (was `constexpr size_t`) -> `size_t score_budget_bytes()`
   that reads env var `QSA_TEST_SCORE_BUDGET_BYTES` (via `std::getenv`/`strtoull`) and
   falls back to the same 128 MiB default when unset/invalid. Production paths never set
   this var, so real behavior is unchanged; only the ONE call site in `score_scratch_rows()`
   needed updating (`score_budget_bytes` -> `score_budget_bytes()`). Added `<cstdlib>`.
2. `qsa_gpu_test.cpp` additions:
   - `make_reference_sharing_weights(rg, shared_ref, past_len, q_len, has_gate)`: like
     `make_reference` but copies `idx_w`/`q_norm`/`k_norm`/`rope_cos`/`rope_sin`/`rotary`
     from an existing ref instead of drawing fresh ones -- required for correctness: one
     QSA primitive is one layer's weights shared by every sequence in a batch, so two
     sequences in one test MUST NOT get independently-random weights.
   - `run_qsa_batch(refs, has_gate)`: generalizes `run_qsa` to N independent sequences in
     ONE `network.execute()` call. Builds `past_lens`/`subsequence_begins` from
     `refs[*].past_len/q_len`, gives every sequence its own contiguous PA-page range in a
     SHARED cache (`page_begin[s]`, non-overlapping), concatenates per-sequence
     mixed/query/key/value-current tensors and position_ids in sequence order, and reuses
     `refs.front()`'s weights for the single `idx_w`/`q_norm`/`k_norm`/`rotary` topology
     inputs (all refs must share these; only checks `D`/`Hq`/`Hkv` match, via `ADD_FAILURE()`
     since the function returns non-void and cannot use `ASSERT_*`).
   - `TEST(qsa_gpu_test, mixed_batch_prefill_and_decode_together)`: seq0 = chunked-prefill
     continuation (`past_len=40,q_len=37`, `any_multi`), seq1 = decode append
     (`past_len=63,q_len=1`, `any_single`, shares seq0's weights via
     `make_reference_sharing_weights`), both in `prod_gqa_dims(200)`. This is the FIRST
     test in the repo that ever constructs `num_seqs>1` for QSA, and the first real
     exercise of `QSAStage::MIXED` on GPU. Compares each sequence's output slice against
     its own independent CPU reference (`ref.attention()`), margin-filtered exactly like
     `expect_qsa_matches`. PASS in 3.4 s.
   - `score_budget_override` RAII class (mirrors the `_WIN32`/`setenv` pattern already used
     in `tests/functional/subgraph_tests/dynamic/moe.cpp`'s
     `set_disable_moe_fusion`/`unset_disable_moe_fusion`): sets/clears
     `QSA_TEST_SCORE_BUDGET_BYTES` for the guard's lifetime.
   - `TEST(qsa_gpu_test, dpas_score_multi_chunk_matches_single_chunk)`: `dpas_dims()`,
     q_len=64, `score_budget_override guard(64)` (64 BYTES, one BLOCK_SCORES row is
     `max_blocks*4=64` bytes here) forces `score_scratch_rows()` down to 1 row/chunk, i.e.
     64 chunks instead of 1, then reuses `expect_qsa_matches` unchanged. PASS in 6.5 s.
   Both new tests placed right after their thematically-closest existing test
   (`dpas_prefill_and_scalar_decode_agree` / `dpas_chunked_prefill_with_past`).
3. Rebuilt clean on remote BMG (incremental, ~40 s), ran the 2 new tests alone (PASS,
   9.9 s total) then the full `*qsa*:*gated_residual*:-DISABLED_*` filter: 47/47 PASS (was
   45; +2 new). No regressions, no leftover env var effects on later tests (RAII-scoped).

Key lesson for next time: "the dispatch code is written generically for N" and "N>1 has
actually been run through the real kernels on GPU" are DIFFERENT claims -- always check
the test harness's hard-coded shapes (grep for `{ref.past_len}`-style single-element
vectors), not just the C++ loop bounds in the impl, before asserting a batching feature is
verified.
