---
name: dev_onednn_gemm
description: Study oneDNN GEMM operations for better performance.
---

## Workflow

1. **Read code**: Read all oneDNN GEMM-related sources (see `reference.md`) and `SUMMARY.md` for architecture/history
2. **Investigation**: Investigate and understand oneDNN GEMM kernel design details and optimizations strategies.
      1) kernel design and data flow
      2) vectorization and instruction usage
      3) memory access patterns and blocking strategies
      4) threading and parallelization approaches
3. **Applying**: Apply insights to optimize our own GEMM implementation and write kernels that better utilize hardware capabilities:
      1) Adopting similar blocking and tiling strategies
      2) Leveraging DPAS instructions for better performance
      3) Optimizing memory access patterns to improve cache utilization
      4) Enhancing threading and parallelization to fully utilize hardware resources

4. **Documentation**: 
      - Update `SUMMARY.md` with investigation results; keep this file as concise reference
      - Update `OPTIMIZATION_TIPS.md` when optimizing our GEMM kernel with insights from oneDNN


## Related Docs

- `SUMMARY.md` — Full work summary with detailed architecture, file map, performance tables, and history
- `OPTIMIZATION_TIPS.md` — Key optimization tips distilled from oneDNN GEMM investigation and verified by new GEMM kernel
- `reference.md` — Complete file listing of oneDNN GEMM sources for quick reference
- `remote_machine.md` — Instructions for setting up remote machine access for testing
