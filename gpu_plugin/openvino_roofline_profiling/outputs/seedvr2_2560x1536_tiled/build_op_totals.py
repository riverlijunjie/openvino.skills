#!/usr/bin/env python3
"""Aggregate per-op totals (summed across kernels) from a parsed metrics JSON +
kernel_table_config.json, for a clean top-level bottleneck summary.
-- 2560x1536 TILED (K=4, 2x2 grid) decoder variant."""
import json
import sys
sys.path.insert(0, "utils")
from build_model_report import ReportContext, eval_expr, PLATFORMS

model_dir = "models/seedvr2_2560x1536_tiled"
cfg = json.loads(open(f"{model_dir}/ops_mapping.json").read())
ktc = json.loads(open(f"{model_dir}/kernel_table_config.json").read())
metrics = json.loads(open("outputs/seedvr2_2560x1536_tiled/ptl_metrics.json").read())

ctx = ReportContext(metrics, cfg, "PTL", PLATFORMS["PTL"])
env = ctx.env()

for mode in ktc["modes"]:
    mode_env = dict(env)
    totals = {}
    for spec in mode["specs"]:
        log = eval_expr(spec["log"], mode_env)
        entry = metrics.get(log, {})
        total_ns = entry.get("total_kernel_ns", 0)
        calls = eval_expr(spec["calls"], mode_env)
        op = eval_expr(spec["op"], mode_env)
        total_ms = (total_ns / 1e6) * calls
        totals[op] = totals.get(op, 0.0) + total_ms
    grand_total = sum(totals.values())
    print(f"\n=== {mode['name']} (grand total {grand_total:.1f} ms) ===")
    for op, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {op:<28} {t:>12.2f} ms  {100*t/grand_total:5.1f}%")
