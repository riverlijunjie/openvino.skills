#!/usr/bin/env python3
"""
Parse a directory of cliloader Device Performance Timing logs and
extract GPU kernel stats (avg ns per iteration) per log.

For each log, we sum avg(ns) for all GPU kernels (excluding clEnqueue*
host-side API calls and memcpy/memfill) to get the "pure kernel time"
per iteration.  This handles benches where the op decomposes into
multiple kernels (e.g. PA = kv_update + attention + finalization).
"""
import re
import sys
import json
from pathlib import Path
from collections import defaultdict


KERNEL_EXCLUDES = (
    "clEnqueue", "clFinish", "clWait",
    "clFlush", "clRelease", "clRetain",
    "clSetKernel",
    # L2/L3 cache-flush helper kernels. fc_bench/pa_bench enqueue a large Relu
    # (Parameter → Relu → Result) between every infer to evict cached weights
    # so the measured kernel actually reads from VRAM. The Relu is compiled by
    # the GPU plugin into an `activation_*` primitive. Neither FC nor PA nor
    # small-ops benches produce activation kernels in their own data paths
    # (those use gemm/fc, pa_*, rmsnorm/rope/eltwise respectively), so
    # excluding "activation" globally is safe.
    "activation",
)


def parse_device_timing(path: Path):
    """Return dict of {kernel_name: {count,avg_ns,min_ns,max_ns,total_ns}}.
    Only GPU kernels (lines after 'Device Performance Timing Results').
    """
    text = path.read_text(errors="ignore")
    # Find the Device section
    dev_idx = text.find("Device Performance Timing Results")
    if dev_idx < 0:
        return {}
    section = text[dev_idx:]
    kernels = {}
    for line in section.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        # Format: "Function Name,  Calls,     Time (ns), Time (%),  Average (ns), Min, Max"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        name = parts[0]
        if not name or name.startswith("Function Name"):
            continue
        # Skip host API calls we don't want (memfill, memcpy are OK to track separately)
        if any(name.startswith(x) for x in KERNEL_EXCLUDES):
            continue
        try:
            calls = int(parts[1])
            total_ns = int(parts[2])
            avg_ns = int(parts[4])
            min_ns = int(parts[5])
            max_ns = int(parts[6])
        except (ValueError, IndexError):
            continue
        kernels[name] = dict(calls=calls, total_ns=total_ns,
                             avg_ns=avg_ns, min_ns=min_ns, max_ns=max_ns)
    return kernels


def classify(name: str) -> str:
    """Return a coarse category: 'kernel' for real GPU compute, 'io' for memcpy/memfill."""
    if name.startswith("clEnqueueMem"):
        return "io"
    return "kernel"


def summarize_log(path: Path):
    kernels = parse_device_timing(path)
    # Determine expected iterations-per-bench from max call count among GPU kernels.
    # Per-iteration contribution = total_ns / max_calls. This correctly excludes
    # one-off kernels like reorder_data (weight-layout conversion, calls=1).
    gpu_kernels = {n: k for n, k in kernels.items() if classify(n) == "kernel"}
    max_calls = max((k["calls"] for k in gpu_kernels.values()), default=1)
    total_kernel_ns = 0
    per_kernel = []
    for name, stat in gpu_kernels.items():
        per_iter_ns = stat["total_ns"] / max_calls if max_calls > 0 else 0
        total_kernel_ns += per_iter_ns
        per_kernel.append((name, int(per_iter_ns), stat["calls"]))
    per_kernel.sort(key=lambda x: -x[1])
    return {
        "total_kernel_ns": int(total_kernel_ns),
        "iters_detected": max_calls,
        "per_kernel": per_kernel,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_logs_v2.py <logdir> [output.json]")
        sys.exit(1)
    log_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    results = {}
    for log in sorted(log_dir.glob("*.log")):
        r = summarize_log(log)
        results[log.stem] = r

    if out_path:
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_path}")

    # Print summary
    print(f"{'Log':<45} {'Total kernel ns':>15}  Top kernel")
    print("-" * 100)
    for name in sorted(results):
        r = results[name]
        top = r["per_kernel"][0] if r["per_kernel"] else ("", 0, 0)
        print(f"{name:<45} {r['total_kernel_ns']:>15,}  {top[0][:50]} ({top[1]:,}ns ×{top[2]})")


if __name__ == "__main__":
    main()
