#!/usr/bin/env python3
"""Aggregate peak GPU memory (usm_device + usm_host) reported by conv_bench /
groupnorm_bench across a parsed metrics JSON (output of utils/parse_logs.py).

Usage: python3 build_memory_report.py <parsed_metrics.json> [budget_gib=20]
"""
import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: build_memory_report.py <parsed_metrics.json> [budget_gib=20]")
        return 1
    path = sys.argv[1]
    budget_gib = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    data = json.load(open(path))

    rows = []
    for log, entry in data.items():
        dev = entry.get("peak_usm_device_bytes", 0) or 0
        host = entry.get("peak_usm_host_bytes", 0) or 0
        if dev == 0 and host == 0:
            continue  # bench doesn't report memory stats (fc/sdpa/vlsdpa) or log empty
        rows.append((log, dev, host, dev + host))

    rows.sort(key=lambda r: -r[3])

    GIB = 1024.0 ** 3
    print(f"{'Op/log':<38} {'usm_device (GiB)':>16} {'usm_host (GiB)':>15} {'total (GiB)':>12}")
    print("-" * 85)
    worst = 0.0
    for log, dev, host, total in rows:
        print(f"{log:<38} {dev/GIB:>16.3f} {host/GIB:>15.3f} {total/GIB:>12.3f}")
        worst = max(worst, total / GIB)

    print("-" * 85)
    print(f"Peak single-op GPU memory observed: {worst:.3f} GiB (budget: {budget_gib:.1f} GiB)")
    if worst > budget_gib:
        print(f"WARNING: exceeded {budget_gib:.1f} GiB budget!")
    else:
        print(f"OK: within {budget_gib:.1f} GiB budget ({budget_gib - worst:.3f} GiB headroom).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
