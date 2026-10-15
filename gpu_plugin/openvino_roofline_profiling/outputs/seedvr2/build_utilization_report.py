#!/usr/bin/env python3
"""Time-weighted hardware-utilization (Eff%) summary, per stage and overall,
split by compute-bound (% of FP16 XMX peak) vs memory-bound (% of peak BW)
kernel time. Reads the already-generated kernel_tables.md (no remote re-run).

Usage: python3 build_utilization_report.py [kernel_tables.md path]
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "kernel_tables.md"
    text = path.read_text()
    row_re = re.compile(
        r"^\| (\S.*?) \| `(\S+)` \| ([\d.]+) \| (\d+) \| ([\d.]+) \| (\d+) \| ([\d.]+) \| ([\d.]+)% \| (compute|memory) \|$",
        re.M,
    )

    grand_c_time = grand_c_eff = grand_m_time = grand_m_eff = grand_total = 0.0

    for block in re.split(r"^### ", text, flags=re.M)[1:]:
        stage = block.split()[0]
        c_time = c_eff = m_time = m_eff = 0.0
        for (_op, _kernel, _single, _calls, total_ms, _gflops, _gbs, eff, bound) in row_re.findall(block):
            total_ms = float(total_ms)
            eff = float(eff)
            if bound == "compute":
                c_time += total_ms
                c_eff += total_ms * eff
            else:
                m_time += total_ms
                m_eff += total_ms * eff
        stage_total = c_time + m_time
        if stage_total == 0:
            continue
        grand_total += stage_total
        grand_c_time += c_time
        grand_c_eff += c_eff
        grand_m_time += m_time
        grand_m_eff += m_eff

        print(f"--- {stage} (total {stage_total:,.1f} ms) ---")
        if c_time:
            print(f"  compute-bound: {c_time:,.1f} ms ({c_time/stage_total*100:.1f}% of stage) "
                  f"-- avg {c_eff/c_time:.1f}% of FP16 XMX peak")
        if m_time:
            print(f"  memory-bound:  {m_time:,.1f} ms ({m_time/stage_total*100:.1f}% of stage) "
                  f"-- avg {m_eff/m_time:.1f}% of peak BW")
        print(f"  blended time-weighted utilization: {(c_eff+m_eff)/stage_total:.1f}%")
        print()

    print("=== OVERALL (all stages) ===")
    print(f"total kernel time: {grand_total:,.1f} ms")
    if grand_c_time:
        print(f"compute-bound: {grand_c_time:,.1f} ms ({grand_c_time/grand_total*100:.1f}%) "
              f"-- avg {grand_c_eff/grand_c_time:.1f}% of FP16 XMX peak")
    if grand_m_time:
        print(f"memory-bound:  {grand_m_time:,.1f} ms ({grand_m_time/grand_total*100:.1f}%) "
              f"-- avg {grand_m_eff/grand_m_time:.1f}% of peak BW")
    print(f"BLENDED overall utilization: {(grand_c_eff+grand_m_eff)/grand_total:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
