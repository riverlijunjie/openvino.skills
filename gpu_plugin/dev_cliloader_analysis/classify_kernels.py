import re
import sys
from collections import defaultdict


def extract_base_name(name):
    """Strip numeric hash + SIMD/GWS/LWS config to get base kernel name."""
    if name.startswith('clEnqueue'):
        m = re.match(r'(clEnqueue\w+)\(\s*(\w+)', name)
        if m:
            return f"{m.group(1)}({m.group(2)})"
        return name.split('(')[0]
    base = re.split(r'\s+SIMD', name)[0]
    base = re.sub(r'_\d{8,}.*$', '', base)
    base = base.rstrip('_')
    return base


def parse_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Extract total time from log
    total_time_ns = None
    for l in lines:
        m = re.match(r'Total Time \(ns\):\s*(\d+)', l.strip())
        if m:
            total_time_ns = int(m.group(1))
            break
    if total_time_ns is None:
        print("ERROR: Could not find 'Total Time (ns)' in log file.")
        sys.exit(1)

    # Find data start
    data_start = None
    for i, l in enumerate(lines):
        if 'Function Name' in l and 'Calls' in l:
            data_start = i + 1
            break
    if data_start is None:
        print("ERROR: Could not find header line.")
        sys.exit(1)

    records = []
    for l in lines[data_start:]:
        l = l.strip()
        if not l or 'CLIntercept' in l or 'shutdown' in l:
            continue
        parts = l.rsplit(',', 6)
        if len(parts) < 7:
            continue
        name = parts[0].strip()
        try:
            calls = int(parts[1].strip())
            time_ns = int(parts[2].strip())
            avg_ns = int(parts[4].strip())
            min_ns = int(parts[5].strip())
            max_ns = int(parts[6].strip())
        except (ValueError, IndexError):
            continue
        records.append((name, calls, time_ns, avg_ns, min_ns, max_ns))

    return total_time_ns, records


def classify_kernels(total_time_ns, records):
    groups = defaultdict(lambda: {
        'time': 0, 'calls': 0,
        'min': float('inf'), 'max': 0, 'entries': 0
    })
    for r in records:
        bn = extract_base_name(r[0])
        groups[bn]['time'] += r[2]
        groups[bn]['calls'] += r[1]
        groups[bn]['min'] = min(groups[bn]['min'], r[4])
        groups[bn]['max'] = max(groups[bn]['max'], r[5])
        groups[bn]['entries'] += 1
    return groups


def format_output(total_time_ns, groups):
    out = []
    sep = '=' * 120
    out.append(sep)
    out.append("KERNEL CLASSIFICATION BY BASE NAME (stripping numeric hash + SIMD/GWS/LWS config)")
    out.append(f"Total GPU Time: {total_time_ns / 1e9:.3f}s")
    out.append(sep)
    out.append("")
    out.append(
        f"{'#':<4} {'Base Kernel Name':<55} {'Total(ms)':>10} {'Time%':>7} "
        f"{'Calls':>8} {'Configs':>7} {'Avg(us)':>10} {'Min(us)':>10} {'Max(us)':>10}"
    )
    out.append('-' * 120)

    sorted_groups = sorted(groups.items(), key=lambda x: -x[1]['time'])
    for idx, (bn, v) in enumerate(sorted_groups):
        avg = v['time'] / v['calls'] if v['calls'] else 0
        pct = v['time'] / total_time_ns * 100
        out.append(
            f"{idx + 1:<4} {bn:<55} {v['time'] / 1e6:>10.1f} {pct:>6.2f}% "
            f"{v['calls']:>8,} {v['entries']:>7} {avg / 1e3:>10.1f} "
            f"{v['min'] / 1e3:>10.1f} {v['max'] / 1e3:>10.1f}"
        )

    out.append("")
    out.append("CUMULATIVE TIME:")
    out.append('-' * 80)
    cum = 0
    for idx, (bn, v) in enumerate(sorted_groups):
        cum += v['time']
        pct = v['time'] / total_time_ns * 100
        cum_pct = cum / total_time_ns * 100
        out.append(f"{idx + 1:<4} {bn:<55} {pct:>6.2f}%  cum={cum_pct:>6.2f}%")
        if cum_pct > 99.5:
            out.append("     ... remaining kernels < 0.5% ...")
            break

    return '\n'.join(out)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <cliloader_log> [output_file]")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    total_time_ns, records = parse_log(log_path)
    groups = classify_kernels(total_time_ns, records)
    result = format_output(total_time_ns, groups)

    print(result)
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)
        print(f"\nWritten to {output_path}")


if __name__ == '__main__':
    main()
