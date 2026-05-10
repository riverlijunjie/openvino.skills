#!/usr/bin/env python3
"""
ov_verbose_weight_analyze.py
============================
Analyzes OpenVINO GPU verbose log files to extract model weight constants
and produce a comprehensive report: per-category table, per-weight detail,
precision summary, architecture insights and key observations.

Usage:
    python ov_verbose_weight_analyze.py <log_file> [--encoding utf-16|utf-8] [--out report.txt]

Example:
    python ov_verbose_weight_analyze.py qwen3.5_verbose.log
"""

import re
import sys
import os
import argparse
from collections import defaultdict

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# ANSI color helpers (auto-disabled when stdout is redirected)
# ---------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def bold(t):    return _c("1",    t)
def cyan(t):    return _c("1;36", t)
def green(t):   return _c("1;32", t)
def yellow(t):  return _c("1;33", t)
def magenta(t): return _c("1;35", t)
def red(t):     return _c("1;31", t)


# ---------------------------------------------------------------------------
# Regex pattern  –  matches create_data lines like:
#   [constant:<name>: constant] layout: <prec>:<fmt>:<shape>:<pad>, mem_ptr(..., <N> bytes)
# ---------------------------------------------------------------------------
PATTERN = re.compile(
    r'\[constant:([^:\]]+):\s*constant\]\s+'
    r'layout:\s*([a-zA-Z0-9_]+)'   # precision  e.g. f16, u4, i8, bf16
    r':([a-zA-Z0-9_]+)'            # layout fmt e.g. bfyx, bfzyx
    r':([^,]+?)'                   # shape      e.g. 256x512x2048
    r'(?::nopad|:nopd|:noppad)?'   # optional pad token
    r',\s*mem_ptr\([^,]+,\s*(\d+)\s*bytes\)'  # byte count
)

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


# ---------------------------------------------------------------------------
# Weight categorisation
# ---------------------------------------------------------------------------
def categorize(name: str) -> str:
    n = name.lower()

    # Vision encoder / merger
    if any(x in n for x in ('patch_embed', 'patch embed', 'conv_proj')):
        return 'Vision Patch Embed'
    if 'merger' in n and 'fc1' in n:  return 'Vision Merger FC1'
    if 'merger' in n and 'fc2' in n:  return 'Vision Merger FC2'
    if 'merger' in n:                 return 'Vision Merger Other'

    # Embedding / LM head
    if 'embed_tokens' in n or ('embed' in n and 'weight' in n and 'vision' not in n):
        return 'Token Embedding'
    if 'lm_head' in n or 'output.weight' in n:
        return 'LM Head'

    # MoE expert weights (routed)
    if 'expert' in n and 'shared' not in n:
        if 'gate_proj' in n or ('gate' in n and 'proj' in n):
            return 'MoE Expert gate_proj'
        if 'up_proj'   in n:   return 'MoE Expert up_proj'
        if 'down_proj' in n:   return 'MoE Expert down_proj'
        if 'scale' in n:       return 'MoE Expert Scale'
        if 'zp' in n or 'zero' in n: return 'MoE Expert ZeroPoint'
        return 'MoE Expert Other'

    # Shared expert
    if 'shared_expert' in n or ('shared' in n and 'expert' in n):
        if 'gate_proj' in n:   return 'Shared Expert gate_proj'
        if 'up_proj'   in n:   return 'Shared Expert up_proj'
        if 'down_proj' in n:   return 'Shared Expert down_proj'
        return 'Shared Expert Other'

    # MoE gate (routing)
    if 'mlp.gate' in n or ('gate' in n and 'weight' in n and 'proj' not in n
                            and 'expert' not in n):
        return 'MoE Gate'

    # Standard attention projections (non-vision)
    if 'vision' not in n and 'vit' not in n:
        if 'q_proj' in n and ('scale' not in n and 'zp' not in n):
            return 'Attn Q Proj'
        if 'k_proj' in n and ('scale' not in n and 'zp' not in n):
            return 'Attn K Proj'
        if 'v_proj' in n and ('scale' not in n and 'zp' not in n):
            return 'Attn V Proj'
        if ('o_proj' in n or ('self' in n and 'proj' in n)) and \
           ('scale' not in n and 'zp' not in n):
            return 'Attn O Proj'

    # Linear / RWKV attention
    if 'in_proj_qkv' in n: return 'Linear Attn in_proj_qkv'
    if 'in_proj_z'   in n: return 'Linear Attn in_proj_z'
    if 'out_proj'    in n and 'vision' not in n: return 'Linear Attn out_proj'
    if 'in_proj_a'   in n: return 'Linear Attn in_proj_a'
    if 'in_proj_b'   in n: return 'Linear Attn in_proj_b'
    if 'rwkv' in n or 'linear_attn' in n: return 'Linear Attn Other'

    # Vision encoder attention / MLP
    if 'vision' in n or 'vit' in n:
        if any(x in n for x in ('q_proj', 'k_proj', 'v_proj', 'qkv')):
            return 'ViT Attn QKV'
        if 'proj' in n or 'out' in n:   return 'ViT Attn Proj'
        if 'fc1' in n or 'mlp.0' in n:  return 'ViT MLP FC1'
        if 'fc2' in n or 'mlp.2' in n:  return 'ViT MLP FC2'
        if 'scale' in n:                return 'ViT Scale/ZP'
        return 'ViT Other'

    # Norm layers
    if any(x in n for x in ('norm', 'ln_', 'layernorm', 'rmsnorm')):
        return 'LayerNorm / RMSNorm'

    # Quantisation artefacts
    if 'scale' in n:  return 'Quant Scale'
    if 'zp' in n or 'zero_point' in n: return 'Quant ZeroPoint'

    # Bias / RoPE
    if 'bias' in n:             return 'Bias'
    if 'rope' in n or 'rotary' in n: return 'RoPE / Positional'
    if 'proj.weight' in n:      return 'Vision Proj Weight'

    return 'Other / Misc Constant'


# ---------------------------------------------------------------------------
# Byte -> human readable
# ---------------------------------------------------------------------------
def fmt_size(b: int) -> str:
    if b == 0:        return "0 B"
    if b < 1024:      return f"{b} B"
    if b < 1 << 20:   return f"{b/1024:.2f} KB"
    if b < 1 << 30:   return f"{b/1024/1024:.2f} MB"
    return f"{b/1024/1024/1024:.3f} GB"


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def hr(char='=', width=130):
    return char * width

def section(title: str) -> str:
    pad = (128 - len(title)) // 2
    return f"\n{hr()}\n{' '*pad}{bold(title)}\n{hr()}"


# ---------------------------------------------------------------------------
# Main parse
# ---------------------------------------------------------------------------
def parse_log(log_path: str, encoding: str) -> dict:
    """Return dict: name -> {precision, layout_format, shape, bytes}"""
    weights = {}
    total_lines = 0
    matched = 0

    print(cyan(f"[INFO] Scanning: {log_path}"))
    print(cyan(f"[INFO] Encoding: {encoding}"))
    print(cyan(f"[INFO] File size: {fmt_size(os.path.getsize(log_path))}"))
    print()

    with open(log_path, 'r', encoding=encoding, errors='replace') as fh:
        for line in fh:
            total_lines += 1
            if 'create_data' not in line:
                continue
            clean = ANSI_RE.sub('', line)
            m = PATTERN.search(clean)
            if m:
                name      = m.group(1).strip()
                precision = m.group(2).strip()
                fmt       = m.group(3).strip()
                shape     = m.group(4).strip()
                nbytes    = int(m.group(5))
                weights[name] = dict(precision=precision, layout_format=fmt,
                                     shape=shape, bytes=nbytes,
                                     category=categorize(name))
                matched += 1

    print(green(f"[INFO] Lines scanned : {total_lines:,}"))
    print(green(f"[INFO] Constants found: {matched:,}  |  unique names: {len(weights):,}"))
    return weights


# ---------------------------------------------------------------------------
# Build category summary
# ---------------------------------------------------------------------------
def build_category_summary(weights: dict):
    """Return list of dicts, sorted by total_bytes desc."""
    by_cat = defaultdict(list)
    for name, info in weights.items():
        by_cat[info['category']].append((name, info))

    rows = []
    for cat, items in by_cat.items():
        total_bytes = sum(i['bytes'] for _, i in items)
        # representative = largest single tensor
        rep_name, rep_info = max(items, key=lambda x: x[1]['bytes'])
        # precision frequency
        prec_freq = defaultdict(int)
        for _, i in items:
            prec_freq[i['precision']] += 1
        main_prec = max(prec_freq, key=prec_freq.get)
        rows.append(dict(
            category   = cat,
            count      = len(items),
            total_bytes= total_bytes,
            rep_name   = rep_name,
            rep_prec   = rep_info['precision'],
            rep_shape  = rep_info['shape'],
            rep_bytes  = rep_info['bytes'],
            main_prec  = main_prec,
            items      = items,
        ))
    rows.sort(key=lambda r: -r['total_bytes'])
    return rows


# ---------------------------------------------------------------------------
# Precision overview
# ---------------------------------------------------------------------------
def build_precision_summary(weights: dict):
    prec_info = defaultdict(lambda: dict(count=0, total_bytes=0, categories=set()))
    for name, info in weights.items():
        p = info['precision']
        prec_info[p]['count']      += 1
        prec_info[p]['total_bytes'] += info['bytes']
        prec_info[p]['categories'].add(info['category'])
    return dict(sorted(prec_info.items(), key=lambda x: -x[1]['total_bytes']))


# ---------------------------------------------------------------------------
# Print functions
# ---------------------------------------------------------------------------
def print_category_table(rows, total_bytes, out):
    W = 160
    out.write(f"\n{'='*W}\n")
    out.write(bold("SECTION 1 — WEIGHT SUMMARY BY CATEGORY  (sorted by total size)\n"))
    out.write(f"{'='*W}\n")

    hdr = (f"{'#':>3}  {'Category':<30}  {'Count':>6}  {'Main Prec':<10}  "
           f"{'Largest Shape':<38}  {'Largest Tensor':>16}  "
           f"{'Category Total':>16}  {'% of All':>8}")
    out.write(bold(hdr) + "\n")
    out.write(f"{'-'*W}\n")

    grand = total_bytes
    for i, r in enumerate(rows, 1):
        pct = 100.0 * r['total_bytes'] / grand if grand else 0
        shape_str = r['rep_shape']
        if len(shape_str) > 37:
            shape_str = shape_str[:34] + '...'
        line = (f"{i:>3}  {r['category']:<30}  {r['count']:>6}  "
                f"{r['main_prec']:<10}  {shape_str:<38}  "
                f"{fmt_size(r['rep_bytes']):>16}  "
                f"{fmt_size(r['total_bytes']):>16}  {pct:>7.2f}%")
        out.write(line + "\n")

    out.write(f"{'-'*W}\n")
    out.write(bold(f"{'GRAND TOTAL':<49}{sum(r['count'] for r in rows):>6}"
                   f"{'':>67}{fmt_size(grand):>16}{'':>10}\n"))
    out.write(f"{'='*W}\n")


def print_precision_table(prec_summary, total_bytes, out):
    W = 120
    out.write(f"\n{'='*W}\n")
    out.write(bold("SECTION 2 — PRECISION BREAKDOWN\n"))
    out.write(f"{'='*W}\n")

    hdr = f"{'Precision':<12}  {'Count':>6}  {'Total Size':>16}  {'% of All':>9}  {'Used for (categories)'}"
    out.write(bold(hdr) + "\n")
    out.write(f"{'-'*W}\n")

    for prec, info in prec_summary.items():
        pct  = 100.0 * info['total_bytes'] / total_bytes if total_bytes else 0
        cats = ', '.join(sorted(info['categories'])[:4])
        if len(info['categories']) > 4:
            cats += f" (+{len(info['categories'])-4} more)"
        out.write(f"{prec:<12}  {info['count']:>6}  {fmt_size(info['total_bytes']):>16}"
                  f"  {pct:>8.2f}%  {cats}\n")

    out.write(f"{'='*W}\n")


def print_detailed_table(weights, out):
    W = 160
    out.write(f"\n{'='*W}\n")
    out.write(bold("SECTION 3 — ALL UNIQUE WEIGHT CONSTANTS  (sorted by size desc)\n"))
    out.write(f"{'='*W}\n")

    hdr = (f"{'#':>5}  {'Weight Name':<65}  {'Category':<28}  "
           f"{'Prec':<8}  {'Format':<8}  {'Shape':<38}  {'Size':>14}")
    out.write(bold(hdr) + "\n")
    out.write(f"{'-'*W}\n")

    sorted_w = sorted(weights.items(), key=lambda x: -x[1]['bytes'])
    for idx, (name, info) in enumerate(sorted_w, 1):
        short = name if len(name) <= 64 else '...' + name[-61:]
        shape = info['shape']
        if len(shape) > 37:
            shape = shape[:34] + '...'
        out.write(f"{idx:>5}  {short:<65}  {info['category']:<28}  "
                  f"{info['precision']:<8}  {info['layout_format']:<8}  "
                  f"{shape:<38}  {fmt_size(info['bytes']):>14}\n")

    out.write(f"{'='*W}\n")


def print_top_per_category(rows, n, out):
    W = 140
    out.write(f"\n{'='*W}\n")
    out.write(bold(f"SECTION 4 — TOP {n} TENSORS PER CATEGORY\n"))
    out.write(f"{'='*W}\n")

    for r in rows:
        out.write(f"\n  {cyan(r['category'])}  "
                  f"({r['count']} tensors, total {fmt_size(r['total_bytes'])})\n")
        out.write(f"  {'-'*135}\n")
        top = sorted(r['items'], key=lambda x: -x[1]['bytes'])[:n]
        for name, info in top:
            short = name if len(name) <= 64 else '...' + name[-61:]
            out.write(f"    {short:<65}  {info['precision']:<8}  "
                      f"{info['shape']:<40}  {fmt_size(info['bytes']):>14}\n")

    out.write(f"\n{'='*W}\n")


def print_observations(weights, rows, prec_summary, total_bytes, out):
    W = 120
    out.write(f"\n{'='*W}\n")
    out.write(bold("SECTION 5 — KEY OBSERVATIONS\n"))
    out.write(f"{'='*W}\n\n")

    # compute top-3 categories
    top3 = rows[:3]
    for rank, r in enumerate(top3, 1):
        pct = 100.0 * r['total_bytes'] / total_bytes if total_bytes else 0
        out.write(f"  {rank}. [{r['category']}] is the #{rank} largest consumer: "
                  f"{fmt_size(r['total_bytes'])} ({pct:.1f}% of all weights)\n")
        out.write(f"     -> {r['count']} tensors, representative shape: {r['rep_shape']}, "
                  f"precision: {r['rep_prec']}\n\n")

    # precision observation
    out.write("  Precision usage:\n")
    for prec, info in prec_summary.items():
        pct = 100.0 * info['total_bytes'] / total_bytes if total_bytes else 0
        if pct >= 0.1:
            out.write(f"    * {prec:<8}: {fmt_size(info['total_bytes']):>12}  ({pct:.2f}%)\n")

    # count by precision
    out.write("\n  Tensor count by precision:\n")
    prec_cnt = defaultdict(int)
    for _, info in weights.items():
        prec_cnt[info['precision']] += 1
    for prec, cnt in sorted(prec_cnt.items(), key=lambda x: -x[1]):
        out.write(f"    * {prec:<8}: {cnt:>6} tensors\n")

    # rough param count estimate (based on bytes & precision)
    out.write("\n  Estimated parameter count (by precision element size):\n")
    prec_elem = {'f32':4,'f16':2,'bf16':2,'i8':1,'u8':1,'i4':0.5,'u4':0.5,'i32':4,'i64':8}
    total_params = 0
    by_prec_params = defaultdict(int)
    for _, info in weights.items():
        es = prec_elem.get(info['precision'], 1)
        params = int(info['bytes'] / es)
        total_params += params
        by_prec_params[info['precision']] += params
    for prec, params in sorted(by_prec_params.items(), key=lambda x: -x[1]):
        out.write(f"    * {prec:<8}: ~{params/1e9:.3f} B parameters\n")
    out.write(f"    >> Total (all precisions): ~{total_params/1e9:.3f} B parameter equivalents\n")

    out.write(f"\n{'='*W}\n")


# ---------------------------------------------------------------------------
# Report header / footer
# ---------------------------------------------------------------------------
def print_header(log_path, weights, total_bytes, out):
    out.write(f"\n{'#'*130}\n")
    out.write(bold("  OpenVINO Verbose Log — Model Weight Analysis Report\n"))
    out.write(f"{'#'*130}\n\n")
    out.write(f"  Log file  : {log_path}\n")
    out.write(f"  File size : {fmt_size(os.path.getsize(log_path))}\n")
    out.write(f"  Unique constants  : {len(weights):,}\n")
    out.write(f"  Grand total weight: {fmt_size(total_bytes)}\n\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze OpenVINO GPU verbose log to extract and summarize model weights."
    )
    parser.add_argument("log_file", help="Path to the verbose log file")
    parser.add_argument(
        "--encoding", default="utf-16",
        choices=["utf-16", "utf-8", "utf-16-le", "utf-16-be", "latin-1"],
        help="Log file text encoding (default: utf-16)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to write the report (default: print to stdout)"
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top tensors to show per category (default: 5)"
    )
    parser.add_argument(
        "--no-detail", action="store_true",
        help="Skip the full per-tensor detail table (Section 3) to keep output compact"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        print(red(f"[ERROR] File not found: {args.log_file}"), file=sys.stderr)
        sys.exit(1)

    # --- Parse ---
    weights = parse_log(args.log_file, args.encoding)
    if not weights:
        print(red("[ERROR] No weight constants found. Check encoding or log format."),
              file=sys.stderr)
        sys.exit(1)

    total_bytes   = sum(i['bytes'] for i in weights.values())
    cat_rows      = build_category_summary(weights)
    prec_summary  = build_precision_summary(weights)

    # --- Render ---
    if args.out:
        fout = open(args.out, 'w', encoding='utf-8')
        # Disable ANSI in file output
        global USE_COLOR
        USE_COLOR = False
    else:
        fout = sys.stdout

    try:
        print_header(args.log_file, weights, total_bytes, fout)
        print_category_table(cat_rows, total_bytes, fout)
        print_precision_table(prec_summary, total_bytes, fout)
        if not args.no_detail:
            print_detailed_table(weights, fout)
        print_top_per_category(cat_rows, args.top, fout)
        print_observations(weights, cat_rows, prec_summary, total_bytes, fout)

        fout.write("\n[Done]\n")

    finally:
        if args.out:
            fout.close()
            print(green(f"\n[INFO] Report written to: {args.out}"))


if __name__ == "__main__":
    main()
