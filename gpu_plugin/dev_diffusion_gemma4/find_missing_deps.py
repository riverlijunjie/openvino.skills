#!/usr/bin/env python3
"""Recursive include-closure to find files the ported MoE code needs but that are
absent in ovmx while present in the source-of-truth (SoT). Basename-based heuristic."""
import os, re, subprocess, sys
from collections import defaultdict, deque

OVMX = "/home/ov2022/workspace/ovmx/openvino.mx"
SOT  = "/home/ov2022/workspace/code_review/openvino"
INC = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)

def ls(repo):
    tracked = subprocess.check_output(["git", "ls-files", "src/"], cwd=repo, text=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src/"], cwd=repo, text=True)
    return [l for l in (tracked + untracked).splitlines() if l]

def index(files):
    idx = defaultdict(list)
    for f in files:
        idx[os.path.basename(f)].append(f)
    return idx

ovmx_files = ls(OVMX); sot_files = ls(SOT)
ovmx_idx = index(ovmx_files); sot_idx = index(sot_files)
ovmx_base = set(ovmx_idx); sot_base = set(sot_idx)

# Seed: all moe-named files present in ovmx, plus the two known failing files.
seed = [f for f in ovmx_files if "moe" in f.lower() and "onednn" not in f]
worklist = deque(seed)
seen = set(seed)
missing = {}          # basename -> sot path
visited_includes = set()

def read(repo, path):
    p = os.path.join(repo, path)
    try:
        with open(p, "r", errors="ignore") as fh: return fh.read()
    except OSError: return ""

while worklist:
    f = worklist.popleft()
    # read from ovmx if present else SoT
    text = read(OVMX, f) if os.path.exists(os.path.join(OVMX, f)) else read(SOT, f)
    for inc in INC.findall(text):
        base = os.path.basename(inc)
        if base in visited_includes: continue
        visited_includes.add(base)
        if base in ovmx_base:
            continue  # resolvable in ovmx (good enough heuristic)
        if base in sot_base:
            # missing in ovmx, present in SoT -> needs porting
            # choose the SoT path whose suffix best matches the include string
            cands = sot_idx[base]
            best = next((c for c in cands if c.endswith(inc)), cands[0])
            if base not in missing:
                missing[base] = best
                if best not in seen:
                    seen.add(best); worklist.append(best)
        # else: system/header not in either -> ignore

print("=== MISSING files (absent in ovmx, present in SoT), include-closure of MoE code ===")
for base in sorted(missing):
    print(missing[base])
print(f"\nTOTAL MISSING: {len(missing)}")
