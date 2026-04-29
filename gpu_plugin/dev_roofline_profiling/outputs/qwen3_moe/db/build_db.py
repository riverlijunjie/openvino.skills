#!/usr/bin/env python3
"""Build a SQLite database from parsed cliloader metrics for query-friendly analysis.

Schema (per SKILL.md §9 'Put all logs data into a database or structured format'):
  runs(model, platform, config, mode, kv_or_S, total_kernel_ns_per_iter, iters, log_path)
  kernels(model, platform, config, mode, kv_or_S, kernel,
          calls_total, calls_per_iter, per_iter_ns, avg_per_call_ns, share_in_run)

Use:
  python3 build_db.py
  sqlite3 metrics.db 'SELECT * FROM kernels WHERE platform="BMG" AND mode="decode"
                      ORDER BY per_iter_ns DESC LIMIT 20'
"""
import json, sqlite3, re
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent          # outputs/<model>/db/
OUT    = DB_DIR.parent                             # outputs/<model>/
DB     = DB_DIR / "metrics.db"
DB.unlink(missing_ok=True)
conn = sqlite3.connect(DB)
c = conn.cursor()
c.executescript("""
CREATE TABLE runs (
  model TEXT, platform TEXT, config TEXT,
  mode TEXT, kv_or_S INTEGER,
  total_kernel_ns_per_iter INTEGER,
  iters INTEGER,
  log_path TEXT,
  PRIMARY KEY(model, platform, config)
);
CREATE TABLE kernels (
  model TEXT, platform TEXT, config TEXT,
  mode TEXT, kv_or_S INTEGER,
  kernel TEXT,
  calls_total INTEGER, calls_per_iter INTEGER,
  per_iter_ns INTEGER, avg_per_call_ns INTEGER,
  share_in_run REAL
);
CREATE INDEX idx_runs_pm ON runs(platform, mode, kv_or_S);
CREATE INDEX idx_kern_pm ON kernels(platform, mode, kv_or_S);
CREATE INDEX idx_kern_name ON kernels(kernel);
""")

MODEL = "qwen3_moe"

def classify(cfg):
    if "decode" in cfg or cfg.endswith("_T1") or cfg.endswith("_M1"):
        m = re.search(r'kv(\d+)', cfg)
        return "decode", int(m.group(1)) if m else 1
    if "prefill" in cfg or re.search(r'_S\d+$', cfg):
        m = re.search(r'_S(\d+)', cfg)
        return "prefill", int(m.group(1)) if m else 0
    return "other", 0

for plat, fname in [("BMG", "bmg_metrics.json"), ("PTL", "ptl_metrics.json")]:
    data = json.loads((OUT/fname).read_text())
    for cfg, e in data.items():
        mode, x = classify(cfg)
        total = e.get("total_kernel_ns", 0)
        iters = e.get("iters_detected") or 1
        c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)",
                  (MODEL, plat, cfg, mode, x, total, iters, str(OUT/'logs'/f'{cfg}.log')))
        for entry in e.get("per_kernel", []):
            if not entry or len(entry) < 3: continue
            n, per_iter_ns, calls_total = entry
            kn = re.sub(r'_\d{15,}', '', n).strip('_')
            calls_per_iter = max(calls_total // iters, 1)
            avg = per_iter_ns // max(calls_per_iter, 1)
            share = per_iter_ns / total if total else 0
            c.execute("INSERT INTO kernels VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (MODEL, plat, cfg, mode, x, kn,
                       calls_total, calls_per_iter, per_iter_ns, avg, share))

conn.commit()
nr = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
nk = c.execute("SELECT COUNT(*) FROM kernels").fetchone()[0]
print(f"DB: {DB}")
print(f"  runs: {nr}, kernels: {nk}\n")

print("=== Sample queries ===\n")

print("Top BMG decode kernels (per-iter ms):")
for r in c.execute("""
  SELECT config, kernel, per_iter_ns/1e6 AS ms, calls_per_iter, share_in_run*100 AS pct
  FROM kernels WHERE platform='BMG' AND mode='decode'
  ORDER BY ms DESC LIMIT 8""").fetchall():
    print(f"  {r[0]:30s} {r[1]:50s} {r[2]:8.4f} ms calls/iter={r[3]} share={r[4]:5.1f}%")

print("\nMoE prefill scaling (BMG vs PTL, per-iter ms):")
for r in c.execute("""
  SELECT kv_or_S,
    SUM(CASE WHEN platform='BMG' THEN per_iter_ns ELSE 0 END)/1e6 AS bmg_ms,
    SUM(CASE WHEN platform='PTL' THEN per_iter_ns ELSE 0 END)/1e6 AS ptl_ms
  FROM kernels WHERE config LIKE 'moe_prefill_S%'
  GROUP BY kv_or_S ORDER BY kv_or_S""").fetchall():
    s, bmg, ptl = r
    ratio = (ptl/bmg) if bmg else 0
    print(f"  S={s:6d}  BMG {bmg:9.2f} ms  PTL {ptl:9.2f} ms  ratio {ratio:.2f}x")

print("\nPA decode vs kv (BMG, per-iter ms):")
for r in c.execute("""
  SELECT kv_or_S, SUM(per_iter_ns)/1e6 AS ms
  FROM kernels WHERE platform='BMG' AND config LIKE 'pa_decode_kv%'
  GROUP BY kv_or_S ORDER BY kv_or_S""").fetchall():
    print(f"  kv={r[0]:6d}  {r[1]:.4f} ms")

conn.close()
