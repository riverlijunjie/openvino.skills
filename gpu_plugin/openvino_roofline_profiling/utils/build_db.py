#!/usr/bin/env python3
"""Build a SQLite database from parsed cliloader metrics. Model-agnostic.

Per SKILL.md §9 ("Put all logs data into a database or structured format for
easy querying and analysis"). Schema below; one DB can hold many models so the
same `db/metrics.db` file lets you compare across models and platforms.

Usage:
    # Default: ingest every model under outputs/ into db/metrics.db
    python3 utils/build_db.py

    # Single model
    python3 utils/build_db.py --model qwen3_moe

    # Custom DB location / output dir
    python3 utils/build_db.py --out my.db --outputs-dir outputs

Schema:
    runs(model, platform, config, mode, kv_or_S,
         total_kernel_ns_per_iter, iters, log_path)
    kernels(model, platform, config, mode, kv_or_S, kernel,
            calls_total, calls_per_iter,
            per_iter_ns, avg_per_call_ns, share_in_run)

Sample queries:
    sqlite3 db/metrics.db \\
      "SELECT model, kernel, per_iter_ns/1e6 AS ms FROM kernels
       WHERE platform='BMG' AND mode='decode'
       ORDER BY ms DESC LIMIT 10;"
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # skill root
DEFAULT_OUTPUTS = ROOT / "outputs"
DEFAULT_DB = ROOT / "db" / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  model TEXT, platform TEXT, config TEXT,
  mode TEXT, kv_or_S INTEGER,
  total_kernel_ns_per_iter INTEGER,
  iters INTEGER,
  log_path TEXT,
  PRIMARY KEY(model, platform, config)
);
CREATE TABLE IF NOT EXISTS kernels (
  model TEXT, platform TEXT, config TEXT,
  mode TEXT, kv_or_S INTEGER,
  kernel TEXT,
  calls_total INTEGER, calls_per_iter INTEGER,
  per_iter_ns INTEGER, avg_per_call_ns INTEGER,
  share_in_run REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_pm ON runs(platform, mode, kv_or_S);
CREATE INDEX IF NOT EXISTS idx_kern_pm ON kernels(platform, mode, kv_or_S);
CREATE INDEX IF NOT EXISTS idx_kern_name ON kernels(kernel);
CREATE INDEX IF NOT EXISTS idx_kern_model ON kernels(model);
"""


def classify(cfg: str):
    """Derive (mode, kv_or_S) from a bench-config name like 'pa_decode_kv4096' or 'moe_prefill_S8192'."""
    if "decode" in cfg or cfg.endswith("_T1") or cfg.endswith("_M1"):
        m = re.search(r'kv(\d+)', cfg)
        return "decode", int(m.group(1)) if m else 1
    if "prefill" in cfg or re.search(r'_S\d+$', cfg):
        m = re.search(r'_S(\d+)', cfg)
        return "prefill", int(m.group(1)) if m else 0
    return "other", 0


def ingest_model(c, model: str, model_dir: Path) -> tuple[int, int]:
    """Insert all (BMG/PTL/...) metrics JSONs found under outputs/<model>/. Returns (#runs, #kernels)."""
    nr = nk = 0
    for fname in sorted(model_dir.glob("*_metrics.json")):
        plat = fname.stem.replace("_metrics", "").upper()
        data = json.loads(fname.read_text())
        for cfg, e in data.items():
            mode, x = classify(cfg)
            total = e.get("total_kernel_ns", 0)
            iters = e.get("iters_detected") or 1
            log_path = str(model_dir / "logs" / f"{cfg}.log")
            c.execute(
                "INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?)",
                (model, plat, cfg, mode, x, total, iters, log_path),
            )
            nr += 1
            for entry in e.get("per_kernel", []) or []:
                if not entry or len(entry) < 3:
                    continue
                n, per_iter_ns, calls_total = entry
                kn = re.sub(r'_\d{15,}', '', n).strip('_')
                calls_per_iter = max(calls_total // iters, 1)
                avg = per_iter_ns // max(calls_per_iter, 1)
                share = per_iter_ns / total if total else 0
                c.execute(
                    "INSERT INTO kernels VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (model, plat, cfg, mode, x, kn,
                     calls_total, calls_per_iter, per_iter_ns, avg, share),
                )
                nk += 1
    return nr, nk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_DB, help="SQLite output (default: db/metrics.db)")
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS, help="Outputs root (default: outputs/)")
    ap.add_argument("--model", action="append", help="Restrict ingest to specific model(s). Repeatable. Default: all models found under outputs/.")
    ap.add_argument("--reset", action="store_true", help="Delete the DB file before rebuilding (instead of upserting).")
    args = ap.parse_args()

    if args.reset and args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.out)
    c = conn.cursor()
    c.executescript(SCHEMA)

    if args.model:
        models = args.model
    else:
        models = [p.name for p in args.outputs_dir.iterdir() if p.is_dir()]

    total_r = total_k = 0
    for m in sorted(models):
        mdir = args.outputs_dir / m
        if not mdir.is_dir():
            print(f"  skip: {m} (no directory under {args.outputs_dir})")
            continue
        # purge prior rows for this model on rebuild
        c.execute("DELETE FROM runs    WHERE model=?", (m,))
        c.execute("DELETE FROM kernels WHERE model=?", (m,))
        nr, nk = ingest_model(c, m, mdir)
        print(f"  {m:20s}  runs={nr:4d}  kernels={nk:5d}")
        total_r += nr
        total_k += nk

    conn.commit()
    print(f"DB: {args.out}")
    print(f"  TOTAL runs={total_r}  kernels={total_k}")

    # Optional friendly preview
    print("\nSample: top-5 BMG decode kernels per model")
    for r in c.execute("""
        SELECT model, kernel, per_iter_ns/1e6 AS ms, share_in_run*100 AS pct
        FROM kernels
        WHERE platform='BMG' AND mode='decode'
        ORDER BY ms DESC LIMIT 10
    """).fetchall():
        print(f"  {r[0]:15s} {r[1]:50s}  {r[2]:8.4f} ms  share={r[3]:5.1f}%")
    conn.close()


if __name__ == "__main__":
    main()
