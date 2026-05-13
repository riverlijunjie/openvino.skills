#!/usr/bin/env python3
"""Generic model report builder for OpenVINO roofline profiling.

This utility replaces per-model `build_report.py` logic with a shared engine plus
model-specific `report_config.json` files.

Usage:
    python3 utils/build_model_report.py \
        --model-dir models/qwen3_moe \
        --output-dir outputs/qwen3_moe \
        --platform BMG \
        --out outputs/qwen3_moe/performance_metrics_bmg.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = json.loads((SKILL_ROOT / "utils" / "platforms.json").read_text())


def fc_bytes_int4(m: int, k: int, n: int, g: int = 128) -> int:
    return k * n // 2 + (k // g) * n * 2 + (k // g) * n // 2 + m * k * 2 + m * n * 2


def fc_bytes_int8(m: int, k: int, n: int, g: int = 128) -> int:
    return k * n + (k // g) * n * 2 + m * k * 2 + m * n * 2


def fc_flops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def platform_desc(platform_name: str, pdata: dict[str, Any]) -> str:
    name = pdata.get("name", platform_name)
    arch = pdata.get("arch")
    bw = pdata.get("bw_gbs_measured_read") or pdata.get("bw_gbs_spec")
    fp16 = pdata.get("fp16_xmx_tflops")
    int8 = pdata.get("int8_xmx_tops")
    parts = [name]
    if arch:
        parts.append(arch)
    extras = []
    if bw:
        extras.append(f"{bw:g} GB/s")
    if fp16:
        extras.append(f"FP16 {fp16:g} TFLOPS")
    if int8:
        extras.append(f"INT8 {int8:g} TOPS")
    return f"{platform_name} ({', '.join(parts)}; {', '.join(extras)})"


SAFE_BUILTINS = {
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "sorted": sorted,
}


class ReportContext:
    def __init__(self, metrics: dict[str, Any], cfg: dict[str, Any], platform_name: str, pdata: dict[str, Any]):
        self.metrics = metrics
        self.cfg = cfg
        self.platform_name = platform_name
        self.pdata = pdata
        self.bw = pdata.get("bw_gbs_measured_read") or pdata.get("bw_gbs_spec") or 0.0
        self.fp16 = pdata.get("fp16_xmx_tflops") or 0.0
        self.int8 = pdata.get("int8_xmx_tops") or 0.0
        self.platform_desc = platform_desc(platform_name, pdata)

    def metric_entry(self, log: str) -> dict[str, Any]:
        return self.metrics.get(log, {})

    def metric_ns(self, log: str) -> float:
        return float(self.metric_entry(log).get("total_kernel_ns", 0) or 0)

    def metric_ms(self, log: str) -> float:
        return self.metric_ns(log) / 1e6

    def metric_present(self, log: str) -> bool:
        return self.metric_ns(log) > 0

    def env(self) -> dict[str, Any]:
        return {
            **SAFE_BUILTINS,
            "cfg": self.cfg,
            "platform_name": self.platform_name,
            "platform_desc": self.platform_desc,
            "platform": self.pdata,
            "BW": self.bw,
            "FP16": self.fp16,
            "INT8": self.int8,
            "metric_entry": self.metric_entry,
            "metric_ns": self.metric_ns,
            "metric_ms": self.metric_ms,
            "metric_present": self.metric_present,
            "fc_bytes_int4": fc_bytes_int4,
            "fc_bytes_int8": fc_bytes_int8,
            "fc_flops": fc_flops,
        }


def eval_expr(expr: str, env: dict[str, Any]) -> Any:
    return eval(expr, {"__builtins__": {}}, env)


def format_value(value: Any, fmt: str | None) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if fmt:
        return format(value, fmt)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:" if i else "---" for i, _ in enumerate(headers)]) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def build_decode_rows(spec: dict[str, Any], env: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    table_env = dict(env)
    for key, expr in spec.get("variables", {}).items():
        table_env[key] = eval_expr(expr, table_env)

    rows = []
    for row_spec in spec.get("rows", []):
        row_env = dict(table_env)
        if row_spec.get("when") and not eval_expr(row_spec["when"], row_env):
            continue
        log = eval_expr(row_spec["log"], row_env)
        ms = row_env["metric_ms"](log)
        if ms == 0 and row_spec.get("skip_if_missing", True):
            continue
        calls = eval_expr(row_spec["calls"], row_env)
        bytes_ = eval_expr(row_spec["bytes"], row_env)
        flops = eval_expr(row_spec["flops"], row_env)
        gbs = bytes_ / (ms * 1e6) if ms else 0.0
        tops = flops / (ms * 1e9) / 1000 if ms else 0.0
        ai = flops / bytes_ if bytes_ else 0.0
        eff = gbs / row_env["BW"] * 100 if row_env["BW"] else 0.0
        rows.append({
            "name": eval_expr(row_spec["name"], row_env),
            "log": log,
            "ms": ms,
            "calls": calls,
            "total": ms * calls,
            "ai": ai,
            "gbs": gbs,
            "tops": tops,
            "eff": eff,
            "bound": eval_expr(row_spec.get("bound", "'memory'"), row_env),
        })
    rows.sort(key=lambda r: -r["total"])
    total = sum(r["total"] for r in rows)
    return rows, total


def render_decode_section(spec: dict[str, Any], env: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    table_env = dict(env)
    for key, expr in spec.get("variables", {}).items():
        table_env[key] = eval_expr(expr, table_env)
    title = eval_expr(spec["title"], table_env)
    subtitle = eval_expr(spec.get("subtitle", "repr(platform_desc)"), table_env)
    rows, total = build_decode_rows(spec, env)

    print(f"\n{title}")
    print(f"_{subtitle}_")
    print()
    md_rows = []
    for r in rows:
        md_rows.append([
            r["name"],
            f"{r['ms']:.4f}",
            str(r["calls"]),
            f"{r['total']:.3f}",
            f"{r['ai']:.2f}",
            f"{r['gbs']:.1f}",
            f"{r['eff']:.1f}%",
            str(r["bound"]),
        ])
    render_markdown_table(["Op", "Avg ms", "Calls", "Total ms", "AI", "GB/s", "Eff%", "Bound"], md_rows)
    if total:
        print(f"\n**Decode total** = {total:.3f} ms → **{1000/total:.1f} tok/s**")
    return rows, total


def render_sweep(sweep: dict[str, Any], env: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sweep_env = dict(env)
    for key, expr in sweep.get("variables", {}).items():
        sweep_env[key] = eval_expr(expr, sweep_env)

    axis_name = sweep["axis_name"]
    headers = [col["header"] for col in sweep["columns"]]
    title = eval_expr(sweep["title"], sweep_env)
    print(f"\n{title}\n")

    rows_md = []
    stored_map: dict[str, Any] = {}
    stored_rows: list[dict[str, Any]] = []

    for axis_value in sweep["axis_values"]:
        row_env = dict(sweep_env)
        row_env[axis_name] = axis_value
        for key, expr in sweep.get("row_variables", {}).items():
            row_env[key] = eval_expr(expr, row_env)
        if sweep.get("skip_if") and eval_expr(sweep["skip_if"], row_env):
            continue

        save_row = True
        override = None
        for candidate in sweep.get("overrides", []):
            if eval_expr(candidate["when"], row_env):
                override = candidate
                save_row = candidate.get("save", True)
                break

        row_values: dict[str, Any] = {}
        for col in sweep["columns"]:
            row_values[col["header"]] = eval_expr(col["expr"], row_env)

        if sweep.get("total_expr"):
            row_env["total"] = eval_expr(sweep["total_expr"], row_env)
            row_values[sweep.get("total_header", "total")] = row_env["total"]
        if sweep.get("tokens_expr"):
            row_env["tokens_per_sec"] = eval_expr(sweep["tokens_expr"], row_env)
            row_values[sweep.get("tokens_header", "tok/s")] = row_env["tokens_per_sec"]

        if override:
            for header, expr in override.get("cells", {}).items():
                row_values[header] = eval_expr(expr, row_env)

        row_md = []
        for col in sweep["columns"]:
            row_md.append(format_value(row_values[col["header"]], col.get("fmt")))
        if sweep.get("total_expr"):
            row_md.append(format_value(row_values[sweep.get("total_header", "total")], sweep.get("total_fmt")))
        if sweep.get("tokens_expr"):
            row_md.append(format_value(row_values[sweep.get("tokens_header", "tok/s")], sweep.get("tokens_fmt")))
        rows_md.append(row_md)

        stored_row = {axis_name: axis_value, **row_values}
        stored_rows.append(stored_row)
        if save_row and sweep.get("save_key"):
            save_expr = sweep.get("save_expr", "total")
            stored_map[str(axis_value)] = eval_expr(save_expr, row_env)

    display_headers = headers[:]
    if sweep.get("total_expr"):
        display_headers.append(sweep.get("total_header", "total"))
    if sweep.get("tokens_expr"):
        display_headers.append(sweep.get("tokens_header", "tok/s"))
    render_markdown_table(display_headers, rows_md)
    return stored_map, stored_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--report-config", type=Path, default=None)
    parser.add_argument("--ops-mapping", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report_cfg_path = args.report_config or (args.model_dir / "report_config.json")
    ops_mapping_path = args.ops_mapping or (args.model_dir / "ops_mapping.json")
    metrics_path = args.metrics or (args.output_dir / f"{args.platform.lower()}_metrics.json")

    report_cfg = json.loads(report_cfg_path.read_text())
    ops_mapping = json.loads(ops_mapping_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    pdata = PLATFORMS[args.platform]
    ctx = ReportContext(metrics, ops_mapping["config"], args.platform, pdata)
    env = ctx.env()

    for key, expr in report_cfg.get("variables", {}).items():
        env[key] = eval_expr(expr, env)

    decode_rows, decode_total = render_decode_section(report_cfg["decode_table"], env)

    output_payload = {
        "platform": args.platform,
        "platform_desc": ctx.platform_desc,
        "decode_total_ms": decode_total,
        "decode_rows": decode_rows,
    }

    for sweep in report_cfg.get("sweeps", []):
        saved_map, saved_rows = render_sweep(sweep, env)
        if sweep.get("save_key"):
            output_payload[sweep["save_key"]] = saved_map
        output_payload.setdefault("sweeps", {})[sweep.get("name", sweep["axis_name"])] = saved_rows

    if args.out:
        args.out.write_text(json.dumps(output_payload, indent=2, default=str))
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
