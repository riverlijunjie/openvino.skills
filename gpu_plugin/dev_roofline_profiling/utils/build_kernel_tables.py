#!/usr/bin/env python3
"""Generic kernel-table generator for OpenVINO roofline profiling.

This utility replaces per-model `build_kernel_tables.py` logic with a shared
engine plus model-specific `kernel_table_config.json` files.
"""
from __future__ import annotations

import argparse
import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from build_model_report import PLATFORMS, ReportContext, eval_expr, format_value, render_markdown_table

SKILL_ROOT = Path(__file__).resolve().parent.parent


def sanitize_kernel_name(name: str) -> str:
    name = re.sub(r"_\d{15,}", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def get_kernels(metrics: dict[str, Any], log: str, ignore_patterns: list[str] | None = None) -> list[tuple[str, float, int]]:
    entry = metrics.get(log)
    if not entry:
        return []
    ignore_patterns = ignore_patterns or []
    compiled = [re.compile(p) for p in ignore_patterns]
    out = []
    for row in entry.get("per_kernel", []):
        if not row or len(row) < 3:
            continue
        name, ns, calls = row[0], row[1], row[2]
        clean = sanitize_kernel_name(name)
        if any(p.search(clean) for p in compiled):
            continue
        out.append((clean, ns / 1e6, calls))
    out.sort(key=lambda item: -item[1])
    return out


def render_mode(mode_spec: dict[str, Any], env: dict[str, Any], metrics: dict[str, Any], platform_name: str, ctx: ReportContext) -> None:
    mode_env = dict(env)
    for key, expr in mode_spec.get("variables", {}).items():
        mode_env[key] = eval_expr(expr, mode_env)

    print(eval_expr(mode_spec["heading"], mode_env))
    print()

    axis_name = mode_spec["axis_name"]
    axis_label = mode_spec.get("axis_label", axis_name)
    peak_compute = eval_expr(mode_spec.get("peak_compute", "platform.get('fp16_xmx_tflops', 0) * 1000"), mode_env)
    coverage = float(mode_spec.get("coverage", 0.95))

    for axis_value in mode_spec["axis_values"]:
        axis_env = dict(mode_env)
        axis_env[axis_name] = axis_value
        for key, expr in mode_spec.get("row_variables", {}).items():
            axis_env[key] = eval_expr(expr, axis_env)
        if mode_spec.get("skip_if") and eval_expr(mode_spec["skip_if"], axis_env):
            continue

        rows = []
        for spec in mode_spec["specs"]:
            row_env = dict(axis_env)
            if spec.get("when") and not eval_expr(spec["when"], row_env):
                continue
            log = eval_expr(spec["log"], row_env)
            if log is None:
                continue
            ignore_patterns = [eval_expr(p, row_env) if isinstance(p, str) and p.startswith(("'", '"', "f'", 'f\"')) else p for p in spec.get("ignore_kernel_patterns", [])]
            kernels = get_kernels(metrics, log, ignore_patterns)
            if not kernels:
                continue
            op_total_ms = sum(kernel_ms for _, kernel_ms, _ in kernels)
            if op_total_ms <= 0:
                continue
            op_name = eval_expr(spec["op"], row_env)
            op_bytes = eval_expr(spec["bytes"], row_env)
            op_flops = eval_expr(spec["flops"], row_env)
            calls = eval_expr(spec["calls"], row_env)
            bound = eval_expr(spec.get("bound", "'memory'"), row_env)
            op_coverage = float(eval_expr(spec.get("coverage", str(coverage)), row_env)) if isinstance(spec.get("coverage"), str) else float(spec.get("coverage", coverage))

            cumulative = 0.0
            significant = []
            for kernel in kernels:
                significant.append(kernel)
                cumulative += kernel[1]
                if cumulative >= op_coverage * op_total_ms:
                    break

            for kernel_name, kernel_ms, _calls_in_log in significant:
                share = kernel_ms / op_total_ms if op_total_ms else 0.0
                total_ms = kernel_ms * calls
                kernel_bytes = op_bytes * share
                kernel_flops = op_flops * share
                gbs = kernel_bytes / (kernel_ms * 1e6) if kernel_ms else 0.0
                gflops = kernel_flops / (kernel_ms * 1e6) if kernel_ms else 0.0
                eff = gbs / ctx.bw * 100 if bound == "memory" else (gflops / peak_compute * 100 if peak_compute else 0.0)
                rows.append({
                    "op": op_name,
                    "kernel": kernel_name,
                    "single_ms": kernel_ms,
                    "calls": calls,
                    "total_ms": total_ms,
                    "gflops": gflops,
                    "gbs": gbs,
                    "eff": eff,
                    "bound": bound,
                })

        rows.sort(key=lambda row: -row["total_ms"])
        print(f"#### {platform_name} — {mode_spec['name']} {axis_label}={axis_value}")
        print()
        md_rows = []
        total_stage_ms = 0.0
        for row in rows:
            total_stage_ms += row["total_ms"]
            md_rows.append([
                row["op"],
                f"`{row['kernel']}`",
                format_value(row["single_ms"], ".4f"),
                format_value(row["calls"], "d"),
                format_value(row["total_ms"], ".3f"),
                format_value(row["gflops"], ".0f"),
                format_value(row["gbs"], ".1f"),
                format_value(row["eff"], ".1f") + "%",
                row["bound"],
            ])
        render_markdown_table(["Op", "Kernel", "Single ms", "Calls/inf", "Total ms", "GFLOPS", "GB/s", "Eff%", "Bound"], md_rows)
        print(f"\n**Total inference time (this stage)** ≈ **{total_stage_ms:.2f} ms**\n")


def generate_tables(model_dir: Path, output_dir: Path, config_path: Path, ops_mapping_path: Path, platforms: list[str] | None = None) -> str:
    kernel_cfg = json.loads(config_path.read_text())
    ops_mapping = json.loads(ops_mapping_path.read_text())
    model_label = ops_mapping.get("model", model_dir.name)
    if isinstance(model_label, dict):
        model_label = model_label.get("name", model_dir.name)

    requested_platforms = platforms or kernel_cfg.get("platforms")
    if not requested_platforms:
        requested_platforms = []
        for candidate in PLATFORMS:
            metrics_path = output_dir / f"{candidate.lower()}_metrics.json"
            if metrics_path.exists():
                requested_platforms.append(candidate)

    out = StringIO()
    with redirect_stdout(out):
        title = kernel_cfg.get("title", "f'# {model_name} — Per-token-size Kernel Tables'")
        base_env = {
            "model_name": model_label,
        }
        print(eval_expr(title, {**base_env}))
        print()
        for platform_name in requested_platforms:
            metrics_path = output_dir / f"{platform_name.lower()}_metrics.json"
            if not metrics_path.exists():
                continue
            pdata = PLATFORMS[platform_name]
            metrics = json.loads(metrics_path.read_text())
            ctx = ReportContext(metrics, ops_mapping["config"], platform_name, pdata)
            env = ctx.env()
            env.update(base_env)
            for key, expr in kernel_cfg.get("variables", {}).items():
                env[key] = eval_expr(expr, env)
            print(f"## Platform: {platform_name}\n")
            for mode_spec in kernel_cfg["modes"]:
                render_mode(mode_spec, env, metrics, platform_name, ctx)
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-config", type=Path, default=None)
    parser.add_argument("--ops-mapping", type=Path, default=None)
    parser.add_argument("--platform", action="append", dest="platforms")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config_path = args.kernel_config or (args.model_dir / "kernel_table_config.json")
    ops_mapping_path = args.ops_mapping or (args.model_dir / "ops_mapping.json")
    text = generate_tables(args.model_dir, args.output_dir, config_path, ops_mapping_path, args.platforms)
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
