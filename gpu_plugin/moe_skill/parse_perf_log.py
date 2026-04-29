#!/usr/bin/env python3
"""Parse MOE grouped GEMM performance test log and generate comparison tables."""

import re
import sys
from pathlib import Path


def parse_log(log_path):
    """Parse the performance log and extract structured data.

    Returns a dict: {gemm_flag: {input_tokens: {phase: {metric: value}}}}
    where phase is 'warmup' or 'test', metric is prefill/decode/generated.
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    # Split by MOE_USE_GROUPED_GEMM_PREFILL setting
    sections = re.split(r"set MOE_USE_GROUPED_GEMM_PREFILL=(\d+)", text)
    # sections[0] is preamble, then alternating (flag_value, section_text)

    results = {}
    for i in range(1, len(sections), 2):
        flag = int(sections[i])
        section = sections[i + 1]
        results[flag] = parse_section(section)

    return results


def parse_section(section):
    """Parse one section (one GEMM_PREFILL setting) with multiple prompt sizes."""
    data = {}

    # Find all warm-up entries
    warmup_prefill = re.findall(
        r"\[warm-up\]\[P0\] First token latency: ([\d.]+) ms, "
        r"other tokens latency: ([\d.]+) ms/token, "
        r"len of input tokens: (\d+)",
        section,
    )
    # Find all test [1] entries
    test_prefill = re.findall(
        r"\[1\]\[P0\] First token latency: ([\d.]+) ms, "
        r"other tokens latency: ([\d.]+) ms/token, "
        r"len of input tokens: (\d+)",
        section,
    )

    # Find all generated texts: [warm-up][P0] Generated: ... up to next [ INFO ] line
    warmup_gen = re.findall(
        r"\[warm-up\]\[P0\] Generated:(.*?)(?=\[ INFO \] \[warm-up\]\[P0\] start:)",
        section,
        re.DOTALL,
    )
    test_gen = re.findall(
        r"\[1\]\[P0\] Generated:(.*?)(?=\[ INFO \] \[1\]\[P0\] start:)",
        section,
        re.DOTALL,
    )

    # Match input token sizes from warm-up entries
    for first_tok, other_tok, tokens in warmup_prefill:
        tokens = int(tokens)
        if tokens not in data:
            data[tokens] = {"warmup": {}, "test": {}}
        data[tokens]["warmup"]["prefill_ms"] = float(first_tok)
        data[tokens]["warmup"]["decode_ms"] = float(other_tok)

    for first_tok, other_tok, tokens in test_prefill:
        tokens = int(tokens)
        if tokens not in data:
            data[tokens] = {"warmup": {}, "test": {}}
        data[tokens]["test"]["prefill_ms"] = float(first_tok)
        data[tokens]["test"]["decode_ms"] = float(other_tok)

    # Associate generated text with token sizes by order
    sorted_tokens = sorted(data.keys())
    for idx, tokens in enumerate(sorted_tokens):
        if idx < len(warmup_gen):
            data[tokens]["warmup"]["generated"] = warmup_gen[idx].strip()
        if idx < len(test_gen):
            data[tokens]["test"]["generated"] = test_gen[idx].strip()

    return data


def fmt(val, unit="ms"):
    """Format a numeric value."""
    if val is None:
        return "N/A"
    if unit == "ms":
        return f"{val:.2f}"
    if unit == "ratio":
        return f"{val:.4f}"
    return str(val)


def print_latency_table(title, results, phase):
    """Print a latency comparison table for warmup or test phase."""
    all_tokens = set()
    for flag_data in results.values():
        all_tokens.update(flag_data.keys())
    sorted_tokens = sorted(all_tokens)

    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    header = (
        f"{'Input Tokens':>12} | "
        f"{'Prefill(0) ms':>14} | {'Prefill(1) ms':>14} | {'Ratio(1/0)':>10} | "
        f"{'Decode(0) ms':>13} | {'Decode(1) ms':>13} | {'Ratio(1/0)':>10}"
    )
    print(header)
    print("-" * 120)

    for tokens in sorted_tokens:
        d0 = results.get(0, {}).get(tokens, {}).get(phase, {})
        d1 = results.get(1, {}).get(tokens, {}).get(phase, {})

        p0 = d0.get("prefill_ms")
        p1 = d1.get("prefill_ms")
        dec0 = d0.get("decode_ms")
        dec1 = d1.get("decode_ms")

        p_ratio = p1 / p0 if (p0 and p1) else None
        d_ratio = dec1 / dec0 if (dec0 and dec1) else None

        print(
            f"{tokens:>12} | "
            f"{fmt(p0):>14} | {fmt(p1):>14} | {fmt(p_ratio, 'ratio'):>10} | "
            f"{fmt(dec0):>13} | {fmt(dec1):>13} | {fmt(d_ratio, 'ratio'):>10}"
        )

    print()


def print_output_table(title, results):
    """Print generated output comparison table for the test phase."""
    all_tokens = set()
    for flag_data in results.values():
        all_tokens.update(flag_data.keys())
    sorted_tokens = sorted(all_tokens)

    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")

    for tokens in sorted_tokens:
        d0 = results.get(0, {}).get(tokens, {}).get("test", {})
        d1 = results.get(1, {}).get(tokens, {}).get("test", {})

        gen0 = d0.get("generated", "N/A")
        gen1 = d1.get("generated", "N/A")

        # Truncate for display
        max_len = 200
        gen0_disp = (gen0[:max_len] + "...") if len(gen0) > max_len else gen0
        gen1_disp = (gen1[:max_len] + "...") if len(gen1) > max_len else gen1

        match = "YES" if gen0 == gen1 else "NO"

        print(f"\n--- Input Tokens: {tokens} | Output Match: {match} ---")
        print(f"  PREFILL=0: {gen0_disp}")
        print(f"  PREFILL=1: {gen1_disp}")

    print()


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "perf_test_log.log"
    results = parse_log(log_path)

    if not results:
        print("ERROR: No data parsed from log file.")
        sys.exit(1)

    flags_found = sorted(results.keys())
    print(f"Parsed GROUPED_GEMM_PREFILL settings: {flags_found}")
    for flag in flags_found:
        tokens_list = sorted(results[flag].keys())
        print(f"  PREFILL={flag}: token sizes = {tokens_list}")

    # Table 1: Warmup latency comparison
    print_latency_table(
        "Table 1: Warm-up Latency Comparison (MOE_USE_GROUPED_GEMM_PREFILL=0 vs 1)",
        results,
        "warmup",
    )

    # Table 2: Test latency comparison
    print_latency_table(
        "Table 2: Test Latency Comparison (MOE_USE_GROUPED_GEMM_PREFILL=0 vs 1)",
        results,
        "test",
    )

    # Table 3: Output comparison
    print_output_table(
        "Table 3: Test Generated Output Comparison (MOE_USE_GROUPED_GEMM_PREFILL=0 vs 1)",
        results,
    )


if __name__ == "__main__":
    main()
