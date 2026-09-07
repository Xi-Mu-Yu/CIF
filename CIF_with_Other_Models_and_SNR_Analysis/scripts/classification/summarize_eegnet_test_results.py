#!/usr/bin/env python3
"""
Parse result_classification.txt under results/classification/EEGNet and compute
mean and sample standard deviation of test-set metrics across runs (e.g. multiple seeds).
Reported values are multiplied by 100 and rounded to two decimals (e.g. 85.32 for 0.8532).

For each experiment subfolder (.../<exp>/EEGNet/result_classification.txt), writes
test_summary_mean_std.md (Markdown: mean ± std, ×100, two decimals) unless --output-name is set.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

TEST_PATTERN = re.compile(
    r"Test results --- Loss:\s*([\d.]+),\s*Accuracy:\s*([\d.]+),\s*Precision:\s*([\d.]+),\s*"
    r"Recall:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AUROC:\s*([\d.]+),\s*AUPRC:\s*([\d.]+)"
)

METRIC_NAMES = (
    "Loss",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUROC",
    "AUPRC",
)


def parse_test_metrics(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for m in TEST_PATTERN.finditer(text):
        rows.append([float(x) for x in m.groups()])
    return rows


def summarize_array(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        std = float("nan")
    else:
        std = float(np.std(values, ddof=1))
    return mean, std


def format_mean_pm_std(mean: float, std: float) -> str:
    m = mean * 100
    if np.isfinite(std):
        s = std * 100
        return f"{m:.2f} ± {s:.2f}"
    return f"{m:.2f}"


def build_summary_text(exp_dir: str, n: int, arr: np.ndarray) -> str:
    lines: list[str] = [
        f"# {exp_dir}",
        "",
        f"Test set metrics: **n = {n}** runs. Values are **×100** with **two decimal** places.",
        "",
        "| Metric | Test (mean ± std) |",
        "| --- | --- |",
    ]
    for j, name in enumerate(METRIC_NAMES):
        mean, std = summarize_array(arr[:, j])
        cell = format_mean_pm_std(mean, std)
        lines.append(f"| {name} | {cell} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mean and sample standard deviation of EEGNet test metrics from result_classification.txt files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root folder (default: <repo>/results/classification/EEGNet)",
    )
    parser.add_argument(
        "--output-name",
        default="test_summary_mean_std.md",
        help="Written next to result_classification.txt in each .../<exp>/EEGNet/ folder (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print tables to stdout (only write files and short paths).",
    )
    args = parser.parse_args()

    if args.root is None:
        repo = Path(__file__).resolve().parents[2]
        root = repo / "results" / "classification" / "EEGNet"
    else:
        root = args.root.expanduser().resolve()

    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    result_files = sorted(root.glob("**/EEGNet/result_classification.txt"))
    if not result_files:
        print(f"No result_classification.txt found under {root}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Root: {root}\n")
    for path in result_files:
        rel = path.relative_to(root)
        exp_dir = rel.parts[0] if rel.parts else str(path.parent)
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = parse_test_metrics(text)
        out_path = path.parent / args.output_name

        if not rows:
            msg = f"[{exp_dir}] no Test results lines found — skipped (no file written)"
            print(msg, file=sys.stderr if args.quiet else sys.stdout)
            continue

        arr = np.asarray(rows, dtype=np.float64)
        n = arr.shape[0]
        body = build_summary_text(exp_dir, n, arr)
        out_path.write_text(body, encoding="utf-8")

        if args.quiet:
            print(f"Wrote {out_path}")
        else:
            print(f"=== {exp_dir}  (n={n} test runs) → {out_path}")
            print(body, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
