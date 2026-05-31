"""Recompute attack-run summary statistics after dropping degenerate phantoms.

Motivation
----------
The test set contains at least one near-empty phantom (clean_sino_l2 == 0,
||x_gt|| ~ 0). Its relative-l2 error explodes (~1e9), which corrupts every
*_rel_l2_mean in summary.json (mean ~1e8 while the median stays ~0.05).

This script re-reads a per_sample_metrics.csv, drops the degenerate rows, and
recomputes mean / ci95 / median for every numeric column, exactly matching the
aggregation in attack.py (ci95 = 1.96 * std(ddof=1) / sqrt(n)).

Standard library only -- runs anywhere, no numpy/pandas needed.

Usage
-----
    python recompute_summary.py path/to/per_sample_metrics.csv
    python recompute_summary.py path/to/per_sample_metrics.csv --write     # write summary_filtered.json
    python recompute_summary.py runs/ --glob "**/per_sample_metrics.csv"    # batch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median as _median


# Columns attack.py aggregates. Any extra numeric columns are also summarised.
KEY_COLUMNS = [
    "clean_mse", "adv_mse", "mse_ratio",
    "clean_rel_l2", "adv_rel_l2", "rel_l2_ratio",
    "clean_psnr", "adv_psnr",
    "clean_ssim", "adv_ssim",
    "pred_shift_rel_l2", "init_shift_rel_l2",
    "delta_l2", "delta_linf", "delta_mean_abs",
    "success_rel_l2", "success_mse",
]


def confidence_interval_95(vals: list[float]) -> tuple[float, float]:
    """Mirror attack.py: mean and 1.96 * std(ddof=1) / sqrt(n) half-width."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # ddof=1
    half_width = 1.96 * math.sqrt(var) / math.sqrt(n)
    return mean, half_width


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def is_degenerate(row: dict[str, str]) -> bool:
    """True for rows to DROP: empty phantom (no signal) or non-finite metrics."""
    if "clean_sino_l2" in row and _to_float(row["clean_sino_l2"]) <= 0.0:
        return True
    for col in ("clean_rel_l2", "adv_rel_l2"):
        if col in row:
            v = _to_float(row[col])
            if not math.isfinite(v) or abs(v) > 100.0:
                return True
    return False


def recompute(csv_path: Path, write: bool) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_before = len(rows)
    kept = [r for r in rows if not is_degenerate(r)]
    n_dropped = n_before - len(kept)

    print(f"\n=== {csv_path} ===")
    print(f"samples: {n_before} total, {n_dropped} dropped, {len(kept)} kept")

    if not kept:
        print("  (nothing left after filtering)")
        return

    cols = list(kept[0].keys())
    ordered = [c for c in KEY_COLUMNS if c in cols]
    ordered += [c for c in cols if c not in ordered]

    out: dict[str, float] = {"num_examples": len(kept)}
    print(f"{'metric':24s} {'mean':>14s} {'ci95':>12s} {'median':>14s}")
    for col in ordered:
        vals = [v for v in (_to_float(r[col]) for r in kept) if math.isfinite(v)]
        if not vals:
            continue
        mean, ci = confidence_interval_95(vals)
        med = _median(vals)
        out[f"{col}_mean"] = mean
        out[f"{col}_ci95"] = ci
        out[f"{col}_median"] = med
        if col in KEY_COLUMNS:
            print(f"{col:24s} {mean:14.5f} {ci:12.5f} {med:14.5f}")

    if write:
        summary_path = csv_path.parent / "summary_filtered.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"-> wrote {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="per_sample_metrics.csv, or a directory with --glob")
    ap.add_argument("--glob", default=None, help="glob under a directory, e.g. '**/per_sample_metrics.csv'")
    ap.add_argument("--write", action="store_true", help="write summary_filtered.json next to each CSV")
    args = ap.parse_args()

    if args.glob:
        csvs = sorted(args.path.glob(args.glob))
        if not csvs:
            raise SystemExit(f"no CSVs matched {args.path}/{args.glob}")
    else:
        csvs = [args.path]

    for csv_path in csvs:
        recompute(csv_path, write=args.write)


if __name__ == "__main__":
    main()
