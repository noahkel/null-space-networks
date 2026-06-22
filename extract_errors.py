#!/usr/bin/env python3
"""
extract_errors.py -- Aggregate clean vs. adversarial reconstruction error
(relative-L2 and MSE, both mean and median) across every attack run.

It walks

    <root>/attack_runs_*/init_<init>/<model>/<attack>/<norm>_eps_<eps>/summary.json

and produces, for each (shape, init, model[, eps]):
  * clean error      -- mean and median   (eps-independent; repeated per eps row)
  * adversarial error -- mean and median  (one value per eps)

Outputs a tidy CSV (one row per shape/init/model/eps) and prints grouped
console tables.  Clean metrics do not depend on eps, so they are identical
across the eps rows of a given (shape, init, model).

Usage
-----
    python extract_errors.py                       # scan ./, write error_summary.csv
    python extract_errors.py --root . --out err.csv
    python extract_errors.py --metric mse          # console tables use MSE
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Optional

# Error metrics that exist as "<cond>_<metric>_<stat>" in summary.json.
METRICS = ["rel_l2", "mse"]
CONDS = ["clean", "adv"]
STATS = ["mean", "median"]

# Stable colour per model so figures are comparable across inits/shapes.
MODEL_COLORS = {
    "nsn": "#1f77b4",        # blue
    "resnet": "#d62728",     # red
    "dpnsn": "#2ca02c",      # green
    "dpnsn_res": "#ff7f0e",  # orange
}


def parse_run(summary_path: str, root: str) -> Optional[dict]:
    """Recover run metadata from the path of a summary.json file."""
    parts = os.path.normpath(os.path.relpath(summary_path, root)).split(os.sep)
    try:
        i = next(k for k, p in enumerate(parts) if p.startswith("init_"))
    except StopIteration:
        return None
    if i + 3 >= len(parts):
        return None

    tag = parts[i - 1]                  # e.g. attack_runs_ellipses_n0.01
    init = parts[i][len("init_"):]      # e.g. fbp
    model = parts[i + 1]                # e.g. nsn
    attack = parts[i + 2]               # e.g. adam
    eps_dir = parts[i + 3]              # e.g. l2_eps_0.005

    m = re.match(r"(?P<norm>.+?)_eps_(?P<eps>.+)$", eps_dir)
    norm = m.group("norm") if m else ""
    try:
        eps = float(m.group("eps")) if m else float("nan")
    except ValueError:
        eps = float("nan")

    sm = re.search(r"attack_runs_(?P<shape>[A-Za-z]+)_n(?P<noise>[\d.]+)", tag)
    shape = sm.group("shape") if sm else tag
    noise = sm.group("noise") if sm else ""

    return dict(tag=tag, shape=shape, noise=noise, init=init, model=model,
                attack=attack, norm=norm, eps=eps)


def collect(root: str) -> list[dict]:
    pattern = os.path.join(root, "attack_runs_*", "init_*", "*", "*",
                           "*_eps_*", "summary.json")
    rows: list[dict] = []
    for sp in glob.glob(pattern):
        meta = parse_run(sp, root)
        if meta is None:
            continue
        with open(sp) as f:
            d = json.load(f)
        rec = dict(meta)
        rec["n"] = d.get("num_examples")
        for cond in CONDS:
            for met in METRICS:
                for stat in STATS:
                    key = f"{cond}_{met}_{stat}"
                    rec[key] = d.get(key)
        rows.append(rec)
    rows.sort(key=lambda r: (r["shape"], r["init"], r["model"], r["eps"]))
    return rows


def csv_fields() -> list[str]:
    fields = ["shape", "noise", "init", "model", "attack", "norm", "eps", "n"]
    for cond in CONDS:
        for met in METRICS:
            for stat in STATS:
                fields.append(f"{cond}_{met}_{stat}")
    return fields


def write_csv(rows: list[dict], out: str) -> None:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields(), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v) -> str:
    return f"{v:9.4f}" if isinstance(v, (int, float)) else f"{'n/a':>9}"


def print_tables(rows: list[dict], metric: str) -> None:
    if metric not in METRICS:
        raise SystemExit(f"--metric must be one of {METRICS}")
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["shape"], r["init"])].append(r)

    for (shape, init) in sorted(groups):
        rs = groups[(shape, init)]
        models = sorted({r["model"] for r in rs})
        epss = sorted({r["eps"] for r in rs})
        print(f"\n=== {shape} / init={init}  ::  {metric} error  (mean [median]) ===")

        print("  clean (eps-independent):")
        for m in models:
            mr = next(r for r in rs if r["model"] == m)
            print(f"    {m:<11}{_fmt(mr[f'clean_{metric}_mean'])} "
                  f"[{_fmt(mr[f'clean_{metric}_median']).strip()}]")

        print("  adversarial  (rows = eps, cols = model):")
        print("    " + " " * 7 + "| " +
              " | ".join(f"{m:^21}" for m in models))
        for e in epss:
            cells = []
            for m in models:
                match = [r for r in rs if r["model"] == m and r["eps"] == e]
                if match:
                    mean = match[0][f"adv_{metric}_mean"]
                    med = match[0][f"adv_{metric}_median"]
                    cells.append(f"{_fmt(mean)} [{_fmt(med).strip():>8}]")
                else:
                    cells.append(f"{'-':^21}")
                print(f"    {e:<7.3g}| " + " | ".join(cells))

def _model_color(model: str, fallback_idx: int):
    return MODEL_COLORS.get(model, f"C{fallback_idx % 10}")

def make_plots(rows: list[dict], metric: str, plot_dir: str,
                               stat: str = "mean") -> list[str]:
    """One figure per shape (a panel per init): adversarial error vs eps,
    one line per model, with each model's clean error as a faint floor.

    Returns the list of written PNG paths.  Requires matplotlib; if it is
    not installed a clear message is printed and nothing is written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed -- skipping plots "
                              "(pip install matplotlib).")
        return []

    os.makedirs(plot_dir, exist_ok=True)
    by_shape: dict = defaultdict(list)
    for r in rows:
        by_shape[r["shape"]].append(r)

    stats = ["mean", "median"] if stat == "both" else [stat]
    line_for = {"mean": dict(ls="-", marker="o"),
                                "median": dict(ls="--", marker="s")}
    written: list[str] = []

    for shape, srs in sorted(by_shape.items()):
        inits = sorted({r["init"] for r in srs})
        fig, axes = plt.subplots(1, len(inits),
                                                 figsize=(max(7.5, 5.2 * len(inits)), 4.8),
                                                 squeeze=False)
        for col, init in enumerate(inits):
            ax = axes[0][col]
            rs = [r for r in srs if r["init"] == init]
            models = sorted({r["model"] for r in rs})
            epss = sorted({r["eps"] for r in rs})
            for mi, model in enumerate(models):
                color = _model_color(model, mi)
                mr = {r["eps"]: r for r in rs if r["model"] == model}
                for st in stats:
                    ys = [mr[e].get(f"adv_{metric}_{st}") for e in epss]
                    lab = model if st == stats[0] else None
                    ax.plot(epss, ys, color=color, label=lab,
                                            **line_for[st], markersize=4, linewidth=1.6)
                                # clean floor (eps-independent) using the primary stat
                clean = mr[epss[0]].get(f"clean_{metric}_{stats[0]}")
                if isinstance(clean, (int, float)):
                    ax.axhline(clean, color=color, ls="-.", lw=1.0, alpha=0.85)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(epss)
            ax.set_xticklabels([f"{e:g}" for e in epss], fontsize=8)
            ax.set_xlabel(r"attack budget  $\epsilon=\|\delta\|/\|y\|$")
            if col == 0:
                ax.set_ylabel(f"{metric} error")
            ax.set_title(f"init = {init}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, title="")
        stat_note = ("solid=adv mean, dashed=adv median" if stat == "both"
                                     else f"solid=adv {stat}")
        fig.suptitle(f"{shape}: adversarial {metric} error vs attack budget "
                                     f"({stat_note}; dashed-strpied= clean floor)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = os.path.join(plot_dir, f"error_curves_{shape}_{metric}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".",
                                    help="directory containing attack_runs_* folders (default: .)")
    ap.add_argument("--out", default="error_summary.csv",
                                    help="output CSV path (default: error_summary.csv)")
    ap.add_argument("--metric", default="rel_l2", choices=METRICS,
                                    help="metric for the console tables and plots (default: rel_l2)")
    ap.add_argument("--stat", default="mean", choices=["mean", "median", "both"],
                                    help="statistic to plot (default: mean)")
    ap.add_argument("--plot-dir", default="error_plots",
                                    help="directory for PNG plots (default: error_plots)")
    ap.add_argument("--no-print", action="store_true",
                                    help="write the CSV only, skip console tables")
    ap.add_argument("--no-plot", action="store_true",
                                    help="skip generating PNG plots")
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No summary.json found under {args.root}/attack_runs_*")

    write_csv(rows, args.out)
    n_cells = len(rows)
    shapes = sorted({r["shape"] for r in rows})
    inits = sorted({r["init"] for r in rows})
    models = sorted({r["model"] for r in rows})
    print(f"Wrote {args.out}: {n_cells} rows "
                          f"({len(shapes)} shapes x {len(inits)} inits x {len(models)} models x eps)")
    print(f"  shapes={shapes}  inits={inits}  models={models}")

    if not args.no_print:
        print_tables(rows, args.metric)

    if not args.no_plot:
        pngs = make_plots(rows, args.metric, args.plot_dir, args.stat)
        for p in pngs:
            print(f"Wrote {p}")

if __name__ == "__main__":
        main()