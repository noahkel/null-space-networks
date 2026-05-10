#!/usr/bin/env python3
"""
Generate three figures comparing FBP-init vs Pinv-init for NSN vs ResNet:

  1. scatter_quality_vs_robustness.png  — clean PSNR vs adversarial PSNR
  2. error_decomposition.png            — range vs null-space error per model/init
  3. example_grid.png                   — example reconstruction panels (if PNGs exist)

Usage:
    python plot_experiment.py [--results-dir attack_runs_ellipses] [--out-dir figures]

Designed to work with whatever (init, model) combinations are present — missing
pinv results will simply be absent from the plots.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

INIT_COLORS = {
    "fbp":  "#2166ac",   # blue
    "pinv": "#d6604d",   # red-orange
    "lw":   "#4dac26",   # green
    "tv":   "#7b2d8b",   # purple
}
INIT_LABELS = {
    "fbp":  "FBP init",
    "pinv": "Pinv init ($A_{\\mathrm{la}}^+$)",
    "lw":   "Landweber init",
    "tv":   "TV init",
}

MODEL_MARKERS = {
    "resnet":    "o",
    "nsn":       "s",
    "dpnsn":     "^",
    "dpnsn_res": "D",
}
MODEL_LABELS = {
    "resnet":    "ResNet",
    "nsn":       "NSN",
    "dpnsn":     "DPNSN",
    "dpnsn_res": "DPNSN-Res",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def discover_summaries(results_dir: Path):
    """
    Walk results_dir and collect all summary.json files.

    Returns a list of dicts, each with keys:
        init, model, attack, norm, eps, summary (the parsed JSON)
    """
    records = []
    for path in sorted(results_dir.rglob("summary.json")):
        parts = path.relative_to(results_dir).parts
        # expected: init_{init} / {model} / {attack} / {norm}_eps_{eps} / summary.json
        if len(parts) != 5:
            continue
        init_dir, model, attack, norm_eps, _ = parts
        if not init_dir.startswith("init_"):
            continue
        init = init_dir[len("init_"):]
        norm, eps_str = norm_eps.rsplit("_eps_", 1)
        with open(path) as f:
            summary = json.load(f)
        records.append(dict(
            init=init, model=model, attack=attack,
            norm=norm, eps=float(eps_str),
            summary=summary, path=path,
        ))
    return records


# ── Figure 1: Scatter — clean PSNR vs adversarial PSNR ───────────────────────

def plot_scatter(records, out_path: Path, attack="adam", norm="l2"):
    recs = [r for r in records if r["attack"] == attack and r["norm"] == norm]
    if not recs:
        print(f"[scatter] no records for attack={attack} norm={norm}, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    seen_inits = set()
    seen_models = set()

    for r in recs:
        s = r["summary"]
        x = s["clean_psnr_mean"]
        y = s["adv_psnr_mean"]
        xerr = s["clean_psnr_ci95"]
        yerr = s["adv_psnr_ci95"]
        color = INIT_COLORS.get(r["init"], "#888888")
        marker = MODEL_MARKERS.get(r["model"], "x")
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt=marker, color=color, markersize=9,
            markeredgecolor="white", markeredgewidth=0.8,
            ecolor=color, elinewidth=1.2, capsize=3,
        )
        # label the point
        label = f"{MODEL_LABELS.get(r['model'], r['model'])}\n({INIT_LABELS.get(r['init'], r['init'])})"
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(7, 3),
            fontsize=7.5, color=color,
        )
        seen_inits.add(r["init"])
        seen_models.add(r["model"])

    # diagonal reference: y = x (perfectly robust)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]) - 1,
        max(ax.get_xlim()[1], ax.get_ylim()[1]) + 1,
    ]
    ax.plot(lims, lims, "--", color="#aaaaaa", linewidth=1, label="clean = adv (ideal robust)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # legend patches for init color
    init_patches = [
        mpatches.Patch(color=INIT_COLORS[i], label=INIT_LABELS.get(i, i))
        for i in sorted(seen_inits)
        if i in INIT_COLORS
    ]
    # legend entries for model markers
    model_lines = [
        plt.Line2D([0], [0], marker=MODEL_MARKERS.get(m, "x"), color="#555555",
                   linestyle="None", markersize=8,
                   label=MODEL_LABELS.get(m, m))
        for m in sorted(seen_models)
        if m in MODEL_MARKERS
    ]

    ax.legend(
        handles=init_patches + model_lines + [
            plt.Line2D([0], [0], linestyle="--", color="#aaaaaa", label="clean = adv")
        ],
        loc="upper left", framealpha=0.85,
    )

    ax.set_xlabel("Clean PSNR (dB)  ↑ better quality")
    ax.set_ylabel("Adversarial PSNR (dB)  ↑ more robust")
    ax.set_title("Reconstruction quality vs adversarial robustness")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[scatter] saved → {out_path}")


# ── Figure 2: Error decomposition bar chart ───────────────────────────────────

def plot_error_decomposition(records, out_path: Path, attack="adam", norm="l2"):
    recs = [r for r in records if r["attack"] == attack and r["norm"] == norm]
    if not recs:
        print(f"[error_decomp] no records for attack={attack} norm={norm}, skipping")
        return

    # Sort: inits first, then models within each init
    recs = sorted(recs, key=lambda r: (r["init"], r["model"]))

    labels, ran_vals, ran_errs, nul_vals, nul_errs = [], [], [], [], []
    for r in recs:
        s = r["summary"]
        model_str = MODEL_LABELS.get(r["model"], r["model"])
        init_str = INIT_LABELS.get(r["init"], r["init"])
        labels.append(f"{model_str}\n({init_str})")
        ran_vals.append(s["clean_e_ran_l2_mean"])
        ran_errs.append(s["clean_e_ran_l2_ci95"])
        nul_vals.append(s["clean_e_nul_l2_mean"])
        nul_errs.append(s["clean_e_nul_l2_ci95"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))

    bars_ran = ax.bar(
        x - width / 2, ran_vals, width,
        yerr=ran_errs, capsize=4,
        label="Range-space error  $\\|e_{\\mathrm{ran}}\\|$",
        color="#d6604d", alpha=0.85, error_kw=dict(elinewidth=1.2),
    )
    bars_nul = ax.bar(
        x + width / 2, nul_vals, width,
        yerr=nul_errs, capsize=4,
        label="Null-space error  $\\|e_{\\mathrm{nul}}\\|$",
        color="#4393c3", alpha=0.85, error_kw=dict(elinewidth=1.2),
    )

    # shade background by init to group visually
    init_groups = {}
    for i, r in enumerate(recs):
        init_groups.setdefault(r["init"], []).append(i)
    for init, idxs in init_groups.items():
        lo, hi = min(idxs) - 0.5, max(idxs) + 0.5
        ax.axvspan(lo, hi, alpha=0.06,
                   color=INIT_COLORS.get(init, "#cccccc"),
                   label=f"_{init}")  # underscore → hidden from legend

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$\\ell_2$ error norm  ↓ better")
    ax.set_title("Clean reconstruction: range vs null-space error components")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)

    # add value labels on bars
    for bar in list(bars_ran) + list(bars_nul):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.02,
            f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[error_decomp] saved → {out_path}")


# ── Figure 3: Example reconstruction grid ────────────────────────────────────

def plot_example_grid(records, out_path: Path, attack="adam", norm="l2", eps=1, example_idx=0):
    """
    Load saved example_NNN.png panels for each (init, model) combo and arrange them
    in a grid. Skips missing combinations gracefully.
    """
    recs = [
        r for r in records
        if r["attack"] == attack and r["norm"] == norm and r["eps"] == eps
    ]
    if not recs:
        print(f"[example_grid] no records, skipping")
        return

    recs = sorted(recs, key=lambda r: (r["init"], r["model"]))

    # load example images — each PNG is a multi-panel figure from the attack script
    loaded = []
    for r in recs:
        img_path = r["path"].parent / f"example_{example_idx:03d}.png"
        if not img_path.exists():
            print(f"  [example_grid] missing {img_path}, skipping")
            continue
        img = plt.imread(str(img_path))
        loaded.append((r, img))

    if not loaded:
        print(f"[example_grid] no example images found, skipping")
        return

    n = len(loaded)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    for ax, (r, img) in zip(axes, loaded):
        ax.imshow(img)
        ax.axis("off")
        init_label = INIT_LABELS.get(r["init"], r["init"])
        model_label = MODEL_LABELS.get(r["model"], r["model"])
        s = r["summary"]
        subtitle = (
            f"clean PSNR={s['clean_psnr_mean']:.1f} dB   "
            f"adv PSNR={s['adv_psnr_mean']:.1f} dB   "
            f"MSE ratio={s['mse_ratio_mean']:.1f}×"
        )
        ax.set_title(
            f"{model_label}  |  {init_label}\n{subtitle}",
            fontsize=10, pad=4,
        )

    fig.suptitle(
        f"Reconstruction examples — attack: {attack.upper()}, ε={eps} (L2)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[example_grid] saved → {out_path}")

    # ── Figure 4 (bonus): per-sample strip plots — robustness distribution ────────

def plot_mse_ratio_distribution(records, out_path: Path, attack="adam", norm="l2"):
    """
    Violin / strip plot showing the per-sample MSE ratio distribution per model/init.
    Requires per_sample_metrics.csv alongside each summary.json.
    """
    import csv

    recs = [r for r in records if r["attack"] == attack and r["norm"] == norm]
    recs = sorted(recs, key=lambda r: (r["init"], r["model"]))

    all_data = []
    for r in recs:
        csv_path = r["path"].parent / "per_sample_metrics.csv"
        if not csv_path.exists():
            continue
        ratios = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ratios.append(float(row["mse_ratio"]))
        if ratios:
            all_data.append((r, ratios))

    if not all_data:
        print(f"[mse_ratio_dist] no per-sample CSVs found, skipping")
        return

    labels = []
    data = []
    colors = []
    for r, ratios in all_data:
        model_str = MODEL_LABELS.get(r["model"], r["model"])
        init_str = INIT_LABELS.get(r["init"], r["init"])
        labels.append(f"{model_str}\n({init_str})")
        data.append(ratios)
        colors.append(INIT_COLORS.get(r["init"], "#888888"))

    fig, ax = plt.subplots(figsize=(max(6, len(all_data) * 1.4), 5))

    # boxplot on log scale (violins don't render well with log y-axis)
    bp = ax.boxplot(
        data, positions=range(len(data)),
        widths=0.4, patch_artist=True,
        medianprops=dict(color="#111111", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.4, linestyle="none"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    # overlay jittered dots
    rng = np.random.default_rng(0)
    for i, (ratios, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(ratios))
        ax.scatter(np.full(len(ratios), i) + jitter, ratios,
                   color=color, alpha=0.45, s=16, zorder=3)

    ax.axhline(1.0, linestyle="--", color="#aaaaaa", linewidth=1, label="ratio = 1 (no effect)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MSE ratio (adv / clean)  ↓ more robust")
    ax.set_title("Per-sample adversarial MSE ratio distribution")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[mse_ratio_dist] saved → {out_path}")

    # ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="attack_runs_ellipses",
                        help="Root directory of attack result folders")
    parser.add_argument("--out-dir", default="figures",
                        help="Where to save the output figures")
    parser.add_argument("--attack", default="adam")
    parser.add_argument("--norm", default="l2")
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--example-idx", type=int, default=0,
                        help="Which example image index to use in the grid plot")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = discover_summaries(results_dir)
    if not records:
        print(f"No summary.json files found under {results_dir}")
        return
    print(f"Found {len(records)} result(s):")
    for r in records:
        print(f"  init={r['init']}  model={r['model']}  attack={r['attack']}  "
              f"norm={r['norm']}  eps={r['eps']}")
    plot_scatter(
        records, out_dir / "scatter_quality_vs_robustness.png",
        attack=args.attack, norm=args.norm,
    )
    plot_error_decomposition(
        records, out_dir / "error_decomposition.png",
        attack=args.attack, norm=args.norm,
    )
    plot_example_grid(
        records, out_dir / "example_grid.png",
        attack=args.attack, norm=args.norm, eps=args.eps,
        example_idx=args.example_idx,
    )
    plot_mse_ratio_distribution(
        records, out_dir / "mse_ratio_distribution.png",
        attack=args.attack, norm=args.norm,
    )
if __name__ == "__main__":
    main()