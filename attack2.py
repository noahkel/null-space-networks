#!/usr/bin/env python3
"""
attack2.py — Image-domain (network-input) adversarial robustness benchmark.

This is the *second* attack script in the repo and is intentionally
complementary to ``attack.py``:

    attack.py   perturbs the **measurement** y (the limited-angle sinogram),
                re-runs the full reconstruction pipeline, and measures the
                downstream damage.  It asks: "how robust is the whole CT
                pipeline to a perturbation of the physical measurement?"

    attack2.py  perturbs the **network input image** x directly — the
                FBP/pinv reconstruction that is actually fed to the model —
                inside an Lp norm ball, and measures how the *learned map*
                x -> f(x) amplifies that perturbation.  It asks: "how stable
                is the trained network itself?"  This is the classic
                white-box adversarial-example setting (cf. Antun et al.,
                "On instabilities of deep learning in image reconstruction").

Why this is a clean test of architecture
----------------------------------------
Both wrappers ignore the sinogram argument and are *pure functions of the
input image*:

    RESNET : f(x) = x + UNet(x)
    NSN    : f(x) = x + P_null( UNet(x) )            (P_null = proj onto null(A_la))

So perturbing x isolates the network.  The identity skip ``x`` passes the
perturbation through identically in both models; the only architectural
difference is the *learned correction*, which the NSN confines to the null
space.  That makes the null/range error decomposition the natural lens.

Fair, channel-aware comparison (see repo memory)
------------------------------------------------
Total reconstruction error is an unfair yardstick for the NSN: it has
deliberately weak clean values (it only writes the null space) and its range
error is a locked inversion floor it cannot correct.  So every plot here
reports the attack **against each model's own clean baseline** (the
attack-induced delta) and decomposes the error into the range (measured,
shared floor) and null (structural, learned) channels.

Outputs (under ``attack2_runs_<tag>/init_<init>/``)
---------------------------------------------------
  * robustness_multimetric.png   headline: PSNR/SSIM/relL2/MAE vs budget,
                                  clean baseline (dashed) vs adversarial (solid)
  * clean_vs_adv_bars.png         per-metric clean-vs-adv bars at a fixed budget
  * error_decomposition.png       range vs null error split, clean vs adv
  * sensitivity_scatter.png       per-sample adv error vs realised perturbation
  * qualitative_examples.png      GT | clean recon | adv recon, both models
  * <model>/<eps>/summary.json    aggregated metrics
  * <model>/<eps>/per_sample.csv  raw per-sample metrics

Usage (mirrors attack.py / slurm.sh conventions)
-------------------------------------------------
  python attack2.py --type ellipses --init pinv --models resnet,nsn \
      --eps 0.01,0.02,0.05,0.1 --norm l2 --steps 40 \
      --data-root /scratch/noah/data/ellipses_out --model-dir /scratch/noah/models
"""
import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless / cluster safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Reuse the *verified* data/model plumbing from attack.py so this script cannot
# drift from the real pipeline.  The attack itself and every plot below are
# written from scratch — only the loaders/operator builders are shared.
from attack import (
    build_radon,
    get_loader,
    load_model_checkpoint,
    load_summary,
)
from src.utils import (
    decompose_error,
    mae,
    max_abs_err,
    nrmse,
    psnr,
    rel_l2_np,
    set_seed,
    ssim,
    to_4d,
)

# Metric registry: key -> (pretty label, direction).  "higher" = larger is
# better (robust networks keep these high under attack); "lower" = smaller is
# better.  Used to drive every plot and the success criterion.
METRICS: List[Tuple[str, str, str]] = [
    ("psnr", "PSNR (dB)", "higher"),
    ("ssim", "SSIM", "higher"),
    ("rel_l2", "Relative L2 error", "lower"),
    ("mae", "MAE", "lower"),
]

# Stable per-model colours so every figure is visually consistent.
MODEL_COLORS = {"resnet": "#1f77b4", "nsn": "#d62728", "dpnsn": "#2ca02c", "dpnsn_res": "#9467bd"}


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def l2_norm_batch(x: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm of a [B, C, H, W] batch -> [B]."""
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#555555")


# --------------------------------------------------------------------------- #
# Norm-ball projection (image domain)                                         #
# --------------------------------------------------------------------------- #
def proj_l2_ball(delta: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Project per-sample so ||delta_i|| <= eps_i (eps is a [B] tensor)."""
    norms = l2_norm_batch(delta).clamp_min(1e-12)
    eps_v = eps.to(norms).reshape(-1).clamp_min(0.0)
    scale = torch.minimum(torch.ones_like(norms), eps_v / norms)
    return delta * scale.view(-1, 1, 1, 1)


def proj_linf_ball(delta: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    e = eps.to(delta).reshape(-1, 1, 1, 1).clamp_min(0.0)
    return torch.max(torch.min(delta, e), -e)


def project_ball(delta: torch.Tensor, eps: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "l2":
        return proj_l2_ball(delta, eps)
    if norm == "linf":
        return proj_linf_ball(delta, eps)
    raise ValueError(f"Unsupported norm '{norm}'")


def random_start(x: torch.Tensor, eps: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "linf":
        e = eps.to(x).reshape(-1, 1, 1, 1)
        return (torch.rand_like(x) * 2.0 - 1.0) * e
    # l2: random direction projected into the ball
    return proj_l2_ball(torch.randn_like(x), eps)


# --------------------------------------------------------------------------- #
# Attack objective (per-sample, to be *maximised*)                            #
# --------------------------------------------------------------------------- #
def attack_loss(pred: torch.Tensor, x_gt: torch.Tensor, objective: str, radon) -> torch.Tensor:
    """Per-sample loss vector [B] the attacker maximises.

    mse  : total reconstruction error ||pred - x_gt||^2.
    null : only the null-space component ||P_null(pred - x_gt)||^2 — the
           structural/learned channel.  P_null is computed exactly as the NSN
           computes it internally (radon.proj_null_image), so the objective is
           consistent with the model and forces the budget onto the component
           the network is actually responsible for instead of the trivial,
           data-consistent range channel.
    """
    err = pred - x_gt
    if objective == "null":
        if radon is None:
            raise ValueError("objective 'null' requires a radon operator")
        err = radon.proj_null_image(err)
    elif objective != "mse":
        raise ValueError(f"Unknown objective '{objective}'")
    return (err ** 2).reshape(err.shape[0], -1).mean(dim=1)


@dataclass
class AttackResult:
    x_adv: torch.Tensor          # adversarial network input  x + delta
    delta: torch.Tensor          # the perturbation
    runtime_sec: float


def pgd_image_attack(
    model: nn.Module,
    x_init: torch.Tensor,
    y: torch.Tensor,
    x_gt: torch.Tensor,
    eps: torch.Tensor,
    alpha_frac: float,
    steps: int,
    restarts: int,
    norm: str,
    objective: str,
    radon,
    random_init: bool,
) -> AttackResult:
    """White-box PGD in image space on the network *input* x_init.

    The perturbation budget ``eps`` is a per-sample [B] tensor (eps_i scaled to
    a fixed fraction of ||x_init_i||), so every image receives the same
    *relative* perturbation regardless of its intensity.  Per-restart we keep
    the per-sample best (highest loss) adversarial input.  FGSM is the special
    case steps=1, restarts=1, random_init=False.
    """
    start = time.perf_counter()
    batch = x_init.shape[0]
    step_size = (alpha_frac * eps).view(-1, 1, 1, 1)
    best_delta = torch.zeros_like(x_init)
    best_score = torch.full((batch,), -float("inf"), device=x_init.device, dtype=x_init.dtype)

    for _ in range(max(1, restarts)):
        delta = random_start(x_init, eps, norm) if random_init else torch.zeros_like(x_init)
        delta = project_ball(delta, eps, norm)

        for _ in range(steps):
            delta.requires_grad_(True)
            pred = model(x_init + delta, y)
            loss = attack_loss(pred, x_gt, objective, radon).sum()
            grad = torch.autograd.grad(loss, delta)[0]
            with torch.no_grad():
                if norm == "linf":
                    delta = delta + step_size * grad.sign()
                else:  # l2: normalise the gradient per sample
                    g = grad / l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)
                    delta = delta + step_size * g
                delta = project_ball(delta, eps, norm)
            delta = delta.detach()

        with torch.no_grad():
            score = attack_loss(model(x_init + delta, y), x_gt, objective, radon)
            improved = score > best_score
            best_score = torch.where(improved, score, best_score)
            mask = improved.view(-1, 1, 1, 1)
            best_delta = torch.where(mask, delta, best_delta)

    x_adv = (x_init + best_delta).detach()
    return AttackResult(x_adv=x_adv, delta=best_delta.detach(), runtime_sec=time.perf_counter() - start)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def _img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()


def image_metrics(recon_np: np.ndarray, gt_np: np.ndarray) -> Dict[str, float]:
    return {
        "rel_l2": rel_l2_np(recon_np, gt_np),
        "psnr": psnr(recon_np, gt_np),
        "ssim": ssim(recon_np, gt_np),
        "mae": mae(recon_np, gt_np),
        "nrmse": nrmse(recon_np, gt_np),
        "max_err": max_abs_err(recon_np, gt_np),
    }


def error_channels(recon: torch.Tensor, gt: torch.Tensor, radon) -> Dict[str, float]:
    """Range/null decomposition of a single-sample error (recon, gt are 4D, B=1)."""
    e_ran, e_nul = decompose_error(recon - gt, radon)
    ran = float(np.linalg.norm(e_ran.numpy().ravel()))
    nul = float(np.linalg.norm(e_nul.numpy().ravel()))
    total = max(math.hypot(ran, nul), 1e-12)
    return {"e_ran_l2": ran, "e_nul_l2": nul, "e_nul_frac": nul / total, "e_ran_frac": ran / total}


def evaluate_batch(
    x_gt: torch.Tensor,
    x_init: torch.Tensor,
    clean_pred: torch.Tensor,
    adv_pred: torch.Tensor,
    delta: torch.Tensor,
    radon,
    success_factor: float,
) -> List[Dict[str, float]]:
    """One row per sample with clean & adversarial metrics, the channel
    decomposition, and the realised perturbation size."""
    rows: List[Dict[str, float]] = []
    delta_l2 = l2_norm_batch(delta)
    input_l2 = l2_norm_batch(x_init).clamp_min(1e-12)
    for i in range(x_gt.shape[0]):
        gt_np = _img(x_gt[i])
        clean = image_metrics(_img(clean_pred[i]), gt_np)
        adv = image_metrics(_img(adv_pred[i]), gt_np)
        ch_clean = error_channels(clean_pred[i : i + 1], x_gt[i : i + 1], radon)
        ch_adv = error_channels(adv_pred[i : i + 1], x_gt[i : i + 1], radon)

        row: Dict[str, float] = {"gt_norm": float(np.linalg.norm(gt_np.ravel()))}
        for k, v in clean.items():
            row[f"clean_{k}"] = v
        for k, v in adv.items():
            row[f"adv_{k}"] = v
        for k, v in ch_clean.items():
            row[f"clean_{k}"] = v
        for k, v in ch_adv.items():
            row[f"adv_{k}"] = v
        # Attack-induced degradation vs this model's own clean baseline (the
        # fair, channel-aware quantity).  Sign convention: positive = worse.
        row["d_rel_l2"] = adv["rel_l2"] - clean["rel_l2"]
        row["d_mae"] = adv["mae"] - clean["mae"]
        row["d_psnr"] = clean["psnr"] - adv["psnr"]       # dB lost
        row["d_ssim"] = clean["ssim"] - adv["ssim"]
        row["d_e_nul_l2"] = ch_adv["e_nul_l2"] - ch_clean["e_nul_l2"]
        # Realised perturbation magnitude (should match the requested budget).
        row["delta_l2"] = float(delta_l2[i].item())
        row["delta_rel_l2"] = float((delta_l2[i] / input_l2[i]).item())
        # "Success" = the attack at least doubled (success_factor) the rel-L2
        # error over the clean baseline for this model.
        row["success"] = float(adv["rel_l2"] >= success_factor * max(clean["rel_l2"], 1e-12))
        rows.append(row)
    return rows


def aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan"), "n": 0}
    mean = float(arr.mean())
    ci = float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "ci95": ci, "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)), "q75": float(np.percentile(arr, 75)), "n": int(arr.size)}


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    return {key: aggregate([r[key] for r in rows if key in r]) for key in rows[0].keys()}


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _is_better_high(direction: str) -> bool:
    return direction == "higher"


def plot_multimetric_robustness(
    out_dir: Path,
    eps_list: List[float],
    adv_stats: Dict[str, Dict[float, Dict[str, Dict[str, float]]]],
    clean_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """HEADLINE FIGURE.  One panel per metric.  For each model: a dashed
    horizontal line at the clean baseline (median) and a solid marker line of
    the adversarial median across the budget sweep, with an IQR band.  The gap
    between dashed and solid *is* the robustness loss — small gap = robust."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    eps_sorted = sorted(eps_list)
    for ax, (key, label, direction) in zip(axes.ravel(), METRICS):
        adv_key = f"adv_{key}"  # evaluate_batch prefixes adversarial metrics
        for model, per_eps in adv_stats.items():
            color = model_color(model)
            xs = [e for e in eps_sorted if e in per_eps and adv_key in per_eps[e]]
            med = [per_eps[e][adv_key]["median"] for e in xs]
            q25 = [per_eps[e][adv_key]["q25"] for e in xs]
            q75 = [per_eps[e][adv_key]["q75"] for e in xs]
            if xs:
                ax.fill_between(xs, q25, q75, color=color, alpha=0.15)
                ax.plot(xs, med, "-o", color=color, lw=2, ms=5, label=f"{model} (adversarial)")
            # clean baseline (independent of eps)
            base = clean_stats.get(model, {}).get(key, {}).get("median", float("nan"))
            if np.isfinite(base) and xs:
                ax.hlines(base, min(xs), max(xs), color=color, ls="--", lw=1.5,
                          label=f"{model} (clean)")
        arrow = "↑ better" if _is_better_high(direction) else "↓ better"
        ax.set_title(f"{label}   ({arrow})")
        ax.set_xlabel("Perturbation budget  ε = ‖δ‖ / ‖x‖")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if len(eps_sorted) > 1 and min(eps_sorted) > 0:
            ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Image-domain robustness: adversarial (solid) vs clean baseline (dashed)\n"
                 "smaller clean→adv gap = more robust network", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "robustness_multimetric.png", dpi=150)
    plt.close(fig)


def plot_clean_vs_adv_bars(
    out_dir: Path,
    eps: float,
    adv_stats: Dict[str, Dict[float, Dict[str, Dict[str, float]]]],
    clean_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Per-metric grouped bars at a single budget: each model gets a clean bar
    and an adversarial bar.  Shows the absolute level *and* the drop, so the
    NSN's deliberately-weak clean baseline is visible rather than hidden."""
    models = list(adv_stats.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    width = 0.35
    for ax, (key, label, direction) in zip(axes.ravel(), METRICS):
        x = np.arange(len(models))
        clean_vals = [clean_stats.get(m, {}).get(key, {}).get("median", float("nan")) for m in models]
        adv_vals = [adv_stats.get(m, {}).get(eps, {}).get(f"adv_{key}", {}).get("median", float("nan"))
                    for m in models]
        ax.bar(x - width / 2, clean_vals, width, label="clean", color="#1D9E75")
        ax.bar(x + width / 2, adv_vals, width, label="adversarial", color="#D4537E")
        arrow = "↑ better" if _is_better_high(direction) else "↓ better"
        ax.set_title(f"{label}   ({arrow})")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Clean vs adversarial reconstruction quality at ε = {eps:g}  (median over test samples)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "clean_vs_adv_bars.png", dpi=150)
    plt.close(fig)


def plot_error_decomposition(
    out_dir: Path,
    eps: float,
    chan_stats: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]],
) -> None:
    """Range vs null decomposition of the reconstruction error, clean vs adv,
    at a fixed budget.  Left: stacked absolute magnitudes ‖e_ran‖ (shared
    inversion floor) + ‖e_nul‖ (structural/learned channel).  Right: the null
    fraction.  This is the architecture story — where each model's error lives
    and where the attack pushes it."""
    models = list(chan_stats.keys())
    fig, (ax_abs, ax_frac) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(models))
    width = 0.35
    c_ran, c_nul = "#1f77b4", "#d62728"

    def med(model: str, cond: str, key: str) -> float:
        return chan_stats[model][cond].get(eps, {}).get(key, {}).get("median", float("nan"))

    for offset, cond, hatch in [(-width / 2, "clean", ""), (width / 2, "adv", "//")]:
        ran = [med(m, cond, "e_ran_l2") for m in models]
        nul = [med(m, cond, "e_nul_l2") for m in models]
        ax_abs.bar(x + offset, ran, width, color=c_ran, hatch=hatch,
                   label=f"range ‖e_ran‖ ({cond})", edgecolor="white")
        ax_abs.bar(x + offset, nul, width, bottom=ran, color=c_nul, hatch=hatch,
                   label=f"null ‖e_nul‖ ({cond})", edgecolor="white")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(models)
    ax_abs.set_ylabel("‖error component‖ (median)")
    ax_abs.set_title(f"Error split: range (floor) + null (structural) at ε = {eps:g}")
    ax_abs.legend(fontsize=8)
    ax_abs.grid(True, axis="y", alpha=0.3)

    for offset, cond in [(-width / 2, "clean"), (width / 2, "adv")]:
        frac = [med(m, cond, "e_nul_frac") for m in models]
        ax_frac.bar(x + offset, frac, width, label=cond,
                    color="#1D9E75" if cond == "clean" else "#D4537E")
    ax_frac.set_xticks(x)
    ax_frac.set_xticklabels(models)
    ax_frac.set_ylim(0, 1.05)
    ax_frac.set_ylabel("‖e_nul‖ / ‖e‖ (median)")
    ax_frac.set_title("Null-space fraction of the error")
    ax_frac.legend(fontsize=8)
    ax_frac.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Where the error lives: range = shared inversion floor, null = learned channel",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "error_decomposition.png", dpi=150)
    plt.close(fig)


def plot_sensitivity_scatter(out_dir: Path, pool: Dict[str, List[Dict[str, float]]]) -> None:
    """Per-sample cloud: adversarial rel-L2 vs the realised relative
    perturbation, pooled over the whole budget sweep.  Reveals spread and
    worst-case behaviour the medians hide."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, rows in pool.items():
        xs = [r["delta_rel_l2"] for r in rows]
        ys = [r["adv_rel_l2"] for r in rows]
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.5, color=model_color(model), label=model)
    ax.set_xlabel("Realised perturbation  ‖δ‖ / ‖x‖")
    ax.set_ylabel("Adversarial reconstruction error  (rel L2)")
    ax.set_title("Per-sample sensitivity to image-domain perturbation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_scatter.png", dpi=150)
    plt.close(fig)


def plot_qualitative(out_dir: Path, eps: float, examples: "Examples") -> None:
    """GT | (per model) clean recon | adversarial recon, for a few samples,
    at a fixed budget.  Lets a human *see* the robustness; per-panel rel-L2 in
    the title quantifies it."""
    models = examples.models
    gts = examples.gt
    if not gts or any(m not in examples.clean for m in models):
        return
    # Clamp to the smallest captured count so every row is complete.
    n = min([len(gts)] + [len(examples.clean[m]) for m in models] + [len(examples.adv[m]) for m in models])
    if n == 0:
        return
    ncols = 1 + 2 * len(models)
    fig, axes = plt.subplots(n, ncols, figsize=(2.6 * ncols, 2.7 * n), squeeze=False)
    for r in range(n):
        ax = axes[r][0]
        ax.imshow(gts[r], cmap="gray")
        ax.set_title("Ground truth" if r == 0 else "")
        ax.axis("off")
        col = 1
        for m in models:
            clean_img, clean_rl = examples.clean[m][r]
            adv_img, adv_rl = examples.adv[m][r]
            for img, rl, tag in [(clean_img, clean_rl, "clean"), (adv_img, adv_rl, "adv")]:
                a = axes[r][col]
                a.imshow(img, cmap="gray")
                head = f"{m} {tag}" if r == 0 else ""
                a.set_title(f"{head}\nrelL2={rl:.3f}", fontsize=8)
                a.axis("off")
                col += 1
    fig.suptitle(f"Qualitative reconstructions at ε = {eps:g}  (clean vs adversarial)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "qualitative_examples.png", dpi=150)
    plt.close(fig)


@dataclass
class Examples:
    """Holds a handful of images for the qualitative figure, captured at the
    representative budget and for the same sample indices across all models."""
    models: List[str]
    gt: List[np.ndarray] = field(default_factory=list)
    clean: Dict[str, List[Tuple[np.ndarray, float]]] = field(default_factory=dict)
    adv: Dict[str, List[Tuple[np.ndarray, float]]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Image-domain adversarial robustness for Radon reconstruction models.")
    parser.add_argument("--type", required=True, choices=["ellipses", "lodopab"])
    parser.add_argument("--init", default="pinv", choices=["fbp", "pinv", "tv", "lw"])
    parser.add_argument("--models", default="resnet,nsn")
    parser.add_argument("--eps", type=str, default="0.01,0.02,0.05,0.1",
                        help="Comma-separated budget sweep. eps is the perturbation size as a "
                             "fraction of the input-image norm ‖x‖ (per sample).")
    parser.add_argument("--norm", default="l2", choices=["l2", "linf"])
    parser.add_argument("--objective", default="mse", choices=["mse", "null"],
                        help="mse = total error; null = only the null-space (structural) error component.")
    parser.add_argument("--steps", type=int, default=40, help="PGD steps (1 => FGSM).")
    parser.add_argument("--alpha-frac", type=float, default=0.2,
                        help="PGD step size as a fraction of eps per step.")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-factor", type=float, default=2.0,
                        help="Attack 'succeeds' on a sample if adv rel-L2 >= factor * clean rel-L2.")
    parser.add_argument("--num-examples", type=int, default=3, help="Samples in the qualitative figure.")
    parser.add_argument("--data-root", default=None, help="Path to the {type}_out data dir (noise subfolder).")
    parser.add_argument("--model-dir", default=None, help="Base dir holding the checkpoints.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tag", default=None, help="Label for the output dir (default: --type).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    example = args.type
    init_method = args.init.lower()
    eps_list = parse_float_list(args.eps)
    model_names = parse_list_arg(args.models)
    rep_eps = max(eps_list)  # budget used for the fixed-eps figures / examples

    # --- operator, data, output dir (mirrors attack.py's single-noise path) ---
    summary = load_summary(example, "", data_root=args.data_root)
    beta = float(summary["mean_norm_y_minus_y_delta"])  # only needed to build dpnsn; harmless otherwise
    radon = build_radon(summary, device=device)
    loader = get_loader(
        example=example, init_method=init_method, batch_size=args.batch_size, split=args.split,
        n_train=args.n_train, n_test=args.n_test, num_workers=args.num_workers,
        data_root=args.data_root, noise="",
    )

    tag = args.tag or example
    out_root = Path(args.out_dir) if args.out_dir else Path(f"attack2_runs_{tag}") / f"init_{init_method}"
    out_root.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "eps_list": eps_list, "rep_eps": rep_eps})
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Cache the test batches once (deterministic order; shared across models so
    # the qualitative figure compares the same samples).
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seen = 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = to_4d(y_delta).to(device)
        batches.append((x_gt, x_init, y_delta))
        seen += x_gt.shape[0]
        if seen >= args.max_samples:
            break

    random_init = not args.no_random_start

    # Result accumulators ----------------------------------------------------
    adv_stats: Dict[str, Dict[float, Dict[str, Dict[str, float]]]] = {}
    clean_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    chan_stats: Dict[str, Dict[str, Dict[float, Dict[str, Dict[str, float]]]]] = {}
    pool: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_names}
    examples = Examples(models=model_names)

    for model_name in model_names:
        print(f"loading {model_name}")
        model = load_model_checkpoint(
            example=example, init_method=init_method, model_name=model_name, radon=radon,
            beta=beta, device=device, model_dir=args.model_dir, noise="",
        )
        adv_stats[model_name] = {}
        chan_stats[model_name] = {"clean": {}, "adv": {}}

        # Clean predictions (cached once per model) and clean metrics.
        clean_preds: List[torch.Tensor] = []
        clean_rows: List[Dict[str, float]] = []
        with torch.no_grad():
            for x_gt, x_init, y_delta in batches:
                cp = model(x_init, y_delta)
                clean_preds.append(cp)
                for i in range(x_gt.shape[0]):
                    row = image_metrics(_img(cp[i]), _img(x_gt[i]))
                    row.update(error_channels(cp[i : i + 1], x_gt[i : i + 1], radon))
                    clean_rows.append(row)
        clean_stats[model_name] = summarize(clean_rows)

        for eps_nominal in eps_list:
            rows: List[Dict[str, float]] = []
            runtime = 0.0
            captured = 0
            result_dir = out_root / model_name / f"{args.norm}_eps_{eps_nominal:g}"
            result_dir.mkdir(parents=True, exist_ok=True)

            for (x_gt, x_init, y_delta), clean_pred in zip(batches, clean_preds):
                eps_batch = eps_nominal * l2_norm_batch(x_init)  # per-sample budget
                res = pgd_image_attack(
                    model=model, x_init=x_init, y=y_delta, x_gt=x_gt, eps=eps_batch,
                    alpha_frac=args.alpha_frac, steps=args.steps, restarts=args.restarts,
                    norm=args.norm, objective=args.objective, radon=radon, random_init=random_init,
                )
                runtime += res.runtime_sec
                with torch.no_grad():
                    adv_pred = model(res.x_adv, y_delta)
                batch_rows = evaluate_batch(
                    x_gt=x_gt, x_init=x_init, clean_pred=clean_pred, adv_pred=adv_pred,
                    delta=res.delta, radon=radon, success_factor=args.success_factor,
                )
                rows.extend(batch_rows)

                # Capture a few images for the qualitative figure (rep eps only).
                if eps_nominal == rep_eps and captured < args.num_examples:
                    for i in range(x_gt.shape[0]):
                        if captured >= args.num_examples:
                            break
                        if model_name == model_names[0]:
                            examples.gt.append(_img(x_gt[i]))
                        examples.clean.setdefault(model_name, []).append(
                            (_img(clean_pred[i]), batch_rows[i]["clean_rel_l2"]))
                        examples.adv.setdefault(model_name, []).append(
                            (_img(adv_pred[i]), batch_rows[i]["adv_rel_l2"]))
                        captured += 1

            stats = summarize(rows)
            adv_stats[model_name][eps_nominal] = stats
            chan_stats[model_name]["clean"][eps_nominal] = {
                k: clean_stats[model_name][k] for k in ("e_ran_l2", "e_nul_l2", "e_nul_frac")
                if k in clean_stats[model_name]
            }
            # Strip the 'adv_' prefix so clean and adv share the plot helper's keys.
            chan_stats[model_name]["adv"][eps_nominal] = {
                "e_ran_l2": stats.get("adv_e_ran_l2", {}),
                "e_nul_l2": stats.get("adv_e_nul_l2", {}),
                "e_nul_frac": stats.get("adv_e_nul_frac", {}),
            }
            pool[model_name].extend(rows)

            # Persist per-(model,eps) outputs.
            out_summary = {
                "model_name": model_name, "attack": "image_pgd", "norm": args.norm,
                "objective": args.objective, "eps": eps_nominal, "steps": args.steps,
                "num_examples": len(rows), "runtime_total_sec": runtime,
                "runtime_per_example_sec": runtime / max(len(rows), 1),
                "metrics": stats,
            }
            with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(out_summary, f, indent=2)
            if rows:
                with open(result_dir / "per_sample.csv", "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

            adv_rl = stats.get("adv_rel_l2", {}).get("median", float("nan"))
            clean_rl = clean_stats[model_name].get("rel_l2", {}).get("median", float("nan"))
            succ = stats.get("success", {}).get("mean", float("nan"))
            print(f"[{model_name} eps={eps_nominal:g}] n={len(rows)} "
                  f"clean_relL2={clean_rl:.4f} adv_relL2={adv_rl:.4f} success={succ:.2f}")

    # ----------------------------- figures ---------------------------------- #
    plot_multimetric_robustness(out_root, eps_list, adv_stats, clean_stats)
    plot_clean_vs_adv_bars(out_root, rep_eps, adv_stats, clean_stats)
    plot_error_decomposition(out_root, rep_eps, chan_stats)
    plot_sensitivity_scatter(out_root, pool)
    plot_qualitative(out_root, rep_eps, examples)
    print(f"\nDone. Figures and per-run metrics written under: {out_root}")


if __name__ == "__main__":
    main()
