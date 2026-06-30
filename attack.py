#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from src.ellipse_dataloader import get_ellipse_dataloader
from src.landweber import landweber
from src.radon import AstraRadonAdapter
from src.radon_matrix import MatrixRadonAdapter
from src.total_variation import tv_cp
from src.utils import (
    build_models,
    decompose_error,
    mae,
    max_abs_err,
    nrmse,
    psnr,
    rel_l2_np,
    set_seed,
    ssim,
    to_4d,
    visualise_decomposition,
    decompose_metrics
)

def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]

def l2_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)

def linf_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).abs().max(dim=1).values

def proj_l2_ball(delta: torch.Tensor, eps) -> torch.Tensor:
    # eps may be a scalar (one global budget) or a per-sample 1-D tensor of shape [B]
    # (per-sample budget eps_i = eps_frac * ||y_i||). A zero budget maps to zeros.
    norms = l2_norm_batch(delta).clamp_min(1e-12)
    if torch.is_tensor(eps):
        eps_vec = eps.to(device=norms.device, dtype=norms.dtype).reshape(-1).clamp_min(0.0)
    else:
        if eps <= 0:
            return torch.zeros_like(delta)
        eps_vec = torch.full_like(norms, float(eps))
    scale = torch.minimum(torch.ones_like(norms), eps_vec / norms)
    return delta * scale.view(-1, 1, 1, 1)

def proj_linf_ball(delta: torch.Tensor, eps) -> torch.Tensor:
    if torch.is_tensor(eps):
        eps_t = eps.to(device=delta.device, dtype=delta.dtype).reshape(-1, 1, 1, 1).clamp_min(0.0)
        return torch.max(torch.min(delta, eps_t), -eps_t)
    if eps <= 0:
        return torch.zeros_like(delta)
    return delta.clamp(-eps, eps)

def project_delta(delta: torch.Tensor, eps: float, norm: str, projector: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    delta = projector(delta)
    if norm == "l2":
        return projector(proj_l2_ball(delta, eps))
    if norm == "linf":
        return projector(proj_linf_ball(delta, eps))
    raise ValueError(f"Unsupported norm '{norm}'")

def random_start_like(y: torch.Tensor, eps: float, norm: str, projector: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if not torch.is_tensor(eps) and eps <= 0:
        return torch.zeros_like(y)
    if norm == "linf":
        if torch.is_tensor(eps):
            eps_t = eps.to(device=y.device, dtype=y.dtype).reshape(-1, 1, 1, 1)
            delta = (torch.rand_like(y) * 2.0 - 1.0) * eps_t
        else:
            delta = torch.empty_like(y).uniform_(-eps, eps)
    elif norm == "l2":
        delta = torch.randn_like(y)
        delta = proj_l2_ball(delta, eps)
    else:
        raise ValueError(f"Unsupported norm '{norm}'")
    return projector(delta)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean isotropic total variation over a batch [B, C, H, W]."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

def reduce_loss(loss_map: torch.Tensor) -> torch.Tensor:
    if loss_map.ndim <= 1:
        return loss_map.mean()
    return loss_map.reshape(loss_map.shape[0], -1).mean(dim=1).mean()

def per_example_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).reshape(x.shape[0], -1).mean(dim=1)

def batch_mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.abs().reshape(x.shape[0], -1).mean(dim=1)

def confidence_interval_95(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    half_width = float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, half_width

def _median_of(rows: List[Dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return float(np.median(vals)) if vals else float("nan")

@dataclass
class AttackResult:
    y_adv: torch.Tensor
    delta: torch.Tensor
    runtime_sec: float

class InitReconstructor:
    def __init__(self, example: str, init_method: str, summary: Dict, radon):
        self.example = example
        self.init_method = init_method
        self.summary = summary
        self.radon = radon
        self.l_value = float(summary.get("operator_norm_A2") or radon.norm_A2 or 1.0)
        self.tau = 1.0 / max(self.l_value, 1e-6)
        self.sigma = 1.0 / max(self.l_value, 1e-6)
        self.theta = 1.0
        _tv_alpha = summary.get("tv_best_alpha")
        self.tv_alpha = float(_tv_alpha) if _tv_alpha not in (None, "None") else 0.0
        self.tv_iters = int(summary.get("tv_iters_final") or 0)
        self.lw_iters = int(summary.get("lw_iters") or 0)
        self.lw_omega = float(summary.get("lw_omega") or 1.0 / max(self.l_value, 1e-6))
        self.noise_sigma_rel = float(summary.get("noise_sigma_rel") or 0.0)

    def _fbp_seed(self, y: torch.Tensor) -> torch.Tensor:
        if self.init_method in ("fbp", "pinv") and self.example == "ellipses":
            return self.radon.fbp_la(y)
        return self.radon.fbp(y, filter_name="ram-lak")

    def surrogate(self, y: torch.Tensor) -> torch.Tensor:
        return self._fbp_seed(y)

    def exact(self, y: torch.Tensor) -> torch.Tensor:
        if self.init_method == "fbp":
            return self._fbp_seed(y)

        if self.init_method == "pinv":
            return self.radon.backward_la(y)

        x0 = self._fbp_seed(y)

        if self.init_method == "lw":
            return landweber(
                A=self.radon.forward_la,
                AT=self.radon.backward_la,
                g=y,
                x0=x0,
                omega=self.lw_omega,
                n_iter=self.lw_iters,
            )

        if self.init_method == "tv":
            return tv_cp(
                x0=x0,
                A=self.radon.forward_la,
                AT=self.radon.backward_la,
                g=y,
                alpha=self.tv_alpha,
                tau=self.tau,
                sigma=self.sigma,
                theta=self.theta,
                Niter=self.tv_iters,
                print_flag=False,
            )

        raise ValueError(f"Unsupported init method '{self.init_method}'")

class ModelAttackAdapter:
    def __init__(
        self,
        model: nn.Module,
        init_reconstructor: InitReconstructor,
        projector: Callable[[torch.Tensor], torch.Tensor],
        attack_init_mode: str
    ):
        self.model = model
        self.init_reconstructor = init_reconstructor
        self.projector = projector
        self.attack_init_mode = attack_init_mode
        self._clean_init_ref: Optional[torch.Tensor] = None

    def build_inputs(self, y_adv: torch.Tensor, mode: Optional[str] = None, project: bool = True,) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = mode or self.attack_init_mode
        if project:
            y_adv = self.projector(y_adv)
        if mode == "surrogate":
            x_init = self.init_reconstructor.surrogate(y_adv)
        elif mode == "exact":
            x_exact = self.init_reconstructor.exact(y_adv)
            x_surrogate = self.init_reconstructor.surrogate(y_adv)
            x_init = x_exact + (x_surrogate - x_surrogate.detach())
        else:
            raise ValueError(f"Unknown init mode '{mode}'")

        return x_init, y_adv

    def forward(self, y_adv: torch.Tensor, mode: Optional[str] = None, project: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_init, y_adv = self.build_inputs(y_adv, mode=mode, project=project)
        pred = self.model(x_init, y_adv)
        return pred, x_init, y_adv

def attack_objective(
    pred: torch.Tensor,
    x_gt: torch.Tensor,
    clean_pred: torch.Tensor,
    objective: str,
    shift_weight: float,
    target: torch.Tensor = None,
    radon=None,
) -> torch.Tensor:
    """
    Attack loss to be *maximised*.

    Notation:  e = pred - x_gt          (error vs ground truth)
             s = pred - clean_pred    (shift vs the clean / unattacked prediction)
             P_null  = projection onto the null space of the forward operator
             P_range = I - P_null     (data-consistent / range component)
             λ       = shift_weight

    Objectives:

      mse          : ‖e‖²                        (total error vs GT)
      targeted     : -‖pred - target‖²           (distance to a chosen target image)
      shift        : ‖s‖²                        (deviation from the clean prediction)
      hybrid       : ‖e‖² + λ‖s‖²                (GT error + weighted shift)

      null         : ‖P_null e‖²                 (null-space error vs GT)
      null_shift   : ‖P_null s‖²                 (null-space shift vs clean pred)
      null_hybrid  : ‖P_null e‖² - λ‖P_range e‖² (null error, penalising the range
                                                  component so budget isn't wasted on
                                                  the data-consistent channel the NSN
                                                  cannot correct by design)

      range        : ‖P_range e‖²                (range / data-consistent error vs GT)
      range_shift  : ‖P_range s‖²                (range shift vs clean pred)
      range_hybrid : ‖P_range e‖² - λ‖P_null e‖²  (range error, penalising the null
                                                  component, mirror of null_hybrid)
    """
    gt_term = reduce_loss((pred - x_gt) ** 2)
    shift_term = reduce_loss((pred - clean_pred.detach()) ** 2)
    target_term = reduce_loss((pred - target.detach()) **2)
    if objective == "targeted":
        return -target_term
    if objective == "mse":
        return gt_term
    if objective == "shift":
        return shift_term
    if objective == "hybrid":
        return gt_term + shift_weight * shift_term
    if objective in ("range", "range_shift", "range_hybrid"):
        if radon is None:
            raise ValueError(f"Objective '{objective}' requires a radon operator.")
        err = (pred - x_gt) if objective != "range_shift" else (pred - clean_pred.detach())
        err_range = err - radon.proj_null_image(err)
        range_term = reduce_loss(err_range**2)
        if objective != "range_hybrid":
            return range_term
        err_null = err - err_range
        null_term = reduce_loss(err_null **2)
        return range_term - shift_weight * null_term
    if objective in ("null", "null_shift", "null_hybrid"):
        if radon is None:
            raise ValueError(f"Objective '{objective}' requires a radon operator.")
        err = (pred - x_gt) if objective != "null_shift" else (pred - clean_pred.detach())
        err_null = radon.proj_null_image(err)
        null_term = reduce_loss(err_null ** 2)
        if objective != "null_hybrid":
            return null_term
        # Penalise the range component so budget is not wasted on the trivial,
        # data-consistent channel the NSN cannot correct by design.
        err_range = err - err_null
        range_term = reduce_loss(err_range ** 2)
        return null_term - shift_weight * range_term

    raise ValueError(f"Unknown objective '{objective}'")

def fgsm_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: float,
    norm: str,
    objective: str,
    shift_weight: float,
    stealth_weight: float = 0.0,
) -> AttackResult:
    start = time.perf_counter()
    with torch.no_grad():
        y_proj = adapter.projector(y_clean)
    y_adv = y_clean.detach().clone().requires_grad_(True)
    pred, _, y_adv = adapter.forward(y_adv, project=False)
    loss = attack_objective(pred, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon)
    if stealth_weight > 0.0:
        loss = loss - stealth_weight * reduce_loss((y_adv - y_clean.detach()).abs())
    grad = torch.autograd.grad(loss, y_adv, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        eps_b = eps.view(-1, 1, 1, 1) if torch.is_tensor(eps) else eps
        if norm == "linf":
            delta = eps_b * grad.sign()
        elif norm == "l2":
            grad_norm = l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)
            delta = eps_b * grad / grad_norm
        else:
            raise ValueError(f"Unsupported norm '{norm}'")
        y_adv = adapter.projector(y_proj + delta)
        delta = y_adv - y_clean
    return AttackResult(y_adv=y_adv.detach(), delta=delta.detach(), runtime_sec=time.perf_counter() - start)

def pgd_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    restarts: int,
    norm: str,
    objective: str,
    shift_weight: float,
    random_start: bool,
    stealth_weight: float = 0.0,
) -> AttackResult:
    start = time.perf_counter()
    best_y_adv = y_clean.detach().clone()
    best_delta = torch.zeros_like(y_clean)
    best_score = -float("inf")

    for _ in range(restarts):
        delta = random_start_like(y_clean, eps, norm, adapter.projector) if random_start else torch.zeros_like(y_clean)
        delta = project_delta(delta, eps, norm, adapter.projector)

        for _ in range(steps):
            with torch.no_grad():
                y_proj = adapter.projector(y_clean + delta)
            y_adv = (y_clean + delta).detach().requires_grad_(True)
            pred, _, _ = adapter.forward(y_adv, project=False)
            loss = attack_objective(pred, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon)
            if stealth_weight > 0.0:
                loss = loss - stealth_weight * reduce_loss((y_adv - y_clean.detach()).abs())
            grad = torch.autograd.grad(loss, y_adv, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                if norm == "linf":
                    delta = (y_proj + alpha * grad.sign()) - y_clean
                elif norm == "l2":
                    grad_norm = l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)
                    delta = (y_proj + alpha * grad / grad_norm) - y_clean
                else:
                    raise ValueError(f"Unsupported norm '{norm}'")

                delta = project_delta(delta, eps, norm, adapter.projector)

        with torch.no_grad():
            y_adv = adapter.projector(y_clean + delta)
            pred, _, _ = adapter.forward(y_adv, project=False)
            score = float(attack_objective(pred, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon).item())
            if score > best_score:
                best_score = score
                best_y_adv = y_adv.detach().clone()
                best_delta = (best_y_adv - y_clean).detach().clone()

    return AttackResult(y_adv=best_y_adv, delta=best_delta, runtime_sec=time.perf_counter() - start)

def spsa_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    samples: int,
    sigma: float,
    norm: str,
    objective: str,
    shift_weight: float,
    stealth_weight: float = 0.0,
) -> AttackResult:
    start = time.perf_counter()
    delta = torch.zeros_like(y_clean)

    for _ in range(steps):
        grad_est = torch.zeros_like(y_clean)
        for _ in range(samples):
            direction = torch.empty_like(y_clean).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            direction = adapter.projector(direction)

            y_plus = adapter.projector(y_clean + delta + sigma * direction)
            y_minus = adapter.projector(y_clean + delta - sigma * direction)

            with torch.no_grad():
                pred_plus, _, _ = adapter.forward(y_plus, mode="exact")
                pred_minus, _, _ = adapter.forward(y_minus, mode="exact")
                loss_plus = attack_objective(pred_plus, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon)
                loss_minus = attack_objective(pred_minus, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon)

            grad_est = grad_est + ((loss_plus - loss_minus) / (2.0 * sigma)) * direction

        grad_est = grad_est / max(samples, 1)

        # Stealthiness: add analytic gradient of -stealth_weight * mean(|δ|)
        # No finite differences needed since δ is known analytically
        if stealth_weight > 0.0:
            grad_est = grad_est - stealth_weight * delta.sign()

        with torch.no_grad():
            if norm == "linf":
                delta = delta + alpha * grad_est.sign()
            elif norm == "l2":
                grad_norm = l2_norm_batch(grad_est).clamp_min(1e-12).view(-1, 1, 1, 1)
                delta = delta + alpha * grad_est / grad_norm
            else:
                raise ValueError(f"Unsupported norm '{norm}'")
            delta = project_delta(delta, eps, norm, adapter.projector)

    with torch.no_grad():
        y_adv = adapter.projector(y_clean + delta)
        delta = y_adv - y_clean

    return AttackResult(y_adv=y_adv.detach(), delta=delta.detach(), runtime_sec=time.perf_counter() - start)

def adam_attack(
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    eps: float,
    norm: str,
    objective: str,
    shift_weight: float,
    tv_weight: float,
    consistency_weight: float,
    steps:  int = 500,
    lr: float = 0.01,
    scheduler_patience: int = 50,
    stealth_weight: float = 0.0,
    restarts: int = 4,
    random_start: bool = True,
) -> AttackResult:
    start = time.perf_counter()
    best_y_adv = y_clean.detach().clone()
    best_delta = torch.zeros_like(y_clean)
    best_score = -float("inf")
    for _ in range(restarts):
        delta = random_start_like(y_clean, eps, norm, adapter.projector) if random_start else torch.zeros_like(y_clean)
        delta = project_delta(delta, eps, norm, adapter.projector)
        opt = torch.optim.Adam([delta], lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=scheduler_patience)

        # Pre-compute clean init once for consistency term
        with torch.no_grad():
            clean_init = adapter.init_reconstructor.exact(adapter.projector(y_clean))

        for _ in range(steps):
            opt.zero_grad()

            y_adv = adapter.projector(y_clean + delta)
            pred, x_init, _ = adapter.forward(y_adv, project=False)

            # Negate because Adam minimizes, but we want to maximize reconstruction error
            loss = -attack_objective(pred, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon)

            if tv_weight > 0.0:
                loss = loss + tv_weight * total_variation(pred)

            if consistency_weight > 0.0:
                loss = loss + consistency_weight * reduce_loss((x_init - clean_init) ** 2)

            if stealth_weight > 0.0:
                loss = loss + stealth_weight * reduce_loss(delta.abs())

            loss.backward()
            opt.step()
            sched.step(loss.item())

            with torch.no_grad():
                delta.data = project_delta(delta.data, eps, norm, adapter.projector)
        with torch.no_grad():
            y_adv = adapter.projector(y_clean + delta)
            pred, _, _ = adapter.forward(y_adv, project=False)
            score = float(attack_objective(pred, x_gt, clean_pred, objective, shift_weight, radon=adapter.init_reconstructor.radon).item())
        if score > best_score:
                best_score = score
                best_y_adv = y_adv.detach().clone()
                best_delta = (best_y_adv - y_clean).detach().clone()

    return AttackResult(y_adv=best_y_adv.detach(), delta=best_delta, runtime_sec=time.perf_counter() - start)

def load_summary(example: str, data_root: Optional[str] = None) -> Dict:
    root = data_root or f"{example}_out"
    summary_path = Path(root) / f"summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_radon(summary: Dict, device: torch.device):
    angles = np.asarray(summary["angles"], dtype=np.float64)
    phi = tuple(summary["phi"])  # already in radians
    matrix_mode = int(summary.get("matrix_mode", 0))
    if matrix_mode == 1:
        return MatrixRadonAdapter(
            resolution=int(summary["img_size"]),
            angles=angles,
            det_count=int(summary["det_count"]),
            dx=float(summary["dx"]),
            estimate_norm=False,
            device=device,
            dtype=torch.float64,
            phi=phi,
            svd_threshold=float(summary.get("svd_threshold", 4e-3)),
            cache_dir="radon_cache",
        )
    return AstraRadonAdapter(
        resolution=int(summary["img_size"]),
        angles=angles,
        det_count=int(summary["det_count"]),
        clip_to_circle=False,
        dx=float(summary["dx"]),
        estimate_norm=False,
        device=device,
        dtype=torch.float64,
        phi=phi,
    )

def load_model_checkpoint(
    example: str,
    init_method: str,
    model_name: str,
    radon,
    beta: float,
    device: torch.device,
    model_dir: Optional[str] = None,
) -> nn.Module:
    base = Path(model_dir) if model_dir else None
    candidates = [
        *(
            [base / f"init_{init_method}" / f"checkpoints" / f"{model_name}_best.pt"]
            if base else []
        ),
        Path(f"runs_{example}") / f"init_{init_method}" / f"checkpoints" / f"{model_name}_best.pt",
        Path(f"checkpoints") / f"{model_name}_best.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_name}' and init '{init_method}'. "
            f"Searched:\n  {searched}"
        )

    model = build_models([model_name], radon=radon, beta=beta)[model_name].to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze().numpy()

def evaluate_batch(
    x_gt: torch.Tensor,
    clean_init: torch.Tensor,
    clean_y: torch.Tensor,
    clean_pred: torch.Tensor,
    adv_init: torch.Tensor,
    adv_y: torch.Tensor,
    adv_pred: torch.Tensor,
    delta: torch.Tensor,
    success_rel_l2_factor: float,
    radon=None,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    batch_size = x_gt.shape[0]

    clean_mse_batch = per_example_mse(clean_pred, x_gt)
    adv_mse_batch = per_example_mse(adv_pred, x_gt)
    delta_l2_batch = l2_norm_batch(delta)
    delta_linf_batch = linf_norm_batch(delta)
    sino_shift_batch = batch_mean_abs(delta)

    for i in range(batch_size):
        gt_np = to_numpy_img(x_gt[i])
        clean_pred_np = to_numpy_img(clean_pred[i])
        adv_pred_np = to_numpy_img(adv_pred[i])
        clean_init_np = to_numpy_img(clean_init[i])
        adv_init_np = to_numpy_img(adv_init[i])
        clean_y_np = to_numpy_img(clean_y[i])
        adv_y_np = to_numpy_img(adv_y[i])

        clean_rel_l2 = rel_l2_np(clean_pred_np, gt_np)
        adv_rel_l2 = rel_l2_np(adv_pred_np, gt_np)
        init_shift = rel_l2_np(adv_init_np, clean_init_np)
        pred_shift = rel_l2_np(adv_pred_np, clean_pred_np)

        # Image-comparison metrics for the *initialisation* reconstruction
        # (FBP/pinv/tv/lw output, i.e. the network input before the NSN).
        # These quantify how much the attack already corrupts the recon that
        # is fed into the network, separately from the final prediction.
        clean_init_rel_l2 = rel_l2_np(clean_init_np, gt_np)
        adv_init_rel_l2 = rel_l2_np(adv_init_np, gt_np)

        clean_mse = float(clean_mse_batch[i].item())
        adv_mse = float(adv_mse_batch[i].item())
        clean_sino_l2 = float(np.linalg.norm(clean_y_np.reshape(-1)))
        delta_l2_i = float(delta_l2_batch[i].item())

        row: Dict[str, float] = {
            "gt_norm": float(np.linalg.norm(gt_np.ravel())),
            "clean_mse": clean_mse,
            "adv_mse": adv_mse,
            "mse_ratio": adv_mse / max(clean_mse, 1e-12),
            "clean_rel_l2": clean_rel_l2,
            "adv_rel_l2": adv_rel_l2,
            "rel_l2_ratio": adv_rel_l2 / max(clean_rel_l2, 1e-12),
            "clean_psnr": psnr(clean_pred_np, gt_np),
            "adv_psnr": psnr(adv_pred_np, gt_np),
            "clean_ssim": ssim(clean_pred_np, gt_np),
            "adv_ssim": ssim(adv_pred_np, gt_np),
            "clean_mae": mae(clean_pred_np, gt_np),
            "adv_mae": mae(adv_pred_np, gt_np),
            "clean_nrmse": nrmse(clean_pred_np, gt_np),
            "adv_nrmse": nrmse(adv_pred_np, gt_np),
            "clean_max_err": max_abs_err(clean_pred_np, gt_np),
            "adv_max_err": max_abs_err(adv_pred_np, gt_np),
            # Init-reconstruction metrics (network input, before the NSN)
            "clean_init_rel_l2": clean_init_rel_l2,
            "adv_init_rel_l2": adv_init_rel_l2,
            "init_rel_l2_ratio": adv_init_rel_l2 / max(clean_init_rel_l2, 1e-12),
            "clean_init_psnr": psnr(clean_init_np, gt_np),
            "adv_init_psnr": psnr(adv_init_np, gt_np),
            "clean_init_ssim": ssim(clean_init_np, gt_np),
            "adv_init_ssim": ssim(adv_init_np, gt_np),
            "clean_init_mae": mae(clean_init_np, gt_np),
            "adv_init_mae": mae(adv_init_np, gt_np),
            "pred_shift_rel_l2": pred_shift,
            "init_shift_rel_l2": init_shift,
            "delta_l2": delta_l2_i,
            "delta_linf": float(delta_linf_batch[i].item()),
            "delta_mean_abs": float(sino_shift_batch[i].item()),
            "delta_rel_l2": delta_l2_i / max(clean_sino_l2, 1e-12),
            "clean_sino_l2": clean_sino_l2,
            "adv_sino_l2": float(np.linalg.norm(adv_y_np.reshape(-1))),
            "success_rel_l2": float(adv_rel_l2 >= success_rel_l2_factor * max(clean_rel_l2, 1e-12)),
        }

        if radon is not None:
            e_ran_c, e_nul_c = decompose_error(clean_pred[i: i + 1] - x_gt[i: i + 1], radon)
            e_ran_a, e_nul_a = decompose_error(adv_pred[i: i + 1] - x_gt[i: i + 1], radon)
            clean_e_l2 = max(float(np.linalg.norm((clean_pred_np - gt_np).ravel())), 1e-12)
            adv_e_l2 = max(float(np.linalg.norm((adv_pred_np - gt_np).ravel())), 1e-12)
            clean_e_ran_l2 = float(np.linalg.norm(e_ran_c.numpy().ravel()))
            clean_e_nul_l2 = float(np.linalg.norm(e_nul_c.numpy().ravel()))
            adv_e_ran_l2 = float(np.linalg.norm(e_ran_a.numpy().ravel()))
            adv_e_nul_l2 = float(np.linalg.norm(e_nul_a.numpy().ravel()))
            row.update({
                "clean_e_ran_l2": clean_e_ran_l2,
                "clean_e_nul_l2": clean_e_nul_l2,
                "clean_e_ran_l2_frac": clean_e_ran_l2 / max(clean_e_l2, 1e-12),
                "clean_e_nul_l2_frac": clean_e_nul_l2 / max(clean_e_l2, 1e-12),
                "adv_e_ran_l2": adv_e_ran_l2,
                "adv_e_nul_l2": adv_e_nul_l2,
                "adv_e_ran_l2_frac": adv_e_ran_l2 / max(adv_e_l2, 1e-12),
                "adv_e_nul_l2_frac": adv_e_nul_l2 / max(adv_e_l2, 1e-12), 
            })

            #   ||proj_ran(A x_hat) - y|| / ||y||   on the measured angles.
            # Damaging adversarial reconstructions stay measurement-consistent, i.e. the
            # error lives in the small-singular-value / null subspace the data
            # cannot constrain. A data-consistent model (NSN) should keep this
            # near zero; an unconstrained ResNet need not.
            with torch.no_grad():
                def _consistency(pred_t, y_t):
                    y_hat = radon.proj_ran(radon.forward(pred_t))
                    num = float(torch.linalg.norm((y_hat - y_t).reshape(-1)).item())
                    den = float(torch.linalg.norm(y_t.reshape(-1)).item())
                    return num / max(den, 1e-12)
                clean_consistency = _consistency(clean_pred[i: i + 1], clean_y[i: i + 1])
                adv_consistency = _consistency(adv_pred[i: i + 1], adv_y[i: i + 1])
            row.update({
                "clean_consistency_rel": clean_consistency,
                "adv_consistency_rel": adv_consistency,
            })
            # Per-metric range/null decomposition. ‖e‖² = ‖e_ran‖² + ‖e_nul‖², but
            # SSIM/PSNR/MAE/… are non-additive and cannot be split from the L2 norms
            # above. Instead we rebuild the reconstruction that carries *only* the
            # range (x_gt + e_ran) resp. null (x_gt + e_nul) component of the error and
            # score it with the same image metrics as the full prediction. This shows
            # how much each error subspace degrades each metric on its own — e.g. how
            # much of the SSIM/PSNR drop is structural (null) vs data-consistent (range).
            row.update({f"clean_{k}": v for k, v in decompose_metrics(clean_pred[i:i+1] - x_gt[i:i+1], x_gt[i:i+1], radon).items()})
            row.update({f"adv_{k}":   v for k, v in decompose_metrics(adv_pred[i:i+1]   - x_gt[i:i+1], x_gt[i:i+1], radon).items()})

            # Decompose the *init-reconstruction* error too, so we can see how the
            # attack distributes range vs null energy in the network input,
            # before the NSN is applied.
            e_ran_ic, e_nul_ic = decompose_error(clean_init[i: i + 1] - x_gt[i: i + 1], radon)
            e_ran_ia, e_nul_ia = decompose_error(adv_init[i: i + 1] - x_gt[i: i + 1], radon)
            clean_ie_l2 = max(float(np.linalg.norm((clean_init_np - gt_np).ravel())), 1e-12)
            adv_ie_l2 = max(float(np.linalg.norm((adv_init_np - gt_np).ravel())), 1e-12)
            clean_ie_ran_l2 = float(np.linalg.norm(e_ran_ic.numpy().ravel()))
            clean_ie_nul_l2 = float(np.linalg.norm(e_nul_ic.numpy().ravel()))
            adv_ie_ran_l2 = float(np.linalg.norm(e_ran_ia.numpy().ravel()))
            adv_ie_nul_l2 = float(np.linalg.norm(e_nul_ia.numpy().ravel()))
            row.update({
                "clean_init_e_ran_l2": clean_ie_ran_l2,
                "clean_init_e_nul_l2": clean_ie_nul_l2,
                "clean_init_e_ran_frac": clean_ie_ran_l2 / max(clean_ie_l2, 1e-12),
                "clean_init_e_nul_frac": clean_ie_nul_l2 / max(clean_ie_l2, 1e-12),
                "adv_init_e_ran_l2": adv_ie_ran_l2,
                "adv_init_e_nul_l2": adv_ie_nul_l2,
                "adv_init_e_ran_frac": adv_ie_ran_l2 / max(adv_ie_l2, 1e-12),
                "adv_init_e_nul_frac": adv_ie_nul_l2 / max(adv_ie_l2, 1e-12),
            })

        rows.append(row)

    return rows

def save_examples(
    out_dir: Path,
    example_rows: List[Dict],
) -> None:
    if not example_rows:
        return

    for idx, row in enumerate(example_rows):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        images = [
            (row["x_gt"], "Ground Truth", "gray"),
            (row["clean_init"], "Clean Init", "gray"),
            (row["adv_init"], "Adv Init", "gray"),
            (row["delta"], "Sinogram Delta", "viridis"),
            (row["clean_pred"], "Clean Pred", "gray"),
            (row["adv_pred"], "Adv Pred", "gray"),
            (row["clean_y"], "Clean Sino", "gray"),
            (row["adv_y"], "Adv Sino", "gray"),
        ]
        for ax, (img, title, cmap) in zip(axes.reshape(-1), images):
            im = ax.imshow(img, cmap=cmap, aspect="auto" if img.ndim == 2 and img.shape[0] != img.shape[1] else None)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(out_dir / f"example_{idx:03d}.png", dpi=160)
        plt.close(fig)

        if "e_ran_init_clean" in row:
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["clean_init"],
                e_ran=row["e_ran_init_clean"],
                e_nul=row["e_nul_init_clean"],
                out_path=out_dir / f"decomp_init_clean_{idx:03d}.png",
                title=f"Clean — init error decomposition, before Network (example {idx})",
            )
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["adv_init"],
                e_ran=row["e_ran_init_adv"],
                e_nul=row["e_nul_init_adv"],
                out_path=out_dir / f"decomp_init_adv_{idx:03d}.png",
                title=f"Adversarial — init error decomposition, before NSN (example {idx})",
            )

        if "e_ran_clean" in row:
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["clean_pred"],
                e_ran=row["e_ran_clean"],
                e_nul=row["e_nul_clean"],
                out_path=out_dir / f"decomp_clean_{idx:03d}.png",
                title=f"Clean — error decomposition (example {idx})",
            )
            visualise_decomposition(
                gt=row["x_gt"],
                recon=row["adv_pred"],
                e_ran=row["e_ran_adv"],
                e_nul=row["e_nul_adv"],
                out_path=out_dir / f"decomp_adv_{idx:03d}.png",
                title=f"Adversarial — error decomposition (example {idx})",
            )
            e_ran_c = row["e_ran_clean"]
            e_ran_a = row["e_ran_adv"]
            e_ran_diff = e_ran_a - e_ran_c
            e_abs = max(np.abs(e_ran_c).max(), np.abs(e_ran_a).max(), 1e-12)
            diff_abs = max(np.abs(e_ran_diff).max(), 1e-12)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            panels = [
                (e_ran_c, f"e_ran clean  ‖·‖={np.linalg.norm(e_ran_c.ravel()):.3f}", e_abs),
                (e_ran_a, f"e_ran adv    ‖·‖={np.linalg.norm(e_ran_a.ravel()):.3f}", e_abs),
                (e_ran_diff, f"Δe_ran (adv − clean)  ‖·‖={np.linalg.norm(e_ran_diff.ravel()):.3f}", diff_abs),
            ]
            if "proj_ran_fbp_delta" in row:
                p = row["proj_ran_fbp_delta"]
                panels.append((p, f"proj_ran(FBP(δ))  ‖·‖={np.linalg.norm(p.ravel()):.3f}", diff_abs))
            ncols = len(panels)
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            if ncols == 1:
                axes = [axes]
            for ax, (img, title, vabs) in zip(axes, panels):
                im = ax.imshow(img, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(
                f"Range-space error shift (example {idx}) — "
                "should equal proj_range(FBP(δ)) for NSN",
                fontsize=9,
            )
            plt.tight_layout()
            plt.savefig(out_dir / f"range_diff_{idx:03d}.png", dpi=150)
            plt.close(fig)

def save_scatter_plot(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    x_key: str = "delta_rel_l2",
    y_key: str = "adv_rel_l2",
) -> None:
    """Per-sample sensitivity cloud: adversarial error vs the *relative* perturbation
    size (‖delta‖/‖y‖). Plotting against clean error collapses every point onto the
    y-axis because clean error ~ 0; plotting against perturbation size spreads the
    points across the sweep and reveals how steeply each model degrades."""
    if not rows_by_model:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, (model_name, rows) in enumerate(rows_by_model.items()):
        xs = [r[x_key] for r in rows if x_key in r and y_key in r]
        ys = [r[y_key] for r in rows if x_key in r and y_key in r]
        if xs:
            ax.scatter(xs, ys, label=model_name, s=20, alpha=0.6, color=colors[i % len(colors)])
    if x_key == "delta_rel_l2":
        ax.set_xlabel("Relative perturbation  ‖delta‖ / ‖y‖")
    else:
        ax.set_xlabel(f"Clean error ({x_key})")
    ax.set_ylabel(f"Adv error ({y_key})")
    ax.set_title("Per-sample sensitivity: adversarial error vs perturbation size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_perturbation_vs_adv.png", dpi=150)
    plt.close(fig)

def save_robustness_curve(
    out_dir: Path,
    curve_by_model: Dict[str, List[Dict]],
    y_key: str = "adv_rel_l2",
) -> None:
    """Plot the median of y_key (with an inter-quartile band) against the nominal attack
    budget eps, one line per model. Median + IQR is robust to the small-||gt|| / small-||y||
    outlier samples that skew the mean; it shows the perturbation size at which each
    model's reconstruction actually breaks."""
    if not curve_by_model:
        return
    med_key, q25_key, q75_key = f"{y_key}_median", f"{y_key}_q25", f"{y_key}_q75"
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10.colors
    xs: List[float] = []
    plotted = False
    for i, (model_name, points) in enumerate(curve_by_model.items()):
        pts = sorted((p for p in points if med_key in p), key=lambda p: p["eps"])
        if not pts:
            continue
        xs = [p["eps"] for p in pts]
        ys = [p[med_key] for p in pts]
        lo = [max(0.0, p[med_key] - p.get(q25_key, p[med_key])) for p in pts]
        hi = [max(0.0, p.get(q75_key, p[med_key]) - p[med_key]) for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, lw=1.5,
                    label=model_name, color=colors[i % len(colors)])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    if len(xs) > 1 and min(xs) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("Attack budget eps  (fraction of signal norm ‖y‖)")
    ax.set_ylabel(f"{y_key}  (median, IQR band)")
    ax.set_title("Robustness curve: reconstruction error vs attack budget")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"robustness_{y_key}.png", dpi=150)
    plt.close(fig)

def save_error_components_curve(
    out_dir: Path,
    curve_by_model: Dict[str, List[Dict]],
    stat: str = "median",
    fname: str = "error_components_clean_vs_adv.png",
) -> None:
    """Per-model curves of clean vs adversarial error magnitude against the attack
    budget eps, split into the range and null-space components. One panel per model
    (e.g. resnet, nsn); dashed = clean baseline, solid = adversarial; blue = range
    component ‖e_ran‖, red = null component ‖e_nul‖.

    Makes the NSN-vs-ResNet contrast explicit: the NSN's adversarial error grows in the
    range component it passes through by design (data consistency), while the ResNet's
    grows in the (hallucinated) null component. Uses the median by default so the small
    ‖gt‖ / ‖y‖ outliers do not skew the curve, matching save_robustness_curve."""
    present = [m for m in curve_by_model if curve_by_model[m]]
    if not present:
        return
    c_ran, c_nul = "#1f77b4", "#d62728"
    series = [
        ("clean_e_ran_l2", "clean range", c_ran, "--", None),
        ("clean_e_nul_l2", "clean null",  c_nul, "--", None),
        ("adv_e_ran_l2",   "adv range",   c_ran, "-",  "o"),
        ("adv_e_nul_l2",   "adv null",    c_nul, "-",  "o"),
    ]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    drew = False
    for ax, model_name in zip(axes[0], present):
        pts = sorted(curve_by_model[model_name], key=lambda p: p["eps"])
        xs = [p["eps"] for p in pts]
        for base, label, color, ls, marker in series:
            ys = [p.get(f"{base}_{stat}", float("nan")) for p in pts]
            if all(math.isnan(v) for v in ys):
                continue
            ax.plot(xs, ys, ls, marker=marker, color=color, lw=1.5, label=label)
            drew = True
        if len(xs) > 1 and min(xs) > 0:
            ax.set_xscale("log")
        ax.set_xlabel("Attack budget eps  (fraction of signal norm ‖y‖)")
        ax.set_ylabel(f"‖error component‖  ({stat})")
        ax.set_title(model_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    if not drew:
        plt.close(fig)
        return
    fig.suptitle("Error components vs attack budget: clean (dashed) vs adversarial (solid)")
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close(fig)

def save_decomposition_bar(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    clean_key: str = "clean_e_nul_frac",
    adv_key: str = "adv_e_nul_frac",
    fname: str = "decomp_nul_frac.png",
    title: str = "Null-space fraction of error: clean vs adversarial (median)",
) -> None:
    """Grouped bar chart of the null-space fraction of the error (‖e_nul‖/‖e‖), clean
    vs adversarial, per model. Shows that the attack pushes error out of the null space
    and into the (data-consistent) range space that the NSN passes through unchanged.

    Call with the ``*_init_e_nul_frac`` keys to draw the same chart for the
    initialisation reconstruction (the network input, before the NSN)."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return

    def median_of(rows: List[Dict], key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(np.median(vals)) if vals else float("nan")

    clean_nul = [median_of(rows_by_model[m], clean_key) for m in models]
    adv_nul = [median_of(rows_by_model[m], adv_key) for m in models]
    if all(math.isnan(v) for v in clean_nul + adv_nul):
        return

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, clean_nul, w, label="clean", color="#1D9E75")
    ax.bar(x + w / 2, adv_nul, w, label="adversarial", color="#D4537E")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("‖e_nul‖ / ‖e‖  (median)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close(fig)

def summarize_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {"num_examples": len(rows)}
    if not rows:
        return metrics

    keys = [
        "gt_norm",
        "clean_mse",
        "adv_mse",
        "mse_ratio",
        "clean_rel_l2",
        "adv_rel_l2",
        "rel_l2_ratio",
        "clean_psnr",
        "adv_psnr",
        "clean_ssim",
        "adv_ssim",
        "clean_mae",
        "adv_mae",
        "clean_nrmse",
        "adv_nrmse",
        "clean_max_err",
        "adv_max_err",
        "clean_init_rel_l2",
        "adv_init_rel_l2",
        "init_rel_l2_ratio",
        "clean_init_psnr",
        "adv_init_psnr",
        "clean_init_ssim",
        "adv_init_ssim",
        "clean_init_mae",
        "adv_init_mae",
        "pred_shift_rel_l2",
        "init_shift_rel_l2",
        "delta_l2",
        "delta_linf",
        "delta_mean_abs",
        "delta_rel_l2",
        "success_rel_l2",
        "success_mse",
    ]

    decomp_keys = [
        "clean_e_ran_l2", "clean_e_nul_l2", "clean_e_ran_frac", "clean_e_nul_frac",
        "adv_e_ran_l2", "adv_e_nul_l2", "adv_e_ran_frac", "adv_e_nul_frac",
        "clean_init_e_ran_l2", "clean_init_e_nul_l2",
        "clean_init_e_ran_frac", "clean_init_e_nul_frac",
        "adv_init_e_ran_l2", "adv_init_e_nul_l2",
        "adv_init_e_ran_frac", "adv_init_e_nul_frac",
        "clean_consistency_rel", "adv_consistency_rel",
    ]
    # Per-metric range/null decomposition emitted by evaluate_batch: clean/adv ×
    # range/null × {rel_l2,psnr,ssim,mae,nrmse,max_err}. Aggregated like everything
    # else; absent (and silently skipped) when the attack runs without a radon op.
    decomp_keys += [
        f"{cond}_{metric}_{sub}"
        for cond in ("clean", "adv")
        for metric in ("rel_l2", "psnr", "ssim", "mae", "nrmse", "max_err")
        for sub in ("ran", "nul")
    ]
    keys = keys + [k for k in decomp_keys if k in rows[0]]
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean, half_width = confidence_interval_95(values)
        metrics[f"{key}_mean"] = mean
        metrics[f"{key}_ci95"] = half_width
        metrics[f"{key}_median"] = float(np.median(values))
        metrics[f"{key}_q25"] = float(np.percentile(values, 25))
        metrics[f"{key}_q75"] = float(np.percentile(values, 75))

    return metrics

def run_attack(
    attack_name: str,
    adapter: ModelAttackAdapter,
    x_gt: torch.Tensor,
    y_clean: torch.Tensor,
    clean_pred: torch.Tensor,
    args,
    eps,
) -> AttackResult:
    objective = args.objective
    shift_weight = args.shift_weight
    if attack_name == "fgsm":
        return fgsm_attack(
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            eps=eps,
            norm=args.norm,
            objective=objective,
            shift_weight=shift_weight,
            stealth_weight=args.stealth_weight,
        )

    if attack_name == "pgd":
        return pgd_attack(
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            eps=eps,
            alpha=args.alpha,
            steps=args.steps,
            restarts=args.restarts,
            norm=args.norm,
            objective=objective,
            shift_weight=shift_weight,
            random_start=not args.no_random_start,
            stealth_weight=args.stealth_weight,
        )

    if attack_name == "spsa":
        return spsa_attack(
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            eps=eps,
            alpha=args.alpha,
            steps=args.steps,
            samples=args.spsa_samples,
            sigma=args.spsa_sigma,
            norm=args.norm,
            objective=objective,
            shift_weight=shift_weight,
            stealth_weight=args.stealth_weight,
        )

    if attack_name == "adam":
        return adam_attack(
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            eps=eps,
            steps=args.steps,
            lr=args.adam_lr,
            scheduler_patience=args.adam_patience,
            norm=args.norm,
            objective=objective,
            shift_weight=shift_weight,
            tv_weight=args.adam_tv_weight,
            consistency_weight=args.adam_consistency_weight,
            stealth_weight=args.stealth_weight,
        )

    raise ValueError(f"Unknown attack '{attack_name}'")

def build_clean_cache(adapter: ModelAttackAdapter, loader, max_samples: int, device) -> List[Tuple]:
    """Cache (x_gt, x_init, y_clean, clean_pred) for the first max_samples examples.
    Mirrors the clean-caching block in main() so the analyses below operate on the
    exact same inputs as the primary benchmark."""
    cache: List[Tuple] = []
    n = 0
    with torch.no_grad():
        for x_gt, x_init, y_delta in loader:
            x_gt = to_4d(x_gt).to(device)
            x_init = to_4d(x_init).to(device)
            y_delta = to_4d(y_delta).to(device)
            y_clean = adapter.projector(y_delta)
            clean_pred = adapter.model(x_init, y_clean)
            cache.append((x_gt, x_init, y_clean, clean_pred))
            n += x_gt.shape[0]
            if n >= max_samples:
                break
    return cache

def attack_over_cache(
    adapter: ModelAttackAdapter,
    clean_cache: List[Tuple],
    radon,
    args,
    attack_name: str,
    eps_nominal: float,
    objective: str,
    shift_weight: float,
    max_samples: int,
) -> List[Dict[str, float]]:
    """Run one attack (given objective/shift_weight) over the cached clean batches
    and return the pooled per-sample evaluation rows. No image dumping - just the
    metrics needed to aggregate channel energies."""
    rows: List[Dict[str, float]] = []
    processed = 0
    for x_gt, clean_init, y_clean, clean_pred in clean_cache:
        if processed >= max_samples:
            break
        eps_batch = eps_nominal * l2_norm_batch(y_clean)
        result = run_attack(
            attack_name=attack_name,
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            args=args,
            eps=eps_batch,
        )
        with torch.no_grad():
            adv_pred, adv_init, y_adv = adapter.forward(result.y_adv, mode=args.eval_init_mode)
        rows.extend(evaluate_batch(
            x_gt=x_gt,
            clean_init=clean_init,
            clean_y=y_clean,
            clean_pred=clean_pred,
            adv_init=adv_init,
            adv_y=y_adv,
            adv_pred=adv_pred,
            delta=result.delta,
            success_rel_l2_factor=args.success_rel_l2_factor,
            radon=radon,
        ))
        processed += x_gt.shape[0]
    return rows

def save_null_growth_headline(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    eps: float,
    attack_name: str,
) -> None:
    """headline: for each model, the null-space error magnitude ||e_nul|| clean vs
    adversarial, with the adversarial range floor ||e_ran|| shown alongside. The
    growth of the null channel is the fair robustness signal; the range floor is the
    shared inversion error every data-consistent method carries."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    clean_nul = [_median_of(rows_by_model[m], "clean_e_nul_l2") for m in models]
    adv_nul = [_median_of(rows_by_model[m], "adv_e_nul_l2") for m in models]
    adv_ran = [_median_of(rows_by_model[m], "adv_e_ran_l2") for m in models]
    if all(math.isnan(v) for v in clean_nul + adv_nul):
        return
    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 3, 5))
    ax.bar(x - w, clean_nul, w, label="||e_nul|| clean", color="#9ecae1")
    ax.bar(x, adv_nul, w, label="||e_nul|| adversarial", color="#d62728")
    ax.bar(x + w, adv_ran, w, label="||e_ran|| adv (shared floor)", color="#7f7f7f", alpha=0.7)
    for xi, c, a in zip(x, clean_nul, adv_nul):
        if not (math.isnan(c) or math.isnan(a)):
            ax.annotate("d_null=%+.3g" % (a - c), (xi - w / 2, max(c, a)),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||error component||  (median)")
    ax.set_title("null-space error growth under attack "
                 "(%s, eps=%g)\nfair signal = growth of ||e_nul||; ||e_ran|| is the shared floor"
                 % (attack_name, eps), fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "headline_null_growth.png", dpi=150)
    plt.close(fig)

def save_consistency_plot(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    eps: float,
) -> None:
    """measurement (data) consistency ||proj_ran(A x_hat) - y||/||y||, clean vs adv,
    per model. Low/flat under attack => the adversarial error hides in the null /
    small-singular-value subspace (the literature finding), which is exactly why a
    null-targeted attack/metric is the fair lens."""
    models = [m for m in rows_by_model if rows_by_model[m]]
    if not models:
        return
    clean_c = [_median_of(rows_by_model[m], "clean_consistency_rel") for m in models]
    adv_c = [_median_of(rows_by_model[m], "adv_consistency_rel") for m in models]
    if all(math.isnan(v) for v in clean_c + adv_c):
        return
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.bar(x - w / 2, clean_c, w, label="clean", color="#1D9E75")
    ax.bar(x + w / 2, adv_c, w, label="adversarial", color="#D4537E")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||proj_ran(A x_hat) - y|| / ||y||  (median)")
    ax.set_title("Measurement consistency: clean vs adversarial (eps=%g)\n"
                 "stays low => adversarial error lives in the null subspace" % eps, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "measurement_consistency.png", dpi=150)
    plt.close(fig)

# (matrix_res metric label, function: channel sub -> per-sample row key in evaluate_batch)
# L2 is the absolute Euclidean norm of the error component; the rest are image
# metrics scored on (x_gt + e_component) vs x_gt, so they read the same channel
# through a different lens. Used by run_extra_analyses (collection) and
# save_attack_channel_matrix (plotting) so the two stay in sync.
_CHANNEL_METRIC_SOURCES = [
    ("l2",      lambda sub: "adv_e_%s_l2" % sub),
    ("rel_l2",  lambda sub: "adv_rel_l2_%s" % sub),
    ("psnr",    lambda sub: "adv_psnr_%s" % sub),
    ("ssim",    lambda sub: "adv_ssim_%s" % sub),
    ("mae",     lambda sub: "adv_mae_%s" % sub),
    ("nrmse",   lambda sub: "adv_nrmse_%s" % sub),
    ("max_err", lambda sub: "adv_max_err_%s" % sub),
]

def save_attack_channel_matrix(
    out_dir: Path,
    matrix_res: Dict[str, Dict[str, Dict[str, float]]],
    eps: float,
) -> None:
    """merged range/null channel matrix, one heatmap per image metric.

    Rows are (model, channel) pairs -- e.g. nsn null, nsn range, resnet null,
    resnet range -- and columns are the attack objectives. One figure is written
    per metric: L2 (absolute Euclidean norm of the error component), rel-L2, PSNR,
    SSIM, MAE, NRMSE and max-abs-err. The metric and whether higher means more or
    less damage are written in the title (PSNR/SSIM: lower = more damage; the rest:
    higher = more damage). Cells are the median over the evaluated samples at the
    analysis budget eps; the true value is printed in every cell because the four
    rows can span very different scales, so colour alone is not comparable across
    rows. A white separator divides the models.

    matrix_res[model][objective] must contain '<metric>_nul' / '<metric>_ran'
    keys for each metric label (filled in run_extra_analyses via
    _CHANNEL_METRIC_SOURCES).
    """
    models = list(matrix_res)
    if not models:
        return
    objs = list(matrix_res[models[0]])
    row_specs = [(m, sub) for m in models for sub in ("nul", "ran")]
    sub_name = {"nul": "null", "ran": "range"}
    metrics = [
        ("l2",      "L2 (abs)",     "higher = more damage"),
        ("rel_l2",  "rel-L2",       "higher = more damage"),
        ("psnr",    "PSNR (dB)",    "lower = more damage"),
        ("ssim",    "SSIM",         "lower = more damage"),
        ("mae",     "MAE",          "higher = more damage"),
        ("nrmse",   "NRMSE",        "higher = more damage"),
        ("max_err", "max-abs-err",  "higher = more damage"),
    ]
    for label, nice, direction in metrics:
        try:
            M = np.array([[matrix_res[m][o]["%s_%s" % (label, sub)] for o in objs]
                          for (m, sub) in row_specs], dtype=float)
        except KeyError:
            continue
        if np.all(np.isnan(M)):
            continue
        fig, ax = plt.subplots(figsize=(1.9 * len(objs) + 3.0,
                                        0.8 * len(row_specs) + 2.5))
        im = ax.imshow(M, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(objs)))
        ax.set_xticklabels(objs, rotation=30, ha="right")
        ax.set_yticks(range(len(row_specs)))
        ax.set_yticklabels(["%s %s" % (m, sub_name[sub]) for (m, sub) in row_specs])
        for k in range(2, len(row_specs), 2):  # separate the models
            ax.axhline(k - 0.5, color="white", lw=1.5)
        for r in range(len(row_specs)):
            for c in range(len(objs)):
                ax.text(c, r, "%.3g" % M[r, c], ha="center", va="center",
                        color="white", fontsize=8)
        ax.set_title("Adversarial channel error -- metric: %s (%s)\n"
                     "rows = model x {null, range}, cols = attack objective "
                     "(median, eps=%g)" % (nice, direction, eps), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / ("channel_matrix_%s.png" % label), dpi=150)
        plt.close(fig)

def save_shift_weight_sweep(
    out_dir: Path,
    sweep_res: Dict[str, List[Dict[str, float]]],
    objective: str,
    eps: float,
) -> None:
    """null (solid) and range (dashed) error-channel magnitude vs the hybrid
    shift-weight, one colour per model. Locates the weight that maximises structural
    (null) damage while suppressing the trivial range channel."""
    models = [m for m in sweep_res if sweep_res[m]]
    if not models:
        return
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(7.5, 5))
    drew = False
    for i, m in enumerate(models):
        pts = sorted(sweep_res[m], key=lambda p: p["w"])
        xs = [p["w"] for p in pts]
        nul = [p["nul"] for p in pts]
        ran = [p["ran"] for p in pts]
        col = colors[i % len(colors)]
        ax.plot(xs, nul, "-o", color=col, lw=1.6, label="%s ||e_nul||" % m)
        ax.plot(xs, ran, "--s", color=col, lw=1.2, alpha=0.7, label="%s ||e_ran||" % m)
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("shift-weight  (range-error penalty in the hybrid objective)")
    ax.set_ylabel("||error component||  (median)")
    ax.set_title("shift-weight sweep (%s, eps=%g)\n"
                 "solid = null (structural), dashed = range (floor)" % (objective, eps), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=max(1, len(models)))
    plt.tight_layout()
    plt.savefig(out_dir / "shift_weight_sweep.png", dpi=150)
    plt.close(fig)

def estimate_lipschitz_nullspace(
    model: nn.Module,
    adapter: ModelAttackAdapter,
    clean_cache: List[Tuple],
    radon,
    n_samples: int = 4,
    n_iter: int = 8,
) -> Dict[str, float]:
    """operator-norm (local Lipschitz) estimate of the *learned correction*
    restricted to the null space of A_la.

    We linearise the correction  g(x) = f(x) - x  (= P_null(UNet(x)) for the NSN,
    UNet(x) for the ResNet) around the clean init x0, restrict both input and output
    to null(A_la) with the same projector P = radon.proj_null_image, and estimate the
    largest singular value of  M = P . J_g . P  by power iteration:

        d <- P d / ||.|| ;   repeat:  u = M d ,  d = M^T u / ||.|| ;   sigma ~ ||M d||.

    This is an attack-independent, architecture-comparable measure of how strongly a
    null-space input perturbation can be amplified into null-space output error - the
    quantity that governs worst-case robustness of the learned channel. Falls back to
    a finite-difference random-direction lower bound if double-backward is unavailable.
    """
    device = next(model.parameters()).device
    proj = radon.proj_null_image

    samples: List[float] = []
    count = 0
    for x_gt, x_init, y_clean, clean_pred in clean_cache:
        B = x_init.shape[0]
        for b in range(B):
            if count >= n_samples:
                break
            x0 = x_init[b: b + 1].detach()
            y0 = y_clean[b: b + 1].detach()

            def G(x: torch.Tensor) -> torch.Tensor:
                # learned correction, output restricted to the null space
                return proj(model(x, y0) - x)

            sigma = float("nan")
            try:
                d = proj(torch.randn_like(x0))
                d = d / (torch.linalg.norm(d.reshape(-1)) + 1e-12)
                for _ in range(n_iter):
                    _, u = torch.autograd.functional.jvp(G, x0, d, create_graph=False, strict=False)
                    u = proj(u)
                    _, w = torch.autograd.functional.vjp(G, x0, u, create_graph=False, strict=False)
                    w = proj(w)
                    nw = torch.linalg.norm(w.reshape(-1))
                    if nw < 1e-12:
                        break
                    d = w / nw
                _, u = torch.autograd.functional.jvp(G, x0, d, create_graph=False, strict=False)
                sigma = float(torch.linalg.norm(proj(u).reshape(-1)).item())
            except Exception:
                # finite-difference lower bound over a few random null directions
                best = 0.0
                with torch.no_grad():
                    g0 = G(x0)
                    scale = float(torch.linalg.norm(x0.reshape(-1)).item()) / math.sqrt(x0.numel())
                    h = 1e-3 * max(scale, 1e-6)
                    for _ in range(max(4, n_iter)):
                        d = proj(torch.randn_like(x0))
                        d = d / (torch.linalg.norm(d.reshape(-1)) + 1e-12)
                        diff = proj((G(x0 + h * d) - g0) / h)
                        best = max(best, float(torch.linalg.norm(diff.reshape(-1)).item()))
                sigma = best

            if not math.isnan(sigma):
                samples.append(sigma)
            count += 1
        if count >= n_samples:
            break

    if not samples:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(samples)),
        "max": float(np.max(samples)),
        "std": float(np.std(samples)),
        "n": len(samples),
    }

def save_lipschitz_plot(out_dir: Path, lip_res: Dict[str, Dict[str, float]]) -> None:
    """P6 bar chart: null-restricted local Lipschitz constant per model (mean +/- std,
    max marked). Higher => the learned channel amplifies null-space perturbations more,
    i.e. is intrinsically less robust there - independent of any particular attack."""
    models = [m for m in lip_res if lip_res[m].get("n", 0) > 0]
    if not models:
        return
    means = [lip_res[m]["mean"] for m in models]
    stds = [lip_res[m]["std"] for m in models]
    maxes = [lip_res[m]["max"] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.bar(x, means, 0.5, yerr=stds, capsize=4, color="#4C72B0", label="mean +/- std")
    ax.scatter(x, maxes, color="#C44E52", zorder=3, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("||P.J_g.P||  (null-restricted local Lipschitz)")
    ax.set_title("P6 - null-restricted Lipschitz of the learned correction\n"
                 "operator norm of P_null . J(f-x) . P_null (power iteration)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "lipschitz_nullspace.png", dpi=150)
    plt.close(fig)

def run_extra_analyses(
    args,
    radon,
    beta: float,
    device,
    loader,
    init_reconstructor: "InitReconstructor",
    projector: Callable[[torch.Tensor], torch.Tensor],
    out_root: Path,
    example: str,
    init_method: str,
) -> None:
    """Drive the opt-in analyses. Loops each model once, builds one clean
    cache, and runs whichever analyses are enabled."""
    matrix_objs = parse_list_arg(args.objective_matrix) if args.objective_matrix else []
    sweep_weights = parse_float_list(args.shift_weight_sweep) if args.shift_weight_sweep else []
    do_lip = bool(args.lipschitz)
    if not (matrix_objs or sweep_weights or do_lip):
        return

    eps_list = parse_float_list(args.eps)
    eps_nominal = args.analysis_eps if args.analysis_eps is not None else max(eps_list)
    attack_names = parse_list_arg(args.attacks)
    attack_name = args.analysis_attack or attack_names[0]
    max_samples = args.analysis_max_samples if args.analysis_max_samples is not None else args.max_samples
    model_names = parse_list_arg(args.models)

    matrix_res: Dict[str, Dict[str, Dict[str, float]]] = {}
    sweep_res: Dict[str, List[Dict[str, float]]] = {}
    lip_res: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        print("[analysis] loading " + model_name)
        model = load_model_checkpoint(
            example=example, init_method=init_method, model_name=model_name,
            radon=radon, beta=beta, device=device, model_dir=args.model_dir,
        )
        adapter = ModelAttackAdapter(
            model=model, init_reconstructor=init_reconstructor, projector=projector,
            attack_init_mode=args.attack_init_mode,
        )
        cache = build_clean_cache(adapter, loader, max_samples, device)

        if matrix_objs:
            d: Dict[str, Dict[str, float]] = {}
            for obj in matrix_objs:
                rows = attack_over_cache(adapter, cache, radon, args, attack_name,
                                         eps_nominal, obj, args.shift_weight, max_samples)
                md: Dict[str, float] = {}
                for label, keyfn in _CHANNEL_METRIC_SOURCES:
                    for sub in ("nul", "ran"):
                        md["%s_%s" % (label, sub)] = _median_of(rows, keyfn(sub))
                d[obj] = md
                print("[analysis][matrix] %s obj=%s l2_nul=%.4g l2_ran=%.4g"
                      % (model_name, obj, md["l2_nul"], md["l2_ran"]))
            matrix_res[model_name] = d

        if sweep_weights:
            lst: List[Dict[str, float]] = []
            for wv in sweep_weights:
                rows = attack_over_cache(adapter, cache, radon, args, attack_name,
                                         eps_nominal, args.objective, wv, max_samples)
                lst.append({"w": wv, "ran": _median_of(rows, "adv_e_ran_l2"),
                            "nul": _median_of(rows, "adv_e_nul_l2")})
                print("[analysis][sweep] %s w=%g nul=%.4g ran=%.4g"
                      % (model_name, wv, lst[-1]["nul"], lst[-1]["ran"]))
            sweep_res[model_name] = lst

        if do_lip:
            lip_res[model_name] = estimate_lipschitz_nullspace(
                model, adapter, cache, radon,
                n_samples=args.lipschitz_samples, n_iter=args.lipschitz_iters,
            )
            print("[analysis][lipschitz] %s mean=%.4g max=%.4g"
                  % (model_name, lip_res[model_name]["mean"], lip_res[model_name]["max"]))

    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if matrix_res:
        save_attack_channel_matrix(analysis_dir, matrix_res, eps_nominal)
    if sweep_res:
        save_shift_weight_sweep(analysis_dir, sweep_res, args.objective, eps_nominal)
    if lip_res:
        save_lipschitz_plot(analysis_dir, lip_res)
    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({"eps": eps_nominal, "attack": attack_name,
                   "matrix": matrix_res, "shift_weight_sweep": sweep_res,
                   "lipschitz_nullspace": lip_res}, f, indent=2)
def parse():
    parser = argparse.ArgumentParser(description="Adversarial attacks for Radon reconstruction models.")
    parser.add_argument("--type", required=True, choices=["ellipses", "lodopab"])
    parser.add_argument("--init", default="pinv", choices=["fbp", "pinv", "tv", "lw"], help="Initialization method")
    parser.add_argument("--models", default="nsn", choices=["resnet","nsn","dpnsn","dpnsn_res"])
    parser.add_argument("--attacks", default="pgd", choices=["pgd", "spsa", "fgsm", "adam"])
    parser.add_argument("--norm", default="l2", choices=["l2", "linf"])
    parser.add_argument("--eps", type=str, default="0.01",  help="Attack budget multiplier(s). Comma-separated for a sweep, e.g.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--objective", default="mse", choices=["mse", "shift", "hybrid", "null", "null_shift", "null_hybrid"], help="Attack target. mse/shift/hybrid reward total error (on a data-consistent "
             "NSN this is solved trivially by range-space corruption). null/null_shift "
             "reward only the null-space (structural/learned) error component; "
             "null_hybrid additionally penalises range-space error so the budget is "
             "spent on structural rather than trivial, data-consistent corruption.",
    )
    parser.add_argument("--shift-weight", type=float, default=0.25)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-rel-l2-factor", type=float, default=1.5)
    parser.add_argument("--spsa-samples", type=int, default=16)
    parser.add_argument("--spsa-sigma", type=float, default=1e-2)
    parser.add_argument(
        "--objective-matrix", default="",
        help="comma-separated objectives to sweep for the model x objective channel "
             "matrix, e.g. 'mse,null,null_hybrid'. Empty disables the matrix.",
    )
    parser.add_argument("--shift-weight-sweep", default="",
        help="comma-separated shift-weight values to sweep for the hybrid/null_hybrid "
             "objective, e.g. '0,0.25,1,4'. Empty disables the sweep.",
    )
    parser.add_argument("--lipschitz", action="store_true", help="estimate the null-restricted local Lipschitz constant of each model.")
    parser.add_argument("--lipschitz-samples", type=int, default=4, help="number of examples to average the Lipschitz estimate over.")
    parser.add_argument("--lipschitz-iters", type=int, default=8, help="power-iteration steps for the Lipschitz estimate.")
    parser.add_argument("--analysis-max-samples", type=int, default=None, help="Sample budget for the analyses (default: --max-samples).")
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--adam-patience", type=int, default=50)
    parser.add_argument("--adam-tv-weight", type=float, default=0.0)
    parser.add_argument("--adam-consistency-weight", type=float, default=0.0)
    parser.add_argument("--save-examples", type=int, default=6)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-root", default=None, help="Path to {example}_out data directory (default: ./{type}_out)")
    parser.add_argument("--model-dir", default=None, help="Base dir containing runs_{type}/ checkpoints (default: .)")
    return parser.parse_args()
def main() -> None:
    
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    example = args.type
    init_method = args.init.lower()
    tag = args.tag or example

    out_root = Path(args.out_dir) if args.out_dir else Path(f"attack_runs_{tag}") / f"init_{init_method}"
    out_root.mkdir(parents=True, exist_ok=True)

    eps_list = parse_float_list(args.eps)
     
    attack_names = parse_list_arg(args.attacks)
    model_names = parse_list_arg(args.models)
    
    summary = load_summary(example, data_root=args.data_root)
    average_l2_noise = float(summary["mean_norm_y_minus_y_delta"])
    mean_sino_norm = float(summary.get("mean_norm_y") or 0.0)
    eps_scale = mean_sino_norm if mean_sino_norm > 0 else 1.0

    print(f"Exporting config to {out_root}/config.json")
    config = vars(args).copy()
    config["device"] = str(device)
    config["eps_list"] = eps_list
    config["summary_path"] = str(Path(f"{example}_out") / f"summary.json")
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Constructing Radon")
    radon = build_radon(summary, device=device)
    projector = lambda y: radon.proj_ran(y)

    print("Constructing Dataloader")
    if example == "ellipses":
        loader = get_ellipse_dataloader(
            init_recon=init_method,
            batch_size=args.batch_size,
            split=args.split,
            n_train=args.n_train,
            n_test=args.n_test,
            num_workers=args.num_workers,
            data_root=args.data_root,
            shuffle=False,
            device=device
        )
    else:
        raise NotImplementedError("Lodopab not implemented")
        loader = get_lodopab_dataloader(
        init_recon=init_method,
        batch_size=args.batch_size,
        split=args.split,
        n_train=args.n_train,
        n_test=args.n_test,
        data_root=args.root,
        shuffle=False,
        num_workers=args.num_workers,
        device=None,
    )

    init_reconstructor = InitReconstructor(example=example, init_method=init_method, summary=summary, radon=radon)
   
    print("Data collection for attack runs")
    scatter_all: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    decomp_by_eps: Dict[Tuple[str, float], Dict[str, List[Dict]]] = defaultdict(dict)
    curve_rows: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for model_name in model_names:
        print("loading " + model_name)
        model = load_model_checkpoint(
            example=example,
            init_method=init_method,
            model_name=model_name,
            radon=radon,
            beta=average_l2_noise,
            device=device,
            model_dir=args.model_dir
        )
        adapter = ModelAttackAdapter(
            model=model,
            init_reconstructor=init_reconstructor,
            projector=projector,
            attack_init_mode=args.init,
        )

        print("started caching of clean model outputs")
        with torch.no_grad():
            clean_rows_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            sample_count = 0
            for x_gt, x_init, y_delta in loader:
                x_gt = to_4d(x_gt).to(device)
                x_init = to_4d(x_init).to(device)
                y_delta = to_4d(y_delta).to(device)
                if torch.linalg.norm(y_delta - y_clean) > 1e-8:
                    print("Something went wrong with the noise")
                    raise ValueError("Problem.")
                y_clean = adapter.projector(y_delta)
                clean_pred = adapter.model(x_init, y_clean)
                clean_rows_cache.append((x_gt, x_init, y_clean, clean_pred))
                sample_count += x_gt.shape[0]
                if sample_count >= args.max_samples:
                    break
        print("starting attack")
        for attack_name in attack_names:
            for eps_nominal in eps_list:
                # Reference (dataset-mean) budget, recorded for context only. The
                # budget actually applied is per-sample (eps_nominal * ||y_i||), below.
                eps_actual = eps_nominal * eps_scale
                result_dir = out_root / model_name / attack_name / f"{args.norm}_eps_{eps_nominal:g}"
                result_dir.mkdir(parents=True, exist_ok=True)
                rows: List[Dict[str, float]] = []
                example_rows: List[Dict] = []
                total_runtime = 0.0
                processed = 0
                for x_gt, clean_init, y_clean, clean_pred in clean_rows_cache:
                    if processed >= args.max_samples:
                        break
                    # Per-sample L2 budget:
                    eps_batch = eps_nominal * l2_norm_batch(y_clean)
                    
                    attack_result = run_attack(
                        attack_name=attack_name,
                        adapter=adapter,
                        x_gt=x_gt,
                        y_clean=y_clean,
                        clean_pred=clean_pred,
                        args=args,
                        eps=eps_batch,
                    )
                    total_runtime += attack_result.runtime_sec

                    print("Evaluating attack")
                    with torch.no_grad():
                        adv_pred, adv_init, y_adv = adapter.forward(attack_result.y_adv, mode=args.eval_init_mode)
                    batch_rows = evaluate_batch(
                        x_gt=x_gt,
                        clean_init=clean_init,
                        clean_y=y_clean,
                        clean_pred=clean_pred,
                        adv_init=adv_init,
                        adv_y=y_adv,
                        adv_pred=adv_pred,
                        delta=attack_result.delta,
                        success_rel_l2_factor=args.success_rel_l2_factor,
                        radon=radon,
                    )
                    rows.extend(batch_rows)
                    remaining_slots = args.save_examples - len(example_rows)
                    if remaining_slots > 0:
                        for j in range(min(x_gt.shape[0], remaining_slots)):
                            e_ran_clean, e_nul_clean = decompose_error(
                                clean_pred[j: j + 1] - x_gt[j: j + 1], radon
                            )
                            e_ran_adv, e_nul_adv = decompose_error(
                                adv_pred[j: j + 1] - x_gt[j: j + 1], radon
                            )
                            # Decomposition of the init-reconstruction error
                            # (network input, before the NSN).
                            e_ran_init_clean, e_nul_init_clean = decompose_error(
                                clean_init[j: j + 1] - x_gt[j: j + 1], radon
                            )
                            e_ran_init_adv, e_nul_init_adv = decompose_error(
                                adv_init[j: j + 1] - x_gt[j: j + 1], radon
                            )
                            fbp_delta = radon.fbp_la(attack_result.delta[j: j + 1])
                            e_ran_fbp_d, _ = decompose_error(fbp_delta, radon)
                            example_rows.append(
                                {
                                    "x_gt": to_numpy_img(x_gt[j]),
                                    "clean_init": to_numpy_img(clean_init[j]),
                                    "adv_init": to_numpy_img(adv_init[j]),
                                    "clean_pred": to_numpy_img(clean_pred[j]),
                                    "adv_pred": to_numpy_img(adv_pred[j]),
                                    "clean_y": to_numpy_img(y_clean[j]),
                                    "adv_y": to_numpy_img(y_adv[j]),
                                    "delta": to_numpy_img(attack_result.delta[j]),
                                    "e_ran_clean": e_ran_clean.squeeze().numpy(),
                                    "e_nul_clean": e_nul_clean.squeeze().numpy(),
                                    "e_ran_adv": e_ran_adv.squeeze().numpy(),
                                    "e_nul_adv": e_nul_adv.squeeze().numpy(),
                                    "e_ran_init_clean": e_ran_init_clean.squeeze().numpy(),
                                    "e_nul_init_clean": e_nul_init_clean.squeeze().numpy(),
                                    "e_ran_init_adv": e_ran_init_adv.squeeze().numpy(),
                                    "e_nul_init_adv": e_nul_init_adv.squeeze().numpy(),
                                    "proj_ran_fbp_delta": e_ran_fbp_d.squeeze().numpy(),
                                }
                            )
                    processed += x_gt.shape[0]
                    if processed >= args.max_samples:
                        break
                summary_metrics = summarize_metrics(rows)
                summary_metrics["attack_runtime_total_sec"] = total_runtime
                summary_metrics["attack_runtime_per_example_sec"] = total_runtime / max(len(rows), 1)
                summary_metrics["model_name"] = model_name
                summary_metrics["attack_name"] = attack_name
                summary_metrics["norm"] = args.norm
                summary_metrics["eps"] = eps_nominal
                summary_metrics["eps_actual"] = eps_actual  # dataset-mean reference; budget is per-sample
                summary_metrics["eps_budget_mode"] = "per_sample_l2"
                summary_metrics["attack_init_mode"] = args.attack_init_mode
                summary_metrics["eval_init_mode"] = args.eval_init_mode
                with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary_metrics, f, indent=2)
                if rows:
                    fieldnames = list(rows[0].keys())
                    with open(result_dir / "per_sample_metrics.csv", "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                save_examples(result_dir, example_rows)
                scatter_all[attack_name][model_name].extend(rows)
                decomp_by_eps[(attack_name, eps_nominal)][model_name] = rows
                curve_rows[attack_name][model_name].append({"eps": eps_nominal, **summary_metrics})
                print(
                    f"[model={model_name} attack={attack_name} eps={eps_nominal:g}] "
                    f"n={len(rows)} adv_rel_l2={summary_metrics.get('adv_rel_l2_mean', float('nan')):.4f} "
                    f"success_rel_l2={summary_metrics.get('success_rel_l2_mean', float('nan')):.3f} "
                    f"clean_rel_l2={summary_metrics.get('clean_rel_l2_mean', float('nan')):.3f}"
                )
    # After every model/attack/eps for this noise level, write the cross-run plots.
    # Per attack: the robustness curve (error vs eps) and the pooled sensitivity
    # scatter. Per (attack, eps): the range/null error decomposition bar chart.
    for att_name, rows_by_model in scatter_all.items():
        plot_dir = out_root / att_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_scatter_plot(plot_dir, rows_by_model)
        save_robustness_curve(plot_dir, curve_rows[att_name], y_key="adv_rel_l2")
        save_robustness_curve(plot_dir, curve_rows[att_name], y_key="adv_psnr")
        save_robustness_curve(plot_dir, curve_rows[att_name], y_key="adv_ssim")
        save_error_components_curve(plot_dir, curve_rows[att_name])
    for (att_name, eps_nominal), rows_by_model in decomp_by_eps.items():
        decomp_dir = out_root / att_name / f"eps_{eps_nominal:g}"
        decomp_dir.mkdir(parents=True, exist_ok=True)
        save_decomposition_bar(decomp_dir, rows_by_model)
        save_decomposition_bar(
            decomp_dir,
            rows_by_model,
            clean_key="clean_init_e_nul_frac",
            adv_key="adv_init_e_nul_frac",
            fname="decomp_init_nul_frac.png",
            title="Init null-space fraction of error (before NSN): clean vs adversarial (median)",
        )
        save_null_growth_headline(decomp_dir, rows_by_model, eps_nominal, att_name)
        save_consistency_plot(decomp_dir, rows_by_model, eps_nominal)
    run_extra_analyses(
        args=args,
        radon=radon,
        beta=average_l2_noise,
        device=device,
        loader=loader,
        init_reconstructor=init_reconstructor,
        projector=projector,
        out_root=out_root,
        example=example,
        init_method=init_method,
    )


if __name__ == "__main__":
    main()
