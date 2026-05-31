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
from src.utils import build_models, decompose_error, psnr, rel_l2_np, set_seed, ssim, to_4d, visualise_decomposition



def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def l2_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=1)


def linf_norm_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).abs().max(dim=1).values


def proj_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return torch.zeros_like(delta)
    norms = l2_norm_batch(delta).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(norms), torch.full_like(norms, eps) / norms)
    return delta * scale.view(-1, 1, 1, 1)


def proj_linf_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
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
    if eps <= 0:
        return torch.zeros_like(y)
    if norm == "linf":
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
            sigma_sino = self.noise_sigma_rel * float(y.abs().max())
            return self.radon.backward_la_tikhonov(y, lambda_reg=sigma_sino ** 2)

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
        attack_init_mode: str,
    ):
        self.model = model
        self.init_reconstructor = init_reconstructor
        self.projector = projector
        self.attack_init_mode = attack_init_mode

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
) -> torch.Tensor:
    gt_term = reduce_loss((pred - x_gt) ** 2)
    shift_term = reduce_loss((pred - clean_pred.detach()) ** 2)

    if objective == "mse":
        return gt_term
    if objective == "shift":
        return shift_term
    if objective == "hybrid":
        return gt_term + shift_weight * shift_term
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
    loss = attack_objective(pred, x_gt, clean_pred, objective, shift_weight)
    if stealth_weight > 0.0:
        loss = loss - stealth_weight * reduce_loss((y_adv - y_clean.detach()).abs())
    grad = torch.autograd.grad(loss, y_adv, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        if norm == "linf":
            delta = eps * grad.sign()
        elif norm == "l2":
            grad_norm = l2_norm_batch(grad).clamp_min(1e-12).view(-1, 1, 1, 1)
            delta = eps * grad / grad_norm
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
            loss = attack_objective(pred, x_gt, clean_pred, objective, shift_weight)
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
            score = float(attack_objective(pred, x_gt, clean_pred, objective, shift_weight).item())
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
                loss_plus = attack_objective(pred_plus, x_gt, clean_pred, objective, shift_weight)
                loss_minus = attack_objective(pred_minus, x_gt, clean_pred, objective, shift_weight)

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
) -> AttackResult:
    start = time.perf_counter()

    delta = nn.Parameter(torch.zeros_like(y_clean))
    opt = torch.optim.Adam([delta], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=scheduler_patience)

    # Pre-compute clean surrogate init once for consistency term
    with torch.no_grad():
        clean_init = adapter.init_reconstructor.surrogate(adapter.projector(y_clean))

    for _ in range(steps):
        opt.zero_grad()

        y_adv = adapter.projector(y_clean + delta)
        pred, x_init, _ = adapter.forward(y_adv, project=False)

        # Negate because Adam minimizes, but we want to maximize reconstruction error
        loss = -attack_objective(pred, x_gt, clean_pred, objective, shift_weight)

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
        delta_final = (y_adv - y_clean).detach()

    return AttackResult(y_adv=y_adv.detach(), delta=delta_final, runtime_sec=time.perf_counter() - start)


def get_loader(example: str, init_method: str, batch_size: int, split: str, n_train: int, n_test: int, num_workers: int, data_root: Optional[str] = None):
    root = data_root or f"{example}_out"
    if example == "ellipses":
        return get_ellipse_dataloader(
            init_recon=init_method,
            batch_size=batch_size,
            split=split,
            n_train=n_train,
            n_test=n_test,
            data_root=root,
            shuffle=False,
            num_workers=num_workers,
            device=None,
        )
    #return get_lodopab_dataloader(
    #    init_recon=init_method,
    #    batch_size=batch_size,
    #    split=split,
    #    n_train=n_train,
    #    n_test=n_test,
    #    data_root=root,
    #    shuffle=False,
    #    num_workers=num_workers,
    #    device=None,
    #)
    raise NotImplementedError("Lodopab not implemented")


def load_summary(example: str, data_root: Optional[str] = None) -> Dict:
    root = data_root or f"{example}_out"
    summary_path = Path(root) / "summary.json"
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
    noise: str,
    model_dir: Optional[str] = None,
) -> nn.Module:
    base = Path(model_dir) if model_dir else None
    candidates = [
        *(
            [base / f"init_{init_method}{noise}" / f"checkpoints{noise}" / f"{model_name}_best.pt"]
            if base else []
        ),
        Path(f"runs_{example}") / f"init_{init_method}{noise}" / "checkpoints{noise}" / f"{model_name}_best.pt",
        Path(f"checkpoints{noise}") / f"{model_name}_best.pt",
    ]
    print(f"Model path and checkpoint path: {candidates} and  {model_dir}")
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for model '{model_name}' and init '{init_method}'")

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
    success_mse_factor: float,
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

        clean_mse = float(clean_mse_batch[i].item())
        adv_mse = float(adv_mse_batch[i].item())

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
            "pred_shift_rel_l2": pred_shift,
            "init_shift_rel_l2": init_shift,
            "delta_l2": float(delta_l2_batch[i].item()),
            "delta_linf": float(delta_linf_batch[i].item()),
            "delta_mean_abs": float(sino_shift_batch[i].item()),
            "clean_sino_l2": float(np.linalg.norm(clean_y_np.reshape(-1))),
            "adv_sino_l2": float(np.linalg.norm(adv_y_np.reshape(-1))),
            "success_rel_l2": float(adv_rel_l2 >= success_rel_l2_factor * max(clean_rel_l2, 1e-12)),
            "success_mse": float(adv_mse >= success_mse_factor * max(clean_mse, 1e-12)),
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
                "clean_e_ran_frac": clean_e_ran_l2 / max(clean_e_l2, 1e-12),
                "clean_e_nul_frac": clean_e_nul_l2 / max(clean_e_l2, 1e-12),
                "adv_e_ran_l2": adv_e_ran_l2,
                "adv_e_nul_l2": adv_e_nul_l2,
                "adv_e_ran_frac": adv_e_ran_l2 / max(adv_e_l2, 1e-12),
                "adv_e_nul_frac": adv_e_nul_l2 / max(adv_e_l2, 1e-12),
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
    print("finished save")

def save_scatter_plot(
    out_dir: Path,
    rows_by_model: Dict[str, List[Dict]],
    x_key: str = "clean_rel_l2",
    y_key: str = "adv_rel_l2",
) -> None:
    if not rows_by_model:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, (model_name, rows) in enumerate(rows_by_model.items()):
        xs = [r[x_key] for r in rows if x_key in r and y_key in r]
        ys = [r[y_key] for r in rows if x_key in r and y_key in r]
        if xs:
            ax.scatter(xs, ys, label=model_name, s=20, alpha=0.7, color=colors[i % len(colors)])
    lim_lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5, label="y = x")
    ax.set_xlabel(f"Clean error ({x_key})")
    ax.set_ylabel(f"Adv error ({y_key})")
    ax.set_title("Clean vs Adversarial Error per Sample & Network")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_clean_vs_adv.png", dpi=150)
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
        "pred_shift_rel_l2",
        "init_shift_rel_l2",
        "delta_l2",
        "delta_linf",
        "delta_mean_abs",
        "success_rel_l2",
        "success_mse",
    ]

    decomp_keys = [
        "clean_e_ran_l2", "clean_e_nul_l2", "clean_e_ran_frac", "clean_e_nul_frac",
        "adv_e_ran_l2", "adv_e_nul_l2", "adv_e_ran_frac", "adv_e_nul_frac",
    ]

    keys = keys + [k for k in decomp_keys if k in rows[0]]
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean, half_width = confidence_interval_95(values)
        metrics[f"{key}_mean"] = mean
        metrics[f"{key}_ci95"] = half_width
        metrics[f"{key}_median"] = float(np.median(values))

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
    if attack_name == "fgsm":
        return fgsm_attack(
            adapter=adapter,
            x_gt=x_gt,
            y_clean=y_clean,
            clean_pred=clean_pred,
            eps=eps,
            norm=args.norm,
            objective=args.objective,
            shift_weight=args.shift_weight,
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
            objective=args.objective,
            shift_weight=args.shift_weight,
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
            objective=args.objective,
            shift_weight=args.shift_weight,
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
            objective=args.objective,
            shift_weight=args.shift_weight,
            tv_weight=args.adam_tv_weight,
            consistency_weight=args.adam_consistency_weight,
            stealth_weight=args.stealth_weight,
        )

    raise ValueError(f"Unknown attack '{attack_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial attacks for Radon reconstruction models.")
    parser.add_argument("--type", required=True, choices=["ellipses", "lodopab"])
    parser.add_argument("--init", default="fbp,pinv", help="Initialization method: tv,lw,fbp")
    parser.add_argument("--models", default="resnet,nsn,dpnsn,dpnsn_res")
    parser.add_argument("--attacks", default="pgd,fgsm,spsa")
    parser.add_argument("--norm", default="l2", choices=["l2", "linf"])
    parser.add_argument("--eps", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--objective", default="mse", choices=["mse", "shift", "hybrid"])
    parser.add_argument("--shift-weight", type=float, default=0.25)
    parser.add_argument("--attack-init-mode", default="exact", choices=["surrogate", "exact"])
    parser.add_argument("--eval-init-mode", default="exact", choices=["surrogate", "exact"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-rel-l2-factor", type=float, default=1.5)
    parser.add_argument("--success-mse-factor", type=float, default=2.0)
    parser.add_argument("--spsa-samples", type=int, default=16)
    parser.add_argument("--spsa-sigma", type=float, default=1e-2)
    parser.add_argument("--stealth-weight", type=float, default=0.0)
    parser.add_argument("--adam-lr", type=float, default=0.01)
    parser.add_argument("--adam-patience", type=int, default=50)
    parser.add_argument("--adam-tv-weight", type=float, default=0.0)
    parser.add_argument("--adam-consistency-weight", type=float, default=0.0)
    parser.add_argument("--save-examples", type=int, default=6)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-root", default=None, help="Path to {example}_out data directory (default: ./{type}_out)")
    parser.add_argument("--model-dir", default=None, help="Base dir containing runs_{type}/ checkpoints (default: .)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    example = args.type
    init_method = args.init.lower()
    summary = load_summary(example, data_root=args.data_root)
    beta = float(summary["mean_norm_y_minus_y_delta"])
    radon = build_radon(summary, device=device)
    eps = args.eps * beta
    loader = get_loader(
        example=example,
        init_method=init_method,
        batch_size=args.batch_size,
        split=args.split,
        n_train=args.n_train,
        n_test=args.n_test,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    init_reconstructor = InitReconstructor(example=example, init_method=init_method, summary=summary, radon=radon)
    projector = lambda y: radon.proj_ran(y)

    attack_names = parse_list_arg(args.attacks)
    model_names = parse_list_arg(args.models)
    print(model_names)
    for i in ("0.0", "1.0", "2.0"):
        out_root = Path(args.out_dir) if args.out_dir else Path(f"attack_runs_{example}{i}") / f"init_{init_method}{i}"
        out_root.mkdir(parents=True, exist_ok=True)

        config = vars(args).copy()
        config["device"] = str(device)
        config["summary_path"] = str(Path(f"{example}{i}_out") / f"summary{i}.json")
        with open(out_root / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        scatter_rows: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)

        for model_name in model_names:
            print("loading " + model_name)
            model = load_model_checkpoint(
                example=example,
                init_method=init_method,
                model_name=model_name,
                radon=radon,
                beta=beta,
                device=device,
                model_dir=args.model_dir,
            )
            adapter = ModelAttackAdapter(
                model=model,
                init_reconstructor=init_reconstructor,
                projector=projector,
                attack_init_mode=args.attack_init_mode,
            )
            print("started caching of clean model outputs")
            with torch.no_grad():
                clean_rows_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                sample_count = 0
                for x_gt, _, y_delta in loader:
                    x_gt = to_4d(x_gt).to(device)
                    y_delta = to_4d(y_delta).to(device)
                    clean_pred, clean_init, y_clean = adapter.forward(y_delta, mode="exact")
                    clean_rows_cache.append((x_gt, clean_init, y_clean, clean_pred))
                    sample_count += x_gt.shape[0]
                    if sample_count >= args.max_samples:
                        break
            print(attack_names)
            for attack_name in attack_names:
                result_dir = out_root / model_name / attack_name / f"{args.norm}_eps_{args.eps:g}"
                result_dir.mkdir(parents=True, exist_ok=True)

                rows: List[Dict[str, float]] = []
                example_rows: List[Dict] = []
                total_runtime = 0.0
                processed = 0

                for x_gt, clean_init, y_clean, clean_pred in clean_rows_cache:
                    print(f"Processed {processed} samples")
                    if processed >= args.max_samples:
                        break

                    attack_result = run_attack(
                        attack_name=attack_name,
                        adapter=adapter,
                        x_gt=x_gt,
                        y_clean=y_clean,
                        clean_pred=clean_pred,
                        args=args,
                        eps=eps,
                    )
                    total_runtime += attack_result.runtime_sec

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
                        success_mse_factor=args.success_mse_factor,
                        radon = radon,

                    )
                    rows.extend(batch_rows)

                    remaining_slots = args.save_examples - len(example_rows)
                    if remaining_slots > 0:
                        for i in range(min(x_gt.shape[0], remaining_slots)):
                            e_ran_clean, e_nul_clean = decompose_error(
                                clean_pred[i: i + 1] - x_gt[i: i + 1], radon
                            )
                            e_ran_adv, e_nul_adv = decompose_error(
                                adv_pred[i: i + 1] - x_gt[i: i + 1], radon
                            )
                            fbp_delta = radon.fbp_la(attack_result.delta[i: i + 1])
                            e_ran_fbp_d, _ = decompose_error(fbp_delta, radon)
                            example_rows.append(
                                {
                                    "x_gt": to_numpy_img(x_gt[i]),
                                    "clean_init": to_numpy_img(clean_init[i]),
                                    "adv_init": to_numpy_img(adv_init[i]),
                                    "clean_pred": to_numpy_img(clean_pred[i]),
                                    "adv_pred": to_numpy_img(adv_pred[i]),
                                    "clean_y": to_numpy_img(y_clean[i]),
                                    "adv_y": to_numpy_img(y_adv[i]),
                                    "delta": to_numpy_img(attack_result.delta[i]),
                                    "e_ran_clean": e_ran_clean.squeeze().numpy(),
                                    "e_nul_clean": e_nul_clean.squeeze().numpy(),
                                    "e_ran_adv": e_ran_adv.squeeze().numpy(),
                                    "e_nul_adv": e_nul_adv.squeeze().numpy(),
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
                summary_metrics["eps"] = args.eps
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
                scatter_rows[attack_name][model_name] = rows

                print(
                    f"[model={model_name} attack={attack_name}] "
                    f"n={len(rows)} adv_rel_l2={summary_metrics.get('adv_rel_l2_mean', float('nan')):.4f} "
                    f"success_rel_l2={summary_metrics.get('success_rel_l2_mean', float('nan')):.3f}"
                    f"clean_rel_l2={summary_metrics.get('clean_rel_l2_mean', float('nan')):.3f}"
                )
                for attack_name, rows_by_model in scatter_rows.items():
                    scatter_dir = out_root / attack_name
                    scatter_dir.mkdir(parents=True, exist_ok=True)
                    save_scatter_plot(scatter_dir, rows_by_model)

if __name__ == "__main__":
    main()
