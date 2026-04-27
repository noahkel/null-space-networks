import torch
import numpy as np
from pathlib import Path
from src.unet import UNet
from src.wrappers import RESNET, NSN, DPNSN, DPNSN_RES
from typing import List, Union, Dict
from src.radon import _RadonBase
from src.radon_matrix import MatrixRadonAdapter

import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

@torch.no_grad()
def save_example_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    title: str,
):
    model.eval()
    x_gt, x_init, y_delta = next(iter(loader))
    x_gt = to_4d(x_gt).to(device)
    x_init = to_4d(x_init).to(device)
    y_delta = to_4d(y_delta).to(device)

    pred = model(x_init, y_delta)

    gt_np = x_gt[0, 0].detach().cpu().numpy()
    init_np = x_init[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(gt_np, cmap="gray")
    axes[0].set_title("GT")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(init_np, cmap="gray")
    axes[1].set_title("Init")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Model Output")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def rel_l2_np(x: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(x - y)
    den = np.linalg.norm(y)
    return float(num / (den + 1e-12))

def rel_l2(x: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((x - x_gt).reshape(-1))
    den = torch.linalg.norm(x_gt.reshape(-1)).clamp_min(eps)
    return float((num / den).item())

def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0.0:
        return float("inf")
    data_range = float(y.max() - y.min())
    if data_range <= 0.0:
        data_range = 1.0
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    data_range = float(y.max() - y.min())
    if data_range <= 0.0:
        data_range = 1.0
    return float(sk_ssim(y, x, data_range=data_range))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def create_simple_phantom(size: int, device="cpu"):
    """
    Simple phantom:
    - filled disk
    - rectangle
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )

    img = torch.zeros((size, size), device=device)

    # disk
    img[(xx**2 + yy**2) < 0.5**2] = 1.0

    # rectangle
    img[(xx > -0.7) & (xx < -0.3) & (yy > -0.2) & (yy < 0.4)] = 0.7
    
    return img

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is (B, 1, H, W)."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def save_image_with_colorbar(img2d: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    im = plt.imshow(img2d, cmap="gray")
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
def visualise_example(
    gt: np.ndarray,
    sino: np.ndarray,
    fbp: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """
    Save a 1×3 figure: ground truth | sinogram | FBP reconstruction.

    Parameters
    ----------
    gt : np.ndarray, shape (H, W)
        Ground-truth image.
    sino : np.ndarray, shape (n_angles, det_count)
        Measured sinogram (limited-angle, possibly noisy).
    fbp : np.ndarray, shape (H, W)
        FBP reconstruction from the limited-angle sinogram.
    out_path : Path
        Where to save the PNG.
    title : str
        Overall figure title (e.g. geometry / noise description).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im0 = axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sino, cmap="gray", aspect="auto")
    axes[1].set_title("Sinogram (limited angle)")
    axes[1].set_xlabel("Detector")
    axes[1].set_ylabel("Angle index")

    im2 = axes[2].imshow(fbp, cmap="gray")
    axes[2].set_title("FBP Reconstruction")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def visualise_decomposition(
    gt: np.ndarray,
    recon: np.ndarray,
    e_ran: np.ndarray,
    e_nul: np.ndarray,
    out_path: Path,
    title: str = "",
) -> None:
    """
    Save a 1×5 figure: GT | Recon | Error | Range error | Null-space error.

    Parameters
    ----------
    gt, recon : (H, W) arrays
    e_ran, e_nul : (H, W) range and null-space components of (recon - gt)
    out_path : where to save the PNG
    title : overall figure title
    """
    error = recon - gt
    e_abs = np.abs(error).max()
    e_norm = np.linalg.norm(error.ravel())
    ran_frac = np.linalg.norm(e_ran.ravel()) / max(e_norm, 1e-12)
    nul_frac = np.linalg.norm(e_nul.ravel()) / max(e_norm, 1e-12)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    im0 = axes[0].imshow(gt, cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(error, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    axes[2].set_title("Error")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(e_ran, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    axes[3].set_title(f"Range error\n‖e_ran‖/‖e‖={ran_frac:.3f}")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(e_nul, cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    axes[4].set_title(f"Null-space error\n‖e_nul‖/‖e‖={nul_frac:.3f}")
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def decompose_error(
    e: torch.Tensor,
    radon: "_RadonBase",
    iters: int = 50,
    tol: float = 1e-6,
) -> tuple:
    """
    Orthogonal decomposition of image-space error e:

      e_ran = A_la^+ A_la e   — projection onto range(A_la^T)
      e_nul = e - e_ran       — null-space component

    When radon is a MatrixRadonAdapter with LA pseudoinverse factors built
    (svd_threshold > 0 and phi set at construction), uses two sparse matrix
    multiplications instead of CG. Otherwise falls back to CG.
    Returns (e_ran, e_nul) as detached CPU tensors.
    """
    if isinstance(radon, MatrixRadonAdapter) and hasattr(radon, '_U_k_la'):
        e_ran = radon.pseudoinverse_la(radon.forward_la(e))
        e_ran = e_ran.detach().cpu()
        e_nul = (e.cpu() - e_ran)
        return e_ran, e_nul
    def AtA(x: torch.Tensor) -> torch.Tensor:
        return radon.backward_la(radon.forward_la(x))

    b = AtA(e)
    x = torch.zeros_like(e)
    r = b.clone()
    p = r.clone()
    rr = (r * r).sum()

    for _ in range(iters):
        if rr.item() < tol ** 2:
            break
        Ap     = AtA(p)
        alpha  = rr / (p * Ap).sum().clamp_min(1e-12)
        x      = x + alpha * p
        r      = r - alpha * Ap
        rr_new = (r * r).sum()
        p      = r + (rr_new / rr.clamp_min(1e-12)) * p
        rr     = rr_new

    e_ran = x.detach().cpu()
    e_nul = (e - x).detach().cpu()
    return e_ran, e_nul


def build_models(
    which: List[str],
    radon: _RadonBase,
    beta: Union[float, None] = None,
) -> Dict[str, nn.Module]:
    models: Dict[str, nn.Module] = {}
    for name in which:
        name = name.lower()
        if name == "resnet":
            models[name] = RESNET(unet=UNet(in_channels=1, out_channels=1))
        elif name == "nsn":
            models[name] = NSN(unet=UNet(in_channels=1, out_channels=1), radon=radon)
        elif name == "dpnsn":
            models[name] = DPNSN(unet=UNet(in_channels=1, out_channels=1), radon=radon, beta=beta)
        elif name == "dpnsn_res":
            models[name] = DPNSN_RES(unet=UNet(in_channels=1, out_channels=1), radon=radon, beta=beta)
        else:
            raise ValueError(f"Unknown model '{name}'. Use one of: resnet, nsn, dpdnsn")
    return models
