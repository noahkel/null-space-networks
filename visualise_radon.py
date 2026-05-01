"""
visualise_radon.py  –  Compare AstraRadonAdapter vs MatrixRadonAdapter on real data.

Loads one sample from an existing ellipses_out dataset, builds both adapters
from the dataset's summary.json, then displays a 3-row × 5-column figure:

  Row 0  Sinograms : GT image | ASTRA full | ASTRA LA | Matrix full | Matrix LA
  Row 1  FBP recon : stored sino (data) | ASTRA full FBP | ASTRA LA FBP | Matrix full FBP | Matrix LA FBP
  Row 2  Error     : (label) | ASTRA full err | ASTRA LA err | Matrix full err | Matrix LA err

Usage:
    python visualise_radon.py --data-dir C:/GitHub/data/ellipses_out
    python visualise_radon.py --data-dir C:/GitHub/data/ellipses_out --idx 42
    python visualise_radon.py --data-dir C:/GitHub/data/ellipses_out --out out.png --no-show
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _imshow(ax, data: np.ndarray, title: str, cmap: str = "gray",
            vmin=None, vmax=None, aspect: str = "equal") -> None:
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect=aspect, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    if aspect == "equal":
        ax.axis("off")
    else:
        ax.set_xlabel("Detector", fontsize=7)
        ax.set_ylabel("Angle index", fontsize=7)
        ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualise radon transforms: ASTRA vs Matrix, full vs LA")
    p.add_argument("--data-dir",   type=str, required=True,
                   help="Path to ellipses_out directory (must contain summary.json, gt/, sino/)")
    p.add_argument("--idx",        type=int, default=0,      help="Sample index (default 0)")
    p.add_argument("--svd-thresh", type=float, default=1e-6, help="SVD threshold for MatrixRadon")
    p.add_argument("--filter",     type=str, default="ram-lak", help="FBP filter name")
    p.add_argument("--device",     type=str, default=None,   help="cpu / cuda / cuda:0")
    p.add_argument("--out",        type=str, default="radon_comparison.png", help="Output PNG path")
    p.add_argument("--no-show",    action="store_true", help="Save only, do not call plt.show()")
    p.add_argument("--cache-dir",  type=str, default=None,   help="Cache dir for MatrixRadonAdapter")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found")
        sys.exit(1)

    with open(summary_path) as f:
        cfg = json.load(f)

    res       = cfg["img_size"]
    n_angles  = cfg["num_angles"]
    det_count = cfg["det_count"]
    angles    = np.array(cfg["angles"], dtype=np.float64)
    phi       = tuple(cfg["phi"])          # (lo, hi) in radians

    # Count how many angles fall in the LA window
    ang_mask = (angles >= phi[0]) & (angles < phi[1])
    n_la = int(ang_mask.sum())

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype  = torch.float64

    print("=" * 60)
    print("Radon visualisation")
    print(f"  data dir    : {data_dir}")
    print(f"  sample idx  : {args.idx}")
    print(f"  resolution  : {res}×{res}")
    print(f"  angles      : {n_angles} total,  {n_la} limited")
    print(f"  phi (rad)   : [{phi[0]:.4f}, {phi[1]:.4f}]")
    print(f"  phi (deg)   : {np.degrees(phi[0]):.1f}° – {np.degrees(phi[1]):.1f}°")
    print(f"  det_count   : {det_count}")
    print(f"  device      : {device}")
    print("=" * 60)

    # -- Load sample -------------------------------------------------------
    gt_files = sorted((data_dir / "gt").glob("*.npy"))
    if args.idx >= len(gt_files):
        print(f"ERROR: idx={args.idx} out of range (dataset has {len(gt_files)} samples)")
        sys.exit(1)

    fname = gt_files[args.idx].name
    gt_np    = np.load(data_dir / "gt"   / fname).astype(np.float64)  # (H, W)
    sino_np  = np.load(data_dir / "sino" / fname).astype(np.float64)  # (n_la, det_count)
    fbp_np   = np.load(data_dir / "fbp"  / fname).astype(np.float64)  # (H, W)

    print(f"\nLoaded sample: {fname}")
    print(f"  gt shape   : {gt_np.shape}")
    print(f"  sino shape : {sino_np.shape}  (stored LA sinogram)")
    print(f"  fbp shape  : {fbp_np.shape}   (stored FBP init)")

    # Promote to (1, 1, H, W) for adapter calls
    x = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

    # -- Build adapters ----------------------------------------------------
    print("\nBuilding AstraRadonAdapter …")
    try:
        from src.radon import AstraRadonAdapter
        astra_r = AstraRadonAdapter(
            resolution=res, angles=angles, det_count=det_count,
            phi=phi, device=device, dtype=dtype, estimate_norm=False,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    print("Building MatrixRadonAdapter …")
    try:
        from src.radon_matrix import MatrixRadonAdapter
        matrix_r = MatrixRadonAdapter(
            resolution=res, angles=angles, det_count=det_count,
            phi=phi, svd_threshold=args.svd_thresh,
            device=device, dtype=dtype, estimate_norm=False,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    # -- Forward projections and FBP ---------------------------------------
    print("\nComputing sinograms and FBP reconstructions …")
    with torch.no_grad():
        # Full-angle sinograms from both adapters
        sino_astra_full  = astra_r.forward(x)            # (1,1,n_angles,det_count)
        sino_matrix_full = matrix_r.forward(x)

        # LA sinograms: zero out unmeasured angles, keep full shape for fbp_la
        sino_astra_la  = astra_r.proj_ran(sino_astra_full)
        sino_matrix_la = matrix_r.proj_ran(sino_matrix_full)

        # FBP: fbp_la internally applies proj_ran then backward
        fbp_astra_full  = astra_r.fbp(sino_astra_full,    filter_name=args.filter)
        fbp_astra_la    = astra_r.fbp_la(sino_astra_la,  filter_name=args.filter)
        fbp_matrix_full = matrix_r.fbp(sino_matrix_full,  filter_name=args.filter)
        fbp_matrix_la   = matrix_r.fbp_la(sino_matrix_la, filter_name=args.filter)

    # Convert to numpy (2D)
    s_af  = to_np(sino_astra_full[0, 0])
    s_ala = to_np(sino_astra_la[0, 0])
    s_mf  = to_np(sino_matrix_full[0, 0])
    s_mla = to_np(sino_matrix_la[0, 0])

    r_af  = to_np(fbp_astra_full[0, 0])
    r_ala = to_np(fbp_astra_la[0, 0])
    r_mf  = to_np(fbp_matrix_full[0, 0])
    r_mla = to_np(fbp_matrix_la[0, 0])

    e_af  = r_af  - gt_np
    e_ala = r_ala - gt_np
    e_mf  = r_mf  - gt_np
    e_mla = r_mla - gt_np

    def rmse(e): return float(np.sqrt(np.mean(e ** 2)))

    print("\n  RMSE (vs GT)")
    print(f"  ASTRA  full FBP  : {rmse(e_af):.4e}")
    print(f"  ASTRA  LA   FBP  : {rmse(e_ala):.4e}")
    print(f"  Matrix full FBP  : {rmse(e_mf):.4e}")
    print(f"  Matrix LA   FBP  : {rmse(e_mla):.4e}")
    print(f"  Stored FBP init  : {rmse(fbp_np - gt_np):.4e}")

    # -- Colour limits -----------------------------------------------------
    e_abs = max(np.abs(e_af).max(), np.abs(e_ala).max(),
                np.abs(e_mf).max(), np.abs(e_mla).max())

    r_min = min(r_af.min(), r_ala.min(), r_mf.min(), r_mla.min(), gt_np.min())
    r_max = max(r_af.max(), r_ala.max(), r_mf.max(), r_mla.max(), gt_np.max())

    # Shared colour limits for sinograms (use full sinogram range)
    s_min = min(s_af.min(), s_mf.min())
    s_max = max(s_af.max(), s_mf.max())

    # -- Figure (3 rows × 5 cols) ------------------------------------------
    #
    #   Row 0  Sinograms : GT image | ASTRA full | ASTRA LA | Matrix full | Matrix LA
    #   Row 1  FBP recon : stored sino (data ref) | ASTRA full | ASTRA LA | Matrix full | Matrix LA
    #   Row 2  Error     : (label)  | ASTRA full | ASTRA LA | Matrix full | Matrix LA

    sino_ratio = max(n_la / res, 0.4)   # height of LA rows relative to image panels

    fig = plt.figure(figsize=(20, 4 + 4 * sino_ratio + 5))
    gs  = fig.add_gridspec(
        3, 5,
        height_ratios=[sino_ratio, 1, 1],
        hspace=0.5, wspace=0.35,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]

    # ── Row 0: Sinograms ──────────────────────────────────────────────────
    _imshow(axes[0][0], gt_np,  "Ground Truth",   aspect="equal")
    _imshow(axes[0][1], s_af,   "ASTRA\nFull sinogram",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[0][2], s_ala,
            f"ASTRA\nLA sinogram ({n_la}/{n_angles} angles)",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[0][3], s_mf,   "Matrix\nFull sinogram",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[0][4], s_mla,
            f"Matrix\nLA sinogram ({n_la}/{n_angles} angles)",
            vmin=s_min, vmax=s_max, aspect="auto")

    # ── Row 1: FBP Reconstructions ────────────────────────────────────────
    _imshow(axes[1][0], sino_np,
            f"Stored LA sino\n(data, {sino_np.shape[0]} angles, noisy)",
            aspect="auto")
    _imshow(axes[1][1], r_af,  "ASTRA\nFull FBP",  vmin=r_min, vmax=r_max)
    _imshow(axes[1][2], r_ala,
            f"ASTRA\nLA FBP ({n_la}/{n_angles} angles)", vmin=r_min, vmax=r_max)
    _imshow(axes[1][3], r_mf,  "Matrix\nFull FBP", vmin=r_min, vmax=r_max)
    _imshow(axes[1][4], r_mla,
            f"Matrix\nLA FBP ({n_la}/{n_angles} angles)", vmin=r_min, vmax=r_max)

    # ── Row 2: Error maps (recon – GT) ────────────────────────────────────
    axes[2][0].axis("off")
    axes[2][0].text(0.5, 0.5, "Error\n(recon − GT)", ha="center", va="center",
                    fontsize=10, transform=axes[2][0].transAxes)

    _imshow(axes[2][1], e_af,
            f"ASTRA Full FBP\nRMSE={rmse(e_af):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][2], e_ala,
            f"ASTRA LA FBP\nRMSE={rmse(e_ala):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][3], e_mf,
            f"Matrix Full FBP\nRMSE={rmse(e_mf):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][4], e_mla,
            f"Matrix LA FBP\nRMSE={rmse(e_mla):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

    # ── Row labels ────────────────────────────────────────────────────────
    for r, lbl in enumerate(["Sinograms", "FBP Reconstructions", "Errors (recon − GT)"]):
        axes[r][0].set_ylabel(lbl, fontsize=9, fontweight="bold", labelpad=6)

    fig.suptitle(
        f"Radon comparison  –  sample {args.idx} ({fname})  |  "
        f"{res}×{res}  |  {n_angles} angles total,  {n_la} LA  "
        f"|  filter: {args.filter}  |  "
        f"phi: {np.degrees(phi[0]):.0f}°–{np.degrees(phi[1]):.0f}°",
        fontsize=10, y=1.01,
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
