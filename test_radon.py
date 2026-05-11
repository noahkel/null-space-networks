"""
test_radon.py - Verification tests for AstraRadonAdapter and MatrixRadonAdapter.

Run with:
    python test_radon.py                    # fast (64x64, 30 angles)
    python test_radon.py --full             # realistic (256x256, 60 angles)
    python test_radon.py --res 128 --n-angles 40 --n-la 20 --svd-thresh 1e-4

Tests
-----
  [1] Construction and shape checks
  [2] Forward consistency: MatrixRadon ~= AstraRadon
  [3] forward_la returns only the measured rows of forward
  [4] SVD reconstruction quality:  A x ~= U_k S_k Vt_k x
  [5] Pseudoinverse range-consistency:  A A^+ A x ~= A x
  [6] Null-space (la):  ||A_la  proj_null_la(v)|| / ||A_la v|| ~= 0
  [7] Null-space (full): ||A    proj_null(v)||    / ||A v||    ~= 0
  [8] Decomposition:  A_la^+ A_la v + proj_null_la(v) ~= v
  [9] Operator-norm estimate is positive and finite
"""

import argparse
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from create_ellipse_data import single_ellipse_generator
from dival.datasets import EllipsesDataset


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    line = "  [%s] %s" % (tag, name)
    if detail:
        line += "  (%s)" % detail
    print(line)
    _results.append(cond)


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))

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

def _rowlabel(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=9, transform=ax.transAxes)
# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------

def make_phantom(res, device, dtype):
    """Simple disk + rectangle phantom, shape (1, 1, res, res)."""
    lin = torch.linspace(-1, 1, res, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    img = ((xx ** 2 + yy ** 2) < 0.6 ** 2).to(dtype)
    img += 0.5 * ((xx.abs() < 0.3) & (yy.abs() < 0.2)).to(dtype)
    return img.unsqueeze(0).unsqueeze(0)

def _odl_to_4d(img) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img).astype(np.float32)).unsqueeze(0).unsqueeze(0)

def make_phantom_single(res):
    dataset = EllipsesDataset(image_size=res)
    gen = single_ellipse_generator(dataset, 'train')
    return _odl_to_4d(next(gen))

def make_phantom_multiple(res):
    dataset = EllipsesDataset(image_size=res)
    return _odl_to_4d(next(dataset.generator('train')))
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shapes(astra_r, matrix_r, x):
    print("\n[1] Shape checks")
    B, C, H, W = x.shape
    n_a = len(astra_r.angles)
    n_la = matrix_r.n_la
    nd = astra_r.det_count

    y_a = astra_r.forward(x)
    y_m = matrix_r.forward(x)
    y_la = matrix_r.forward_la(x)

    check("astra  forward  shape (B,C,n_a,det)", tuple(y_a.shape) == (B, C, n_a, nd), str(tuple(y_a.shape)))
    check("matrix forward  shape (B,C,n_a,det)", tuple(y_m.shape) == (B, C, n_a, nd), str(tuple(y_m.shape)))
    check("matrix forward_la shape (B,C,n_a,det)", tuple(y_la.shape) == (B, C, n_a, nd), str(tuple(y_la.shape)))

    x_back = matrix_r.backward(y_m)
    x_backla = matrix_r.backward_la(y_la)
    check("matrix backward    shape (B,C,H,W)", tuple(x_back.shape) == (B, C, H, W), str(tuple(x_back.shape)))
    check("matrix backward_la shape (B,C,H,W)", tuple(x_backla.shape) == (B, C, H, W), str(tuple(x_backla.shape)))

    return y_m, y_la


def test_forward_consistency(astra_r, matrix_r, x):
    print("\n[2] Forward consistency: matrix ~= astra")
    y_astra = astra_r.forward(x).to(dtype=torch.float64)
    y_matrix = matrix_r.forward(x)
    err = rel(y_matrix, y_astra)
    check("||A_matrix x - A_astra x|| / ||A_astra x|| < 1e-3", err < 1e-3, "%.2e" % err)


def test_forward_la_rows(matrix_r, x):
    print("\n[3] forward_la == selected rows of forward")
    y_full = matrix_r.forward(x)
    y_la = matrix_r.forward_la(x)
    ang_mask = (matrix_r.angles >= matrix_r.phi[0]) & (matrix_r.angles < matrix_r.phi[1])
    y_full_la = y_full[:, :, ang_mask, :]
    err = rel(y_la[:, :, ang_mask, :], y_full_la)
    check("forward_la == forward[:, la_angles, :]", err < 1e-8, "%.2e" % err)


def test_svd_reconstruction(matrix_r, x):
    print("\n[4] SVD reconstruction quality")
    if not hasattr(matrix_r, "_U_k"):
        print("   SKIP - SVD not built (svd_threshold == 0)")
        return

    B, C, H, W = x.shape
    x_flat = x.reshape(1, H * W).to(dtype=matrix_r.dtype, device=matrix_r.device)

    # Full A
    y_sparse = torch.sparse.mm(matrix_r._A, x_flat.t()).t()
    y_svd    = (matrix_r._U_k * matrix_r._s_k) @ (matrix_r._Vt_k @ x_flat.t())
    err = rel(y_svd.t(), y_sparse)
    check("A  ~= U_k S_k Vt_k", err < 1e-2, "%.2e" % err)

    # Limited-angle A_la
    y_la_sparse = torch.sparse.mm(matrix_r._A_la, x_flat.t()).t()
    y_la_svd    = (matrix_r._U_k_la * matrix_r._s_k_la) @ (matrix_r._Vt_k_la @ x_flat.t())
    err_la = rel(y_la_svd.t(), y_la_sparse)
    check("A_la ~= U_k_la S_k_la Vt_k_la", err_la < 1e-2, "%.2e" % err_la)


def test_pseudoinverse_range_consistency(matrix_r, x, svd_thresh):
    print("\n[5] Pseudoinverse range-consistency:  A A^+ A x ~= A x")
    if not hasattr(matrix_r, "_U_k"):
        print("   SKIP - SVD not built")
        return

    tol = max(1e-6, svd_thresh)

    # Full
    y  = matrix_r.forward(x)
    x_back = matrix_r.backward(y)
    finite_full = torch.isfinite(x_back).all().item()
    check("backward() is finite (no NaN/Inf)", finite_full, "non-finite!" if not finite_full else "")
    y2 = matrix_r.forward(x_back)
    err = rel(y2, y)
    check("||A A^+ A x - A x|| / ||A x||  (full)", err < tol, "%.2e" % err)

    # Limited-angle
    y_la  = matrix_r.forward_la(x)
    x_back_la = matrix_r.backward_la(y_la)
    finite_la = torch.isfinite(x_back_la).all().item()
    check("backward_la() is finite (no NaN/Inf)", finite_la, "non-finite!" if not finite_la else "")
    y_la2 = matrix_r.forward_la(x_back_la)
    err_la = rel(y_la2, y_la)
    check("||A_la A_la^+ A_la x - A_la x|| / ||A_la x||  (la)", err_la < tol, "%.2e" % err_la)


def test_null_space(matrix_r, v, svd_thresh):
    print("\n[6,7] Null-space projections")
    if not hasattr(matrix_r, "_U_k"):
        print("   SKIP - SVD not built")
        return

    tol = max(1e-6, svd_thresh)

    # null(A_la)
    v_null_la  = matrix_r.proj_null_la(v)
    Av_null_la = matrix_r.forward_la(v_null_la)
    Av_la      = matrix_r.forward_la(v)
    err_la = float(Av_null_la.norm()) / (float(Av_la.norm()) + 1e-12)
    check("[6] ||A_la  proj_null_la(v)|| / ||A_la v|| ~= 0", err_la < tol, "%.2e" % err_la)

    # null(A)
    v_null  = matrix_r.proj_null(v)
    Av_null = matrix_r.forward(v_null)
    Av      = matrix_r.forward(v)
    err = float(Av_null.norm()) / (float(Av.norm()) + 1e-12)
    check("[7] ||A     proj_null(v)||    / ||A v||    ~= 0", err    < tol, "%.2e" % err)


def test_decomposition(matrix_r, v):
    print("\n[8] Decomposition: A_la^+ A_la v + proj_null_la(v) ~= v")
    if not hasattr(matrix_r, "_U_k_la"):
        print("   SKIP - SVD not built")
        return

    range_comp = matrix_r.backward_la(matrix_r.forward_la(v))
    null_comp  = matrix_r.proj_null_la(v)
    err = rel(range_comp + null_comp, v)
    check("||range_comp + null_comp - v|| / ||v||", err < 1e-6, "%.2e" % err)


def test_operator_norm(astra_r, matrix_r):
    print("\n[9] Operator norm estimates")
    check("astra  norm_A > 0",
          astra_r.norm_A is not None and astra_r.norm_A > 0,
          "norm_A=%.4e" % astra_r.norm_A if astra_r.norm_A else "None")
    check("matrix norm_A > 0",
          matrix_r.norm_A is not None and matrix_r.norm_A > 0,
          "norm_A=%.4e" % matrix_r.norm_A if matrix_r.norm_A else "None")
    if astra_r.norm_A and matrix_r.norm_A:
        ratio = matrix_r.norm_A / astra_r.norm_A
        check("norm_A ratio in (0.8, 1.2)", 0.8 < ratio < 1.2, "ratio=%.3f" % ratio)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Radon adapter verification tests")
    p.add_argument("--res",        type=int,   default=64,   help="Image resolution (default 64)")
    p.add_argument("--n-angles",   type=int,   default=30,   help="Total projection angles (default 30)")
    p.add_argument("--n-la",       type=int,   default=15,   help="Limited-angle count (default 15)")
    p.add_argument("--svd-thresh", type=float, default=4e-3, help="SVD relative threshold (default 4e-3)")
    p.add_argument("--cache-dir",  type=str,   default="radon_cache", help="Cache directory for matrix/SVD files")
    p.add_argument("--device",     type=str,   default=None, help="Device: cpu / cuda / cuda:0 ...")
    p.add_argument("--full",       action="store_true",      help="Params: 128x128, 180 angles")
    return p.parse_args()

def run_tests(x, astra_r, matrix_r, svd_thresh):
    test_shapes(astra_r, matrix_r, x)
    test_forward_consistency(astra_r, matrix_r, x)
    test_forward_la_rows(matrix_r, x)
    test_svd_reconstruction(matrix_r, x)
    test_pseudoinverse_range_consistency(matrix_r, x, svd_thresh)
    v = torch.randn_like(x)
    test_null_space(matrix_r, v, svd_thresh)
    test_decomposition(matrix_r, v)
    test_operator_norm(astra_r, matrix_r)

    n_pass = sum(_results)
    n_fail = len(_results) - n_pass
    print("\n" + "=" * 60)
    if n_fail == 0:
        print("Results: %d/%d passed - all OK" % (n_pass, len(_results)))
    else:
        print("Results: %d/%d passed - %d FAILED" % (n_pass, len(_results), n_fail))
    print("=" * 60)
    return n_fail

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()

def visualise_results(x, astra_r, matrix_r, n_la, res, n_angles, fname):
    # ── Forward passes ────────────────────────────────────────────────────
    sino_astra_x = astra_r.forward(x)
    sino_matrix_x = matrix_r.forward(x)
    sino_astra_la_x = astra_r.proj_ran(sino_astra_x)
    sino_matrix_la_x = matrix_r.proj_ran(sino_matrix_x)

    fbp_astra_x = astra_r.fbp(sino_astra_x)
    fbp_astra_la_x = astra_r.fbp_la(sino_astra_la_x)
    fbp_matrix_la_x = matrix_r.fbp_la(sino_matrix_la_x)
    fm_x = to_np(matrix_r.fbp(sino_matrix_x))
    fp_x = to_np(matrix_r.backward(sino_matrix_x))
     # Pseudoinverse: A_la^+ y_la  (exact via truncated SVD, matrix adapter only)

    pinv_matrix_la_x = matrix_r.backward_la(sino_matrix_la_x)

    # ── Convert to numpy ──────────────────────────────────────────────────
    x_gt    = to_np(x[0, 0])
    sa_x    = to_np(sino_astra_x[0, 0])
    sm_x    = to_np(sino_matrix_x[0, 0])
    sa_la_x = to_np(sino_astra_la_x[0, 0])
    sm_la_x = to_np(sino_matrix_la_x[0, 0])
    fa_x    = to_np(fbp_astra_x[0, 0])
    fa_la_x = to_np(fbp_astra_la_x[0, 0])
    fm_la_x = to_np(fbp_matrix_la_x[0, 0])

    fp_la_x = to_np(pinv_matrix_la_x[0, 0])

    # ── Errors ────────────────────────────────────────────────────────────
    e_af = fa_x - x_gt  # ASTRA full FBP
    e_ala = fa_la_x - x_gt  # ASTRA LA  FBP
    e_mla = fm_la_x - x_gt  # Matrix LA FBP
    e_pinv = fp_la_x - x_gt  # Matrix LA Pinv

    # ── Error decomposition for Matrix LA FBP and Pinv ────────────────────
    # Range component: A_la^+ A_la e  (what the scanner can see in the error)
    # Null component:  e - range       (what the scanner cannot see)
    def decompose(e_np: np.ndarray):
        e_t = torch.from_numpy(e_np).to(dtype=matrix_r.dtype,
                                        device=matrix_r.device).unsqueeze(0).unsqueeze(0)
        e_ran_t = matrix_r.backward_la(matrix_r.forward_la(e_t))
        e_nul_t = matrix_r.proj_null_la(e_t)
        return to_np(e_ran_t[0, 0]), to_np(e_nul_t[0, 0])

    e_mla_ran, e_mla_nul = decompose(e_mla)
    e_pinv_ran, e_pinv_nul = decompose(e_pinv)

    # ── Shared colour limits ───────────────────────────────────────────────
    def rmse(e): return float(np.sqrt(np.mean(e ** 2)))

    e_abs = max(np.abs(e_af).max(), np.abs(e_ala).max(), np.abs(e_mla).max(), np.abs(e_pinv).max())
    # decomposition panels share a separate scale (range/null errors are smaller)
    d_abs = max(np.abs(e_mla_ran).max(), np.abs(e_mla_nul).max(),
                np.abs(e_pinv_ran).max(), np.abs(e_pinv_nul).max())

    r_min = min(fa_x.min(), fa_la_x.min(), fm_la_x.min(), fp_la_x.min(), x_gt.min())
    r_max = max(fa_x.max(), fa_la_x.max(), fm_la_x.max(), fp_la_x.max(), x_gt.max())
    s_min = min(sa_x.min(), sm_x.min())
    s_max = max(sa_x.max(), sm_x.max())

    # -- Figure layout (5 rows × 5 cols) -----------------------------------
    #
    #  Cols: label/GT | ASTRA full (ref) | ASTRA LA FBP | Matrix LA FBP | Matrix LA Pinv
    #
    #  Row 0  Sinograms
    #  Row 1  Reconstructions
    #  Row 2  Error  (recon − GT)
    #  Row 3  Range-space error  A_la^+ A_la e       ← what the scanner sees in the error
    #  Row 4  Null-space  error  e − A_la^+ A_la e   ← what the scanner cannot see

    sino_ratio = max(n_la / res, 0.4)

    fig = plt.figure(figsize=(20, sino_ratio * 4 + 4 * 4 + 1))
    gs  = fig.add_gridspec(
        5, 5,
        height_ratios=[sino_ratio, 1, 1, 1, 1],
        hspace=0.55, wspace=0.35,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(5)]

    # ── Row 0: Sinograms ──────────────────────────────────────────────────
    _imshow(axes[0][0], x_gt,
            "Ground Truth",
            aspect="equal")
    _imshow(axes[0][1], sa_x,
            "ASTRA\nFull sinogram",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[0][2], sa_la_x,
            f"ASTRA\nLA sinogram ({n_la}/{n_angles})",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[0][3], fa_x, "ASTRA Full FBP\n(reference)", vmin=r_min, vmax=r_max)
    _imshow(axes[0][4], fa_la_x, f"ASTRA LA FBP\n({n_la}/{n_angles})", vmin=r_min, vmax=r_max)

    axes[0][4].axis("off")

    # ── Row 1: Reconstructions ────────────────────────────────────────────
    axes[1][0].axis("off")
    _imshow(axes[1][1], sm_x,
            f"Matrix\nFull sinogram ({n_la}/{n_angles} angles)",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[1][2], sm_la_x,
            f"Matrix\nLA sinogram ({n_la}/{n_angles})",
            vmin=s_min, vmax=s_max, aspect="auto")
    _imshow(axes[1][3], fm_la_x, f"Matrix LA FBP\n({n_la}/{n_angles})", vmin=r_min, vmax=r_max)
    _imshow(axes[1][4], fp_la_x, f"Matrix LA Pinv  $A_{{la}}^+y$\n({n_la}/{n_angles})",
            vmin=r_min, vmax=r_max)

    # ── Row 2: Total error (recon − GT) ───────────────────────────────────
    _rowlabel(axes[2][0], "Total error\n(recon − GT)")

    _imshow(axes[2][1], e_af,
            f"ASTRA Full FBP\nRMSE={rmse(e_af):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][2], e_ala,
            f"ASTRA LA FBP\nRMSE={rmse(e_ala):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][3], e_mla,
            f"Matrix LA FBP\nRMSE={rmse(e_mla):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
    _imshow(axes[2][4], e_pinv,
            f"Matrix LA Pinv\nRMSE={rmse(e_pinv):.3e}",
            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

    # ── Row 3: Range-space error component ───────────────────────────────
    _rowlabel(axes[3][0],
              "Range error\n$A_{la}^+ A_{la}\, e$\n(visible to scanner)")
    _imshow(axes[3][1], fm_x, f"Matrix FBP\n({n_la}/{n_angles})", vmin=r_min, vmax=r_max)
    _imshow(axes[3][2], fp_x, f"Matrix Pinv  $A_{{la}}^+y$\n({n_la}/{n_angles})",
            vmin=r_min, vmax=r_max)
    _imshow(axes[3][3], e_mla_ran,
            f"Matrix LA FBP\nRMSE={rmse(e_mla_ran):.3e}",
            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)
    _imshow(axes[3][4], e_pinv_ran,
            f"Matrix LA Pinv\nRMSE={rmse(e_pinv_ran):.3e}  (≈ 0)",
            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)

    # ── Row 4: Null-space error component ────────────────────────────────
    _rowlabel(axes[4][0],
              "Null error\n$e - A_{la}^+ A_{la}\, e$\n(invisible to scanner)")
    axes[4][1].axis("off")
    axes[4][2].axis("off")
    _imshow(axes[4][3], e_mla_nul,
            f"Matrix LA FBP\nRMSE={rmse(e_mla_nul):.3e}",
            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)
    _imshow(axes[4][4], e_pinv_nul,
            f"Matrix LA Pinv\nRMSE={rmse(e_pinv_nul):.3e}",
            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)

    # ── Row labels ────────────────────────────────────────────────────────
    row_titles = [
        "Sinograms",
        "Reconstructions",
        "Total error (recon − GT)",
        "Range-space error\n(scanner-visible)",
        "Null-space error\n(scanner-invisible)",
    ]
    for r, lbl in enumerate(row_titles):
        axes[r][0].set_ylabel(lbl, fontsize=8, fontweight="bold", labelpad=6)

    fig.suptitle(
        f"Radon comparison  ({fname})  |  "
        f"{res}×{res}  |  {n_angles} angles total,  {n_la} LA",
        fontsize=10, y=1.01,
    )

    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {fname}")
    plt.close(fig)

def main():
    args = parse_args()

    if args.full:
        res, n_angles, n_la = 128, 180, 120
    else:
        res, n_angles, n_la = args.res, args.n_angles, args.n_la
    svd_thresh = args.svd_thresh

    det_count = math.ceil(math.sqrt(2) * res)
    angles    = np.linspace(0, np.pi, n_angles, endpoint=False)
    phi_hi    = float(angles[n_la])    # first n_la angles form the LA window
    phi       = (0.0, phi_hi)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64

    print("=" * 60)
    print("Radon adapter tests")
    print("  resolution : %dx%d" % (res, res))
    print("  angles     : %d total, %d limited  phi=(0.0, %.4f)" % (n_angles, n_la, phi_hi))
    print("  det_count  : %d" % det_count)
    print("  svd_thresh : %s" % svd_thresh)
    print("  device     : %s" % device)
    print("=" * 60)

    print("\nBuilding AstraRadonAdapter ...")
    try:
        from src.radon import AstraRadonAdapter
        astra_r = AstraRadonAdapter(
            resolution=res, angles=angles, det_count=det_count,
            phi=phi, device=device, dtype=dtype, estimate_norm=True,
        )
    except Exception as e:
        print("  ERROR constructing AstraRadonAdapter: %s" % e)
        sys.exit(1)

    print("\nBuilding MatrixRadonAdapter ...")
    try:
        from src.radon_matrix import MatrixRadonAdapter
        matrix_r = MatrixRadonAdapter(
            resolution=res, angles=angles, det_count=det_count,
            phi=phi, svd_threshold=svd_thresh,
            device=device, dtype=dtype, estimate_norm=True,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        print("  ERROR constructing MatrixRadonAdapter: %s" % e)
        sys.exit(1)
    
    n_fail = 0

    print("\nRunning single phantom test ...")
    x = make_phantom(res, device, dtype)
    n_fail += run_tests(x, astra_r, matrix_r, svd_thresh)
    visualise_results(x, astra_r, matrix_r, n_la, res, n_angles, fname="radon_test.png")

    print("\nRunning single phantom test from create_ellipse_data ...")
    x = make_phantom_single(res)
    n_fail += run_tests(x, astra_r, matrix_r, svd_thresh)
    visualise_results(x, astra_r, matrix_r, n_la, res, n_angles, fname="radon_test_single.png")

    
    print("\nRunning multiple phantom test from create_ellipse_data ...")
    x = make_phantom_multiple(res)
    n_fail += run_tests(x, astra_r, matrix_r, svd_thresh)
    visualise_results(x, astra_r, matrix_r, n_la, res, n_angles, fname="radon_test_multiple.png")
    import matplotlib.pyplot as plt

    s = matrix_r._s_k_la.cpu().numpy()          # singular values of A_la, sorted descending
    s_max = s[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: full spectrum
    ax1.semilogy(s / s_max)
    ax1.axhline(1e-3, color='r', linestyle='--', label='1e-3')
    ax1.axhline(1e-2, color='orange', linestyle='--', label='1e-2')
    ax1.axhline(0.5 / s_max, color='g', linestyle='--', label=f'σ_noise/s_max ≈ {0.5/s_max:.4f}')
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("s_k / s_max")
    ax1.set_title("A_la singular value spectrum")
    ax1.legend()

    # Right: how many dims in null space at each threshold
    thresholds = np.logspace(-4, -1, 100)
    null_dims = [(s < t * s_max).sum() for t in thresholds]
    ax2.semilogx(thresholds, null_dims)
    ax2.axhline(16384 / 3, color='k', linestyle='--', label='Expected (1/3 of pixels)')
    ax2.axvline(1e-3, color='r', linestyle='--', label='1e-3')
    ax2.set_xlabel("SVD threshold")
    ax2.set_ylabel("Null space dimension")
    ax2.set_title("Null space size vs threshold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("svd_spectrum.png", dpi=150)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
