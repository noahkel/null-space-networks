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
from pathlib import Path
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
    p.add_argument("--res",        type=int,   default=128,   help="Image resolution (default 64)")
    p.add_argument("--n-angles",   type=int,   default=180,   help="Total projection angles (default 30)")
    p.add_argument("--n-la",       type=int,   default=120,   help="Limited-angle count (default 15)")
    p.add_argument("--svd-thresh", type=float, default=4e-3, help="SVD relative threshold (default 4e-3)")
    p.add_argument("--cache-dir",  type=str,   default="radon_cache", help="Cache directory for matrix/SVD files")
    p.add_argument("--device",     type=str,   default="cuda", help="Device: cpu / cuda / cuda:0 ...")
    p.add_argument("--full",       action="store_true",      help="Params: 128x128, 180 angles")
    p.add_argument("--model-dir",  type=str, default="/home/noah/noah/models_matrices",
                   help="Base directory containing init_*/checkpoints/ sub-folders "
                        "(e.g. /home/noah/noah/models_matrices).  Each init row "
                        "looks for init_{key}/checkpoints/{model-type}_best.pt.")
    p.add_argument("--model-type", type=str,   default="nsn",
                   choices=["nsn", "resnet", "dpnsn", "dpnsn_res"],
                   help="Model architecture to load (default: nsn)")
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


def _find_and_load_model(model_dir, init_key, model_type, matrix_r, noise):
    """
    Look for  {model_dir}/init_{init_key}/checkpoints/{model_type}_best.pt
    and load it.  Returns the model in eval mode, or None if not found.
    """
    if model_dir is None:
        return None
    ckpt = Path(model_dir) / f"init_{init_key}{noise}" / f"checkpoints{noise}" / f"{model_type}_best.pt"
    if not ckpt.exists():
        print(f"  [model] no checkpoint: {ckpt}")
        return None
    return _load_model(str(ckpt), model_type, matrix_r, matrix_r.device)


def visualise_results(x_is, astra_r, matrix_r, matrix_r_full, n_la, res, n_angles, fname,
                      model_dir=None, model_type="nsn", device="cuda", noise="0.0"):
    """
    Grid: one row per init method, columns show the reconstruction, its error
    against ground truth, and the error decomposed into range- and null-space
    components.

    For each init method the corresponding trained model is looked up at
        {model_dir}/init_{init_key}/checkpoints/{model_type}_best.pt
    If the checkpoint exists, four additional columns are shown for that row;
    if not, those columns are blank ("no checkpoint").  If *model_dir* is None
    or no checkpoint is found for any row, the model columns are omitted
    entirely.

    Layout
    ------
    Col 0  : row label (init name)
    Col 1  : ground truth
    Col 2  : init reconstruction
    Col 3  : total error  (recon − GT)
    Col 4  : range error  A_la^+ A_la e
    Col 5  : null error   e − A_la^+ A_la e
    Col 6  : model output          ┐
    Col 7  : model total error     │ present only when at least one
    Col 8  : model range error     │ init has a checkpoint
    Col 9  : model null error      ┘
    """
    if isinstance(x_is, torch.Tensor):
        x_is = [x_is]
    sino_a_full = []
    sino_a_la = []
    sino_m_la = []
    x_gt_np = []
    samples = []
    r_min = 1
    r_max = -1
    e_abs = 0
    d_abs = 0
    for x in x_is:
        x_gt_np.append(to_np(x[0, 0]))
        def add_noise(sino: torch.Tensor, level: float = 0.01) -> torch.Tensor:
            sigma = level * sino.norm() / sino.numel() ** 0.5
            return sino + torch.randn_like(sino) * sigma
        # ── Sinograms ─────────────────────────────────────────────────────────────
        sino_a_full.append(astra_r.forward(x))
        if float(noise) > 0:
            sino_a_full[-1] = add_noise(sino_a_full[-1], level=float(noise) / 100)
        sino_a_la.append(astra_r.proj_ran(sino_a_full[-1]))
        sino_m_la.append(matrix_r.proj_ran(matrix_r.forward(x)))  # shared LA measurement
        if float(noise) > 0:
            sino_m_la[-1] = add_noise(sino_m_la[-1], level=float(noise) / 100)

        # ── Init methods: (display_name, init_key_for_checkpoint, recon_tensor) ──
        init_tensors = [
            ("FBP\n(Astra, LA)",  "fbp",  astra_r.fbp_la(sino_a_full[-1])),
            #("Tikh\n(Matrik, LA)",  "tikh", matrix_r.backward_la_tikhonov(sino_a_la, 4e-3)),
            ("FBP\n(Matrix, LA)",   "fbp",  matrix_r.fbp_la(sino_m_la[-1])),
            ("Pinv\n(Matrix, LA)",  "pinv", matrix_r.backward_la(sino_m_la[-1])),
            ("Pinv\n(Full Matrix, LA)",  "pinv_full", matrix_r_full.backward_la(sino_m_la[-1])),
        ]

        # ── Load one model per unique init_key (cache to avoid re-loading) ────────
        model_cache: dict = {}
        for _, init_key, _ in init_tensors:
            if init_key not in model_cache:
                if "full" in init_key:
                    model_cache[init_key] = _find_and_load_model(
                        model_dir, init_key, model_type, matrix_r_full, noise
                    )
                else:
                    model_cache[init_key] = _find_and_load_model(
                        model_dir, init_key, model_type, matrix_r, noise
                    )

        # ── Error decomposition helper ────────────────────────────────────────────
        def decomp(recon_t, radon):
            """Returns (recon_np, err_np, e_ran_np, e_nul_np) — all (H, W) float32."""
            r64 = recon_t.to(dtype=radon.dtype, device=radon.device)
            x64 = x.to(dtype=radon.dtype, device=radon.device)
            e_t     = r64 - x64
            e_nul_t = radon.proj_null_la(e_t)
            e_ran_t = e_t - e_nul_t

            return (
                to_np(recon_t[0, 0]),
                to_np(e_t[0, 0]),
                to_np(e_ran_t[0, 0]),
                to_np(e_nul_t[0, 0]),
            )

        # ── Per-row data ──────────────────────────────────────────────────────────
        rows = []
        for name, init_key, recon_t in init_tensors:
            if "full" in init_key:
                init_data = decomp(recon_t, matrix_r_full)
            else:
                init_data = decomp(recon_t, matrix_r)

            row_model = model_cache[init_key]
            model_data = None
            if row_model is not None:
                row_model.eval()
                with torch.no_grad():
                    out = row_model(recon_t.float().to(device), sino_m_la[-1].float().to(device))
                if "full" in init_key:
                    model_data = decomp(out.to(dtype=matrix_r.dtype), matrix_r_full)
                else:
                    model_data = decomp(out.to(dtype=matrix_r.dtype), matrix_r)

            rows.append((name, init_data, model_data))

        samples.append(rows)
        # ── Shared colour scales ───────────────────────────────────────────────────
        def rmse(e): return float(np.sqrt(np.mean(e ** 2)))

        all_imgs   = [x_gt_np[-1]] + [r[1][0] for r in rows]
        all_errs   = [r[1][1] for r in rows]
        all_decomp = [r[1][2] for r in rows] + [r[1][3] for r in rows]

        # at least one row has model data → show model columns
        has_model = any(r[2] is not None for r in rows)
        if has_model:
            all_imgs   += [r[2][0] for r in rows if r[2] is not None]
            all_errs   += [r[2][1] for r in rows if r[2] is not None]
            all_decomp += [r[2][2] for r in rows if r[2] is not None]
            all_decomp += [r[2][3] for r in rows if r[2] is not None]

        r_min  = min(min(a.min() for a in all_imgs), r_min)
        r_max  = max(max(a.max() for a in all_imgs), r_max)
        e_abs  = max(max(np.abs(a).max() for a in all_errs), e_abs)   if all_errs   else 1.0
        d_abs  = max(max(np.abs(a).max() for a in all_decomp), d_abs) if all_decomp else 1.0
    has_model = any(r[2] is not None for r in samples[-1])
    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(samples[-1])*len(samples)
    n_cols = 10 if has_model else 6        # label + GT + 4 init + [4 model]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    INIT_HDRS = [
        "",
        "Ground Truth",
        "Init recon",
        "Total error\n(recon − GT)",
        "Range error\n$A_{la}^+A_{la}\\,e$\n(scanner-visible)",
        "Null error\n$e - A_{la}^+A_{la}\\,e$\n(scanner-invisible)",
    ]
    MODEL_HDRS = [
        f"Model output\n({model_type})",
        "Model error",
        "Model range\nerror",
        "Model null\nerror",
    ]
    ALL_HDRS = INIT_HDRS + (MODEL_HDRS if has_model else [])
    for i, rows in enumerate(samples):
        for ri, (name, (recon_np, err_np, e_ran_np, e_nul_np), mdata) in enumerate(rows):
            is_top = ri == 0

            def title(col_i, metric="", _top=is_top):
                hdr = ALL_HDRS[col_i]
                if _top:
                    return f"{hdr}\n{metric}" if metric else hdr
                return metric

            # Col 0: row label
            axes[i * len(rows) + ri, 0].axis("off")
            axes[i * len(rows) + ri, 0].text(0.5, 0.5, name, ha="center", va="center",
                             fontsize=9, fontweight="bold")

            # Col 1: ground truth
            _imshow(axes[i * len(rows) + ri, 1], x_gt_np[i], title(1), vmin=r_min, vmax=r_max)

            # Col 2: init reconstruction
            _imshow(axes[i * len(rows) + ri, 2], recon_np,
                    title(2, f"RMSE Error={rmse(err_np):.3e}"),
                    vmin=r_min, vmax=r_max)

            # Col 3: total error
            _imshow(axes[i * len(rows) + ri, 3], err_np,
                    title(3, f"RMSE Error={rmse(err_np):.3e}"),
                    cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)

            # Col 4: range error
            _imshow(axes[i * len(rows) + ri, 4], e_ran_np,
                    title(4, f"RMSE Error={rmse(e_ran_np):.3e}"),
                    cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)

            # Col 5: null error
            _imshow(axes[i * len(rows) + ri, 5], e_nul_np,
                    title(5, f"RMSE Error={rmse(e_nul_np):.3e}"),
                    cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)

            # Cols 6–9: model
            if has_model:
                if mdata is not None:
                    m_img, m_err, m_ran, m_nul = mdata
                    _imshow(axes[i * len(rows) + ri, 6], m_img,
                            title(6, f"RMSE Error={rmse(m_err):.3e}"),
                            vmin=r_min, vmax=r_max)
                    _imshow(axes[i * len(rows) + ri, 7], m_err,
                            title(7, f"RMSE Error={rmse(m_err):.3e}"),
                            cmap="RdBu_r", vmin=-e_abs, vmax=e_abs)
                    _imshow(axes[i * len(rows) + ri, 8], m_ran,
                            title(8, f"RMSE Error={rmse(m_ran):.3e}"),
                            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)
                    _imshow(axes[i * len(rows) + ri, 9], m_nul,
                            title(9, f"RMSE Error={rmse(m_nul):.3e}"),
                            cmap="RdBu_r", vmin=-d_abs, vmax=d_abs)
                else:
                    for ci in range(6, 10):
                        axes[i * len(rows) + ri, ci].axis("off")
                        axes[i * len(rows) + ri, ci].text(
                            0.5, 0.5, "no checkpoint", ha="center", va="center",
                            fontsize=8, color="#888888",
                            transform=axes[i * len(rows) + ri, ci].transAxes,
                        )

    fig.suptitle(
        f"Init × decomposition  ({fname})  |  "
        f"{res}×{res}  |  {n_angles} angles total, {n_la} LA",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {fname}")
    plt.close(fig)
def _load_model(checkpoint_path, model_type, matrix_r, device):
    """Load a model checkpoint and return the model in eval mode."""
    from src.utils import build_models
    beta = 1.0  # placeholder — not used by NSN/ResNet
    model = build_models([model_type], radon=matrix_r, beta=beta)[model_type].to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {model_type} from {checkpoint_path}")
    return model


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
    if args.model_dir:
        print("  model-dir  : %s  (%s)" % (args.model_dir, args.model_type))
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
        matrix_r_full = MatrixRadonAdapter(
            resolution=res, angles=angles, det_count=det_count,
            phi=phi, svd_threshold=1e-15,
            device=device, dtype=dtype, estimate_norm=True,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        print("  ERROR constructing MatrixRadonAdapter: %s" % e)
        sys.exit(1)

    n_fail = 0

    vis_kwargs = dict(model_dir=args.model_dir, model_type=args.model_type)
    xsyn = make_phantom(res, device, dtype)
    xsin = []
    xmul = []
    for i in range(10):
        xsin.append(make_phantom_single(res))
        xmul.append(make_phantom_multiple(res))

    for i in ("0.0","1.0", "2.0"):
        print("\nRunning synthetic phantom test ...")
        n_fail += run_tests(xsyn, astra_r, matrix_r, svd_thresh)
        visualise_results(xsyn, astra_r, matrix_r, matrix_r_full, n_la, res, n_angles,
                          fname=f"radon_test_{i}.png", noise=i, **vis_kwargs)

        print("\nRunning single-ellipse phantom test ...")
        for xs in xsin:
            n_fail += run_tests(xs, astra_r, matrix_r, svd_thresh)
        visualise_results(xsin, astra_r, matrix_r, matrix_r_full, n_la, res, n_angles,
                          fname=f"radon_test_single_{i}.png", noise=i, **vis_kwargs)

        print("\nRunning multi-ellipse phantom test ...")
        for xm in xmul:
            n_fail += run_tests(xm, astra_r, matrix_r, svd_thresh)
        visualise_results(xmul, astra_r, matrix_r, matrix_r_full, n_la, res, n_angles,
                          fname=f"radon_test_multiple_{i}.png", noise=i, **vis_kwargs)
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
