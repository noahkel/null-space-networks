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
    check("matrix forward_la shape (B,C,n_la,det)", tuple(y_la.shape) == (B, C, n_la, nd), str(tuple(y_la.shape)))

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
    y2 = matrix_r.forward(matrix_r.backward(y))
    err = rel(y2, y)
    check("||A A^+ A x - A x|| / ||A x||  (full)", err < tol, "%.2e" % err)

    # Limited-angle
    y_la  = matrix_r.forward_la(x)
    y_la2 = matrix_r.forward_la(matrix_r.backward_la(y_la))
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
    p.add_argument("--svd-thresh", type=float, default=1e-6, help="SVD relative threshold (default 1e-6)")
    p.add_argument("--cache-dir",  type=str,   default=None, help="Cache directory for matrix/SVD files")
    p.add_argument("--device",     type=str,   default=None, help="Device: cpu / cuda / cuda:0 ...")
    p.add_argument("--full",       action="store_true",      help="Params: 128x128, 180 angles")
    return p.parse_args()


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

    x = make_phantom(res, device, dtype)

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
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
