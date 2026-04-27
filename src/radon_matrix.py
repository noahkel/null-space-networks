"""
Sparse-matrix Radon adapter.

Precomputes the system matrix A via ASTRA
    forward(x)  = A x          — sparse matmul
    backward(y) = A_la^+ y     — dense matmul with precomputed pseudoinverse
Usage
-----
    from src.radon_matrix import MatrixRadonAdapter

    adapter = MatrixRadonAdapter(
        resolution=256,
        angles=np.linspace(0, np.pi, 60, endpoint=False),
        det_count=363,
        phi=(lo, hi),
        svd_threshold=1e-5,
    )
    y = adapter.forward(x)   # (B, C, n_angles, det_count)
    x = adapter.backward(y)  # (B, C, resolution, resolution)
"""

import hashlib
import math
import warnings
import numpy as np
import torch
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from pathlib import Path
from typing import Optional, Tuple, Union

from src.radon import _RadonBase, filter_sinogram


class MatrixRadonAdapter(_RadonBase):
    """
    Radon adapter backed by a precomputed sparse system matrix.

    On construction the system matrix A is built once via ASTRA.
    forward is a single sparse matmul; backward is a single dense matmul
    against the precomputed limited-angle pseudoinverse A_la^+.

    Parameters
    ----------
    resolution : int
        Square image side length (pixels).
    angles : np.ndarray
        Projection angles in radians.
    det_count : int
        Number of detector elements.
    dataset : str or None
        Optional label (stored, not used internally).
    dx : float
        Pixel-spacing scale factor applied to forward output.
    estimate_norm : bool
        Run power iteration to estimate ``norm_A`` and ``norm_A2``.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for tensors.
    dtype : torch.dtype
        Floating-point dtype.
    phi : (float, float) or None
        Limited-angle window ``[lo, hi)`` in radians.
    svd_threshold : float
        If > 0, compute a truncated SVD of A_la and store the pseudoinverse
        matrix A_la^+ = V_k diag(1/s_k) U_k^T for use in backward.
    cache_dir : str or Path or None
        Directory for caching A and A_la^+.

    Notes
    -----
    The sparse matrix has shape ``(n_angles * det_count, resolution**2)``.
    The pseudoinverse matrix has shape ``(resolution**2, n_la * det_count)``.
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        phi: Optional[Tuple[float, float]] = None,
        svd_threshold: float = 0.0,
        cache_dir: Optional[Union[str, Path]] = None,
    ):

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.phi = phi
        self.svd_threshold = float(svd_threshold)
        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        # Warn if detector array is too narrow to cover image corners
        min_det = math.ceil(math.sqrt(2) * self.resolution)
        if self.det_count < min_det:
            warnings.warn(
                f"det_count={self.det_count} may clip image corners. "
                f"Recommended minimum: {min_det} = ceil(sqrt(2) * {self.resolution}).",
                UserWarning,
                stacklevel=2,
            )

        # Build masks (inherited from _RadonBase)
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()
        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        # Precompute system matrix
        # Build or load system matrix (and SVD factors if svd_threshold > 0)
        cache_path = Path(cache_dir) / self._cache_key() if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            print(f"Loading matrix cache from {cache_path}")
            self._load_cache(cache_path)
        else:
            try:
                import astra as _astra
            except ImportError:
                raise ImportError(
                    "astra-toolbox is required. Install with:\n"
                    "  conda install -c astra-toolbox astra-toolbox"
                )
            self._build_matrices(_astra)
            if cache_path is not None:
                print(f"Saving matrix cache to {cache_path}")
                self._save_cache(cache_path)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def _build_matrices(self, astra) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Build the sparse system matrix A and the dense pseudoinverse A_la^+. """
        vol_geom = astra.create_vol_geom(self.resolution, self.resolution)
        proj_geom = astra.create_proj_geom(
            'parallel', 1.0, self.det_count, self.angles
        )

        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        try:
            matrix_id = astra.projector.matrix(proj_id)
            try:
                csr: scipy.sparse.csr_matrix = astra.matrix.get(matrix_id)
            finally:
                astra.matrix.delete(matrix_id)
        finally:
            astra.projector.delete(proj_id)

        # csr shape: (n_angles * det_count, resolution**2)
        csr = csr.astype(np.float64)
        self._A = self._scipy_csr_to_torch(csr)
        print("Built sparse system matrix A")

        if self.svd_threshold > 0 and self.phi is not None:
            ang_mask = (self.angles >= self.phi[0]) & (self.angles < self.phi[1])
            row_mask = np.repeat(ang_mask, self.det_count)
            self._la_AP = self._compute_pseudoinverse(csr[row_mask, :])
            print(f"Built pseudoinverse A_la^+, shape {tuple(self._la_AP.shape)}")

    def _compute_pseudoinverse(self, csr: scipy.sparse.csr_matrix) -> torch.Tensor:
        """
        Compute the truncated pseudoinverse A^+ = V_k diag(1/s_k) U_k^T as a dense matrix.
        Retains singular vectors whose singular value >= svd_threshold * s_max.
        """
        dense = csr.toarray().astype(np.float64)
        m, n = dense.shape
        mem_gb = dense.nbytes / 1e9
        if mem_gb > 4.0:
            warnings.warn(
                f"SVD of {m}×{n} dense matrix ({mem_gb:.1f} GB) may be slow. "
                "Consider reducing problem size or using a coarser threshold.",
                UserWarning,
                stacklevel=3,
            )

        U, s, Vt = scipy.linalg.svd(dense, full_matrices=False)
        cutoff = self.svd_threshold * s[0]
        mask = s >= cutoff
        print(f"  SVD: {len(s)} singular values, s_max={s[0]:.3e}, cutoff={cutoff:.3e}, {mask.sum()} retained")

        U_k, s_k, Vt_k = U[:, mask], s[mask], Vt[mask, :]
        AP = (Vt_k.T / s_k) @ U_k.T  # (n, m)
        return torch.from_numpy(AP).to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Cache key / save / load
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.resolution).encode())
        h.update(str(self.det_count).encode())
        h.update(repr(self.dx).encode())
        h.update(repr(self.phi).encode())
        h.update(repr(self.svd_threshold).encode())
        h.update(self.angles.tobytes())
        return h.hexdigest()[:16]

    def _save_cache(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        t = self._A.cpu()
        csr = scipy.sparse.csr_matrix(
            (t.values().numpy(), t.col_indices().numpy(), t.crow_indices().numpy()),
            shape=t.shape,
        )
        scipy.sparse.save_npz(str(path / "A.npz"), csr)
        if hasattr(self, "_la_AP"):
            np.save(str(path / "la_AP.npy"), self._la_AP.cpu().numpy())


    def _load_cache(self, path: Path) -> None:
        csr = scipy.sparse.load_npz(str(path / "A.npz")).astype(np.float64)
        self._A = self._scipy_csr_to_torch(csr)
        la_ap_path = path / "la_AP.npy"
        if la_ap_path.exists():
            self._la_AP = torch.from_numpy(np.load(str(la_ap_path))).to(device=self.device, dtype=self.dtype)

    def _scipy_csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.Tensor:
        """Convert a scipy CSR matrix to a torch sparse_csr_tensor on self.device."""
        crow = torch.from_numpy(mat.indptr.astype(np.int64))
        col = torch.from_numpy(mat.indices.astype(np.int64))
        val = torch.from_numpy(mat.data.astype(np.float64))
        t = torch.sparse_csr_tensor(crow, col, val, size=mat.shape, dtype=self.dtype)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Core forward / backward (used by _RadonBase helpers)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply A x (sparse matmul), scaled by dx.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, resolution, resolution)

        Returns
        -------
        torch.Tensor, shape (B, C, n_angles, det_count)
        """

        B, C, H, W = x.shape
        n_angles = len(self.angles)

        # Flatten spatial dims: (B*C, resolution**2)
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)

        # (n_angles*det_count, res**2) @ (res**2, B*C) -> (B*C, n_angles*det_count)
        y_flat = torch.sparse.mm(self._A, x_flat.t()).t()
        return y_flat.reshape(B, C, n_angles, self.det_count).to(device=x.device, dtype=x.dtype) * self.dx

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply A_la^+ y (dense matmul with precomputed pseudoinverse).

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
        """
        return self.pseudoinverse_la(y)

    def pseudoinverse_la(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the limited-angle pseudoinverse A_la^+ to the measured-angle rows of y.

        Extracts the measured-angle rows of y (via phi mask), then applies the
        precomputed dense matrix A_la^+ = V_k diag(1/s_k) U_k^T.
        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
        """
        if not hasattr(self, '_la_AP'):
            raise RuntimeError(
                "Pseudoinverse not built. Pass svd_threshold > 0 and phi at construction."
            )
        orig_device, orig_dtype = y.device, y.dtype
        # Extract measured-angle rows; _ran_mask_np shape: (1,1,n_angles,det_count)
        ang_mask = self._ran_mask_np[0, 0, :, 0].astype(bool)
        y_la = y[:, :, ang_mask, :]  # (B, C, n_la, det_count)
        B, C, n_la, nd = y_la.shape

        y_flat = (y_la / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)

        x_flat = (self._la_AP @ y_flat.t()).t()
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def proj_null_image(self, v: torch.Tensor) -> torch.Tensor:
        """Project image v onto null(A_la): v - A_la^+ A_la v """
        if not hasattr(self, '_la_AP'):
            return super().proj_null_image(v)
        return v - self.pseudoinverse_la(self.forward_la(v))

    @torch.no_grad()
    def _estimate_operator_norm(self, iters: int = 20, tol: float = 1e-6, seed: int = 0) -> None:
        """
        Estimate ||A|| and ||A||^2 via power iteration on A^T  A(using sparse adjoint).
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randn(
            (self.resolution ** 2, 1), device=self.device, dtype=self.dtype, generator=g
        )
        x /= x.norm() + 1e-12

        last_lam = None
        lam = None
        for _ in range(iters):
            # y = A x  (forward)
            y = torch.sparse.mm(self._A, x)           # (n_angles*det_count, 1)
            # x_new = A^T y  (adjoint via sparse transpose)
            x_new = torch.sparse.mm(self._A.t(), y)   # (res**2, 1)
            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)
            if last_lam is not None and abs(lam - last_lam) / max(lam, 1e-12) < tol:
                break
            last_lam = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))