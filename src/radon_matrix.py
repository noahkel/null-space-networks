"""
Sparse-matrix Radon adapter.

Stores two system matrices:
    A    — full forward operator (all angles), shape (n_angles*det, res²)
    A_la — limited-angle operator  (phi angles only), shape (n_la*det, res²)

For each matrix a truncated SVD is computed:
    A    = U_k  S_k  Vt_k      (rank-k approximation)
    A_la = U_kl S_kl Vt_kl

These factors are used for:
    backward     : A^+   y    = Vt_k.T  @ (U_k.T  @ y  / s_k)
    backward_la  : A_la^+ y_la = Vt_kl.T @ (U_kl.T @ y_la / s_kl)
    proj_null    : v - Vt_k.T  @ (Vt_k  @ v)   — projection onto null(A)
    proj_null_la : v - Vt_kl.T @ (Vt_kl @ v)   — projection onto null(A_la)

Shapes
------
    forward(x)       x: (B,C,res,res)    →  y:    (B,C,n_angles,det_count)
    forward_la(x)    x: (B,C,res,res)    →  y_la: (B,C,n_la,det_count)
    backward(y)      y: (B,C,n_angles,det_count) →  (B,C,res,res)
    backward_la(y_la) y_la: (B,C,n_la,det_count) →  (B,C,res,res)
"""

import hashlib
import math
import warnings
import numpy as np
import torch
import scipy.linalg
import scipy.sparse
from pathlib import Path
from typing import Optional, Tuple, Union

from src.radon import _RadonBase, filter_sinogram


class MatrixRadonAdapter(_RadonBase):
    """
    Radon adapter backed by precomputed sparse system matrices A and A_la,
    with truncated SVD factors stored for pseudoinverse and null-space operations.

    Parameters
    ----------
    resolution : int
        Square image side length (pixels).
    angles : np.ndarray
        All projection angles in radians.
    det_count : int
        Number of detector elements.
    phi : (float, float)
        Limited-angle window [lo, hi) in radians.  Required.
    svd_threshold : float
        Relative singular-value cutoff: retain singular values >= threshold * s_max.
        Must be > 0 to build SVD factors and enable backward / null-space methods.
    dataset : str or None
        Optional label (stored, not used internally).
    dx : float
        Pixel-spacing scale factor applied to forward output.
    estimate_norm : bool
        Run power iteration to estimate norm_A and norm_A2 from the sparse A.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for tensors.
    dtype : torch.dtype
        Floating-point dtype.
    cache_dir : str or Path or None
        Directory for caching matrices and SVD factors.
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        phi: Tuple[float, float],
        svd_threshold: float = 0.0,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if phi is None:
            raise ValueError("phi=(lo, hi) is required for MatrixRadonAdapter.")

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.phi = phi
        self.svd_threshold = float(svd_threshold)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        min_det = math.ceil(math.sqrt(2) * self.resolution)
        if self.det_count < min_det:
            warnings.warn(
                f"det_count={self.det_count} may clip image corners "
                f"(recommended minimum: {min_det}).",
                UserWarning, stacklevel=2,
            )

        # Limited-angle angle subset
        _la_mask = (self.angles >= phi[0]) & (self.angles < phi[1])
        self.angles_la: np.ndarray = self.angles[_la_mask]
        self._la_row_mask: np.ndarray = np.repeat(_la_mask, self.det_count)

        # Masks used by _RadonBase helpers (proj_ran / proj_nsn)
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()
        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

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
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_la(self) -> int:
        """Number of limited-angle projections."""
        return int(self._la_row_mask.sum()) // self.det_count

    # ------------------------------------------------------------------
    # Matrix / SVD construction
    # ------------------------------------------------------------------

    def _build_matrices(self, astra) -> None:
        """Build sparse A, sparse A_la, and (if svd_threshold > 0) their SVD factors."""
        vol_geom = astra.create_vol_geom(self.resolution, self.resolution)
        proj_geom = astra.create_proj_geom('parallel', 1.0, self.det_count, self.angles)

        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        try:
            matrix_id = astra.projector.matrix(proj_id)
            try:
                csr: scipy.sparse.csr_matrix = astra.matrix.get(matrix_id)
            finally:
                astra.matrix.delete(matrix_id)
        finally:
            astra.projector.delete(proj_id)

        csr = csr.astype(np.float64)

        # Full system matrix
        self._A = self._csr_to_torch(csr)
        print(f"Built sparse A, shape {tuple(csr.shape)}")

        # Limited-angle submatrix
        csr_la = csr[self._la_row_mask, :]
        self._A_la = self._csr_to_torch(csr_la)
        print(f"Built sparse A_la, shape {tuple(csr_la.shape)}")

        if self.svd_threshold > 0:
            print("Computing SVD of A ...")
            self._U_k, self._s_k, self._Vt_k = self._truncated_svd(csr)

            print("Computing SVD of A_la ...")
            self._U_k_la, self._s_k_la, self._Vt_k_la = self._truncated_svd(csr_la)

    def _truncated_svd(
        self, csr: scipy.sparse.csr_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the truncated SVD of a sparse matrix.

        Returns U_k (m,k), s_k (k,), Vt_k (k,n) as torch tensors on self.device,
        retaining singular values >= svd_threshold * s_max.

        GPU strategy: densify as float32, run torch.linalg.svd under the MAGMA
        backend (full thin SVD in one pass — avoids cuSOLVER gesvdj which overflows
        its int32 workspace counter for large sketch matrices).  Falls back to CPU
        LAPACK / ARPACK if MAGMA is unavailable or the matrix is too large for VRAM.

        CPU strategy: dense scipy.linalg.svd for matrices that fit in RAM,
        otherwise scipy.sparse.linalg.svds (ARPACK) directly on the sparse matrix.
        """
        m, n = csr.shape

        def _t(arr) -> torch.Tensor:
            if isinstance(arr, torch.Tensor):
                return arr.to(device=self.device, dtype=self.dtype)
            return torch.from_numpy(np.asarray(arr, dtype=np.float64)).to(
                device=self.device, dtype=self.dtype
            )

        # --- GPU path: ----------

        def _cut_and_return(U_np, s_np, Vt_np, source: str):
            s_np = np.asarray(s_np, dtype=np.float64)
            cutoff = self.svd_threshold * s_np[0]
            keep = s_np >= cutoff
            print(f"  {m}×{n}: {keep.sum()}/{len(s_np)} singular values retained "
                  f"(s_max={s_np[0]:.3e}, cutoff={cutoff:.3e})  [{source}]")
            return _t(U_np[:, keep]), _t(s_np[keep]), _t(Vt_np[keep, :])

        # ------------------------------------------------------------------
        # GPU path: full thin SVD via MAGMA (single pass, no cuSOLVER limits)
        # ------------------------------------------------------------------
        
        
        if self.device.type == "cuda":
            mem_gb = m * n * 4 / 1e9  # float32
            print(f"  densifying {m}×{n} on GPU ({mem_gb:.1f} GB fp32)")
            dense_f32 = None
            try:
                dense_f32 = torch.from_numpy(csr.toarray().astype(np.float32)).to(self.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    torch.backends.cuda.preferred_linalg_library("magma")
                    U_t, s_t, Vh_t = torch.linalg.svd(dense_f32, full_matrices=False)
                torch.backends.cuda.preferred_linalg_library("default")
                result = _cut_and_return(
                    U_t.cpu().numpy(), s_t.cpu().numpy(), Vh_t.cpu().numpy(), "GPU MAGMA"
                )
                del dense_f32, U_t, s_t, Vh_t
                torch.cuda.empty_cache()
                return result
            except Exception as exc:
                torch.backends.cuda.preferred_linalg_library("default")
                del dense_f32
                torch.cuda.empty_cache()
                print(f"  MAGMA SVD failed ({exc}); falling back to CPU ...")

        # ------------------------------------------------------------------
        # CPU path: dense LAPACK for small matrices, ARPACK for large ones
        # ------------------------------------------------------------------
        mem_gb_f64 = m * n * 8 / 1e9
        if mem_gb_f64 <= 4.0:
            print(f"  densifying {m}×{n} on CPU ({mem_gb_f64:.1f} GB fp64) ...")
            dense = csr.toarray().astype(np.float64)
            U, s_cpu, Vt = scipy.linalg.svd(dense, full_matrices=False)
            del dense
            return _cut_and_return(U, s_cpu, Vt, "CPU LAPACK")

        from scipy.sparse.linalg import svds as sp_svds

        sparse_max_q = min(m, n) - 1
        sparse_q = min(256, sparse_max_q)
        while True:
            k = min(sparse_q, sparse_max_q)
            print(f"  sparse CPU SVD (ARPACK), k={k} ...")
            # Try PROPACK first (faster for large k), fall back to ARPACK.
            try:
                U_cpu, s_arr, Vt_cpu = sp_svds(csr, k=k, which="LM", solver="propack")
            except Exception:
                U_cpu, s_arr, Vt_cpu = sp_svds(csr, k=k, which="LM")

            # svds returns singular values in ascending order — reverse them.
            idx = np.argsort(s_arr)[::-1]
            s_arr = s_arr[idx]
            U_cpu = U_cpu[:, idx]
            Vt_cpu = Vt_cpu[idx, :]

            if s_arr[-1] < self.svd_threshold * s_arr[0] or k >= sparse_max_q:
                break
            next_q = min(k * 2, sparse_max_q)
            print(f"  all {k} values above threshold, retrying with k={next_q} ...")
            sparse_q = next_q

        return _cut_and_return(U_cpu, s_arr, Vt_cpu, "sparse CPU")
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

        for name, mat in [("A", self._A), ("A_la", self._A_la)]:
            t = mat.cpu()
            csr = scipy.sparse.csr_matrix(
                (t.values().numpy(), t.col_indices().numpy(), t.crow_indices().numpy()),
                shape=t.shape,
            )
            scipy.sparse.save_npz(str(path / f"{name}.npz"), csr)

        for name, tensor in [
            ("U_k",    getattr(self, "_U_k",    None)),
            ("s_k",    getattr(self, "_s_k",    None)),
            ("Vt_k",   getattr(self, "_Vt_k",   None)),
            ("U_k_la", getattr(self, "_U_k_la", None)),
            ("s_k_la", getattr(self, "_s_k_la", None)),
            ("Vt_k_la",getattr(self, "_Vt_k_la",None)),
        ]:
            if tensor is not None:
                np.save(str(path / f"{name}.npy"), tensor.cpu().numpy())

    def _load_cache(self, path: Path) -> None:
        self._A    = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A.npz")).astype(np.float64))
        self._A_la = self._csr_to_torch(scipy.sparse.load_npz(str(path / "A_la.npz")).astype(np.float64))

        for name in ("U_k", "s_k", "Vt_k", "U_k_la", "s_k_la", "Vt_k_la"):
            p = path / f"{name}.npy"
            if p.exists():
                setattr(self, f"_{name}",
                        torch.from_numpy(np.load(str(p))).to(device=self.device, dtype=self.dtype))

    def _csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.Tensor:
        crow = torch.from_numpy(mat.indptr.astype(np.int64))
        col  = torch.from_numpy(mat.indices.astype(np.int64))
        val  = torch.from_numpy(mat.data.astype(np.float64))
        t = torch.sparse_csr_tensor(crow, col, val, size=mat.shape, dtype=self.dtype)
        return t.to(self.device)

    # ------------------------------------------------------------------
    # Forward operators
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full-angle forward projection: y = A x.

        Parameters
        ----------
        x : (B, C, res, res)

        Returns
        -------
        y : (B, C, n_angles, det_count)
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        y_flat = torch.sparse.mm(self._A, x_flat.t()).t()
        return (y_flat.reshape(B, C, len(self.angles), self.det_count)
                .to(device=x.device, dtype=x.dtype) * self.dx)

    def forward_la(self, x: torch.Tensor) -> torch.Tensor:
        """
        Limited-angle forward projection: y_la = A_la x.

        Parameters
        ----------
        x : (B, C, res, res)

        Returns
        -------
        y_la : (B, C, n_la, det_count)
        """
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        y_flat = torch.sparse.mm(self._A_la, x_flat.t()).t()
        return (y_flat.reshape(B, C, self.n_la, self.det_count)
                .to(device=x.device, dtype=x.dtype) * self.dx)

    # ------------------------------------------------------------------
    # Pseudoinverse (backward) operators  —  A^+ and A_la^+
    # ------------------------------------------------------------------

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the full-angle pseudoinverse: x = A^+ y.

        Uses the truncated SVD: A^+ = Vt_k.T diag(1/s_k) U_k.T

        Parameters
        ----------
        y : (B, C, n_angles, det_count)

        Returns
        -------
        x : (B, C, res, res)
        """
        self._require_svd("_U_k", "backward (full pseudoinverse)")
        orig_device, orig_dtype = y.device, y.dtype
        B, C, n_a, nd = y.shape
        y_flat = (y / self.dx).reshape(B * C, n_a * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._apply_pseudoinverse(y_flat, self._U_k, self._s_k, self._Vt_k)
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def backward_la(self, y_la: torch.Tensor) -> torch.Tensor:
        """
        Apply the limited-angle pseudoinverse: x = A_la^+ y_la.

        Uses the truncated SVD: A_la^+ = Vt_kl.T diag(1/s_kl) U_kl.T

        Parameters
        ----------
        y_la : (B, C, n_la, det_count)

        Returns
        -------
        x : (B, C, res, res)
        """
        self._require_svd("_U_k_la", "backward_la (limited-angle pseudoinverse)")
        orig_device, orig_dtype = y_la.device, y_la.dtype
        B, C, n_la, nd = y_la.shape
        y_flat = (y_la / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)
        x_flat = self._apply_pseudoinverse(y_flat, self._U_k_la, self._s_k_la, self._Vt_k_la)
        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    @staticmethod
    def _apply_pseudoinverse(
        y_flat: torch.Tensor,
        U_k: torch.Tensor,
        s_k: torch.Tensor,
        Vt_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply A^+ = Vt_k.T diag(1/s_k) U_k.T to a batch of flat measurement vectors.

        Parameters
        ----------
        y_flat : (batch, m)
        U_k    : (m, k)
        s_k    : (k,)
        Vt_k   : (k, n)

        Returns
        -------
        x_flat : (batch, n)
        """
        z = (y_flat @ U_k) / s_k   # (batch, k)
        return z @ Vt_k             # (batch, n)

    # ------------------------------------------------------------------
    # Null-space projections
    # ------------------------------------------------------------------

    def proj_null_la(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project image v onto null(A_la): v_n = v - V_kl V_kl^T v.

        Equivalently: v - Vt_kl.T @ (Vt_kl @ v_flat.T)

        Parameters
        ----------
        v : (B, C, res, res)

        Returns
        -------
        v_n : (B, C, res, res)  — component of v in null(A_la)
        """
        self._require_svd("_Vt_k_la", "proj_null_la")
        return self._proj_null(v, self._Vt_k_la)

    def proj_null_image(self, v: torch.Tensor) -> torch.Tensor:
        """Alias for proj_null_la (overrides _RadonBase CG fallback)."""
        return self.proj_null_la(v)

    def proj_null(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project image v onto null(A): v_n = v - V_k V_k^T v.

        Parameters
        ----------
        v : (B, C, res, res)

        Returns
        -------
        v_n : (B, C, res, res)  — component of v in null(A)
        """
        self._require_svd("_Vt_k", "proj_null")
        return self._proj_null(v, self._Vt_k)

    def _proj_null(self, v: torch.Tensor, Vt_k: torch.Tensor) -> torch.Tensor:
        """
        Shared null-space projection: v - Vt_k.T @ (Vt_k @ v_flat.T).

        Parameters
        ----------
        v    : (B, C, res, res)
        Vt_k : (k, n)

        Returns
        -------
        (B, C, res, res)
        """
        orig_device, orig_dtype = v.device, v.dtype
        B, C, H, W = v.shape
        v_flat = v.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)
        coeffs = v_flat @ Vt_k.t()                 # (B*C, k)
        v_range = coeffs @ Vt_k                    # (B*C, n)  — range component
        result = (v_flat - v_range).reshape(B, C, H, W)
        return result.to(device=orig_device, dtype=orig_dtype)

    # ------------------------------------------------------------------
    # Operator norm estimation  (sparse power iteration on A, not A^+)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _estimate_operator_norm(self, iters: int = 20, tol: float = 1e-6, seed: int = 0) -> None:
        """Estimate ||A|| and ||A||² via power iteration using the sparse matrix."""
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        x = torch.randn((self.resolution ** 2, 1), device=self.device, dtype=self.dtype, generator=g)
        x /= x.norm() + 1e-12

        lam, last_lam = None, None
        for _ in range(iters):
            y = torch.sparse.mm(self._A, x)
            x_new = torch.sparse.mm(self._A.t(), y)
            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)
            if last_lam is not None and abs(lam - last_lam) / max(lam, 1e-12) < tol:
                break
            last_lam = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_svd(self, attr: str, method: str) -> None:
        if not hasattr(self, attr):
            raise RuntimeError(
                f"{method} requires SVD factors. "
                "Pass svd_threshold > 0 at construction (and ensure phi is set for _la variants)."
            )
