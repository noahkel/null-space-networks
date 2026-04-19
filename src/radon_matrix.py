"""
Sparse-matrix Radon adapter.

Precomputes the system matrix A via ASTRA
PyTorch's native autograd handles gradients

Usage
-----
    from src.radon_matrix import MatrixRadonAdapter

    adapter = MatrixRadonAdapter(
        resolution=256,
        angles=np.linspace(0, np.pi, 60, endpoint=False),
        det_count=363,
        phi=(lo, hi),
    )
    y = adapter.forward(x)   # (B, C, n_angles, det_count)
    x = adapter.backward(y)  # (B, C, resolution, resolution)
"""

import math
import warnings
import numpy as np
import torch
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from typing import Optional, Tuple, Union

from src.radon import _RadonBase, filter_sinogram


class MatrixRadonAdapter(_RadonBase):
    """
    Radon adapter backed by a precomputed sparse system matrix.

    On construction the full system matrix A is computed once via ASTRA's
    ``astra.matrix`` interface.  Every subsequent forward/backward call is a
    single ``torch.sparse.mm`` — fully differentiable through PyTorch autograd
    with no custom backward required.

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
        Pixel-spacing scale factor applied to forward output (same
        convention as the other adapters).
    estimate_norm : bool
        Run power iteration to estimate ``norm_A`` and ``norm_A2``.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for the sparse matrix and output tensors.
    dtype : torch.dtype
        Floating-point dtype.
    phi : (float, float) or None
        Limited-angle window ``[lo, hi)`` in radians.
    svd_threshold : float
        If > 0, compute a full SVD of A and retain singular values >= this
        threshold.  The retained vectors are stored as pseudoinverse factors
        used by ``pseudoinverse`` / ``pseudoinverse_la`` / ``proj_null_image``.

    Notes
    -----
    The sparse matrix has shape ``(n_angles * det_count, resolution**2)``.
    Its transpose is used for backprojection.

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
    ):
        try:
            import astra as _astra
        except ImportError:
            raise ImportError(
                "astra-toolbox is required. Install with:\n"
                "  conda install -c astra-toolbox astra-toolbox"
            )

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self._A, self._AT = self._build_sparse_matrix(_astra)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    # ------------------------------------------------------------------
    # System matrix construction
    # ------------------------------------------------------------------

    def _build_sparse_matrix(self, astra) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the sparse system matrix via ASTRA and return (A, A^T) as
        CSR tensors on self.device.

        Returns
        -------
        A : torch.Tensor (sparse CSR), shape (n_angles * det_count, resolution**2)
        AT : torch.Tensor (sparse CSR), shape (resolution**2, n_angles * det_count)
        """
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

        if self.svd_threshold > 0 and self.phi is not None:
            ang_mask = ((self.angles >= self.phi[0]) & (self.angles < self.phi[1]))
            row_mask = np.repeat(ang_mask, self.det_count)
            self._build_pseudoinverse_la(csr[row_mask, :])


        A = self._scipy_csr_to_torch(csr)
        AT = self._scipy_csr_to_torch(csr.T.tocsr())
        print("Built sparse Matrix")
        return A, AT

    @staticmethod
    def _full_svd_threshold(
            csr: scipy.sparse.csr_matrix, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a full thin SVD of ``csr`` (converted to dense) and return only
        the components whose singular value >= ``threshold``.

        Returns U_k, s_k, Vt_k in descending singular-value order.
        """
        dense = csr.toarray().astype(np.float64)
        m, n = dense.shape
        mem_gb = dense.nbytes / 1e9
        if mem_gb > 4.0:
            warnings.warn(
                f"SVD of {m}×{n} dense matrix ({mem_gb:.1f} GB) may be slow. "
                "Consider reducing the problem size or using a coarser threshold.",
                UserWarning,
                stacklevel=3,
            )
        # thin SVD: U (m,k), s (k,), Vt (k,n) where k = min(m,n)
        U, s, Vt = scipy.linalg.svd(dense, full_matrices=False)
        cutoff = threshold * s[0]
        mask = s >= cutoff
        kept = int(mask.sum())
        print(f"  SVD: {len(s)} singular values, largest={s[0]:.3e}, cutoff={cutoff:.3e} ({threshold:.2e} * s_max), {kept} retained")
        return U[:, mask], s[mask], Vt[mask, :]

    def _build_pseudoinverse(self, csr: scipy.sparse.csr_matrix) -> None:
        """
        Compute full SVD of A, retain singular values >= svd_threshold, and
        store pseudoinverse factors.

         Stores:
          _pinv_V      : (resolution**2, k)        — right singular vectors
          _pinv_Ut     : (k, n_angles*det_count)   — left singular vectors (transposed)
          _pinv_inv_s  : (k,)                      — reciprocal singular values
        """
        print("Building full pseudoinverse (A)...")
        U, s, Vt = self._full_svd_threshold(csr, self.svd_threshold)
        self._pinv_V = torch.from_numpy(Vt.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._pinv_Ut = torch.from_numpy(U.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._pinv_inv_s = torch.from_numpy((1.0 / s).astype(np.float64)).to(device=self.device, dtype=self.dtype)


    def _build_pseudoinverse_la(self, csr_la: scipy.sparse.csr_matrix) -> None:
        """
        Compute full SVD of A_la (measured-angle submatrix), retain singular
        values >= svd_threshold, and store pseudoinverse factors.

        A_la is the submatrix of A whose rows correspond to measured angles only.
        Stores:
          _la_pinv_V      : (resolution**2, k)
          _la_pinv_Ut     : (k, n_la_angles*det_count)
          _la_pinv_inv_s  : (k,)
        """
        print("Building limited-angle pseudoinverse (A_la)...")
        U, s, Vt = self._full_svd_threshold(csr_la, self.svd_threshold)
        self._la_pinv_V = torch.from_numpy(Vt.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._la_pinv_Ut = torch.from_numpy(U.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._la_pinv_inv_s = torch.from_numpy((1.0 / s).astype(np.float64)).to(device=self.device, dtype=self.dtype)

        self._la_pinv_V = torch.from_numpy(Vt.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._la_pinv_Ut = torch.from_numpy(U.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._la_pinv_inv_s = torch.from_numpy((1.0 / s).astype(np.float64)).to(device=self.device, dtype=self.dtype)

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
        Apply Radon forward projection A x, scaled by dx.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, resolution, resolution)

        Returns
        -------
        torch.Tensor, shape (B, C, n_angles, det_count)
        """
        orig_device = x.device
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        n_angles = len(self.angles)

        # Flatten spatial dims: (B*C, resolution**2)
        x_flat = x.reshape(B * C, H * W).to(dtype=self.dtype, device=self.device)

        # Sparse matmul: (n_angles*det_count, res**2) @ (res**2, B*C) -> (n_angles*det_count, B*C)
        y_flat = torch.sparse.mm(self._A, x_flat.t()).t()  # (B*C, n_angles*det_count)

        return y_flat.reshape(B, C, n_angles, self.det_count).to(device=orig_device, dtype=orig_dtype) * self.dx

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint backprojection A^T y, scaled by 1/dx.

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
        """
        orig_device = y.device
        orig_dtype = y.dtype
        B, C, n_angles, nd = y.shape

        # Flatten sinogram dims: (B*C, n_angles*det_count)
        y_flat = (y / self.dx).reshape(B * C, n_angles * nd).to(dtype=self.dtype, device=self.device)

        # Sparse matmul: (res**2, n_angles*det_count) @ (n_angles*det_count, B*C) -> (res**2, B*C)
        x_flat = torch.sparse.mm(self._AT, y_flat.t()).t()  # (B*C, res**2)

        return x_flat.reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def pseudoinverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the truncated pseudoinverse A^+ y.

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)

        """
        if not hasattr(self, '_pinv_V'):
            raise RuntimeError(
                "Pseudoinverse factors not built. Pass svd_threshold > 0 at construction."
            )
        B, C, n_a, nd = y.shape
        y_flat = (y / self.dx).reshape(B * C, n_a * nd).to(dtype=self.dtype, device=self.device)

        # U_k^T @ y: (svd_rank, B*C)
        intermediate = self._pinv_Ut @ y_flat.t()
        # Scale by 1/sigma_k
        intermediate = self._pinv_inv_s.unsqueeze(1) * intermediate
        # V_k @ result: (res^2, B*C)
        x_flat = self._pinv_V @ intermediate

        return x_flat.t().reshape(B, C, self.resolution, self.resolution)

    def pseudoinverse_la(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the limited-angle pseudoinverse A_la^+ to a masked sinogram.

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)
            Masked sinogram from ``forward_la`` (zeros at unmeasured angles).

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
            Range component A_la^+ A_la e = V_la V_la^T e.
        """
        if not hasattr(self, '_la_pinv_V'):
            raise RuntimeError(
                "LA pseudoinverse factors not built. Pass svd_threshold > 0 and phi at construction."
            )
        orig_device = y.device
        orig_dtype = y.dtype
        B, C, n_a, nd = y.shape

        # Extract measured-angle rows; _ran_mask_np shape: (1,1,n_angles,det_count)
        ang_mask = self._ran_mask_np[0, 0, :, 0].astype(bool)
        y_la = y[:, :, ang_mask, :]  # (B, C, n_la, det_count)
        n_la = int(ang_mask.sum())

        y_flat = (y_la / self.dx).reshape(B * C, n_la * nd).to(dtype=self.dtype, device=self.device)

        intermediate = self._la_pinv_Ut @ y_flat.t()  # (svd_rank, B*C)
        intermediate = self._la_pinv_inv_s.unsqueeze(1) * intermediate
        x_flat = self._la_pinv_V @ intermediate  # (res^2, B*C)

        return x_flat.t().reshape(B, C, self.resolution, self.resolution).to(device=orig_device, dtype=orig_dtype)

    def proj_null_image(self, v: torch.Tensor) -> torch.Tensor:
        """Project image v onto null(A_la): v - A_la^+ A_la v (exact via SVD).

        Falls back to the CG-based base implementation if svd_threshold was 0
        at construction (i.e. LA pseudoinverse factors were not built).
        """
        if not hasattr(self, '_la_pinv_V'):
            return super().proj_null_image(v)
        return v - self.pseudoinverse_la(self.forward_la(v))