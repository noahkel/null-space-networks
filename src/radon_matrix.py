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
    svd_rank : int
        If > 0, compute a truncated SVD of A with this many singular values

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
        svd_rank: int = 0,
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
        self.svd_rank = int(svd_rank)
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

        if self.svd_rank > 0:
            self._build_pseudoinverse(csr)
            if self.phi is not None:
                ang_mask = ((self.angles >= self.phi[0]) & (self.angles < self.phi[1]))
                row_mask = np.repeat(ang_mask, self.det_count)
                self._build_pseudoinverse_la(csr[row_mask, :])


        A = self._scipy_csr_to_torch(csr)
        AT = self._scipy_csr_to_torch(csr.T.tocsr())
        print("Built sparse Matrix")
        return A, AT

    def _build_pseudoinverse(self, csr: scipy.sparse.csr_matrix) -> None:
        """
        Compute truncated SVD of A and store the pseudoinverse factors.

        Stores three dense tensors used by `pseudoinverse`:
          _pinv_V   : (resolution**2, svd_rank)  — right singular vectors
          _pinv_Ut  : (svd_rank, n_angles*det_count) — left singular vectors (transposed)
          _pinv_inv_s : (svd_rank,) — reciprocal singular values

        The pseudoinverse is A^+ = V_k @ diag(1/sigma_k) @ U_k^T, applied as
        three successive matrix multiplies rather than materialised as a dense matrix.
        """
        U, s, Vt = scipy.sparse.linalg.svds(csr, k=self.svd_rank)
        # svds returns singular values in ascending order — reverse to descending
        U  = U[:, ::-1].copy()
        s  = s[::-1].copy()
        Vt = Vt[::-1, :].copy()

        self._pinv_V    = torch.from_numpy(Vt.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._pinv_Ut   = torch.from_numpy(U.T.astype(np.float64)).to(device=self.device, dtype=self.dtype)
        self._pinv_inv_s = torch.from_numpy((1.0 / s).astype(np.float64)).to(device=self.device, dtype=self.dtype)

    def _build_pseudoinverse_la(self, csr_la: scipy.sparse.csr_matrix) -> None:
        """
        Compute truncated SVD of the limited-angle submatrix A_la and store
        pseudoinverse factors used by `pseudoinverse_la`.

        A_la is the submatrix of A whose rows correspond to measured angles only.
        Stores:
          _la_pinv_V      : (resolution**2, svd_rank)
          _la_pinv_Ut     : (svd_rank, n_la_angles*det_count)
          _la_pinv_inv_s  : (svd_rank,)
        """
        U, s, Vt = scipy.sparse.linalg.svds(csr_la, k=self.svd_rank)
        U = U[:, ::-1].copy()
        s = s[::-1].copy()
        Vt = Vt[::-1, :].copy()

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
                "Pseudoinverse factors not built. Pass svd_rank > 0 at construction."
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
                "LA pseudoinverse factors not built. Pass svd_rank > 0 and phi at construction."
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
