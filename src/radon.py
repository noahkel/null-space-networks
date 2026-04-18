import numpy as np
import math
import torch
from typing import Optional, Union, Tuple
import torch.nn.functional as F
try:
    import scipy.fft
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft


def construct_fourier_filter_torch(size: int, filter_name: str, device, dtype=torch.float32) -> torch.Tensor:
    """
    Build the Fourier-domain filter as a 1D torch tensor of shape (size,).
    """
    if size % 2 != 0:
        raise ValueError(f"size must be even, got {size}")

    filter_name = filter_name.lower()

    # Create spatial-domain impulse response f, then FFT -> frequency filter
    n = torch.cat(
        (
            torch.arange(1, size // 2 + 1, 2, device=device, dtype=torch.int64),
            torch.arange(size // 2 - 1, 0, -2, device=device, dtype=torch.int64),
        ),
        dim=0,
    )

    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1.0 / (math.pi * n.to(dtype)) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))

    if filter_name in ("ramp", "ram-lak"):
        pass

    elif filter_name == "shepp-logan":
        # omega = pi * freq, skip DC
        omega = math.pi * torch.fft.fftfreq(size, device=device, dtype=dtype)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega

    elif filter_name == "cosine":
        freq = torch.linspace(0, math.pi, size, device=device, dtype=dtype, requires_grad=False)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif filter_name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, device=device, dtype=dtype))

    elif filter_name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, device=device, dtype=dtype))

    else:
        raise ValueError(
            f"Unknown filter type '{filter_name}'. "
            "Available: 'ramp'/'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'."
        )

    return fourier_filter  # (size,)


def filter_sinogram(
    Y: torch.Tensor,
    filter_name: str = "ramp",
    fourier_filter_cache: Optional[dict] = None,
) -> torch.Tensor:
    """
    Apply FBP-style 1D frequency filtering to sinograms in shape (B, C, H, W),
    filtering along W (detectors). Assumes H = angles.

    Args:
        Y: (B, C, H, W) tensor
        filter_name: filter type
        fourier_filter_cache: optional dict to cache filters by padded_size/device/dtype/name

    Returns:
        Filtered tensor with same shape as Y.
    """
    if Y.ndim != 4:
        raise ValueError(f"Expected input of shape (B, C, H, W), got {tuple(Y.shape)}")

    device = Y.device
    real_dtype = torch.float32 if Y.dtype in (torch.float16, torch.bfloat16) else Y.dtype
    B, C, n_angles, size = Y.shape

    # padded_size = max(64, next_pow2(2*size))
    padded_size = max(64, 1 << math.ceil(math.log2(2 * size)))
    pad = padded_size - size

    # Pad on the last dimension only
    Yf = F.pad(Y.to(real_dtype), (0, pad))  # (B, C, H, padded_size)

    # FFT along detector axis
    sino_fft = torch.fft.fft(Yf, dim=-1)  # complex

    # Build / cache filter
    cache_key = None
    if fourier_filter_cache is not None:
        cache_key = (padded_size, filter_name, device.type, str(device), str(real_dtype))
        f = fourier_filter_cache.get(cache_key)
    else:
        f = None

    if f is None:
        f = construct_fourier_filter_torch(padded_size, filter_name, device=device, dtype=real_dtype)
        # make complex for multiplication with FFT
        f = f.to(torch.complex64 if real_dtype == torch.float32 else torch.complex128)
        if fourier_filter_cache is not None:
            fourier_filter_cache[cache_key] = f

    # Broadcast multiply: (B,C,H,W) * (W,)
    filtered_fft = sino_fft * f.view(1, 1, 1, -1)

    # iFFT back, crop, scale
    filtered = torch.fft.ifft(filtered_fft, dim=-1).real  # (B,C,H,padded_size)
    filtered = filtered[..., :size]  # (B,C,H,W)
    filtered = filtered * (math.pi / (2.0 * n_angles))

    return filtered.to(dtype=Y.dtype)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _RadonBase:
    """
    Base class for Radon adapter implementations.

    Provides limited-angle masking, FBP, power-iteration norm estimation,

    Subclasses must implement `forward` and `backward` and set
    the following attributes in their ``__init__``:

        resolution, det_count, angles (np.ndarray), dx, device, dtype, phi,
        norm_A, norm_A2,
        _ran_mask (torch.Tensor, shape 1×1×n_angles×det_count),
        _nsn_mask (torch.Tensor, same shape).
    """

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_ran_mask_np(self) -> np.ndarray:
        """
        Build a limited-angle mask selecting angles in [lo, hi).

        Returns
        -------
        mask : np.ndarray, shape (1, 1, n_angles, det_count)
            1.0 for angles in-range, 0.0 otherwise.
        """
        lo, hi = self.phi
        ang_mask = ((self.angles >= lo) & (self.angles < hi)).astype(np.float32)
        mask2d = np.repeat(ang_mask.reshape(-1, 1), self.det_count, axis=1)
        return mask2d[None, None, :, :].astype(np.float32)

    def _build_null_mask_np(self) -> np.ndarray:
        """Complement mask: 1 where angles are NOT in [lo, hi), 0 otherwise."""
        return 1.0 - self._build_ran_mask_np()

    # ------------------------------------------------------------------
    # Abstract interface (implemented by subclasses)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Limited-angle and FBP methods
    # ------------------------------------------------------------------

    def forward_la(self, x: torch.Tensor) -> torch.Tensor:
        """Limited-angle forward projection: y = P_phi (A x)."""
        return self.proj_ran(self.forward(x))

    def backward_la(self, y: torch.Tensor) -> torch.Tensor:
        """Limited-angle backprojection: x = A^T (P_phi y)."""
        return self.backward(self.proj_ran(y))

    def fbp(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """Full-angle filtered backprojection: x = A^T( F(y) )."""
        return self.backward(filter_sinogram(y, filter_name=filter_name))

    def fbp_la(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        """Limited-angle filtered backprojection: x = A^T( F(P_phi y) )."""
        return self.backward(filter_sinogram(self.proj_ran(y), filter_name=filter_name))

    def proj_nsn(self, y: torch.Tensor) -> torch.Tensor:
        """Project onto the 'null' (complement) angle set: y * nsn_mask."""
        return y * self._nsn_mask.to(device=y.device, dtype=y.dtype)

    def proj_ran(self, y: torch.Tensor) -> torch.Tensor:
        """Project onto the 'range' (selected) angle set: y * ran_mask."""
        return y * self._ran_mask.to(device=y.device, dtype=y.dtype)

    # ------------------------------------------------------------------
    # Operator norm estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _estimate_operator_norm(
        self,
        iters: int = 20,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        """
        Estimate ||A|| and ||A||^2 via power iteration on A^T A.

        Sets self.norm_A2 and self.norm_A.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        x = torch.randn(
            (1, 1, self.resolution, self.resolution),
            device=self.device,
            dtype=self.dtype,
            generator=g,
        )
        x /= x.norm() + 1e-12

        last_lambda = None
        lam = None

        for _ in range(iters):
            y = self.forward(x)
            x_new = self.backward(y)

            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)

            if last_lambda is not None:
                if abs(lam - last_lambda) / max(lam, 1e-12) < tol:
                    break
            last_lambda = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))


# ---------------------------------------------------------------------------
# torch_radon backend
# ---------------------------------------------------------------------------

class RadonAdapter(_RadonBase):
    """
    Torch_radon-specific convenience wrapper around torch_radon.Radon adds only
    the  __init__, forward, and backward.

    Notes
    -----
    Input / output tensor shapes for torch_radon.Radon are typically:
      x: (B, C, resolution, resolution)
      y: (B, C, n_angles, det_count)
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        clip_to_circle: bool = False,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        phi: Optional[Tuple[float, float]] = None,
    ):
        """
        Parameters
        ----------
        resolution : int
            Image resolution (square image).
        angles : np.ndarray
            Projection angles in radians (torch_radon expects float32).
        det_count : int
            Number of detector bins.
        clip_to_circle : bool
            If True, Radon transform only considers the inscribed circle.
        dataset : str or None
            Optional dataset name (stored but not used by core logic).
        dx : float
            Pixel spacing / scaling. forward multiplies by dx and backward divides by dx.
        estimate_norm : bool
            If True, estimate ||A|| and ||A||^2 via power iteration.
        norm_iters : int
            Max iterations for norm estimation.
        device : torch.device or None
            Where to store masks and run norm estimation.
        dtype : torch.dtype
            dtype for masks and norm estimation.
        phi : (float, float) or None
            Limited-angle interval [lo, hi) in same units as `angles`.
            IMPORTANT: current code assumes phi is not None; otherwise mask building fails.
        """
        try:
            from torch_radon import Radon
        except ImportError:
            raise ImportError(
                "torch_radon is required for RadonAdapter. "
                "Use AstraRadonAdapter instead if torch_radon is not available:\n"
                "  from src.radon import AstraRadonAdapter"
            )
        self.base = Radon(
            resolution=resolution,
            angles=np.asarray(angles, dtype=np.float32),
            det_count=det_count,
            clip_to_circle=clip_to_circle,
        )
        self.dataset = (dataset or "").lower()
        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.dx = float(dx)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        self.phi = phi
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()

        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Radon forward A x, scaled by dx."""
        return self.base.forward(x) * self.dx

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint / backprojection A^T y, scaled by 1/dx."""
        return self.base.backward(y / self.dx)


# ---------------------------------------------------------------------------
# Differentiable ASTRA autograd wrappers
# ---------------------------------------------------------------------------

class _AstraFP(torch.autograd.Function):
    """
    Differentiable forward projection via ASTRA.

    forward : x (B,C,H,W) -> FP(x) (B,C,n_angles,det_count)
    backward: grad_out     -> BP(grad_out)   [adjoint of FP]
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, adapter: "AstraRadonAdapter") -> torch.Tensor:
        ctx.adapter = adapter
        B, C, _H, _W = x.shape
        n_angles = len(adapter.angles)
        x_np = x.detach().cpu().float().numpy()
        out = np.empty((B, C, n_angles, adapter.det_count), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c] = adapter._fp_single(x_np[b, c])
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adapter = ctx.adapter
        B, C, _n_a, _nd = grad_output.shape
        g_np = grad_output.detach().cpu().float().numpy()
        out = np.empty((B, C, adapter.resolution, adapter.resolution), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c] = adapter._bp_single(g_np[b, c])
        # second return value is gradient w.r.t. `adapter` (not a tensor — None)
        return torch.from_numpy(out).to(device=grad_output.device, dtype=grad_output.dtype), None


class _AstraBP(torch.autograd.Function):
    """
    Differentiable backprojection via ASTRA.

    forward : y (B,C,n_angles,det_count) -> BP(y) (B,C,H,W)
    backward: grad_out                   -> FP(grad_out)   [adjoint of BP]
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, adapter: "AstraRadonAdapter") -> torch.Tensor:
        ctx.adapter = adapter
        B, C, _n_a, _nd = y.shape
        y_np = y.detach().cpu().float().numpy()
        out = np.empty((B, C, adapter.resolution, adapter.resolution), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c] = adapter._bp_single(y_np[b, c])
        return torch.from_numpy(out).to(device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adapter = ctx.adapter
        B, C, _H, _W = grad_output.shape
        n_angles = len(adapter.angles)
        g_np = grad_output.detach().cpu().float().numpy()
        out = np.empty((B, C, n_angles, adapter.det_count), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                out[b, c] = adapter._fp_single(g_np[b, c])
        return torch.from_numpy(out).to(device=grad_output.device, dtype=grad_output.dtype), None


# ---------------------------------------------------------------------------
# ASTRA Toolbox backend
# ---------------------------------------------------------------------------

class AstraRadonAdapter(_RadonBase):
    """
    Radon adapter backed by the ASTRA Toolbox.

    The ASTRA Toolbox must be installed separately::

        conda install -c astra-toolbox astra-toolbox          # CPU
        conda install -c astra-toolbox/label/cuda astra-toolbox  # GPU

    Parameters
    ----------
    resolution : int
        Square image side length (pixels).
    angles : np.ndarray
        Projection angles in radians.
    det_count : int
        Number of detector elements.
    clip_to_circle : bool
        Not supported; a warning is issued if True.
    dataset : str or None
        Optional label (stored, not used internally).
    dx : float
        Pixel-spacing scale factor. forward multiplies by dx, backward divides by dx.
    estimate_norm : bool
        Run power iteration to estimate ``norm_A`` and ``norm_A2``.
    norm_iters : int
        Maximum power-iteration steps.
    device : torch.device or None
        Target device for output tensors and norm estimation.
    dtype : torch.dtype
        Floating-point dtype.
    phi : (float, float) or None
        Limited-angle window ``[lo, hi)`` in radians.

    Notes
    -----
    ASTRA operates on NumPy arrays internally.  Input tensors are moved to
    CPU for projection (as float32) and the result is moved back to the
    original device.  For GPU workflows the CUDA algorithms run on ASTRA's
    own GPU memory; the tensor-to-numpy copy is the only overhead.
    """

    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        clip_to_circle: bool = False,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        phi: Optional[Tuple[float, float]] = None,
    ):
        try:
            import astra as _astra
        except ImportError:
            raise ImportError(
                "astra-toolbox is required. Install with:\n"
                "  conda install -c astra-toolbox astra-toolbox\n"
                "  # or for CUDA support:\n"
                "  conda install -c astra-toolbox/label/cuda astra-toolbox"
            )
        self._astra = _astra

        if clip_to_circle:
            import warnings
            warnings.warn(
                "clip_to_circle=True is not supported by AstraRadonAdapter; ignoring.",
                UserWarning,
                stacklevel=2,
            )

        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float64)
        self.dx = float(dx)
        self.dataset = (dataset or "").lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.phi = phi
        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        # ASTRA geometry descriptors
        self._vol_geom = _astra.create_vol_geom(self.resolution, self.resolution)
        self._proj_geom = _astra.create_proj_geom(
            'parallel', 1.0, self.det_count, self.angles
        )

        # Use GPU algorithms only if the device is CUDA and ASTRA has CUDA
        self._use_gpu = (
            str(self.device).startswith('cuda') and _astra.use_cuda()
        )

        # CPU projector (not needed for GPU path)
        if not self._use_gpu:
            self._proj_id: Optional[int] = _astra.create_projector(
                'strip', self._proj_geom, self._vol_geom
            )
        else:
            self._proj_id = None

        # Build masks
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()
        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    # ------------------------------------------------------------------
    # Single-image ASTRA calls
    # ------------------------------------------------------------------

    def _fp_single(self, x_np: np.ndarray) -> np.ndarray:
        """
        Forward project one image.

        Parameters
        ----------
        x_np : np.ndarray, shape (resolution, resolution), float32

        Returns
        -------
        np.ndarray, shape (n_angles, det_count), float32
        """
        astra = self._astra
        vol_id = astra.data2d.create('-vol', self._vol_geom, data=x_np)
        sino_id = astra.data2d.create('-sino', self._proj_geom)
        try:
            if self._use_gpu:
                cfg = astra.astra_dict('FP_CUDA')
            else:
                cfg = astra.astra_dict('FP')
                cfg['ProjectorId'] = self._proj_id
            cfg['VolumeDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                return astra.data2d.get(sino_id).copy()
            finally:
                astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([vol_id, sino_id])

    def _bp_single(self, sino_np: np.ndarray) -> np.ndarray:
        """
        Backproject one sinogram.

        Parameters
        ----------
        sino_np : np.ndarray, shape (n_angles, det_count), float32

        Returns
        -------
        np.ndarray, shape (resolution, resolution), float32
        """
        astra = self._astra
        vol_id = astra.data2d.create('-vol', self._vol_geom)
        sino_id = astra.data2d.create('-sino', self._proj_geom, data=sino_np)
        try:
            if self._use_gpu:
                cfg = astra.astra_dict('BP_CUDA')
            else:
                cfg = astra.astra_dict('BP')
                cfg['ProjectorId'] = self._proj_id
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                return astra.data2d.get(vol_id).copy()
            finally:
                astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([vol_id, sino_id])

    # ------------------------------------------------------------------
    # Batched forward / backward (public interface)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Radon forward projection A x, scaled by dx.

        Differentiable: gradients flow back through ASTRA via BP (the adjoint).

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, resolution, resolution)

        Returns
        -------
        torch.Tensor, shape (B, C, n_angles, det_count)
        """
        return _AstraFP.apply(x, self) * self.dx

    def backward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint / backprojection A^T y, scaled by 1/dx.

        Differentiable: gradients flow back through ASTRA via FP (the adjoint of BP).

        Parameters
        ----------
        y : torch.Tensor, shape (B, C, n_angles, det_count)

        Returns
        -------
        torch.Tensor, shape (B, C, resolution, resolution)
        """
        return _AstraBP.apply(y / self.dx, self)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Release the ASTRA CPU projector if one was created."""
        proj_id = getattr(self, '_proj_id', None)
        if proj_id is not None:
            try:
                self._astra.projector.delete(proj_id)
            except Exception:
                pass