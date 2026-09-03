"""Cartesian multi-coil MRI forward operator, and its adjoint.

    A x = M . F (S_c . x)   c = 1..C

where S_c is the sensitivity map of coil c (pointwise complex multiplication), F is the 2D DFT
and M is the k-space sampling mask. This is the standard forward model of parallel imaging, and
the starting point of the whole CS-MRI / fastMRI line of work.

## Why it sits next to Radon

Radon is integral geometry (integration along lines), MRI is frequency-domain sampling (taking a
subset of Fourier coefficients). Their **null-space structures are completely different** --
sparse-view CT loses information along the angular direction, undersampled MRI loses particular
frequencies. Both are implemented so that the L4 layer, "what lives in the operator null space",
does not end up learning only one geometry.

## Complex numbers and the adjoint

One thing needs care here: in the complex case the adjoint is the **conjugate transpose**, not
the transpose.

    <A x, y> = <x, A^H y>,  where <a,b> = sum(conj(a) . b)

So inside A^H the DFT must use ifft (which conjugates) and the sensitivity maps must be conj(S_c).

**I wrote one sentence here wrongly, and it is kept as a record**: the first version of this
document said "if one conjugation is missing, a real input may still pass (the imaginary parts
happen to cancel), so complex vectors are needed to catch it". Measurement says that is **not
true** -- replacing conj(S) by S gives an error of 1.73 on complex input and **9.61 on real
input, which catches it just as well**. That mechanism was something I assumed rather than
tested. The real reason for using complex random vectors is only broader coverage (real inputs
probe just one real subspace of A), not "real inputs would miss it".
**An explanation that has never been checked numerically should not sit in a file posing as a
reason.**

## Normalization

`torch.fft.fft2(norm="ortho")` makes F unitary (F^H F = I), so the spectral norm of A is
determined only by the mask and the sensitivity maps, and choosing a step size in L2 (which needs
||A||^2) is not disturbed by DFT scale factors. With the default norm, ||A||^2 varies with image
size and the step-size condition drifts along with it -- a very well-hidden trap.
"""
import torch

__all__ = ["CartesianMRI", "birdcage_sensitivities"]


def birdcage_sensitivities(n, n_coils, device="cpu", dtype=torch.complex128):
    """A simplified set of circularly arranged coil sensitivity maps, for self-tests only (not a
    real scanner calibration).

    Real data (fastMRI) comes with ESPIRiT-estimated sensitivity maps; these are synthesized only
    so the operator can still be covered by the adjoint test when no real data is around.
    **Do not pass them off as real sensitivities in experimental conclusions.**
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device, dtype=torch.float64),
        torch.linspace(-1, 1, n, device=device, dtype=torch.float64),
        indexing="ij")
    maps = []
    for c in range(n_coils):
        ang = 2 * torch.pi * c / n_coils
        cx, cy = 1.5 * torch.cos(torch.tensor(ang)), 1.5 * torch.sin(torch.tensor(ang))
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mag = 1.0 / (d2 + 0.5)
        phase = torch.atan2(yy - cy, xx - cx)
        maps.append((mag * torch.exp(1j * phase)).to(dtype))
    S = torch.stack(maps, 0)
    # pointwise normalization: sum_c |S_c|^2 = 1, the common convention for ESPIRiT output
    S = S / torch.sqrt((S.abs() ** 2).sum(0, keepdim=True)).clamp_min(1e-12)
    return S


class CartesianMRI:
    """A x = M . F(S . x), with x an (n,n) complex image and output a (C, n, n) complex k-space."""

    def __init__(self, sens, mask, device="cpu"):
        self.S = sens.to(device)                    # (C, n, n) complex
        self.mask = mask.to(device)                 # (n, n) real 0/1, broadcast over coils
        self.C, self.n, _ = self.S.shape
        self.device = device
        self.dtype = self.S.dtype

    @property
    def in_shape(self):
        return (self.n, self.n)

    @property
    def out_shape(self):
        return (self.C, self.n, self.n)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x.reshape(self.n, self.n).to(self.device, self.dtype)
        coil_imgs = self.S * x.unsqueeze(0)                      # (C,n,n)
        k = torch.fft.fft2(coil_imgs, norm="ortho")
        return k * self.mask.unsqueeze(0)

    def adjoint(self, y):
        """A^H y -- conjugate transpose, not transpose."""
        y = y.reshape(self.C, self.n, self.n).to(self.device, self.dtype)
        k = y * self.mask.unsqueeze(0)                            # M^H = M (real diagonal)
        imgs = torch.fft.ifft2(k, norm="ortho")                   # F^H
        return (self.S.conj() * imgs).sum(0)                      # S^H and sum over coils

    def normal(self, x):
        """A^H A x -- the combination L2 uses most; given separately to save one reshape."""
        return self.adjoint(self.forward(x))
