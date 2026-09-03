"""Parallel-beam Radon with real geometry parameters -- able to match an actual scanner, not
merely to pass the adjoint test.

## Why this file is needed and ops/radon.py is not enough

`ops/radon.py` passes the adjoint test (2.9e-15), but compared against real LoDoPaB observations
its **per-angle shape correlation has a median of only 0.799** (a pure scale difference would
give something close to 1.0 after normalization). The reason is that it hard-codes the number of
detectors to the image side length and has no notion of pixel size or detector spacing.

**This exposed a missing gate in L1**:

  * the adjoint test asks "is A^T the transpose of A" -- it checks the **implementation**
  * the shape comparison asks "is A that scanner" -- it checks the **modelling**

The two are orthogonal. An operator can have a perfect adjoint and completely wrong modelling,
and it will quietly produce plausible-looking reconstructions. So L1 in this repository has
**two** acceptance criteria, not one.

## LoDoPaB geometry (from the paper, Leuschner et al., Sci Data 2021, doi:10.1038/s41597-021-00893-z)

  * image 362 x 362 px, physical domain 26 cm x 26 cm  =>  pixel size 26/362 = 0.071823 cm
  * 513 equispaced detector bins, **span = image diameter** (not the side length) => 26 * sqrt(2) cm
  * 1000 equispaced angles over [0, pi)
  * the observation is a **post-log** quantity: y = -ln(N/N0), N0 = 4096 photons per bin

The first two fix the dimensions: line integral = sum(pixel values) * pixel size. Omit that size
and the result is off by hundreds -- the 700x calibration factor that an earlier hard fit
produced was exactly this.
"""
import math
import torch
import torch.nn.functional as F

__all__ = ["RadonGeom", "LODOPAB"]

LODOPAB = dict(n_img=362, fov_cm=26.0, n_det=513, n_ang=1000)


class RadonGeom:
    """Parallel-beam Radon with physical geometry.

    n_img       image side length (pixels)
    fov_cm      physical side length of the image domain (cm), pixel size = fov_cm / n_img
    n_det       number of detector bins
    det_span_cm total detector span (cm). If None, spans the image diameter, fov_cm*sqrt(2)
    angles      radians

    Every output value is a **physical line integral** (units: image value x cm) and can
    therefore be compared directly against real post-log observations with no calibration factor.
    That is the core difference between this file and ops/radon.py.
    """

    def __init__(self, n_img, fov_cm, n_det, angles,
                 det_span_cm=None, device="cpu", dtype=torch.float64):
        self.n_img, self.n_det = int(n_img), int(n_det)
        self.fov_cm = float(fov_cm)
        self.px_cm = self.fov_cm / self.n_img
        self.det_span_cm = (self.fov_cm * math.sqrt(2.0)
                            if det_span_cm is None else float(det_span_cm))
        self.angles = torch.as_tensor(angles, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

        # physical positions of the detector bin centres (cm), origin at the image centre
        half = self.det_span_cm / 2.0
        det_pos = torch.linspace(-half, half, self.n_det, device=device, dtype=dtype)
        # convert to grid_sample normalized coordinates: the image span fov_cm maps to [-1, 1]
        self._det_norm = (det_pos / (self.fov_cm / 2.0)).clamp(-4.0, 4.0)

        self._grids = [self._sample_grid(a) for a in self.angles]

    def _sample_grid(self, angle):
        """For a given angle, build the (1, n_det, n_img, 2) sampling grid.

        Ray i crosses the image along direction (cos, sin); n_img equispaced points are taken
        along the ray and the line integral is approximated by a trapezoidal sum. Taking n_img
        sample points is standard practice (same order as the image resolution).
        """
        c, s = torch.cos(angle), torch.sin(angle)
        # unit vector along the ray, and the normal
        dirx, diry = -s, c                       # along the ray
        nx, ny = c, s                            # perpendicular to the ray (detector axis)
        # sampling positions along the ray, covering the image diagonal
        t = torch.linspace(-math.sqrt(2.0), math.sqrt(2.0), self.n_img,
                           device=self.device, dtype=self.dtype)
        d = self._det_norm                       # (n_det,)
        # grid point = d * normal + t * ray direction
        gx = d.reshape(-1, 1) * nx + t.reshape(1, -1) * dirx
        gy = d.reshape(-1, 1) * ny + t.reshape(1, -1) * diry
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)     # (1,n_det,n_img,2)

    @property
    def in_shape(self):
        return (self.n_img, self.n_img)

    @property
    def out_shape(self):
        return (len(self.angles), self.n_det)

    def __call__(self, x):
        return self.project(x)

    def project(self, x, length_unit="m"):
        """A x -- the physical line integral.

        length_unit sets the length dimension, default "m". **This is not a decorative
        parameter**: the observations LoDoPaB stores are equivalent to a line integral computed
        "with the normalized image and metres as the length unit", whereas my first version used
        cm, so everything was off by exactly a factor of 100. The measured ratio is 98.63, and
        the 1.4% away from 100 is sampling approximation error (the same error also shows up as a
        shape correlation of 0.976 rather than >0.99).

        Chasing that factor of 100 took three steps, recorded here so it is not chased again:
          1. a hard fit produced a 700x calibration factor -> the geometry itself was wrong (the
             detector span had been taken as the side length)
          2. after fixing the geometry a factor of 99 remained, with a shape correlation of only
             0.976
          3. two sentences in the paper complete the chain: the ground truth is normalized by
             mu/mu_max; the projection uses the **unnormalized** image (units 1/m); and the
             stored observation is divided back by mu_max => equivalent to "normalized image x
             metres"
        """
        x = x.reshape(1, 1, self.n_img, self.n_img).to(self.device, self.dtype)
        scale = 0.01 if length_unit == "m" else 1.0        # cm -> m
        ds = (self.fov_cm * math.sqrt(2.0)) / self.n_img * scale
        rows = []
        for g in self._grids:
            v = F.grid_sample(x, g, align_corners=False, padding_mode="zeros")
            rows.append(v.reshape(self.n_det, self.n_img).sum(dim=1) * ds)
        return torch.stack(rows, dim=0)

    def backproject(self, y):
        """A^T y -- taken via VJP, exact by definition (see the record of the mistake in ops/radon.py)."""
        y = y.reshape(*self.out_shape).to(self.device, self.dtype)
        x0 = torch.zeros(self.n_img, self.n_img, device=self.device,
                         dtype=self.dtype, requires_grad=True)
        out = self.project(x0)
        g, = torch.autograd.grad(out, x0, grad_outputs=y)
        return g.reshape(self.n_img, self.n_img)
