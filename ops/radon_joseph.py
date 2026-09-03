"""Joseph-method parallel-beam Radon -- accurate enough to match real scan data.

## Why radon_geom.py's sampling had to be replaced

`radon_geom.py` takes n_img equispaced points along the ray with a fixed step of `px * sqrt(2)`
(the diagonal / n_img). The problem: **that step does not depend on the angle**, while the
number of pixels a ray crosses does.

  * an axial ray (theta = 0 or pi/2) crosses only n_img pixels, total length n_img * px, yet I
    sampled it with a step sqrt(2) times too large => **undersampled by sqrt(2)**
  * a diagonal ray crosses about sqrt(2) * n_img pixels, where the step is exactly right

Measured consequences: scale ratio 1.0100 (1% off), median per-angle shape correlation 0.9763
(target > 0.99). The two numbers are two symptoms of the same error.

## The Joseph method

For each ray, take the coordinate axis with the larger |d| as the **driving axis**, step along it
one **pixel** at a time, and interpolate linearly in the perpendicular direction. The physical
path length per step is

    ds = px / |d_driving_axis|

That term is the obliquity correction: the more slanted the ray, the longer its path through each
pixel column. Equispaced sampling treats it as a constant, which is why it is wrong.

Joseph, "An Improved Algorithm for Reprojecting Rays Through Pixel Images", IEEE TMI 1982 -- this
is standard practice, not something I invented; it is written out here in order to own it, not
because it is new.

## Difference from Siddon

Siddon computes the exact intersections of the ray with the pixel boundaries (the actual chord
length in each pixel), while Joseph steps along a driving axis and interpolates perpendicular to
it. Joseph is smoother and better suited to interpolation-based implementations, and accurate
enough for parallel beam; Siddon is more "exact" but produces a non-smooth operator. The deep
learning reconstruction community generally uses Joseph-type operators.
"""
import math
import torch
import torch.nn.functional as F

__all__ = ["RadonJoseph", "LODOPAB"]

LODOPAB = dict(n_img=362, fov_cm=26.0, n_det=513, n_ang=1000)


class RadonJoseph:
    """Parallel-beam Radon with Joseph sampling, producing physical line integrals.

    length_unit sets the output dimension. The observations LoDoPaB stores are equivalent to
    "normalized image x metres", hence the default "m". That default was forced by the data, not
    chosen by convention -- see the passage in radon_geom.py on how the factor of 100 was tracked
    down.
    """

    def __init__(self, n_img, fov_cm, n_det, angles, det_span_cm=None,
                 length_unit="m", device="cpu", dtype=torch.float64):
        self.n_img, self.n_det = int(n_img), int(n_det)
        self.fov_cm = float(fov_cm)
        self.px_cm = self.fov_cm / self.n_img
        self.det_span_cm = (self.fov_cm * math.sqrt(2.0)
                            if det_span_cm is None else float(det_span_cm))
        self.angles = torch.as_tensor(angles, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype
        self.unit_scale = 0.01 if length_unit == "m" else 1.0

        half = self.det_span_cm / 2.0
        # detector bin centres, normalized to image coordinates ([-1,1] covers the fov)
        self._s = (torch.linspace(-half, half, self.n_det, device=device, dtype=dtype)
                   / (self.fov_cm / 2.0))
        # pixel centres along the driving axis, align_corners=False convention
        j = torch.arange(self.n_img, device=device, dtype=dtype)
        self._axis = (2.0 * j + 1.0) / self.n_img - 1.0

        self._grids, self._ds = [], []
        for a in self.angles:
            g, ds = self._ray_grid(a)
            self._grids.append(g)
            self._ds.append(ds)

    def _ray_grid(self, theta):
        """Return ((1, n_det, n_img, 2) sampling grid, physical step size ds at this angle).

        Ray: p(t) = s * n + t * d, with n = (cos, sin) the detector axis and d = (-sin, cos) the
        ray direction.
        """
        c, s_ = torch.cos(theta), torch.sin(theta)
        nx, ny = c, s_                     # detector axis
        dx, dy = -s_, c                    # ray direction (unit vector)
        s = self._s.reshape(-1, 1)         # (n_det, 1)
        ax = self._axis.reshape(1, -1)     # (1, n_img)

        if float(dx.abs()) >= float(dy.abs()):
            # x as the driving axis: set p_x = ax, solve for t, then compute p_y
            t = (ax - s * nx) / dx
            gx = ax.expand(self.n_det, self.n_img)
            gy = s * ny + t * dy
            drive = dx
        else:
            t = (ax - s * ny) / dy
            gy = ax.expand(self.n_det, self.n_img)
            gx = s * nx + t * dx
            drive = dy

        # grid_sample's grid[..., 0] is x (width direction), [..., 1] is y (height direction)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        ds = self.px_cm / float(drive.abs()) * self.unit_scale
        return grid, ds

    @property
    def in_shape(self):
        return (self.n_img, self.n_img)

    @property
    def out_shape(self):
        return (len(self.angles), self.n_det)

    def __call__(self, x):
        return self.project(x)

    def project(self, x):
        x = x.reshape(1, 1, self.n_img, self.n_img).to(self.device, self.dtype)
        rows = []
        for g, ds in zip(self._grids, self._ds):
            v = F.grid_sample(x, g, align_corners=False, padding_mode="zeros")
            rows.append(v.reshape(self.n_det, self.n_img).sum(dim=1) * ds)
        return torch.stack(rows, dim=0)

    def backproject(self, y):
        """A^T y -- VJP, exact by definition."""
        y = y.reshape(*self.out_shape).to(self.device, self.dtype)
        x0 = torch.zeros(self.n_img, self.n_img, device=self.device,
                         dtype=self.dtype, requires_grad=True)
        out = self.project(x0)
        g, = torch.autograd.grad(out, x0, grad_outputs=y)
        return g.reshape(self.n_img, self.n_img)
