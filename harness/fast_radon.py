"""Batched GPU forward projection + VJP adjoint -- the engine behind the variational methods
(TV / PnP / unrolled) in the harness.

## Why the harness needs its own forward operator

The Backprojector in harness/lodopab.py is a **one-way** fast backprojection meant for FBP,
whereas TV, PnP and unrolled optimization need a pair (A, A^T) that are **mutually adjoint**,
each called once per iteration over hundreds of iterations. The implementation in
ops/radon_joseph.py is CPU float64 with a per-angle Python loop -- verification grade, far too
slow to run inside a loop.

## Design

  * all angles are stacked into one batch, so a single grid_sample does the whole forward pass
    (1000 angles ~ one large kernel);
  * A^T is taken via VJP -- a direct reuse of the lesson from ops/radon.py: **the adjoint is the
    transpose of the implementation, not the inverse of the geometry**; a VJP is exact by
    definition, and the transpose of a gather is automatically a scatter;
  * float32 + GPU: this is a scoring engine, not a numerical verifier. The adjoint self-check
    threshold is set at 1e-4 for float32 (see the quantification in guards/adjoint.py of why
    "float32 is unsuitable for a strict adjoint test").

## Consistency with the real observations

The geometry matches ops/radon_joseph.py (detector span = image diameter, output units =
normalized image x metres), which was already validated against real LoDoPaB observations at a
scale ratio of 1.0072. There is a further runtime self-check, `self_check`, which compares the
scale of a ground-truth image against the real observation and refuses to start if it falls
outside [0.9, 1.1] -- so a future breaking change does not go unnoticed.
"""
import math
import torch

N_ANG, N_DET, N_IMG = 1000, 513, 362
FOV_CM = 26.0


class FastRadon:
    """A: (362,362) -> (n_ang, 513); A^T is given by VJP. float32, GPU."""

    def __init__(self, device="cuda", n_ang=N_ANG, dtype=torch.float32):
        self.device, self.dtype, self.n_ang = device, dtype, int(n_ang)
        ang = torch.arange(self.n_ang, device=device, dtype=dtype) * math.pi / self.n_ang
        # detector bin centres, normalized to [-1,1] over the FOV; span = diameter => x sqrt(2)
        s = torch.linspace(-1, 1, N_DET, device=device, dtype=dtype) * math.sqrt(2.0)
        # sample points along the ray (simplified Joseph: a fixed n_img points spanning the
        # diagonal; slightly less accurate than the per-axis-driven CPU version, but this is a
        # scoring engine and self_check is the backstop)
        t = torch.linspace(-math.sqrt(2.0), math.sqrt(2.0), N_IMG,
                           device=device, dtype=dtype)
        c, si = torch.cos(ang), torch.sin(ang)
        # grid (n_ang, n_det, n_img, 2): p = s*n + t*d, n=(cos,sin), d=(-sin,cos)
        # Row/column convention (2026-08-27, hypothesis eight): the earlier convention
        # (gx<->cos, gy<->sin) was off by a [transpose] against the real LoDoPaB data. Only
        # after all seven geometry/sampling hypotheses had been ruled out did testing A(gt^T)
        # directly crack it -- ground-truth residuals dropped from 18.8/105/78.7/39.0 to
        # 3.1/0.2/0.6/2.5, i.e. down to the level of pure Poisson noise. It also explains the
        # wide spread of per-sample "accuracy" 0.44-0.98: the less symmetric the anatomy, the
        # larger the transpose mismatch; the near-symmetric sample #0 gave a misleading 0.976.
        # Lesson: when the residual couples to the symmetry of the image content, a high
        # correlation on a single sample proves nothing.
        gy = (s.view(1, -1, 1) * c.view(-1, 1, 1)
              + t.view(1, 1, -1) * (-si).view(-1, 1, 1))
        gx = (s.view(1, -1, 1) * si.view(-1, 1, 1)
              + t.view(1, 1, -1) * c.view(-1, 1, 1))
        self.grid = torch.stack([gx, gy], dim=-1)          # (n_ang,n_det,n_img,2)
        # step size: diagonal length / number of samples, cm -> m (the unit LoDoPaB stores on
        # disk, see ops/radon_geom.py)
        self.ds = (FOV_CM * math.sqrt(2.0) / N_IMG) * 0.01

    @property
    def in_shape(self):
        return (N_IMG, N_IMG)

    @property
    def out_shape(self):
        return (self.n_ang, N_DET)

    def project(self, x):
        x4 = x.reshape(1, 1, N_IMG, N_IMG).to(self.device, self.dtype)
        x4 = x4.expand(self.n_ang, 1, N_IMG, N_IMG)
        v = torch.nn.functional.grid_sample(
            x4, self.grid, align_corners=False, padding_mode="zeros")
        return v.sum(dim=-1).reshape(self.n_ang, N_DET) * self.ds

    def adjoint(self, y):
        x0 = torch.zeros(N_IMG, N_IMG, device=self.device, dtype=self.dtype,
                         requires_grad=True)
        out = self.project(x0)
        g, = torch.autograd.grad(out, x0, grad_outputs=y.to(self.device, self.dtype))
        return g

    # ------------------------------------------------------ self-check ----
    def self_check(self, obs0, gt0, adjoint_tol=1e-3, scale_band=(0.9, 1.1)):
        """Two self-checks: the VJP adjoint (float32 threshold) and scale consistency with the
        real observations.

        Rationale for the adjoint threshold 1e-3: in float32 a correct implementation gives
        ~1e-6..1e-4 (growing with size) and a real bug gives >1e-2 -- an order of magnitude of
        empty space still remains in between.
        """
        g = torch.Generator().manual_seed(0)
        x = torch.randn(*self.in_shape, generator=g).to(self.device, self.dtype)
        y = torch.randn(*self.out_shape, generator=g).to(self.device, self.dtype)
        lhs = float(torch.sum(self.project(x) * y))
        rhs = float(torch.sum(x * self.adjoint(y)))
        adj_err = abs(lhs - rhs) / max(abs(lhs), 1e-30)
        if adj_err > adjoint_tol:
            raise RuntimeError(f"FastRadon adjoint self-check failed: {adj_err:.2e} > {adjoint_tol}")
        mine = self.project(gt0.to(self.device, self.dtype))
        ratio = float(obs0.abs().mean() / mine.abs().mean().clamp_min(1e-30))
        if not (scale_band[0] < ratio < scale_band[1]):
            raise RuntimeError(
                f"FastRadon scale ratio against the real observation {ratio:.3f} is outside the band {scale_band} -- "
                "geometry or units have been broken; refusing to start")
        # Third check: the ground-truth residual must land at the order of the noise level. The
        # scale ratio cannot catch a geometric transpose -- a transpose barely changes mean|.|,
        # so 1.0099 passes anyway, while the ground-truth residual is off by a factor of 6-485.
        resid = float((mine - obs0.to(self.device, self.dtype)).pow(2).sum())
        if resid > 30.0:      # Poisson noise level ~0.2-3; 19-105 under transpose mismatch
            raise RuntimeError(
                f"residual of A(gt) against the real observation {resid:.1f} far exceeds the noise level (<30) -- "
                "geometry mismatch (check whether the row/column convention was reverted); refusing to start")
        return {"adjoint_err": adj_err, "scale_ratio": ratio, "gt_resid": resid}
