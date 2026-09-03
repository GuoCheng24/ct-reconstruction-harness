"""Regularizer family -- the variant space of the first generate-and-filter round (the Adam
route: anything differentiable can be plugged in).

The structural advantage of the TVAdam route pays off here: R in "data term + γ·R(x)" can be
swapped for anything differentiable or subdifferentiable, with no prox to derive. Every
regularizer comes with a reason for "why it might win" -- candidates are generated from strong
priors rather than at random -- and passes a single-image sanity check before entering the loop.

Candidates and their reasons:
  aniso_tv    official baseline (control)
  iso_tv      isotropic -- the official anisotropic choice is an implementation preference;
              isotropic is rotation invariant and often better in CT
  huber_tv    quadratic for small gradients -- no staircasing artefacts in flat regions, which
              is TV's number one weakness
  tgv2        second-order TGV -- an auxiliary field v separates first from second order, far
              better than TV on linear gradients (soft tissue); Adam optimizes (x,v) jointly,
              which is hard to do in prox splitting and free in Adam
  haar_l1     Haar wavelet L1 -- multiscale sparsity, complementary to TV (TV penalizes total
              edge length, wavelets penalize multiscale coefficients)
  tv_haar     the combination -- TV for edges, wavelets for texture
Data term variants:
  poisson     official (control)
  wls         inverse-variance weighted MSE = second-order expansion of Poisson; if the two tie,
              the value of Poisson lies only in the weights
"""
import math
import torch

__all__ = ["REGULARIZERS", "DATA_TERMS", "make_regularizer"]


def aniso_tv(x):
    return (torch.abs(x[..., :, 1:] - x[..., :, :-1]).sum()
            + torch.abs(x[..., 1:, :] - x[..., :-1, :]).sum())


def iso_tv(x, eps=1e-8):
    dx = x[..., :-1, 1:] - x[..., :-1, :-1]
    dy = x[..., 1:, :-1] - x[..., :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + eps).sum()


def huber_tv(x, delta=0.005):
    dx = x[..., :-1, 1:] - x[..., :-1, :-1]
    dy = x[..., 1:, :-1] - x[..., :-1, :-1]
    m = torch.sqrt(dx * dx + dy * dy + 1e-12)
    quad = 0.5 * m * m / delta
    lin = m - 0.5 * delta
    return torch.where(m <= delta, quad, lin).sum()


def haar_l1(x, levels=3):
    """L1 of the detail coefficients of a multilevel Haar decomposition. The crop to an even size is 362->360; the 2px border is not counted."""
    total = x.new_zeros(())
    cur = x[..., :360, :360]
    for _ in range(levels):
        a = 0.5 * (cur[..., ::2, :] + cur[..., 1::2, :])
        d = 0.5 * (cur[..., ::2, :] - cur[..., 1::2, :])
        aa = 0.5 * (a[..., :, ::2] + a[..., :, 1::2])
        ad = 0.5 * (a[..., :, ::2] - a[..., :, 1::2])
        da = 0.5 * (d[..., :, ::2] + d[..., :, 1::2])
        dd = 0.5 * (d[..., :, ::2] - d[..., :, 1::2])
        total = total + ad.abs().sum() + da.abs().sum() + dd.abs().sum()
        cur = aa
    return total


class TGV2:
    """Second-order TGV: alpha1 ||grad x - v||_1 + alpha0 ||E(v)||_1, optimizing (x, v) jointly.

    v is the auxiliary vector field (2 channels) and E is the symmetrized gradient.
    ratio = alpha0/alpha1, commonly taken as 2 in the literature.
    As a "regularizer with parameters of its own", it exposes the (R(x), params) interface via
    make().
    """

    def __init__(self, shape, device, ratio=2.0):
        self.v = torch.zeros(2, *shape, device=device, requires_grad=True)
        self.ratio = ratio

    def params(self):
        return [self.v]

    def __call__(self, x):
        dx = x[..., :-1, 1:] - x[..., :-1, :-1]
        dy = x[..., 1:, :-1] - x[..., :-1, :-1]
        vx = self.v[0][..., :-1, :-1]
        vy = self.v[1][..., :-1, :-1]
        first = (torch.abs(dx - vx) + torch.abs(dy - vy)).sum()
        # the four components of E(v) (forward differences, interior)
        v0, v1 = self.v[0], self.v[1]
        e11 = v0[..., :-1, 1:] - v0[..., :-1, :-1]
        e22 = v1[..., 1:, :-1] - v1[..., :-1, :-1]
        e12 = 0.5 * ((v0[..., 1:, :-1] - v0[..., :-1, :-1])
                     + (v1[..., :-1, 1:] - v1[..., :-1, :-1]))
        second = (e11.abs() + e22.abs() + 2 * e12.abs()).sum()
        return first + self.ratio * second


def make_regularizer(name, shape=(362, 362), device="cuda:0", **kw):
    """Return (reg_fn, extra_params). TGV carries auxiliary field parameters, the rest carry none.

    kw: huber_tv takes delta (default 0.005); tgv2 takes ratio (default 2.0).
    The second loop round promoted these two previously hard-coded constants into sweep
    dimensions.
    """
    if name == "tgv2":
        r = TGV2(shape, device, ratio=float(kw.get("ratio", 2.0)))
        return r, r.params()
    if name == "huber_tv":
        d = float(kw.get("delta", 0.005))
        return (lambda x: huber_tv(x, delta=d)), []
    fns = {"aniso_tv": aniso_tv, "iso_tv": iso_tv,
           "haar_l1": haar_l1,
           "tv_haar": lambda x: aniso_tv(x) + 0.5 * haar_l1(x)}
    return fns[name], []


REGULARIZERS = ["aniso_tv", "iso_tv", "huber_tv", "tgv2", "haar_l1", "tv_haar"]
DATA_TERMS = ["poisson", "wls"]


def wls_loss(y_pred, y_true, mu_max=81.35858, n0=4096.0):
    """Inverse-variance weighted MSE (second-order expansion of Poisson). Var(post-log) ≈ exp(y·mu)/N0/mu^2."""
    w = torch.exp(-y_true * mu_max) * n0 * mu_max * mu_max
    return 0.5 * torch.sum(w * (y_pred - y_true) ** 2)
