"""TV reconstruction baseline -- the harness's second reference method (target: official TV
33.36 / 0.830).

## Why this file has to exist (rather than living inside an experiment script)

This algorithm was improvised twice inside heredoc experiment scripts, and both times it came
out broken: once by stacking a dual warm start on a wrong understanding (PSNR 31.9 -> 18.3), and
once by transcribing the inner sign as + instead of - (PSNR -> negative).
**There is only one verified implementation, it must exist in exactly one place, and changes may
only go through an A/B.**

## Implementation (the version validated on real LoDoPaB data on 2026-08-27)

Outer FISTA (gradient restart), inner Chambolle 2004 dual projection solving prox_{w·TV}, with
the dual variable zeroed at every outer step.
Inner update: p <- Pi_{|p|<=1}( p - 0.245 * grad(v - w * div p) )
This form makes the dual step conservative when w is small (slow convergence), but it is the
version measured at 31.94/0.859.

## Known but unresolved (recorded here, do not treat as solved)

Suspected semi-convergence: 31.94 at n_outer=150, 18.28 at 300 (warm-start version). The number
at 300 is contaminated by the warm-start bug, so "the deeper the iteration, the more it slides
towards the LS solution (9 dB)" is still only a hypothesis.
The depth sweep in tv_selfcheck() exists to answer it -- if PSNR falls monotonically with depth,
insufficient TV strength is confirmed, and the correct fix is to re-derive the inner step size
from the standard Chambolle form rather than forcing λ to compensate.
"""
import math
import torch

__all__ = ["TVReconstructor", "tv_selfcheck"]


def _grad(u):
    gx = torch.zeros_like(u); gy = torch.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div(px, py):
    d = torch.zeros_like(px)
    d[:, :-1] += px[:, :-1]; d[:, 1:] -= px[:, :-1]
    d[:-1, :] += py[:-1, :]; d[1:, :] -= py[:-1, :]
    return d


def _prox_tv(v, w, n_inner, dual_step=None):
    """min_u 0.5||u-v||^2 + w TV(u), isotropic, nonnegative; the dual starts from zero.

    dual_step: the standard Chambolle 2004 dual update is
        p <- Pi_{|p|<=1}( p + tau * grad(div p - v/w) ),  tau <= 1/8
    Rewriting with u = v - w*div p gives p <- Pi( p - (tau/w) * grad u ).
    So the correct coefficient is **tau/w**, not a constant. The first version hard-coded 0.245
    (treating it as an absolute coefficient), which at w=step*lam~5e-3 is only ~1/92 of the
    standard step -- the prox was essentially not solved and the effective TV strength was a few
    percent of the nominal λ, while the outer loop converged all the same (to a problem with
    "almost no TV"). Symptoms: the λ sweep is insensitive and PSNR sticks at 31.94. See the
    commit message for the measured A/B.
    """
    ds = (0.124 / max(w, 1e-9)) if dual_step is None else dual_step
    px = torch.zeros_like(v); py = torch.zeros_like(v)
    for _ in range(n_inner):
        u = v - w * _div(px, py)
        gx, gy = _grad(u)
        px = px - ds * gx
        py = py - ds * gy
        nr = torch.sqrt(px ** 2 + py ** 2).clamp(min=1.0)
        px = px / nr; py = py / nr
    return (v - w * _div(px, py)).clamp(min=0)


class TVReconstructor:
    """method(obs) -> recon, conforming to the harness interface."""

    def __init__(self, A, lam=1e-3, n_outer=150, n_inner=30, L2=0.1794,
                 dual_step=None):
        self.A, self.lam = A, float(lam)
        self.n_outer, self.n_inner = int(n_outer), int(n_inner)
        self.step = 0.99 / L2
        self.dual_step = dual_step

    def __call__(self, obs):
        A, step, lam = self.A, self.step, self.lam
        x = torch.zeros(A.in_shape, device=obs.device, dtype=obs.dtype)
        z = x.clone(); t = 1.0
        for _ in range(self.n_outer):
            g = A.adjoint(A.project(z) - obs)
            xn = _prox_tv(z - step * g, step * lam, self.n_inner, self.dual_step)
            if torch.sum((z - xn) * (xn - x)) > 0:
                t = 1.0; z = xn.clone()
            else:
                tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
                z = xn + ((t - 1) / tn) * (xn - x); t = tn
            x = xn
        return x


def tv_selfcheck(A, obs0, gt0, psnr_fn):
    """Depth sweep: answers "is 31.94 an early-stopping effect". Returns {n_outer: psnr}."""
    out = {}
    for n in (75, 150, 300, 600):
        r = TVReconstructor(A, lam=1e-3, n_outer=n)(obs0)
        out[n] = psnr_fn(r, gt0, "adaptive")
    return out
