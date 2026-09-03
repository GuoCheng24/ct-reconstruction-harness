"""Proximal operators -- the raw material of L2.

    prox_{t g}(v) = argmin_x  g(x) + (1/(2t)) ||x - v||^2

## Why proximal operators deserve a file of their own

The unifying view of modern splitting algorithms is the **monotone inclusion**: find 0 in
A(x) + B(x). In that view a gradient step is an explicit step on B and a proximal step is an
implicit step on A, and FISTA / ADMM / Chambolle-Pock / Condat-Vu differ only in **how the
operator is split and under which metric**.
(Standard survey: Condat, Kitahara, Contreras, Hirabayashi,
 "Proximal Splitting Algorithms: A Tour of Recent Advances, with New Twists", 2020.)

So a proximal operator is not "one step of some algorithm", it is the shared building block of
this whole family. Get it right and every downstream solver reuses it; get it wrong and every
downstream solver is wrong in a way nobody can see.

## The acceptance criterion for each prox

A proximal operator has one hard property that can be checked numerically, independently of g:

  **a prox is firmly nonexpansive**, and therefore 1-Lipschitz:
      ||prox(u) - prox(v)||^2 + ||(u - prox(u)) - (v - prox(v))||^2 <= ||u - v||^2

If that does not hold, it is definitely wrong. It is far stronger than "does the result look
right", because it is an identity rather than an impression.
guards/prox_check.py runs it on every prox.
"""
import torch

__all__ = ["prox_l1", "prox_l2sq", "prox_box", "prox_nonneg", "prox_nuclear",
           "proj_l2ball", "moreau_check"]


def prox_l1(v, t):
    """g(x) = ||x||_1  ->  soft thresholding."""
    return torch.sign(v) * torch.clamp(v.abs() - t, min=0.0)


def prox_l2sq(v, t, y=None):
    """g(x) = (1/2)||x - y||^2  ->  shrink towards y. y=None is treated as 0."""
    y = torch.zeros_like(v) if y is None else y
    return (v + t * y) / (1.0 + t)


def prox_box(v, t, lo=0.0, hi=1.0):
    """g = indicator of [lo,hi]  ->  pointwise clamp. t is unused (the prox of an indicator does not depend on t)."""
    return torch.clamp(v, lo, hi)


def prox_nonneg(v, t):
    return torch.clamp(v, min=0.0)


def proj_l2ball(v, radius, center=None):
    """Projection onto {x : ||x - c|| <= r}. Data fidelity is often written as this constraint."""
    c = torch.zeros_like(v) if center is None else center
    d = v - c
    nrm = torch.linalg.vector_norm(d)
    if float(nrm) <= radius:
        return v.clone()
    return c + d * (radius / nrm)


def prox_nuclear(V, t):
    """g(X) = ||X||_*  ->  soft threshold the singular values. Used for low-rank priors."""
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    return (U * torch.clamp(S - t, min=0.0).unsqueeze(-2)) @ Vh


def moreau_check(prox, v, t, conj_prox=None):
    """Moreau decomposition: v = prox_{t g}(v) + t * prox_{g* / t}(v / t).

    If the prox of the conjugate is supplied, this identity is a second independent correctness
    check -- it tests the relation between the prox and its dual object, and exposes errors that
    are invisible when looking at the prox alone.
    """
    if conj_prox is None:
        return None
    return float(torch.linalg.vector_norm(v - prox(v, t) - t * conj_prox(v / t, 1.0 / t)))
