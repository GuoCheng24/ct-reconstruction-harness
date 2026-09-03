"""Adjoint test: the entry certificate for every operator on the constructive leg.

## Why this has to be the first guard

Every gradient- or duality-based solver downstream -- FISTA uses A^T(Ax-y) as the gradient,
ADMM and Chambolle-Pock use A^T directly in the dual step -- assumes at every step of the
derivation that A^T really is the adjoint of A. If it is not:

  * the code **raises no error**
  * the iteration **still converges to something**
  * the output **still looks like a reconstruction**

Except that something is not the solution of any variational problem. This is the most
dangerous class of failure: no error, real output, entirely wrong conclusions.
The first operator in this repository (ops/radon.py) was wrong in exactly this way in its
first version, with an adjoint error of 0.49 -- and it ran without a hint of trouble.

## Why random vectors

Structured inputs (all-ones, a unit impulse, symmetric patterns) can make the two sides agree
by coincidence and hide a genuine mismatch. Random vectors do not have that coincidence.
Use several of them and take the worst case.

## How the threshold is set

In float64, a correct implementation gives a relative error of order 1e-15 to 1e-13 (growing
slowly with operator size). The default threshold is 1e-10 -- three orders of magnitude looser
than a correct implementation, and two orders of magnitude tighter than any real adjoint bug
(typically 1e-2 or above). **The band in between is empty**, so this threshold needs no tuning.

float32 is unsuitable for this test: its own relative error sits at 1e-6 to 1e-7, which cannot
be distinguished from a mildly wrong adjoint. Test in float64.
"""
import torch

__all__ = ["adjoint_error", "assert_adjoint", "AdjointFailure"]

DEFAULT_TOL = 1e-10


class AdjointFailure(AssertionError):
    pass


def adjoint_error(forward, adjoint, in_shape, out_shape,
                  n_trials=8, seed=0, device="cpu", dtype=torch.float64):
    """Return max |<Ax,y> - <x,A^T y>| / |<Ax,y>|.

    forward/adjoint are two callables; passing functions rather than operator objects lets us
    test arbitrary combinations (including deliberately feeding in a wrong adjoint to verify
    that this guard itself fails).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    worst, detail = 0.0, None
    for t in range(n_trials):
        x = torch.randn(*in_shape, generator=g, dtype=dtype).to(device)
        y = torch.randn(*out_shape, generator=g, dtype=dtype).to(device)
        lhs = float(torch.sum(forward(x) * y))
        rhs = float(torch.sum(x * adjoint(y)))
        denom = max(abs(lhs), 1e-30)
        rel = abs(lhs - rhs) / denom
        if rel > worst:
            worst, detail = rel, (t, lhs, rhs)
    return worst, detail


def assert_adjoint(forward, adjoint, in_shape, out_shape, tol=DEFAULT_TOL, **kw):
    """Raise if the test fails, printing both numbers -- no room for "close enough"."""
    err, detail = adjoint_error(forward, adjoint, in_shape, out_shape, **kw)
    if err > tol:
        t, lhs, rhs = detail
        raise AdjointFailure(
            "adjoint test failed: relative error %.3e > tolerance %.0e\n"
            "  random vector set %d: <Ax,y> = %.12g  but <x,A^T y> = %.12g\n"
            "  A^T is not the adjoint of A. Check: was it written as the geometric inverse (rather than the transpose of the implementation)?\n"
            "  Was filtered backprojection (iradon) mistakenly used as the adjoint?" % (err, tol, t, lhs, rhs))
    return err
