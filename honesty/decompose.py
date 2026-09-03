"""L4 honesty layer: within a reconstruction, which part is supported by the data and which part
was invented by the prior.

## Why this layer exists

The operator A of an inverse problem usually has a nontrivial null space. For any n in null(A),

    A(x + n) = A x

**the measurement cannot distinguish x from x+n at all.** So whatever part of a reconstruction
lies in null(A) was not recovered from the data -- it was chosen by the regularizer, the prior,
or the network weights. A reconstruction that looks clean, sharp and diagnostically useful can
have a substantial share of its structure there.

People doing reconstruction usually do not ask this, and it is exactly the question we should
ask. The shape we keep running into is an overestimated target (for instance an oracle upper
bound often quoted at 72-94% that turns out to be selection bias). Structure in the null space
is the same thing in the reconstruction domain: **it looks like information, and is in fact what
you put there yourself**.

## The decomposition

    x = P_row(x) + P_null(x),   row(A) = range(A^H), the orthogonal complement of null(A)

P_row(x) is determined by the data (within what noise and conditioning allow); P_null(x) is
determined entirely by the prior.

## How to compute it (no need to build A, and no SVD)

Key property: **CGLS started from 0 to solve min ||A z - b|| stays inside row(A) for the whole
iteration**, converging to the minimum-norm solution = A^+ b. So taking b = A x:

    P_row(x) = A^+ (A x)   obtained by CGLS started from 0
    P_null(x) = x - P_row(x)

Only matrix-vector products with A and A^H are needed, matching the L1 operator interface of
this repository.

## Limits that must be stated honestly

1. **Numerical null space vs exact null space**. A finite number of CGLS iterations only covers
   the Krylov subspace; directions with very small singular values are numerically hard to tell
   from the null space, and **physically they are indeed almost unconstrained by the data too**.
   So what this module reports is "the part the data does not actually constrain under a given
   iteration budget", which is closer to practice than the "exact null space" -- but which of
   the two it is has to be spelled out.
2. **A large P_null does not mean the reconstruction is wrong**. The structure the prior fills in
   may well be correct (encoding real knowledge is what a prior is for). This module answers
   "this part is **not what the data said**", not "this part is wrong". Conflating the two turns
   into another kind of overclaiming.
"""
import torch

__all__ = ["cgls", "split_row_null", "null_report"]


def cgls(A_apply, A_adj, b, x0=None, n_iter=60, tol=1e-12, callback=None):
    """CGLS solving min ||A x - b||, started from x0 (0 by default).

    Started from 0 the iterates stay inside row(A), which is exactly why this module uses it as
    a projection.
    """
    x = torch.zeros_like(A_adj(b)) if x0 is None else x0.clone()
    r = b - A_apply(x)
    s = A_adj(r)
    p = s.clone()
    gamma = float(torch.sum(s.conj() * s).real)
    for k in range(n_iter):
        q = A_apply(p)
        qq = float(torch.sum(q.conj() * q).real)
        if qq <= 0:
            break
        alpha = gamma / qq
        x = x + alpha * p
        r = r - alpha * q
        s = A_adj(r)
        gamma_new = float(torch.sum(s.conj() * s).real)
        if callback is not None:
            callback(k, x, gamma_new)
        if gamma_new <= tol * max(gamma, 1e-300):
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x


def split_row_null(A_apply, A_adj, x, n_iter=80):
    """Split x into (the part the data can determine, the part the prior determines)."""
    b = A_apply(x)
    x_row = cgls(A_apply, A_adj, b, n_iter=n_iter)
    return x_row, x - x_row


def null_report(A_apply, A_adj, x_hat, n_iter=80, mask=None):
    """Quantitative honesty report for one reconstruction.

    mask is optional, e.g. "the region inside the body", used to report inside and outside
    separately -- prior structure in air regions is usually harmless, whereas prior structure
    inside a lesion is the thing to worry about.
    """
    x_row, x_null = split_row_null(A_apply, A_adj, x_hat, n_iter=n_iter)
    e_all = float(torch.sum(x_hat.conj() * x_hat).real)
    e_null = float(torch.sum(x_null.conj() * x_null).real)
    out = {
        "energy_total": e_all,
        "energy_null": e_null,
        "null_fraction": e_null / max(e_all, 1e-300),
        "x_row": x_row,
        "x_null": x_null,
    }
    if mask is not None:
        m = mask.to(x_hat.dtype)
        ein = float(torch.sum((x_hat * m).conj() * (x_hat * m)).real)
        enin = float(torch.sum((x_null * m).conj() * (x_null * m)).real)
        out["null_fraction_in_mask"] = enin / max(ein, 1e-300)
    lines = [
        "reconstruction honesty decomposition   %d CGLS iterations" % n_iter,
        "",
        "  energy fraction in directions the data cannot determine   %.1f%%" % (100 * out["null_fraction"]),
    ]
    if mask is not None:
        lines.append("  same fraction inside the mask                             %.1f%%"
                     % (100 * out["null_fraction_in_mask"]))
    lines += [
        "",
        "  This part was not recovered from the measurement; it was chosen by the regularizer / prior / network weights.",
        "  It is not necessarily wrong -- but it is not what the data said.",
    ]
    out["text"] = "\n".join(lines)
    return out
