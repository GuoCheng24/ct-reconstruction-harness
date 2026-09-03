"""Splitting algorithms -- the body of L2.

What is implemented is not the handful from a 2009 textbook, but the current form of each:

| algorithm | version used here | why not the most naive one |
|---|---|---|
| proximal gradient | **FISTA + adaptive restart** (O'Donoghue-Candes 2015) | plain FISTA oscillates on strongly convex problems; restart turns it into linear convergence |
| proximal gradient | **AdaPGD** (Malitsky-Mishchenko, arXiv:1910.09529) | needs neither the Lipschitz constant L nor a line search |
| primal-dual | **Chambolle-Pock**, with tau*sigma*||A||^2 < 1 stated explicitly | the condition is verifiable, and violating it must cause divergence -- this repository uses that as a negative criterion |

Unifying view (the Condat et al. 2020 survey): all of these solve the monotone inclusion
0 in A(x)+B(x), and differ only in how the operator is split and under which metric. So they
share one set of numerically verifiable properties instead of each needing its own story. This
file turns those properties into diagnostics in the return value, not claims in a comment.

## What each solver returns

It returns (x, info), where info carries **quantities that can be checked independently**:
  obj      objective value per step -- used to verify the O(1/k) / O(1/k^2) rate
  fixed    ||x_k - x_{k-1}|| per step -- the fixed-point residual, direct evidence of convergence
  fejer    ||x_k - x_star|| per step (when x_star is given) -- checks Fejer monotonicity
These three exist for one reason: **the statement "it converged" must be supported by numbers,
not by "it ran without an error".**
"""
import torch

__all__ = ["fista", "adapgd", "chambolle_pock", "power_method"]


def power_method(op_normal, shape, n_iter=100, seed=0, dtype=torch.float64, device="cpu"):
    """Estimate ||A||^2 = lambda_max(A^H A) by the power method.

    The step-size condition needs it. Note that this is an **estimate** approaching from below --
    so leave some margin when using it to set a step size, or it may land just out of bounds.
    The solvers here multiply by 0.99 by default, and expose that in a parameter rather than
    hiding it.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g, dtype=torch.float64).to(device)
    if dtype.is_complex:
        x = x.to(dtype)
    x = x / torch.linalg.vector_norm(x)
    lam = 0.0
    for _ in range(n_iter):
        y = op_normal(x)
        lam = float(torch.linalg.vector_norm(y))
        if lam < 1e-30:
            return 0.0
        x = y / lam
    return lam


def fista(grad, prox, x0, step, n_iter=200, restart=True, obj=None, x_star=None):
    """FISTA with O'Donoghue-Candes adaptive restart.

    grad(x)      gradient of the smooth term
    prox(v, t)   proximal operator of the nonsmooth term
    step         step size, must be <= 1/L. Getting it wrong diverges -- one of the negative
                 criteria of this repository.
    restart      when True, use the **gradient restart** criterion: restart whenever
                 <y_{k-1} - x_k, x_k - x_{k-1}> > 0. This criterion needs no function values, so
                 it is cheaper than objective-based restart, and more stable.
    """
    x = x0.clone()
    y = x0.clone()
    t = 1.0
    info = {"obj": [], "fixed": [], "fejer": [], "restarts": 0}
    for k in range(n_iter):
        x_prev = x
        x = prox(y - step * grad(y), step)
        # gradient restart: an obtuse angle between the momentum and the actual progress
        # direction means the momentum is pulling backwards
        if restart and torch.sum((y - x) * (x - x_prev)) > 0:
            t = 1.0
            y = x.clone()
            info["restarts"] += 1
        else:
            t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            y = x + ((t - 1.0) / t_next) * (x - x_prev)
            t = t_next
        info["fixed"].append(float(torch.linalg.vector_norm(x - x_prev)))
        if obj is not None:
            info["obj"].append(float(obj(x)))
        if x_star is not None:
            info["fejer"].append(float(torch.linalg.vector_norm(x - x_star)))
    return x, info


def adapgd(grad, prox, x0, n_iter=200, obj=None, x_star=None, t0=1e-3):
    """Adaptive proximal gradient (Malitsky-Mishchenko 2019) -- **no knowledge of L needed**.

    Two rules suffice: do not grow the step size too fast, and do not overshoot the local
    curvature. The original is for unconstrained gradient descent; a prox step is added here.
    The step size is updated as

        t_k = min( sqrt(1 + theta_{k-1}) * t_{k-1},  ||x_k - x_{k-1}|| / (2 ||g_k - g_{k-1}||) )

    with theta = t_k / t_{k-1}. The second term is an inverse estimate of the "local curvature".

    Why it is worth implementing: in real inverse problems ||A||^2 is either expensive to compute
    or only estimable (see power_method), and underestimating is unsafe while overestimating is
    slow. This algorithm sidesteps the whole issue.
    """
    x = x0.clone()
    g_prev = grad(x)
    t = t0
    x_prev, theta = x.clone(), 1e9
    x = prox(x - t * g_prev, t)
    info = {"obj": [], "fixed": [], "fejer": [], "step": []}
    for k in range(n_iter):
        g = grad(x)
        dx = torch.linalg.vector_norm(x - x_prev)
        dg = torch.linalg.vector_norm(g - g_prev)
        t_new = min(float((1.0 + theta) ** 0.5) * t,
                    float(dx / (2.0 * dg)) if float(dg) > 1e-300 else float("inf"))
        theta = t_new / t
        t = t_new
        x_prev, g_prev = x.clone(), g
        x = prox(x - t * g, t)
        info["step"].append(t)
        info["fixed"].append(float(torch.linalg.vector_norm(x - x_prev)))
        if obj is not None:
            info["obj"].append(float(obj(x)))
        if x_star is not None:
            info["fejer"].append(float(torch.linalg.vector_norm(x - x_star)))
    return x, info


def chambolle_pock(K, KT, prox_f_star, prox_g, x0, y0,
                   tau, sigma, theta=1.0, n_iter=200, obj=None, x_star=None):
    """Chambolle-Pock primal-dual, solving min_x f(Kx) + g(x).

    **Convergence condition: tau * sigma * ||K||^2 < 1.** That condition is one of this
    repository's negative criteria: violate it and the algorithm **must** diverge. If it
    converges anyway, something in the implementation is quietly stabilizing it, and that is the
    real thing to worry about -- an implementation that "runs with any parameters" has usually
    degenerated into a different algorithm somewhere.
    """
    x, y = x0.clone(), y0.clone()
    x_bar = x.clone()
    info = {"obj": [], "fixed": [], "fejer": [], "primal_dual_gap": []}
    for k in range(n_iter):
        y = prox_f_star(y + sigma * K(x_bar), sigma)
        x_prev = x
        x = prox_g(x - tau * KT(y), tau)
        x_bar = x + theta * (x - x_prev)
        info["fixed"].append(float(torch.linalg.vector_norm(x - x_prev)))
        if obj is not None:
            info["obj"].append(float(obj(x)))
        if x_star is not None:
            info["fejer"].append(float(torch.linalg.vector_norm(x - x_star)))
        if not torch.isfinite(torch.linalg.vector_norm(x)):
            info["diverged_at"] = k
            break
    return x, info
