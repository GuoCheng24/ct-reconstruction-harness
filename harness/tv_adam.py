"""The same TV baseline as the official one: TVAdam -- a reimplementation following the
dival.TVAdamCTReconstructor spec.

The complete recipe behind the official TV 33.36 / 0.830 (reconstructed on 2026-08-27 by reading
the dival source and downloading the official hyperparameter json):

    init        x0 = FBP(y; Hann, frequency_scaling=0.1)     <- strongly low-passed FBP warm start
    optimizer   Adam(lr=1e-3), **iterations=5000**, tracking the best-loss iterate
    objective   **poisson_loss**(A x, y)  +  **gamma=20.556** * tv_aniso(x)
    constraints none (not even nonnegativity)

Three layers of truth, uncovered one at a time, each of which changed the numbers:
  1. it is not prox splitting, it is Adam optimizing directly (from the source);
  2. the data term is not MSE but the **Poisson NLL** (found only after downloading the official
     hyperparameter json) -- it weights bins automatically by noise variance, and the "per-sample
     optimal γ drifts by an order of magnitude" seen with MSE tightens sharply under Poisson
     (hardest sample #1: 29.02 with MSE -> 31.40 with Poisson);
  3. γ=20.556 goes with the Poisson sum form, and the units line up exactly with ours
     (post-log/mu_max normalization, N0=4096), so it transfers directly -- unlike the MSE version
     where γ was off by three orders of magnitude.
Hyperparameter tuning is allowed only on the [calibration file file#1]; numbers are reported
only on file#0 (run_tvadam_eval.py).
"""
import math
import torch
from torch.optim import Adam

__all__ = ["TVAdamReconstructor", "poisson_loss"]

MU_MAX = 81.35858
N0 = 4096.0


def poisson_loss(y_pred, y_true, photons=N0, mu_max=MU_MAX):
    """Poisson NLL (an exact reimplementation of dival.util.torch_losses.poisson_loss).

    The loss behind the official TV 33.36 is **not MSE** (discovered on 2026-08-27 only after
    downloading the official hyperparameter json: gamma=20.556, iterations=5000,
    loss_function="poisson"). It converts post-log values back to the count domain and weights
    them by the Poisson likelihood -- highly attenuated bins have few photons and more noise, so
    their weight drops automatically. MSE weights all bins equally, is therefore dragged down by
    the noisy bins, and its per-sample optimal γ drifts.
        loss = sum( N_pred - N_true * log N_pred ),  N(y) = N0 * exp(-y*mu_max)
    """
    n_pred = torch.exp(-y_pred * mu_max) * photons
    n_true = torch.exp(-y_true * mu_max) * photons
    n_pred_log = -y_pred * mu_max + math.log(photons)
    return torch.sum(n_pred - n_true * n_pred_log)


def tv_aniso(x, reduction="sum"):
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    if reduction == "sum":
        return dh.sum() + dw.sum()
    return dh.mean() + dw.mean()


class TVAdamReconstructor:
    """method(obs) -> recon; fbp_init is a callable obs->image implementing FBP."""

    def __init__(self, A, fbp_init, gamma=1e-4, lr=1e-3, iterations=2000,
                 reduction="sum", loss="mse"):
        self.A, self.fbp = A, fbp_init
        self.gamma, self.lr = float(gamma), float(lr)
        self.iterations = int(iterations)
        self.reduction = reduction
        self.loss = loss

    def __call__(self, obs, trace=None):
        x = self.fbp(obs).detach().clone().requires_grad_(True)
        opt = Adam([x], lr=self.lr)
        mse = torch.nn.MSELoss()
        data = ((lambda p, o: mse(p, o)) if self.loss == "mse"
                else (lambda p, o: poisson_loss(p, o)))
        best_loss, best_x = float("inf"), x.detach().clone()
        for i in range(self.iterations):
            opt.zero_grad()
            loss = data(self.A.project(x), obs) + self.gamma * tv_aniso(x, self.reduction)
            loss.backward()
            opt.step()
            l = float(loss.detach())
            if l < best_loss:
                best_loss, best_x = l, x.detach().clone()
            if trace is not None and (i % trace[0] == 0 or i == self.iterations - 1):
                trace[1](i, best_x)
        return best_x
