"""Measure the inverse crime, instead of just name-dropping it.

## What the inverse crime is

Simulate the data with a discrete operator A (y = A x + noise), then invert with that same A.
The modelling error is artificially removed, and reconstruction quality comes out well above
what is achievable in reality -- because the real measurement process is never equal to the
discrete matrix you wrote down. This is the most famous way to fool yourself in inverse
problems; every textbook mentions it, and then **almost nobody measures how large it is**.

## The three tiers measured here

  A. Crime tier   y = A_same x, inverted with A_same       -- same discretization, same geometry
  B. Half-crime   y = A_fine x, inverted with A_coarse     -- different discretization, same physics
  C. Real tier    y = real LoDoPaB low-dose projections, inverted with A -- my operator never
                  took part in generating the data

Put the three numbers side by side and the gap is exactly "how much I would overstate myself
if I only ever ran tier A".

## Why this matters here

The reason this repository exists is to build up the constructive side, and the easiest fake
result that side produces is tier A. The failure shape that keeps recurring is an overestimated
target: a quantity that looks like an upper bound but is really a target you set for yourself
(for instance an oracle bound quoted at 72-94% that turns out to be selection bias). The
inverse crime is the same thing in the reconstruction domain. Measuring it is the job.

## Honest scope

The LoDoPaB observations are themselves **low-dose projections simulated from real patient CT
images** (Poisson noise + known geometry), not scanner raw data. So tier C is one notch stronger
than A/B (real anatomy + real noise model + independent of my discretization), but it is **not
fully real raw data**. Truly raw data would mean fastMRI (registration required). This caveat
must be stated anywhere results from this file are used.
"""
import sys, os, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ops"))
sys.path.insert(0, os.path.join(HERE, "..", "solvers"))
from radon import RadonParallel                      # noqa: E402
from splitting import fista, power_method            # noqa: E402
from prox import prox_nonneg                         # noqa: E402

DATA = os.environ.get("LODOPAB_DIR", "./data/lodopab")


def load_pair(idx=0, file_idx=0):
    """Return (ground-truth image 362x362, real observed sinogram 1000x513)."""
    import h5py
    with h5py.File(f"{DATA}/ground_truth_test_{file_idx:03d}.hdf5", "r") as h:
        gt = np.array(h["data"][idx])
    with h5py.File(f"{DATA}/observation_test_{file_idx:03d}.hdf5", "r") as h:
        obs = np.array(h["data"][idx])
    return gt, obs


def solve_ls(A, y, n_iter=60, nonneg=True):
    """min 0.5||Ax-y||^2 s.t. x>=0, via FISTA with restart. Step size 0.99/L, L from power iteration."""
    L = power_method(lambda v: A.backproject(A.project(v)), A.in_shape,
                     n_iter=40, dtype=torch.float64)
    step = 0.99 / max(L, 1e-30)
    grad = lambda x: A.backproject(A.project(x) - y)
    prox = (lambda v, t: prox_nonneg(v, t)) if nonneg else (lambda v, t: v)
    x, info = fista(grad, prox, torch.zeros(A.in_shape, dtype=torch.float64),
                    step=step, n_iter=n_iter, restart=True)
    return x, L, info


def psnr(x, ref):
    x = np.asarray(x, dtype=np.float64); ref = np.asarray(ref, dtype=np.float64)
    mse = float(np.mean((x - ref) ** 2))
    rng = float(ref.max() - ref.min())
    return 10 * math.log10(rng * rng / mse) if mse > 0 else float("inf")
