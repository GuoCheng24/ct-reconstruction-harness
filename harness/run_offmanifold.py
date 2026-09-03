"""Can the **true distance** to the consistency manifold predict mismatch damage uniformly?

## Why this has to be done (where the previous predictor failed)

Using the Helgason-Ludwig **first-moment** residual as the predictor:
  cor 0.25/1/4 pixels -- predicted 1.80/9.27/16.95 dB, measured 1.78/8.07/16.12 dB (accurate);
  gauge axes (rot/det_scale) -- predicted 0, measured <=0.32 dB after compensation (correct);
  **ang_jitter -- predicted -1.7 dB (harmless), measured +1.7 dB (harmful)**. The sign is flipped.
Reason: the first moment has only 2 basis functions (cosθ, sinθ), and the ~0.3% signature
amplitude of angular jitter falls below the 4% noise floor, so low-order moments are **blind**
to it.

The quantity the theory calls for was never the low-order moment residual anyway; it is the
**true distance** from y to the range R(A_0),
    dist(y) = min_z ‖A_0 z − y‖ ,
which exhausts **all** consistency constraints at once. This script computes that least-squares
residual with CGLS (starting from 0, so the iterates stay inside row(A)), then checks whether it
predicts the damage on all four axes uniformly.

Predictions (if the theory holds):
  gauge axes: dist unchanged (the data is still a legitimate sinogram of some object) => predict
              zero damage => agrees with the compensated measurements;
  cor:        dist grows with δ => predict growing damage;
  jitter:     dist should grow too (it genuinely is not the sinogram of any object at the assumed
              angles), it is only invisible to low-order moments -- if dist catches it, the claim
              "detectable = correctable" is repaired, and the failure of the low-order moments
              was merely **an insufficiently rich test**, not a wrong theory.

Note: the noise itself also places the real observation at a distance from the manifold, so
everything is referenced to **the dist of the unperturbed real observation as the baseline**,
and what is reported is a dist ratio, consistent with the previous convention.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "honesty"))
from lodopab import DATA                      # noqa: E402
import mismatch as M                          # noqa: E402
from decompose import cgls                    # noqa: E402

DEV = "cuda:0"
N_IMG = 362
CG_ITERS = 120


def dist_to_range(y, op, n_iter=CG_ITERS):
    """min_z ‖A z − y‖ / ‖y‖ -- relative distance from y to the range of A (solved by CGLS)."""
    def A(z):
        return op.project(z.reshape(N_IMG, N_IMG))

    def At(r):
        return op.adjoint(r).reshape(-1)

    z = cgls(lambda v: A(v).reshape(-1), lambda r: At(r.reshape(1000, 513)),
             y.reshape(-1), n_iter=n_iter)
    r = A(z).reshape(-1) - y.reshape(-1)
    return float(r.norm() / y.norm())


def main():
    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:3]).copy(), dtype=torch.float32).to(DEV)
    op0 = M.make_operator(device=DEV)
    cell, step = M.det_cell(), M.ANG_STEP

    base = float(np.mean([dist_to_range(obs[i], op0) for i in range(3)]))
    print(f"# relative residual of the unperturbed real observation (noise floor) = {base:.6e}", flush=True)

    # damage uses the "perturbed belief" TV sweep values throughout (results/mismatch_gate.md, Table 1)
    HARM = {("cor", 0.25): 1.785, ("cor", 1.0): 8.071, ("cor", 4.0): 16.121,
            ("rot", 1.0): 0.0, ("rot", 4.0): 0.32,          # gauge: true damage after compensation
            ("det_scale", 0.001): 0.0, ("det_scale", 0.005): -0.06,
            ("ang_jitter", 1.0): 0.708, ("ang_jitter", 4.0): 1.713}

    cases = [("cor", 0.25, {"cor": 0.25 * cell}), ("cor", 1.0, {"cor": 1.0 * cell}),
             ("cor", 4.0, {"cor": 4.0 * cell}),
             ("rot", 1.0, {"rot": 1.0 * step}), ("rot", 4.0, {"rot": 4.0 * step}),
             ("det_scale", 0.001, {"det_scale": 1.001}),
             ("det_scale", 0.005, {"det_scale": 1.005}),
             ("ang_jitter", 1.0, {"ang_jitter": 1.0 * step}),
             ("ang_jitter", 4.0, {"ang_jitter": 4.0 * step})]

    out = []
    for axis, d, kw in cases:
        y = M.perturb_observation(obs, **kw)
        dd = float(np.mean([dist_to_range(y[i], op0) for i in range(3)]))
        ratio = dd / base
        pred = 20 * np.log10(max(ratio, 1e-12))
        harm = HARM.get((axis, d), float("nan"))
        row = {"axis": axis, "delta": d, "dist": dd, "ratio": ratio,
               "pred_dB": pred, "harm_dB": harm}
        out.append(row)
        print(json.dumps(row), flush=True)

    print("\n" + "=" * 68)
    print(f"noise floor dist = {base:.4e}")
    print(f"{'axis':<12}{'delta':>8}{'ratio':>9}{'pred_dB':>10}{'harm_dB':>9}{'diff':>8}")
    for r in out:
        print(f"{r['axis']:<12}{r['delta']:>8.4g}{r['ratio']:>9.4f}"
              f"{r['pred_dB']:>10.2f}{r['harm_dB']:>9.2f}"
              f"{r['pred_dB']-r['harm_dB']:>+8.2f}")
    with open(os.path.join(HERE, "..", "results", "offmanifold.json"), "w") as f:
        json.dump({"base": base, "rows": out}, f, indent=1)


if __name__ == "__main__":
    main()
