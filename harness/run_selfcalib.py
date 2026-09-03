"""Load-bearing experiment: can the deadliest mismatch axis (COR) be solved in closed form by
classical self-calibration?

## Why this experiment decides whether the direction lives or dies

The sweep shows COR is the deadliest of the four axes: 0.25 pixels -1.79 dB, 1 pixel -8.07 dB.
If a paper argues "our method is robust to COR mismatch" while COR can be calibrated away in
closed form by the Helgason-Ludwig consistency conditions known since the 1930s, one sentence
from a reviewer punctures it. So this road has to be **walked to its dead end by us first**,
before deciding what is left.

## Protocol (no simulation, no inverse crime)

The real observation y comes from the official operator. Shifting it along the detector axis by
d pixels gives y' = S_d(y) -- this is not a "simulation", it is exactly what a machine whose
detector is offset by d **would measure** on the same object (S_d is the definition of the
offset). So the true operator of y' is A(cor=d). Three methods:

  oracle    knows the true operator A(cor=d)                       -- the ceiling
  naive     assumes the official operator A(cor=0)                 -- the full mismatch cost
  selfcal   estimates d_hat from y' in closed form via the HL first moment and uses
            A(cor=d_hat) -- no ground truth needed

If selfcal ~ oracle, the COR axis is closed as a research target.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lodopab import psnr, DATA               # noqa: E402
from tv_adam import poisson_loss             # noqa: E402
from regularizers import make_regularizer    # noqa: E402
import mismatch as M                         # noqa: E402
import identifiability as ID                 # noqa: E402

GAMMA, ITERS, N = 20.556, 3000, 4
DEV = "cuda:0"


def tv_recon(y, op, iters=ITERS):
    reg, extra = make_regularizer("aniso_tv", device=DEV)
    x = op.fbp(y).clone().requires_grad_(True)
    opt = torch.optim.Adam([x] + extra, lr=1e-3)
    best_l, best_x = float("inf"), x.detach().clone()
    for _ in range(iters):
        opt.zero_grad()
        loss = poisson_loss(op.project(x), y) + GAMMA * reg(x)
        loss.backward()
        opt.step()
        l = float(loss.detach())
        if l < best_l:
            best_l, best_x = l, x.detach().clone()
    return best_x


def main():
    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:N]).copy(), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_001.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:N]).copy(), dtype=torch.float32)

    geom = M.official_geometry()
    det = np.asarray(geom.det_partition.grid.coord_vectors[0], dtype=np.float64)
    ang = np.asarray(geom.angles, dtype=np.float64)
    cell = M.det_cell()

    out = []
    for d_px in [0.0, 0.25, 1.0, 4.0]:
        # physically offset the detector by d_px: a recalibration of a real measurement, not a simulation
        y_shift = M.shift_detector(obs.to(DEV), d_px)
        d_hats = [ID.estimate_cor(y_shift[i].cpu().numpy().astype(np.float64),
                                  det, ang)["d_hat_m"] / cell for i in range(N)]
        d_hat = float(np.mean(d_hats))

        ops = {
            "oracle":  M.make_operator(device=DEV, cor=d_px * cell),
            "naive":   M.make_operator(device=DEV, cor=0.0),
            "selfcal": M.make_operator(device=DEV, cor=d_hat * cell),
        }
        res = {}
        for name, op in ops.items():
            ps = [psnr(tv_recon(y_shift[i], op).cpu(), gt[i], "adaptive")
                  for i in range(N)]
            res[name] = float(np.mean(ps))
            del op
            torch.cuda.empty_cache()
        row = {"d_px": d_px, "d_hat_px": d_hat,
               "d_hat_err_px": d_hat - d_px, **res}
        out.append(row)
        print(json.dumps(row), flush=True)

    print("\n" + "=" * 74)
    print(f"{'shift':>10} {'HL est':>10} {'est err':>10} "
          f"{'oracle':>9} {'naive':>9} {'selfcal':>9} {'resid':>8}")
    for r in out:
        print(f"{r['d_px']:>10.2f} {r['d_hat_px']:>10.4f} {r['d_hat_err_px']:>+10.4f} "
              f"{r['oracle']:>9.3f} {r['naive']:>9.3f} {r['selfcal']:>9.3f} "
              f"{r['selfcal']-r['oracle']:>+8.3f}")
    with open(os.path.join(HERE, "..", "results", "selfcalib.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
