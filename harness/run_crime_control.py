"""Crime tier A control: synthesize the observations with our own A and run the same TVAdam --
separates operator mismatch from recipe errors.

Background (2026-08-27): on 96 real observations FBP agrees with the official number (30.51 vs
30.19), but the TV gain is only +0.77 dB against the official +3.17. Two suspects:
  (a) our operator has residual mismatch with the real data, and 5000 iterations etch it into
      the reconstruction;
  (b) the TVAdam recipe still has a layer that is not aligned.
This script synthesizes observations on **our own A** following the official noise model
(N ~ Poisson(N0 exp(-A(gt) mu_max)), y = -ln(N/N0)/mu_max) -- in tier A the operator and the
data definition agree, so the mismatch is zero.
  tier A >= 33   =>  (a): the recipe and the solver are innocent, go fix the operator;
  tier A ~ 31.5  =>  (b): keep reading the recipe.
This is exactly the tier A of the three-way control the crime/ layer was designed for, used in
anger for the first time.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_radon import FastRadon
from lodopab import load_batch, psnr, ssim, Backprojector, FBP, ramp_filter, DATA
from tv_adam import TVAdamReconstructor, MU_MAX, N0

os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
LOG = os.path.join(HERE, "..", "results", "crime_control.txt")


def log(m):
    print(m, flush=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def main():
    dev = "cuda:5"
    A = FastRadon(device=dev)
    bp = Backprojector(device=dev)
    obs_real, gt = load_batch(16, 0, dev)
    A.self_check(obs_real[0], gt[0])

    # synthetic observations: official noise model, but the forward model is our A
    g = torch.Generator(device="cpu").manual_seed(0)
    obs_syn = []
    for i in range(16):
        y_clean = A.project(gt[i])
        lam = N0 * torch.exp(-y_clean * MU_MAX)
        n = torch.poisson(lam.cpu(), generator=g).to(dev).clamp(min=1.0)
        obs_syn.append(-torch.log(n / N0) / MU_MAX)
    obs_syn = torch.stack(obs_syn)

    # FBP calibration follows the convention used on the real data (fs=0.1, for initialization)
    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        oc = torch.tensor(np.array(h["data"][:2]), dtype=torch.float32, device=dev)
    with h5py.File(f"{DATA}/ground_truth_test_001.hdf5", "r") as h:
        gc = torch.tensor(np.array(h["data"][:2]), dtype=torch.float32, device=dev)
    num = den = 0.0
    for i in range(2):
        r = bp(ramp_filter(oc[i], 0.1, "hann"))
        num += float((r * gc[i]).sum()); den += float((r * r).sum())
    fbp01 = FBP(bp, calib=num / den, cutoff=0.1, window="hann")

    log("== crime tier A %s ==" % time.strftime("%F %T"))
    log("magnitude of the difference, synthetic vs real observation: mean|y_syn - y_real| = %.5f"
        % float((obs_syn - obs_real).abs().mean()))
    po, so = [], []
    for i in range(16):
        t0 = time.time()
        r = TVAdamReconstructor(A, fbp01, gamma=30.0, iterations=5000,
                                loss="poisson")(obs_syn[i])
        po.append(psnr(r, gt[i], "adaptive"))
        so.append(ssim(r, gt[i], "adaptive"))
        log("  #%-2d  %.2f / %.3f  (%.0fs)" % (i, po[-1], so[-1], time.time() - t0))
    po, so = np.array(po), np.array(so)
    log("tier A (synthetic observation, same TVAdam)  PSNR %.2f ± %.2f   SSIM %.3f"
        % (po.mean(), po.std(), so.mean()))
    log("reference: same 16 real observations with TVAdam = 32.03; official challenge = 33.36")


if __name__ == "__main__":
    main()
