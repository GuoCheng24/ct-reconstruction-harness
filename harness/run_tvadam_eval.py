"""Full evaluation of TVAdam (official recipe): a small γ sweep on file#1 (calibration), final
scoring on file#0 (evaluation).

Run under nohup, results appended to results/tvadam_eval.txt -- part of the harness discipline:
hyperparameters and evaluation use different files, and the numbers are written to disk so they
stay traceable.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_radon import FastRadon                       # noqa: E402
from lodopab import (load_batch, psnr, ssim, Backprojector, FBP,
                     ramp_filter, DATA)                # noqa: E402
from tv_adam import TVAdamReconstructor                # noqa: E402

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "tvadam_eval.txt")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    import sys as _sys
    _a = _sys.argv
    dev = "cuda:5"
    A = FastRadon(device=dev)
    bp = Backprojector(device=dev)

    # calibration file file#1: FBP calibration and the γ sweep both happen here
    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        oc = torch.tensor(np.array(h["data"][:6]), dtype=torch.float32, device=dev)
    with h5py.File(f"{DATA}/ground_truth_test_001.hdf5", "r") as h:
        gc = torch.tensor(np.array(h["data"][:6]), dtype=torch.float32, device=dev)
    num = den = 0.0
    for i in range(2):
        r = bp(ramp_filter(oc[i], 0.1, "hann"))
        num += float((r * gc[i]).sum()); den += float((r * r).sum())
    fbp01 = FBP(bp, calib=num / den, cutoff=0.1, window="hann")

    n_eval = int(_a[2]) if len(_a) > 2 else 16
    obs, gt = load_batch(n_eval, 0, dev)
    A.self_check(obs[0], gt[0])
    log("== %s ==" % time.strftime("%F %T"))

    # γ sweep: 4 images from file#1 (overlapping the 2 used for FBP calibration is fine -- both
    # are on the calibration side)
    log("γ sweep (poisson, 3000 iter, mean over the first 4 images of file#1)")
    best = None
    gammas = ([float(g) for g in _a[1].split(",")] if len(_a) > 1
              else (10.0, 15.0, 20.556, 30.0))
    for gm in gammas:
        ps = []
        for i in range(4):
            r = TVAdamReconstructor(A, fbp01, gamma=gm, iterations=3000,
                                    loss="poisson")(oc[i])
            ps.append(psnr(r, gc[i], "adaptive"))
        m = float(np.mean(ps))
        tag = ""
        if best is None or m > best[1]:
            best = (gm, m); tag = "  <-"
        log("  γ=%-7.3f  calibration mean PSNR %.2f%s" % (gm, m, tag))
    gm_star = best[0]
    log("selected γ=%.3f" % gm_star)
    if gm_star == max(gammas):
        log("⚠ optimum is at the grid boundary -- headroom remains, widen the grid next round")

    # final scoring: 16 images from file#0, official 5000 iter
    log("final (poisson, γ=%.3f, 5000 iter, %d images from file#0)" % (gm_star, n_eval))
    po, so = [], []
    for i in range(n_eval):
        t0 = time.time()
        r = TVAdamReconstructor(A, fbp01, gamma=gm_star, iterations=5000,
                                loss="poisson")(obs[i])
        po.append(psnr(r, gt[i], "adaptive"))
        so.append(ssim(r, gt[i], "adaptive"))
        log("  #%-2d  %.2f / %.3f  (%.0fs)" % (i, po[-1], so[-1], time.time() - t0))
    po, so = np.array(po), np.array(so)
    log("TVAdam (official recipe)  official PSNR %.2f ± %.2f   official SSIM %.3f ± %.3f"
        % (po.mean(), po.std(), so.mean(), so.std()))
    log("official reference        official PSNR 33.36 ± 2.74   official SSIM 0.830")


if __name__ == "__main__":
    main()
