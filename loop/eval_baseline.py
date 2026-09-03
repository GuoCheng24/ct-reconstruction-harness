"""Measure the level of our own trained LPD properly -- numbers from 4 images cannot carry any
conclusion.

The eval in the training log uses only 4 images, and the per-image PSNR spreads over 23.8-41.5 dB
(a range of 17.7 dB), so the error bar on "34.75" may be wider than the 1.5 dB gap we care about.
Any claim of "beating a published baseline" must first have **an own baseline with error bars**.

Usage: python eval_baseline.py --ckpt runs/lpd_true/ckpt.pt --gpu 3 --n 128 --file 1
Discipline: exploration uses file#1; file#0 is reserved for the final confirmation.
"""
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--file", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mismatch as M
    from lodopab import psnr, ssim, DATA, PUBLISHED
    from lpd import LPD

    dev = "cuda:0"
    with h5py.File(f"{DATA}/observation_test_{args.file:03d}.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:args.n]).copy(), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_{args.file:03d}.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:args.n]).copy(), dtype=torch.float32)
    n = obs.shape[0]

    net = LPD(M.make_operator(device=dev), device=dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.net.load_state_dict(ck["model"])
    net.eval()

    po, so = [], []
    with torch.no_grad():
        for i in range(0, n, args.batch):
            r = net(obs[i:i + args.batch].to(dev)).cpu()
            for j in range(r.shape[0]):
                po.append(psnr(r[j], gt[i + j], "adaptive"))
                so.append(ssim(r[j], gt[i + j], "adaptive"))
    po, so = np.array(po), np.array(so)

    # standard error of the mean + bootstrap 95% CI (the per-image distribution is far from
    # normal, so std alone would mislead)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(po, n, replace=True).mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out = {"ckpt_step": ck["step"], "n": n, "file": args.file,
           "psnr_mean": float(po.mean()), "psnr_sem": float(po.std(ddof=1) / np.sqrt(n)),
           "psnr_ci95": [float(lo), float(hi)],
           "psnr_std_across_images": float(po.std(ddof=1)),
           "psnr_min": float(po.min()), "psnr_max": float(po.max()),
           "ssim_mean": float(so.mean())}
    print(json.dumps(out, indent=1))
    print(f"\nour own LPD: {po.mean():.3f} dB  (95% CI [{lo:.3f}, {hi:.3f}], n={n})")
    print(f"per-image spread: std {po.std(ddof=1):.2f} dB, range [{po.min():.2f}, {po.max():.2f}]")
    print("\nagainst the official published numbers (challenge set, a different set, so half a dB of difference is normal):")
    for k in ("FBP", "TV", "DIP + TV", "U-Net", "Learned P.-D."):
        print(f"  {k:16s} {PUBLISHED[k][0]:.2f}")
    with open(os.path.join(HERE, "..", "results",
                           f"lpd_baseline_step{ck['step']}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
