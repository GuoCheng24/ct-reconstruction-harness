"""Deployment-side operator mismatch for a learned reconstructor -- the gate of this direction.

Usage:
    python eval_mismatch.py --ckpt runs/lpd_official/ckpt.pt --gpu 0 [--n 128] [--file 0]

## The question

Whether swapping the correct operator back in rescues an unrolled network has been answered
in print: Gossard & Weiss (arXiv 2202.11342, SIAM J. Imaging Sciences) give the full
training-operator x evaluation-operator matrix on the same LoDoPaB with the same parameters
theta = (projection angles, per-view detector shift) -- CT matched 35.39 -> swapped 32.89 ->
blind 25.70. MetaInv-Net (TMI 2021) and Gilton et al. (TCI 2021) report the same kind of
evidence. The three modes below are still run, but as our own control.

The open question this script measures, which the existing work leaves without any
guarantee:

> Can a quantity computable *without ground truth* -- the Helgason-Ludwig consistency
> residuals of the measured sinogram, order by order -- predict the degradation of a
> *learned* method quantitatively?
>   yes -> it can serve as a per-scan operator-uncertainty error bar;
>   no  -> the direction closes.

On the classical side (TV) the empirical law "loss ~ 20 log10(residual ratio)" holds to
about 1 dB on the visible axes. The learned side is what is measured here. Gossard & Weiss
state the bound explicitly as an open problem.

## Three modes (the data is always a *real* measurement, physically resampled; no simulation)

  naive    data from G(delta), network still holds G(0)         -- "nobody updated the geometry file"
  swap     data from G(delta), network given G(delta) (oracle)  -- upper bound with the true geometry known
  selfcal  data from G(delta), delta estimated in closed form from the data, network given G(delta_hat)
           -- no ground truth, deployable

Shares its ruler with harness/mismatch_eval.py (same perturbations, same metric convention,
same images), so the TV curve and this curve can be compared directly.

## Recipe flags

Defaults are the official LPD recipe (n_layer=3, batch_norm=False, prelu=True). The flags
`--n_layer 4 --batch_norm --no_prelu` reproduce the mis-configured model that an earlier
version of this study used, so that its numbers can be re-derived and compared; see
results/lpd_mismatch_n128.json for both.
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
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

AXES = {
    "none":       lambda d, cell, step: {},
    "cor":        lambda d, cell, step: {"cor": d * cell},
    "rot":        lambda d, cell, step: {"rot": d * step},
    "ang_jitter": lambda d, cell, step: {"ang_jitter": d * step},
    "det_scale":  lambda d, cell, step: {"det_scale": 1.0 + d},
}

GRID = [("cor", [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("rot", [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("ang_jitter", [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("det_scale", [0.0002, 0.0005, 0.001, 0.005])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--file", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--batch_norm", action="store_true")
    ap.add_argument("--no_prelu", action="store_true")
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("EVAL_CHUNK", "8")),
                    help="images per forward pass; an unrolled network on the whole batch OOMs at n=128")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mismatch as M
    import identifiability as ID
    from lodopab import psnr, ssim, DATA
    from lpd import LPD

    dev = "cuda:0"
    cell, step = M.det_cell(), M.ANG_STEP
    geom = M.official_geometry()
    det_coords = np.asarray(geom.det_partition.grid.coord_vectors[0], dtype=np.float64)
    angles = np.asarray(geom.angles, dtype=np.float64)

    with h5py.File(f"{DATA}/observation_test_{args.file:03d}.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:args.n]).copy(), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_{args.file:03d}.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:args.n]).copy(), dtype=torch.float32)

    op0 = M.make_operator(device=dev)
    net = LPD(op0, device=dev, n_layer=args.n_layer,
              batch_norm=args.batch_norm, prelu=not args.no_prelu)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.net.load_state_dict(ck["model"])
    net.eval()
    print(f"# checkpoint step={ck['step']}", flush=True)

    def _forward(y):
        """Chunked forward pass. Per-image metrics averaged afterwards are identical to a
        single-batch pass (verified: same 35.242 at n=16 with chunk 4 and chunk 16)."""
        outs = []
        with torch.no_grad():
            for i in range(0, y.shape[0], args.chunk):
                outs.append(net(y[i:i + args.chunk].to(dev)).cpu())
                torch.cuda.empty_cache()
        return torch.cat(outs, 0)

    def score(y, op):
        net.set_operator(op)
        r = _forward(y)
        return (float(np.mean([psnr(r[i], gt[i], "adaptive") for i in range(args.n)])),
                float(np.mean([ssim(r[i], gt[i], "adaptive") for i in range(args.n)])))

    def hl_spectrum(y):
        """Helgason-Ludwig residuals order by order (computable without ground truth) -- the predictor."""
        return np.mean([ID.hl_residual_spectrum(
            y[i].cpu().numpy().astype(np.float64), det_coords, angles, nmax=6)
            for i in range(args.n)], axis=0)

    rows = []
    p0, s0 = score(obs, op0)
    base_spec = hl_spectrum(obs)
    rows.append({"axis": "none", "delta": 0.0, "naive": p0, "swap": p0,
                 "selfcal": p0, "ssim": s0,
                 "hl_ratio": [1.0] * len(base_spec)})
    print(json.dumps(rows[-1]), flush=True)

    for axis, deltas in GRID:
        for d in deltas:
            kw = AXES[axis](d, cell, step)
            y = M.perturb_observation(obs.to(dev), **kw)
            row = {"axis": axis, "delta": d}
            row["hl_ratio"] = [round(float(v), 4)
                               for v in hl_spectrum(y) / base_spec]
            row["naive"] = score(y, op0)[0]
            # On the gauge axes, separate "metric artefact" from "the network really failed":
            # apply the group action to the output and re-score. A classical method returns
            # to its baseline here (measured: rot -0.32 dB / det_scale +0.06 dB); a CNN, not
            # being equivariant, need not -- and if it does not, the gauge part is *not free*
            # for learned methods, which is the empirical support for "equivariance is a
            # precondition for a computable certificate".
            if axis in ("rot", "det_scale"):
                net.set_operator(op0)
                r = _forward(y).numpy()
                best = -1e9
                for i_sign in (1, -1):
                    vals = []
                    for i in range(args.n):
                        im = r[i]
                        if axis == "rot":
                            im = ndi.rotate(im, i_sign * np.degrees(d * step),
                                            reshape=False, order=3, mode="nearest")
                        else:
                            c = 1.0 + i_sign * d
                            z = ndi.zoom(im, c, order=3)
                            n0, m0 = im.shape[0], z.shape[0]
                            if m0 >= n0:
                                s0 = (m0 - n0) // 2
                                im = z[s0:s0 + n0, s0:s0 + n0] / c
                            else:
                                out = np.zeros_like(im)
                                s0 = (n0 - m0) // 2
                                out[s0:s0 + m0, s0:s0 + m0] = z
                                im = out / c
                        vals.append(psnr(torch.tensor(im), gt[i], "adaptive"))
                    best = max(best, float(np.mean(vals)))
                row["gauge_fixed"] = best
            op_true = M.make_operator(device=dev, **kw)
            row["swap"], row["ssim"] = score(y, op_true)
            del op_true
            torch.cuda.empty_cache()
            if axis == "cor":
                dh = float(np.mean([
                    ID.estimate_cor(y[i].cpu().numpy().astype(np.float64),
                                    det_coords, angles)["d_hat_m"] / cell
                    for i in range(args.n)]))
                op_est = M.make_operator(device=dev, cor=dh * cell)
                row["selfcal"] = score(y, op_est)[0]
                row["d_hat_px"] = dh
                del op_est
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = args.out or os.path.join(RESULTS, f"lpd_mismatch_step{ck['step']}.json")
    with open(out, "w") as f:
        json.dump({"ckpt_step": ck["step"], "n": args.n, "file": args.file,
                   "recipe": {"n_layer": args.n_layer, "batch_norm": args.batch_norm,
                              "prelu": not args.no_prelu},
                   "rows": rows}, f, indent=1)

    print("\n" + "=" * 84)
    print(f"LPD baseline (delta=0) = {p0:.3f}   [checkpoint step {ck['step']}]")
    print("predictor = max over orders of the HL residual ratio (no ground truth); predicted loss = 20 log10(it)")
    print(f"{'axis':<12}{'delta':>8}{'naive':>8}{'measured':>9}{'maxHL':>9}"
          f"{'predicted':>10}{'diff':>7}{'swap':>8}{'selfcal':>8}")
    for r in rows:
        if r["axis"] == "none":
            continue
        mx = max(r["hl_ratio"])
        pred = 20 * np.log10(max(mx, 1e-12))
        meas = p0 - r["naive"]
        sc = (f"{r['selfcal']:>8.2f}" if "selfcal" in r
              else (f"{r['gauge_fixed']:>8.2f}" if "gauge_fixed" in r else f"{'-':>8}"))
        print(f"{r['axis']:<12}{r['delta']:>8.4g}{r['naive']:>8.2f}{meas:>9.2f}"
              f"{mx:>9.3f}{pred:>10.2f}{pred-meas:>+7.2f}{r['swap']:>8.2f}{sc}")
    print("(last column: selfcal on the cor axis; gauge_fixed on the gauge axes = output re-scored after the group action)")
    ms = np.array([p0 - r["naive"] for r in rows if r["axis"] != "none"])
    ps = np.array([20 * np.log10(max(max(r["hl_ratio"]), 1e-12))
                   for r in rows if r["axis"] != "none"])
    if len(ms) > 2:
        from scipy.stats import spearmanr, pearsonr
        print(f"\ngate: Pearson r = {pearsonr(ps, ms)[0]:.3f}, "
              f"Spearman = {spearmanr(ps, ms).statistic:.3f}, "
              f"mean absolute deviation = {np.abs(ps - ms).mean():.2f} dB")


if __name__ == "__main__":
    main()
