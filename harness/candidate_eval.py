"""Evaluation entry point for a single candidate configuration -- called by every search
worker, and by the main loop for final scoring.

Usage:
    python candidate_eval.py '{"reg":"iso_tv","data":"poisson","gamma":20.5,...}'

Config keys:
    reg      regularizer name (regularizers.REGULARIZERS)
    data     poisson | wls
    gamma    regularization weight
    iters    default 3000 (calibration) / 5000 (final)
    n        number of images to evaluate; file=1 calibration / file=0 final
             (discipline: tuning must never touch file 0)
Output: the last line is JSON ({"psnr":..,"ssim":..,"per_image":[..]}), for the caller to parse.

Official operator (odl 0.8.1 + astra_cuda), FBP (Hann, fs=0.1) initialization, best-loss
tracking -- i.e. the pipeline that already matched the reference numbers (33.83 vs 33.36),
with only the regularizer and the data term swapped.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lodopab import psnr, ssim, DATA                    # noqa: E402
from tv_adam import poisson_loss                        # noqa: E402
from regularizers import make_regularizer, wls_loss     # noqa: E402
import h5py                                             # noqa: E402


def main(cfg):
    import odl
    import odl.tomo as odl_tomo
    from odl.contrib.torch import OperatorModule
    dev = "cuda:0"
    domain = odl.uniform_discr([-0.13, -0.13], [0.13, 0.13], (362, 362),
                               dtype=np.float32)
    geometry = odl_tomo.parallel_beam_geometry(domain, num_angles=1000,
                                               det_shape=(513,))
    rt = odl_tomo.RayTransform(domain, geometry, impl="astra_cuda")
    rt_mod = OperatorModule(rt).to(dev)
    fbp = odl_tomo.fbp_op(rt, filter_type="Hann", frequency_scaling=0.1)

    file_idx = int(cfg.get("file", 1))
    n = int(cfg.get("n", 4))
    iters = int(cfg.get("iters", 3000))
    gamma = float(cfg["gamma"])
    with h5py.File(f"{DATA}/observation_test_{file_idx:03d}.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_{file_idx:03d}.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32)

    data_fn = poisson_loss if cfg.get("data", "poisson") == "poisson" else wls_loss
    po, so = [], []
    for i in range(n):
        reg, extra = make_regularizer(cfg["reg"], device=dev,
                                      **{k: cfg[k] for k in ("delta", "ratio")
                                         if k in cfg})
        x = torch.tensor(np.asarray(fbp(obs[i].numpy())), dtype=torch.float32,
                         device=dev).requires_grad_(True)
        y = obs[i].to(dev)
        opt = torch.optim.Adam([x] + extra, lr=float(cfg.get("lr", 1e-3)))
        best_l, best_x = float("inf"), x.detach().clone()
        for _ in range(iters):
            opt.zero_grad()
            loss = data_fn(rt_mod(x[None])[0], y) + gamma * reg(x)
            loss.backward()
            opt.step()
            l = float(loss.detach())
            if l < best_l:
                best_l, best_x = l, x.detach().clone()
        r = best_x.cpu()
        po.append(psnr(r, gt[i], "adaptive"))
        so.append(ssim(r, gt[i], "adaptive"))
    out = {"cfg": cfg, "psnr": float(np.mean(po)), "ssim": float(np.mean(so)),
           "per_image": [round(p, 2) for p in po]}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
