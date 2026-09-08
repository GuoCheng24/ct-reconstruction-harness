"""Train FBP + U-Net (the official dival reference recipe) -- the second learned baseline,
and the faster gate.

Why it is needed, beyond being the 36.00 on the leaderboard:

  An unrolled network (LPD) calls A and A^T explicitly in every layer, so the operator
  slot runs through the whole network; a post-processing network (FBP+U-Net) touches the
  operator in exactly one place, the FBP initialization. Their reactions to "swap the
  operator at deployment" should differ structurally, and that contrast is the core
  experiment of this direction. It also trains about 20x faster (no per-layer operator
  calls) and gives an early answer in about an hour.

Official hyper-parameters (supp.dival lodopab_fbpunet_hyper_params.json, checked):
  scales=5, skip_channels=4, batch_size=32, epochs=250, lr=1e-3,
  filter=Hann, frequency_scaling=1.0 (no low-pass; the U-Net does the denoising),
  init_bias_zero=true, scheduler=cosine, lr_min=1e-4

Usage:
  python train_fbpunet.py --out runs/unet_official --steps 27500 --gpu 0 [--resume]

Training pairs are read from $LODOPAB_VAL_DIR (see train_lpd.LodopabPairs).
"""
import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
sys.path.insert(0, HERE)

VAL_DIR = os.environ.get("LODOPAB_VAL_DIR", "./data/lodopab_val")
CKPT_EVERY = 500
EVAL_EVERY = 1000
FBP_FS = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=27500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", default=VAL_DIR)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mismatch as M
    from lodopab import DATA, psnr
    from dival.reconstructors.networks.unet import get_unet_model
    from train_lpd import LodopabPairs

    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a", buffering=1)

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    ds = LodopabPairs(args.data)
    say(f"training set: {len(ds)} images")

    op = M.make_operator(device=dev)
    fbp = op.make_fbp(FBP_FS)

    # Precompute the FBP once and keep it in memory. Random 2 MB reads from a shared parallel
    # file system cost about 2 s per step at batch 16 (ETA 15 h), and for the fixed official
    # operator the FBP is a constant. With the cache, training is a pure U-Net forward pass and
    # the cache is (362^2 + 362^2) x 3522 x 4 B, about 3.7 GB. (Operator-augmentation runs
    # cannot use this path: there the FBP changes with the geometry and must be computed live.)
    cache = os.path.join(args.out, "fbp_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        X0, GT = z["x0"], z["gt"]
        say(f"loaded FBP cache {X0.shape}")
    else:
        say("precomputing the FBP cache ...")
        X0 = np.empty((len(ds), 362, 362), dtype=np.float32)
        GT = np.empty((len(ds), 362, 362), dtype=np.float32)
        t = time.time()
        for i in range(len(ds)):
            o, g = ds.get(i)
            with torch.no_grad():
                X0[i] = fbp(o[None].to(dev))[0].cpu().numpy()
            GT[i] = g.numpy()
            if (i + 1) % 500 == 0:
                say(f"  {i+1}/{len(ds)}  ({(time.time()-t)/(i+1):.3f} s/image)")
        np.savez(cache, x0=X0, gt=GT)
        say(f"cache written in {(time.time()-t)/60:.1f} min")
    X0t = torch.from_numpy(X0)
    GTt = torch.from_numpy(GT)
    net = get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4,
                         use_sigmoid=False, use_norm=True).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    say(f"U-Net parameters {n_par:,}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr_min)

    step0 = 0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step0 = ck["step"]
        say(f"resuming from step {step0}")

    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        eo = torch.tensor(np.array(h["data"][:4]).copy(), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_001.hdf5", "r") as h:
        eg = torch.tensor(np.array(h["data"][:4]).copy(), dtype=torch.float32)

    rng = np.random.default_rng(1234 + step0)
    t0, run = time.time(), 0.0
    for step in range(step0, args.steps):
        idx = torch.from_numpy(rng.integers(0, len(ds), args.batch))
        x0 = X0t[idx].to(dev, non_blocking=True)
        g = GTt[idx].to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = ((net(x0[:, None])[:, 0] - g) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        run += float(loss.detach())

        if (step + 1) % 50 == 0:
            sps = (step + 1 - step0) / max(time.time() - t0, 1e-9)
            say(f"step {step+1}/{args.steps} loss {run/50:.3e} {sps:.2f} step/s "
                f"eta {(args.steps-step-1)/max(sps,1e-9)/3600:.2f}h")
            run = 0.0
        if (step + 1) % CKPT_EVERY == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if (step + 1) % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                r = net(fbp(eo.to(dev))[:, None])[:, 0]
                ps = [psnr(r[i].cpu(), eg[i], "adaptive") for i in range(4)]
            net.train()
            say(f"  >> eval@{step+1}: PSNR {np.mean(ps):.3f} {[round(p,2) for p in ps]}")
            with open(os.path.join(args.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps({"step": step + 1,
                                    "psnr": float(np.mean(ps))}) + "\n")

    torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": args.steps}, ckpt_path)
    say("training finished")


if __name__ == "__main__":
    main()
