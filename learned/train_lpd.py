"""Train Learned Primal-Dual -- the "learned side" baseline of the mismatch study.

Usage:
    python train_lpd.py --out runs/lpd_official --steps 35220 --gpu 1
    python train_lpd.py --out runs/lpd_official --resume        # continue from checkpoint

The training pairs are read from $LODOPAB_VAL_DIR (the LoDoPaB *validation* split,
extracted; see LodopabPairs for the expected file names).

## Design notes

1. Interruptible: a checkpoint (weights + optimizer + step) is written every CKPT_EVERY
   steps and `--resume` continues from the last one. A six-hour run on a shared node has
   to survive preemption or it starts from zero.
2. Training data = the real (observation, ground truth) pairs of the LoDoPaB validation
   split. Not the train split: that is 30 GB, and the 3522 validation pairs are enough to
   train an LPD that can answer the gate question (is it more fragile to operator mismatch
   than a classical method). No leakage: LoDoPaB splits by patient, validation and test
   share no patient, and evaluation only ever touches test files #0 / #1.
3. Training-side operator = the official operator (delta = 0). Mismatch is injected on the
   deployment side by swapping the network's operator, so one training run serves the whole
   mismatch curve and evaluation always stands on real measurements.
4. Every EVAL_EVERY steps, PSNR on the first 4 images of test file #1 is logged -- for
   watching convergence only; it takes part in no selection.
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
EVAL_EVERY = 2000


class LodopabPairs:
    """(obs, gt) pairs read on demand.

    HDF5 slices must be .copy()'d: `h["data"][i]` is already a fresh array, but slicing a
    large in-memory array (arr[i]) keeps the whole block alive -- that once cost 115 GB of
    RAM on a shared node.
    """

    def __init__(self, split_dir, prefix_obs="observation_validation",
                 prefix_gt="ground_truth_validation"):
        self.dir = split_dir
        self.obs_files = sorted(f for f in os.listdir(split_dir)
                                if f.startswith(prefix_obs))
        self.gt_files = sorted(f for f in os.listdir(split_dir)
                               if f.startswith(prefix_gt))
        assert len(self.obs_files) == len(self.gt_files) and self.obs_files, \
            f"{split_dir}: {len(self.obs_files)} observation / {len(self.gt_files)} ground-truth files"
        self.counts = []
        for f in self.obs_files:
            with h5py.File(os.path.join(split_dir, f), "r") as h:
                self.counts.append(h["data"].shape[0])
        self.offsets = np.cumsum([0] + self.counts)
        self.n = int(self.offsets[-1])
        self._ho, self._hg = {}, {}

    def __len__(self):
        return self.n

    def _handle(self, cache, files, fi):
        """Handle cache. Re-opening the HDF5 file on every get is a real bottleneck: at
        batch 16 that is 32 opens per step, and U-Net training once failed to get past
        50 steps in three minutes. With the cache it is 2 x 28 opens in total."""
        if fi not in cache:
            cache[fi] = h5py.File(os.path.join(self.dir, files[fi]), "r")
        return cache[fi]

    def get(self, idx):
        fi = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        li = idx - int(self.offsets[fi])
        o = np.array(self._handle(self._ho, self.obs_files, fi)["data"][li],
                     dtype=np.float32).copy()
        g = np.array(self._handle(self._hg, self.gt_files, fi)["data"][li],
                     dtype=np.float32).copy()
        return torch.from_numpy(o), torch.from_numpy(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=35220)     # 3522 x 10 epochs
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--data", default=VAL_DIR)
    ap.add_argument("--aug", action="store_true",
                    help="operator augmentation: each sample draws a random geometry, the "
                         "observation is perturbed and the matching operator is swapped into "
                         "the network -- teaching it to use whatever operator it is handed")
    ap.add_argument("--bank", type=int, default=9, help="size of the augmentation operator bank")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import mismatch as M
    from lodopab import DATA, psnr
    from lpd import LPD

    dev = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a", buffering=1)

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    ds = LodopabPairs(args.data)
    say(f"training set: {len(ds)} images from {len(ds.obs_files)} files")

    op = M.make_operator(device=dev)
    net = LPD(op, device=dev)
    say(f"LPD parameters {net.n_params():,}")

    # Operator bank, built once (building a RayTransform per step is too expensive). The
    # real observation is physically resampled by delta and the network gets the matching
    # operator at the same time -- this is operator augmentation, not data augmentation.
    bank = [({}, op)]
    if args.aug:
        cell, astep = M.det_cell(), M.ANG_STEP
        specs = [{"cor": 0.5 * cell}, {"cor": -0.5 * cell},
                 {"cor": 2.0 * cell}, {"cor": -2.0 * cell},
                 {"rot": 1.0 * astep}, {"rot": -1.0 * astep},
                 {"det_scale": 1.002}, {"det_scale": 0.998}]
        for s in specs[:max(0, args.bank - 1)]:
            bank.append((s, M.make_operator(device=dev, **s)))
        say(f"operator augmentation on, bank size {len(bank)}")
    opt = torch.optim.Adam(net.net.parameters(), lr=args.lr)

    step0 = 0
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        say(f"resuming from step {step0}")

    # Convergence watch set (looked at, never selected on)
    with h5py.File(f"{DATA}/observation_test_001.hdf5", "r") as h:
        eo = torch.tensor(np.array(h["data"][:4]).copy(), dtype=torch.float32)
    with h5py.File(f"{DATA}/ground_truth_test_001.hdf5", "r") as h:
        eg = torch.tensor(np.array(h["data"][:4]).copy(), dtype=torch.float32)

    rng = np.random.default_rng(1234 + step0)
    t0, run_loss = time.time(), 0.0
    # Stall detector. Two training processes reading the same HDF5 files on a shared
    # parallel file system both drop to 99% CPU / 0% GPU, the log stops advancing and the
    # processes stay alive -- a silent stall that looks exactly like training. This turns it
    # into a check that shouts: a step slower than 5x the warm-up baseline is logged at once.
    step_ref, stall_warned = None, False
    for step in range(step0, args.steps):
        t_step = time.time()
        idx = int(rng.integers(0, len(ds)))
        o, g = ds.get(idx)
        o, g = o[None].to(dev), g[None].to(dev)
        if len(bank) > 1:
            spec, bop = bank[int(rng.integers(0, len(bank)))]
            if spec:
                o = M.perturb_observation(o, **spec)
            net.set_operator(bop)
        opt.zero_grad(set_to_none=True)
        loss = ((net(o) - g) ** 2).mean()
        loss.backward()
        opt.step()
        run_loss += float(loss.detach())

        dt = time.time() - t_step
        if step_ref is None and step - step0 >= 20:
            step_ref = dt                      # baseline after 20 warm-up steps
        elif step_ref is not None and dt > 5 * step_ref:
            if not stall_warned:
                say(f"!! stall warning: step took {dt:.1f}s, over 5x the baseline {step_ref:.2f}s "
                    f"-- check for another process on the same HDF5 files / the same GPU")
                stall_warned = True
        elif step_ref is not None and dt < 2 * step_ref:
            stall_warned = False

        if (step + 1) % 100 == 0:
            sps = (step + 1 - step0) / max(time.time() - t0, 1e-9)
            say(f"step {step+1}/{args.steps}  loss {run_loss/100:.3e}  "
                f"{sps:.2f} step/s  eta {(args.steps-step-1)/max(sps,1e-9)/3600:.1f}h")
            run_loss = 0.0
        if (step + 1) % CKPT_EVERY == 0:
            torch.save({"model": net.net.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        if (step + 1) % EVAL_EVERY == 0:
            net.set_operator(op)          # in augmentation mode, swap the official operator back before evaluating
            net.eval()
            with torch.no_grad():
                ps = [psnr(net(eo[i:i+1].to(dev))[0].cpu(), eg[i], "adaptive")
                      for i in range(4)]
            net.train()
            say(f"  >> eval@{step+1}: PSNR {np.mean(ps):.3f}  {[round(p,2) for p in ps]}")
            with open(os.path.join(args.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps({"step": step + 1, "psnr": float(np.mean(ps)),
                                    "per_image": [round(p, 3) for p in ps]}) + "\n")

    torch.save({"model": net.net.state_dict(), "opt": opt.state_dict(),
                "step": args.steps}, ckpt_path)
    say("training finished")


if __name__ == "__main__":
    main()
