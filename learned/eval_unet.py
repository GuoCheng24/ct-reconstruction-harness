"""FBP+U-Net on 128 images with a confidence interval. The 36.16 in the training log is a
4-image number; anything that gets cited has to be n = 128."""
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, h5py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness")); sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results"); os.makedirs(RESULTS, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(HERE, "runs/unet_official/ckpt.pt"))
ap.add_argument("--gpu", default="0"); ap.add_argument("--n", type=int, default=128); ap.add_argument("--file", type=int, default=1)
a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import mismatch as M; from lodopab import psnr, ssim, DATA
from dival.reconstructors.networks.unet import get_unet_model
dev = "cuda:0"; op = M.make_operator(device=dev); fbp = op.make_fbp(1.0)
net = get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4, use_sigmoid=False, use_norm=True).to(dev)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False); net.load_state_dict(ck["model"]); net.eval()
with h5py.File(f"{DATA}/observation_test_{a.file:03d}.hdf5", "r") as h: obs = torch.tensor(np.array(h["data"][:a.n]).copy(), dtype=torch.float32)
with h5py.File(f"{DATA}/ground_truth_test_{a.file:03d}.hdf5", "r") as h: gt = torch.tensor(np.array(h["data"][:a.n]).copy(), dtype=torch.float32)
po, so = [], []
with torch.no_grad():
    for i in range(0, a.n, 8):
        x0 = fbp(obs[i:i+8].to(dev)); r = net(x0[:, None])[:, 0].cpu()
        for j in range(r.shape[0]): po.append(psnr(r[j], gt[i+j], "adaptive")); so.append(ssim(r[j], gt[i+j], "adaptive"))
po = np.array(po); rng = np.random.default_rng(0); boot = [rng.choice(po, len(po), replace=True).mean() for _ in range(2000)]
out = dict(ckpt_step=ck["step"], n=len(po), file=a.file, psnr_mean=float(po.mean()),
           psnr_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           psnr_std=float(po.std(ddof=1)), ssim_mean=float(np.mean(so)))
json.dump(out, open(os.path.join(RESULTS, f"unet_baseline_step{ck['step']}.json"), "w"), indent=1)
print(f"FBP+U-Net: {out['psnr_mean']:.3f} dB  95%CI [{out['psnr_ci95'][0]:.3f},{out['psnr_ci95'][1]:.3f}]  n={out['n']}  std {out['psnr_std']:.2f}  SSIM {out['ssim_mean']:.3f}")
