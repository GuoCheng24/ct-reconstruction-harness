"""LoDoPaB-CT harness -- the scoring engine of the fast-iteration loop, first instance.

This is not another audit tool. Audit craft is allowed in only two places, and this is the
first of them: **anti-cheating inside the harness**. It exists so that "generate 20-30 method
variants and reject most of them within seconds to minutes" becomes possible -- that is, to
make each attempt cheap.

## Method interface

A "method" is a callable: method(obs) -> recon
  obs   (n_ang, n_det) torch.float32, real low-dose observation (post-log, normalized units)
  recon (362, 362)     torch.float32

A method **only ever gets the observation**. The ground truth stays with the harness.

## Tiers

  Tier0  seconds  shape / finiteness / value range -- rejects candidates that never ran at all
  Tier1  minutes  PSNR/SSIM over N test images -- rejects mediocre candidates
  Tier2  hours    full test set + null-space hallucination rate + stability -- survivors only

## Anti-cheating (what the harness stands on)

  1. the observations are LoDoPaB's **real files**, generated independently by their pipeline --
     inherently immune to the inverse crime;
  2. null models must score like null models: constant image, unfiltered backprojection. If the
     null model score ever improves, it is the harness that broke -- freeze every conclusion and
     fix it first;
  3. ground-truth sample indices are strictly aligned with the observations (#i to #i), verified
     sample by sample on 2026-08-27 (diagonal correlation clearly above off-diagonal);
  4. the PSNR data_range is fixed at 1.0 (the LoDoPaB ground-truth range is [0,1]) and does not
     float per sample -- a per-sample max-min inflates flat slices.
"""
import math
import os
import numpy as np
import torch
import h5py

DATA = os.environ.get("LODOPAB_DIR", "./data/lodopab")
N_ANG, N_DET, N_IMG = 1000, 513, 362
FOV_CM = 26.0


# ---------------------------------------------------------------- data ----
def load_batch(n=16, file_idx=0, device="cpu"):
    """The first n (observation, ground truth) pairs, float32."""
    with h5py.File(f"{DATA}/observation_test_{file_idx:03d}.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32, device=device)
    with h5py.File(f"{DATA}/ground_truth_test_{file_idx:03d}.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32, device=device)
    return obs, gt


# ------------------------------------- shared operator: backprojection ----
class Backprojector:
    """Explicit FBP-style backprojection (unlike the VJP adjoint in ops/ -- this one has to be
    fast, since every method shares it).

    Pixel (x,y) has detector coordinate s = x cos(theta) + y sin(theta) at angle theta.
    Interpolate and accumulate from the (possibly filtered) sinogram angle by angle, multiplying
    by pi/n_ang as the angular integration measure.
    """

    def __init__(self, device="cpu"):
        self.device = device
        ang = torch.arange(N_ANG, device=device, dtype=torch.float32) * math.pi / N_ANG
        c = torch.linspace(-1, 1, N_IMG, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        # detector span = image diameter => divide normalized coordinates by sqrt(2)
        # Row/column convention: enumerating the 8 rigid transforms as a diagnostic on
        # 2026-08-27 put "transpose" 2.7 dB above "identity" => (yy, xx) must be used here, not
        # (xx, yy). A wrong orientation raises no error and produces no artefact, it just
        # quietly lowers PSNR -- that enumeration was the only way to find it.
        s = (yy.unsqueeze(0) * torch.cos(ang).view(-1, 1, 1)
             + xx.unsqueeze(0) * torch.sin(ang).view(-1, 1, 1)) / math.sqrt(2.0)
        zeros = torch.zeros_like(s)
        # grid_sample grid: x = detector coordinate, y = 0 (single row)
        self.grid = torch.stack([s, zeros], dim=-1)          # (n_ang, H, W, 2)

    def __call__(self, sino):
        """sino (n_ang, n_det) -> (H, W)."""
        rows = sino.reshape(N_ANG, 1, 1, N_DET)
        out = torch.zeros(N_IMG, N_IMG, device=self.device, dtype=sino.dtype)
        B = 100
        for i in range(0, N_ANG, B):
            g = self.grid[i:i + B]
            v = torch.nn.functional.grid_sample(
                rows[i:i + B], g, align_corners=True, padding_mode="zeros")
            out = out + v.sum(dim=0).reshape(N_IMG, N_IMG)
        return out * (math.pi / N_ANG)


def ramp_filter(sino, cutoff=1.0, window="ramp"):
    """Ram-Lak filtering along the detector direction -- using [the FFT of the discrete spatial
    kernel], not by sampling |w| directly in the frequency domain.

    The first version did multiply by |w| directly in the frequency domain, and FBP reached only
    ~19 dB (below a constant image). This is the classic mistake called out in Kak-Slaney
    (Principles of Computerized Tomographic Imaging, ch.3): frequency-domain sampling of the
    continuous ramp is inconsistent with the discrete convolution kernel at DC and low
    frequencies, producing a global bias.
    Discrete Ram-Lak kernel (unit sampling interval): h[0]=1/4, h[even]=0, h[odd]=-1/(pi^2 n^2).
    """
    n = sino.shape[-1]
    pad = 4 * n
    k = torch.arange(pad, device=sino.device, dtype=sino.dtype)
    k = torch.minimum(k, pad - k)                       # circular distance
    h = torch.zeros(pad, device=sino.device, dtype=sino.dtype)
    h[0] = 0.25
    odd = (k % 2 == 1)
    h[odd] = -1.0 / (math.pi ** 2 * k[odd] ** 2)
    H = torch.fft.rfft(h)
    f = torch.fft.rfftfreq(pad, d=1.0, device=sino.device)
    if window == "hann":
        # default window in dival/ODL: 0.5(1+cos(pi f/fc)), f<=fc. The official FBP 30.19 uses it.
        fc = f.max() * cutoff
        W = torch.where(f <= fc, 0.5 * (1 + torch.cos(math.pi * f / fc.clamp_min(1e-9))),
                        torch.zeros_like(f))
        H = H * W
    elif cutoff < 1.0:
        H = H * (f <= f.max() * cutoff)
    S = torch.fft.rfft(sino, n=pad, dim=-1)
    out = torch.fft.irfft(S * H, n=pad, dim=-1)[..., :n]
    return out


# --------------------------------------------------- reference methods ----
class FBP:
    """Filtered backprojection. calib is a one-off global physical calibration (see calibrate_fbp)."""

    def __init__(self, bp, calib=1.0, cutoff=1.0, window="ramp"):
        self.bp, self.calib, self.cutoff, self.window = bp, calib, cutoff, window

    def __call__(self, obs):
        return self.bp(ramp_filter(obs, self.cutoff, self.window)) * self.calib


def calibrate_fbp(bp, calib_file_idx=1, n=4):
    """Fit one global scale for FBP (a discretization constant) on a [dedicated calibration
    file].

    This is calibration of a physical constant, not learning: one number, shared by all samples,
    fitted on a file different from the evaluation file (file_idx=1 vs 0). Turning it into a
    per-sample fit would be cheating -- which is why the interface only allows returning a
    scalar.
    """
    with h5py.File(f"{DATA}/observation_test_{calib_file_idx:03d}.hdf5", "r") as h:
        obs = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32, device=bp.device)
    with h5py.File(f"{DATA}/ground_truth_test_{calib_file_idx:03d}.hdf5", "r") as h:
        gt = torch.tensor(np.array(h["data"][:n]), dtype=torch.float32, device=bp.device)
    num = den = 0.0
    for i in range(n):
        r = bp(ramp_filter(obs[i]))
        num += float((r * gt[i]).sum())
        den += float((r * r).sum())
    return num / max(den, 1e-30)


# --------------------------------------------------------- null models ----
class ConstantImage:
    """Return a constant image (a rough magnitude inferred from the observation mean). Any decent method must crush it."""

    def __call__(self, obs):
        return torch.full((N_IMG, N_IMG), 0.24, device=obs.device, dtype=obs.dtype)


class UnfilteredBP:
    """Unfiltered backprojection -- a physically wrong reconstruction, should be clearly worse than FBP."""

    def __init__(self, bp, calib=1.0):
        self.bp, self.calib = bp, calib

    def __call__(self, obs):
        r = self.bp(obs)
        return r * self.calib


# ------------------------------------------------------------- metrics ----
# Two conventions, both must be reported (pinned down on 2026-08-27 from the Leuschner21
# PMC8321320 text):
#   official PSNR: L = max(gt) - min(gt), adaptive per image -- the convention that is comparable
#   with the LoDoPaB leaderboard and papers;
#   fixed PSNR: L = 1.0 -- our own anti-cheating convention (a per-image range inflates flat
#   slices, see note #4 in the module docstring). Both are legitimate, but comparing numbers
#   across the two is a definition-mismatch accident.
def psnr(x, ref, data_range=1.0):
    mse = float(((x - ref) ** 2).mean())
    if data_range == "adaptive":
        data_range = float(ref.max() - ref.min())
    return 10 * math.log10(max(data_range, 1e-30) ** 2 / max(mse, 1e-30))


def ssim(x, ref, data_range=1.0):
    from skimage.metrics import structural_similarity
    if data_range == "adaptive":
        data_range = float(ref.max() - ref.min())
    return float(structural_similarity(
        x.detach().cpu().numpy(), ref.detach().cpu().numpy(),
        data_range=data_range))


# Officially published reference values (Leuschner et al. 2021, Table 2, LoDoPaB challenge set,
# official convention). Used for cross-checking; the challenge set is not our test set, so half
# a dB of difference is normal.
PUBLISHED = {
    "FBP": (30.19, 0.727), "TV": (33.36, 0.830), "iCTU-Net": (33.70, 0.844),
    "DIP + TV": (34.41, 0.845), "CINN": (35.54, 0.854), "U-Net++": (35.37, 0.861),
    "MS-D-CNN": (35.85, 0.858), "U-Net": (36.00, 0.862),
    "ISTA U-Net": (36.09, 0.862), "Learned P.-D.": (36.25, 0.866),
}


# --------------------------------------------------------------- tiers ----
def tier0(method, obs_sample):
    """Seconds: is the method at least physically sane. Returns (pass/fail, reason)."""
    try:
        r = method(obs_sample)
    except Exception as e:
        return False, f"raised an exception: {e}"
    if tuple(r.shape) != (N_IMG, N_IMG):
        return False, f"shape {tuple(r.shape)} != (362,362)"
    if not torch.isfinite(r).all():
        return False, "output contains nan/inf"
    if float(r.max()) - float(r.min()) < 1e-6:
        return False, "output is constant (unless you are the constant null model)"
    if float(r.abs().max()) > 100:
        return False, f"absurd value range max={float(r.abs().max()):.1f}"
    return True, "ok"


def tier1(method, obs, gt, name="?"):
    """Minutes: N images, both conventions reported. Only official_* is comparable with PUBLISHED."""
    po, pf, so = [], [], []
    for i in range(obs.shape[0]):
        r = method(obs[i])
        po.append(psnr(r, gt[i], "adaptive"))
        pf.append(psnr(r, gt[i], 1.0))
        so.append(ssim(r, gt[i], "adaptive"))
    po, pf, so = np.array(po), np.array(pf), np.array(so)
    return {"name": name, "n": len(po),
            "psnr_official": float(po.mean()), "psnr_official_std": float(po.std()),
            "psnr_fixed": float(pf.mean()),
            "ssim_official": float(so.mean()), "ssim_official_std": float(so.std())}
