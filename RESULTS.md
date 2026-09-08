# Results

All numbers below were produced by the code in this repository on the public
LoDoPaB-CT **test** set, under the official metric convention, with no tuning
on the evaluation files. ± is the standard error over images. Every number is
re-derivable and the per-image values are in
[`results/evaluation_n128.json`](results/evaluation_n128.json); where a
published figure is quoted, it is on the benchmark's **challenge** set, a
different split, and its source is named.

**Revision note (2026-09-08).** The first version of this file reported FBP
31.05, TV 33.83 and TGV 34.51 from the first 16 test images. Those 16 are
0.8 dB easier than the rest, and the FBP row used the TV initialization
filter (fs 0.1) rather than the official FBP setting (fs 0.641). Everything
below is re-measured at n = 128 (iterative methods) or n = 3553 (FBP). The
paired gain of the loop's winner over the matched recipe survived unchanged;
the absolute claim of passing DIP+TV did not.

## The setup

LoDoPaB-CT: 362 × 362 images over a 26 cm field of view, 1000 projection
angles, 513 detector bins spanning the image diagonal, Poisson noise at
N₀ = 4096 incident photons, post-log normalization by µ_max = 81.35858.

Two disjoint file sets are used throughout: **calibration** (all tuning
happens here) and **evaluation** (touched only by final confirmation runs).
The harness owns the ground truth; a method receives observations only.

## Baselines reproduced

| Method | Ours | n | Published (challenge set) |
|---|---|---|---|
| FBP, Hann fs 0.641, **official ODL+ASTRA operator** | **30.52 ± 0.05**, SSIM 0.737 | 3553 | 30.19, SSIM 0.727 |
| FBP, Hann fs 0.641, the operator in `ops/` | 29.33 ± 0.05, SSIM 0.753 | 3553 | |
| TV, Adam, Poisson NLL, γ = 20.556, anisotropic (the official recipe) | **33.00 ± 0.33**, SSIM 0.797 | 128 | 33.36, SSIM 0.830 |

fs 0.641 is the official FBP setting (`lodopab_fbp_hyper_params.json`,
jleuschn/supp.dival); fs 0.1 is the FBP used to *initialize* TV and must not
be reported as an FBP baseline. The FBP row with the official operator
reproduces the published implementation to +0.33 dB across the split change,
which is the size of the test/challenge difference. The `ops/` operator is
1.19 dB behind the official one on identical images: it is the readable
one, not the one the TV and TGV rows are computed with. Its single fitted
calibration scalar moves the number by 0.43 dB depending on whether it is
fitted on 4 or 64 images (29.33 vs 28.91); the 4-image fit is what the code
does and what is reported.

The TV number uses the **published γ untouched**. Getting there took seven
layers of recipe reading; they are logged in [DEBUGGING.md](DEBUGGING.md).
On the test split it lands 0.36 dB below the challenge-split publication,
with the FBP control suggesting the splits differ by about 0.3 dB in the
other direction; so the matched recipe is within roughly half a decibel of
the reference implementation and should not be described as an exact
match.

## Search loop, round 1

Twelve variants, each carrying a one-line reason it might win. Selection on
calibration, confirmation on held-out at full budget.

Surviving gain: **Huber-TV, +0.39 dB** over the matched TV recipe,
reproduced on held-out data.

## Search loop, round 2 — the classical space closes

Variant space, with reasons: Huber δ ∈ {0.002 … 0.008} (the unexplored
parameter of round 1's winner) × TGV ratio ∈ {0.3 … 2.0} (the literature
convention of ratio = 2 is not obviously optimal) × a weighted-least-squares
control (is the value of the Poisson loss only its variance weighting?).

**24 configurations, 4 parallel workers on 2 GPUs, ~13 minutes, zero failed runs.**

Calibration ranking:

| Configuration | PSNR | Note |
|---|---|---|
| **tgv2 r0.5 @ γ20** | **36.52** | ratio < 1 is a real hole in the literature convention |
| tgv2 r0.3 @ γ28 | 36.48 | γ and ratio interact |
| tgv2 r0.5 @ γ20 (wls) | 36.45 | **0.07 dB from Poisson — the loss's value is its weighting** |
| huber δ0.002 @ γ28 | 36.03 | fine-tuning δ yields little (+0.12) |
| official recipe (control) | 35.49 | |

Held-out confirmation on evaluation file 0, **128 images** × 5000 iterations,
zero tuning contact. Paired differences are on the same 128 images:

| Configuration | PSNR | SSIM | paired vs. matched recipe | t | wins |
|---|---|---|---|---|---|
| **tgv2 r0.3 @ γ28** | **33.71 ± 0.36** | 0.804 | **+0.70 ± 0.04** | 16.1 | 125/128 |
| iso_tv @ γ20.556 (the official recipe with isotropic TV) | 33.36 ± 0.34 | 0.801 | +0.36 ± 0.02 | 19.0 | 127/128 |
| aniso_tv @ γ20.556 (the official recipe) | 33.00 ± 0.33 | 0.797 | — | | |

The first 16 of these images, which the previous version of this table used
on their own, average 34.51 for the winner and 33.83 for the matched recipe:
0.8 dB above the 128-image means for both, with the paired gain (+0.68 on
those 16) essentially unchanged. That is the point of pairing.

For context, a published untrained-network result on this benchmark,
DIP + TV, reports **34.41** on the challenge split. The winner here is
**below** that at n = 128; the earlier claim of passing it rested on the easy
subset and is withdrawn. Learned reconstructors occupy 35.4–36.3, and that
is the real frontier.

### Three findings worth stating separately

**TGV with ratio < 1 beats the literature convention.** Second-order total
generalized variation is conventionally used with the ratio of its two
weights set to 2. On this problem the optimum is well below 1, and the
effect survives held-out confirmation at n = 128 by sixteen standard errors.
The convention is not wrong so much as untested here.

**Isotropic TV beats the official anisotropic choice, paired, on 127 of 128
images (+0.36 ± 0.02 dB).** The reference implementation's `tv_loss` is
anisotropic; this repository's docstring called the choice an implementation
preference and predicted isotropic would do better in CT. It does.

**The Poisson likelihood's contribution is its variance weighting, and
nothing else.** On the calibration file (4 images) a weighted least-squares
data term matched it to 0.07 dB; re-checked paired on 32 evaluation images
with `tgv2 r0.5 @ γ20`, the difference is −0.03 ± 0.03 dB (p = 0.28). This
is a negative result about a modeling choice many pipelines treat as
essential, and it is cheap to act on.

## Inverse crime control

A three-tier control asks whether these numbers depend on generating our own
observations. Tier A — observations produced by the official operator, our
reconstruction — scores **33.66** against 33.83 on the real observations of
the same 16 images (both first-16 means; the comparison is like for like even
though those images run easy), i.e. the recipe is not the source of any
advantage. The benchmark's own simulation is structurally an inverse crime
setup: the observations are generated by the same discretized operator the
reconstructor inverts.

The sensitivity is worth recording: a **3% sinogram discrepancy produces a
1.8 dB reconstruction difference**, which is the ill-posedness of the problem
amplifying a small forward-model mismatch. It is also why layer 7 of the
debugging log mattered so much.

## Reproducing these numbers

```bash
export LODOPAB_DIR=/path/to/lodopab       # public dataset
python harness/candidate_eval.py '{"reg":"aniso_tv","data":"poisson","gamma":20.556,"iters":5000,"n":128,"file":0}'   # matched recipe, 33.00
python harness/candidate_eval.py '{"reg":"tgv2","ratio":0.3,"data":"poisson","gamma":28,"iters":5000,"n":128,"file":0}'  # the winner, 33.71
```

Runtime is dominated by the 5000-iteration reconstructions; on one modern GPU
the full evaluation set takes a few hours. `odl` must be pinned to **0.8.1** —
version 1.0 is an API rewrite in which `odl.tomo` moved and numpy interop
breaks.
