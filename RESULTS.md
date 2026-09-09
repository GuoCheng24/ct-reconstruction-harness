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

## Learned side: operator mismatch on a trained LPD

`learned/` trains the official Learned Primal-Dual recipe (dival architecture,
n_layer 3, no batch norm, PReLU, Xavier init, opnorm normalization) on the
3522 validation pairs for 35,220 steps, then applies the same four mismatch
axes as `harness/mismatch.py` at deployment. The first 128 images of test
file 000 — the evaluation file, the same images as the classical tables above
([`results/lpd_mismatch_n128.json`](results/lpd_mismatch_n128.json)):

| axis | delta | naive | swap (true operator) | self-calibrated | gauge-fixed |
|---|---|---|---|---|---|
| — | 0 | **34.98** | | | |
| cor | 1 px | 28.02 | 34.98 | 34.98 | |
| cor | 4 px | 20.65 | 34.97 | 34.97 | |
| rot | 4 steps | 27.91 | 34.97 | | 32.23 |
| det_scale | 0.5 % | 28.99 | 34.78 | | 34.17 |
| ang_jitter | 4 steps | 32.60 | **32.43** | | |

An earlier version of this section carried the same table measured on test
file 001, the calibration file, while saying it was on file 000. No
hyper-parameter was ever selected on file 001 for the learned side, so
nothing leaked; the label was simply wrong, and the classical tables it is
read against are on 000. Re-measured on 000, every conclusion is unchanged
and the numbers move by 0.1–0.5 dB. Both runs are in the results file, each
carrying the file it was measured on.

Three things the table says. Geometric mismatch is fully removable by giving
the trained network the right operator, weights untouched; for the
centre-of-rotation axis the right operator is recoverable from the sinogram
alone (Helgason–Ludwig first moment; the estimate lands within 0.0022 px of the
injected shift across 0.25–4 px), so the self-calibrated column is within
0.0019 dB of the oracle with no ground truth. Two earlier versions of this
paragraph quoted bounds tighter than their own data: 0.003 px, which was the
classical-side figure from the TV experiment rather than this table's, and
0.006 dB against a measured 0.0064. Every bound here is now the measured
maximum itself, and `guards/readme_bounds.py` fails if any quoted bound is
smaller than what the results file shows — rounding a bound down is silent
otherwise.
Per-angle jitter is not removable — it is the one axis that changes the
information content of the measurement, exactly as `identifiability.py`
classifies it. And the gauge axes are *not* free for the network the way they
are for TV: undoing the rotation on the output recovers only part of the loss
(32.23 vs 34.97), because a CNN is not equivariant.

### Several axes wrong at once

The single-axis table moves one parameter at a time, which is not how a
miscalibrated scanner fails. Combining them, same 128 images
([`results/lpd_mismatch_composed_n128.json`](results/lpd_mismatch_composed_n128.json)):

| combination | naive | oracle | self-calibrated | gauge-fixed | error in ĉ |
|---|---|---|---|---|---|
| cor 1 px | 28.02 | 34.98 | 34.98 | | −0.0012 px |
| cor 4 px | 20.65 | 34.97 | 34.97 | | −0.0022 px |
| cor 1 + rot 1 step | 27.91 | 34.98 | 33.76 | 34.68 | −0.0010 px |
| cor 1 + rot 4 steps | 26.37 | 34.97 | 27.92 | 32.23 | −0.0014 px |
| cor 1 + scale 0.1 % | 28.22 | 34.88 | 34.49 | 34.49 | −0.0012 px |
| cor 1 + jitter 1 step | 27.92 | 33.94 | 34.28 | | +0.0029 px |
| cor 1 + jitter 4 steps | 28.24 | 32.43 | 32.59 | | +0.0143 px |
| cor 4 + jitter 4 steps | 20.99 | 32.42 | 32.58 | | +0.0134 px |
| all four | 28.02 | 33.98 | 33.01 | 33.74 | +0.0016 px |

The closed-form estimate of the shift is the part that survives: its error
stays inside 0.0022 pixel whenever jitter is absent and inside 0.0143 pixel
when jitter is present, and it is the same size at a 1-pixel and at a
4-pixel shift, so what is left is a fixed offset of the estimator on these
images and not an error that scales with the quantity being estimated.
Self-calibration then removes exactly the axis it estimates and no other:
against a simultaneous 4-step gantry offset it still lands the
centre-of-rotation and still gives up 7 dB to the oracle, because the leftover
rotation is a gauge that an unrolled CNN cannot absorb.

On the two jitter rows the self-calibrated column reads *above* the oracle.
That ordering is not real — it reverses when the jitter is re-drawn, and the
paired-over-images test that makes it look certain is answering the wrong
question. The arithmetic is in DEBUGGING.md and the numbers in
[`results/lpd_jitter_seed_sweep.json`](results/lpd_jitter_seed_sweep.json).

The gate this direction was built to test — can the consistency residuals,
computable without ground truth, serve as an error bar for the learned
method — **fails**: Pearson 0.85 with the measured loss, but a mean absolute
deviation of 2.6 dB with systematic under-estimation on the gauge axes
(rot 4 steps: predicted 0.0, measured 7.1). Under-estimation is the unsafe
direction. Reported as a negative result.

Provenance: this table was first produced on 2026-08-28 with an LPD that had
five source-only defaults wrong (n_layer 4, batch norm on at batch size 1,
LeakyReLU, Kaiming init) on 4 images, retracted the same evening, and re-run
with the official recipe at n = 128. Re-measuring the mis-configured model at
n = 128 gives a baseline of 30.76 against the 34.75 its 4-image table had
shown — those two runs are both on file 001, so that the mis-configured model
is compared with itself on the same images. The version reported above was
then re-measured on file 000 for the reason given there. The structural
conclusions are the same in all four runs.

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
