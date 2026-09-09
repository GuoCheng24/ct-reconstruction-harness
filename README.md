# inverse-problems-from-scratch

[![test](https://github.com/GuoCheng24/ct-reconstruction-harness/actions/workflows/test.yml/badge.svg)](https://github.com/GuoCheng24/ct-reconstruction-harness/actions/workflows/test.yml)

**Reproduce a published CT reconstruction baseline, then beat it with a
paired test — and show every step of both, including the step where the
first version of this README was wrong.**

Low-dose CT reconstruction, built up from the forward operator: a GPU Radon
transform with an autodiff adjoint, proximal and Adam solvers, a scoring
harness that owns the ground truth, a parametrized family of operator
mismatches with guards that must fail on deliberately broken operators, and a
search loop that proposes variants and confirms them on held-out data.

```
                                                       ours          n      published
  FBP (Hann, fs 0.641)   official ODL+ASTRA operator   30.52 ± 0.05   3553    30.19
  FBP (Hann, fs 0.641)   the operator in ops/          29.33 ± 0.05   3553
  TV  (Adam, Poisson NLL, γ 20.556, anisotropic)       33.00 ± 0.33    128    33.36
  TGV ratio 0.3, γ 28    found by the loop             33.71 ± 0.36    128

  paired gain, TGV over the matched TV recipe,
  same 128 images:                                    +0.70 ± 0.04   t = 16.1   wins 125/128
```

LoDoPaB-CT **test** set, official metric convention, no tuning on the
evaluation files, ± is the standard error over images. The published numbers
are on the benchmark's **challenge** set, a different split: the FBP row shows
the two splits differ by about 0.3 dB with the official implementation, so
the TV row is a reproduction to within roughly that, not an exact match. The
row that took the work is TV; the row that means something is the paired
gain, because a paired comparison on the same images does not care whether
the images are easy.

## What the first version of this README said, and why it changed

The first published version of this table read FBP 31.05, TV 33.83, TGV 34.51
"passing the published DIP+TV 34.41". Every one of those numbers was a mean
over the first 16 of 3553 test images, and those 16 are **easier than the
rest**: by 0.95 dB for TV and 0.92 dB for TGV against the other 112 images
that were evaluated, and by 0.33 dB for FBP against the remaining 3537. The
FBP row was worse than that: its filter setting was the TV *initialization* (frequency scaling 0.1),
not the official FBP setting (0.641, from the benchmark authors' own
`lodopab_fbp_hyper_params.json`), and 31.05 does not occur in any logged run
at all — the nearest thing is four images under the non-official metric.

Re-measured on the full test set for FBP and on 128 images for the iterative
methods, the reproduction is real, the loop's gain is real and larger than
the noise by a factor of 16, and the claim of passing DIP+TV is withdrawn:
33.71 is below 34.41. The numbers, per image, are in
[`results/evaluation_n128.json`](results/evaluation_n128.json).

## Why this repository exists

Most reconstruction code either wraps a framework you cannot see inside, or
reports an improvement over a baseline it never actually matched. The
forward operator in `ops/` is 116 lines you can read; the adjoint is
verified against its own transpose on random inputs; the harness refuses to
report a gain that does not survive on files no tuning has touched. Be
precise about what that operator is: the headline numbers use the official
ODL+ASTRA operator, and the readable one is 1.2 dB behind it on FBP — a gap
that is measured here rather than hidden.

The seven differences between "the method as described" and "the method as
run" are written down in **[DEBUGGING.md](DEBUGGING.md)** — the optimizer that
is not the one in the equations, the loss that is not MSE, the constants that
live in a JSON file, the metric convention worth 3 dB, and the forward-operator
transpose that took eight hypotheses to find. That file is the most useful
thing here.

## What is inside

| Directory | |
|---|---|
| [`ops/`](ops/) | Radon forward operator on GPU, geometry, MRI operator; adjoint by vector-Jacobian product rather than by hand |
| [`solvers/`](solvers/) | Proximal operators and splitting schemes |
| [`harness/`](harness/) | The scoring harness: one evaluation entry point, tiered (seconds / minutes / full budget), null models, both metric conventions, the published-number table |
| [`harness/mismatch.py`](harness/mismatch.py) | A one-parameter family of operator mismatches — gantry angle offset, detector scale, per-angle jitter, centre-of-rotation shift — each reported in pixels or angular steps, with four guards that must fail on deliberately broken operators |
| [`harness/identifiability.py`](harness/identifiability.py) | Which of those mismatches lose information at all: two are coordinate gauges, the centre-of-rotation shift is recovered in closed form from the Helgason–Ludwig moment condition to within 0.0022 pixel with no ground truth, and to within 0.0143 pixel with three other axes wrong at the same time, and only the jitter is a genuine residual |
| [`guards/`](guards/) | The adjoint test — on **random** inputs, because structured ones pass a wrong adjoint with error exactly 0.0 |
| [`honesty/`](honesty/) | Null-space decomposition: how much of a reconstruction is determined by the data and how much is prior |
| [`crime/`](crime/) | Inverse-crime control — does the result depend on who generated the observations? |
| [`loop/`](loop/) | The search loop: propose variants with reasons, sweep, select on calibration, confirm on held-out |
| [`learned/`](learned/) | The learned side: the official Learned Primal-Dual and FBP+U-Net recipes (dival architectures, hyper-parameters read from the published files, five source-only defaults documented), trained from scratch on the validation split with interruptible checkpoints, and an operator-swappable LPD so that the same mismatch family can be run against a network |

Full numbers, per-image spread, and the findings from round 2 are in
**[RESULTS.md](RESULTS.md)**.

## What happens to a learned reconstructor when the operator is wrong

The same four mismatch axes, applied at deployment to an LPD trained on the
official operator, on the first 128 images of test file 000 — the same file
and the same images as the classical table above
([`results/lpd_mismatch_n128.json`](results/lpd_mismatch_n128.json)):

```
                          naive     swap in the        closed-form
  axis         delta   (wrong op)  true operator   self-calibration
  cor          4 px     20.65        34.97            34.97      ← baseline 34.98
  rot          4 steps  27.91        34.97              —
  det_scale    0.5 %    28.99        34.78              —
  ang_jitter   4 steps  32.60        32.43              —        ← not recovered
```

Geometric mismatch is **fully removable**: a 4-pixel centre-of-rotation error
costs 14.3 dB, swapping the correct operator into the trained network (weights
untouched) returns to within 0.009 dB of the baseline, and estimating that
shift from the sinogram alone — the Helgason–Ludwig first moment, no ground
truth — lands within 0.0019 dB of the oracle. The one axis that swapping does
not recover is per-angle jitter, which is the one that actually destroys
measurement information; `identifiability.py` predicts that split before any
network is trained. The hoped-for next step — using the consistency residuals
as a per-scan error bar for the network — does **not** pass: they correlate
with the damage (r = 0.85) but under-estimate it on the gauge axes by up to
7.1 dB, and under-estimation is the unsafe direction.

This table was produced twice before. Once on a mis-configured LPD (five
source-only defaults wrong, see `learned/lpd.py`) and on 4 images; that
version was retracted and re-run with the official recipe at n = 128, and the
conclusions survived — the retracted model's own baseline, re-measured at
n = 128, was 4 dB lower than its 4-image number had suggested. Then once more
on test file 001 rather than 000. File 001 is the file this repository
reserves for calibration, and no hyper-parameter was ever chosen on it here,
so nothing was leaked — but the documentation claimed the table was on file
000 while the data said 001, and the classical table above really is on 000.
It has been re-measured on 000 so that the two tables are on the same images.
Every conclusion is unchanged; the numbers move by 0.1–0.5 dB, file 001 being
the easier of the two. All four runs, and which file each is on, are in the
results file.

## When several axes are wrong at once

One parameter at a time is not how a miscalibrated scanner fails. The
centre-of-rotation shift combined with a gantry-angle offset, a
detector-scale error and per-angle jitter, same 128 images
([`results/lpd_mismatch_composed_n128.json`](results/lpd_mismatch_composed_n128.json)):

```
                            naive   oracle   self-calibrated   error in ĉ
  cor 1 px                  28.02    34.98        34.98        −0.001 px
  cor 1 px + rot 4 steps    26.37    34.97        27.92        −0.001 px
  cor 1 px + scale 0.1 %    28.22    34.88        34.49        −0.001 px
  cor 1 px + jitter 4       28.24    32.43        32.59        +0.014 px
  all four at once          28.02    33.98        33.01        +0.002 px
```

The estimate of the shift survives all of it: across the ten combinations its
error stays between 0.0012 and 0.0143 pixel, and it is the same size at a
1-pixel and at a 4-pixel shift. Most of that is not calibration error at all:
run the estimator on the *unperturbed* sinograms, where the true offset is
zero, and it already reports −0.0012 px. Subtract that floor and what is
genuinely left over is under 4 × 10⁻⁵ px out to a one-pixel shift, reaching
1.0 × 10⁻³ px only at four. The floor belongs to the images — the other test
file gives −0.0049 px, 4.04 times larger — while the part that responds to
the shift agrees between the two files to 6 per cent, so it belongs to the
method ([`harness/run_estimator_floor.py`](harness/run_estimator_floor.py)).
What self-calibration cannot do is repair the axes it does not
estimate. With a 4-step gantry offset also present it still recovers the
centre-of-rotation and still leaves the rotation, which costs 7 dB against
the oracle — a coordinate gauge that the network cannot absorb because it is
not equivariant; re-scoring its output after undoing the rotation recovers
most of that (32.23 against 27.92).

## Three results worth knowing even if you never run this

**TGV with ratio below 1 beats the literature convention.** Second-order
total generalized variation is conventionally run with its two weights in a
2:1 ratio. Here the optimum is well under 1: +0.70 ± 0.04 dB over the matched
TV recipe on 128 held-out images, paired, winning on 125 of them.

**Isotropic TV beats the official anisotropic choice.** The benchmark's
reference recipe uses anisotropic TV. Swapping in isotropic TV, nothing else
changed, is +0.36 ± 0.02 dB, paired, on 127 of 128 images. This repository's
own docstring predicted it; it is now measured.

**The Poisson likelihood buys you its variance weighting and nothing else.**
Swapping in a weighted least-squares data term is −0.03 ± 0.03 dB from
Poisson, paired on 32 images, p = 0.28. If your pipeline carries a Poisson
NLL for principle rather than for the weighting, it is carrying the cost for
none of the benefit.

## One thing the guards found outside this repository

Guard 2 of `harness/mismatch.py` requires that every mismatch axis actually
changes the sinogram at the one-pixel scale. On first run the
centre-of-rotation axis changed it by exactly 0.0: ODL's ASTRA backend
builds a 2D parallel-beam geometry from a pixel width, a pixel count and the
angles, so a shifted detector partition is silently dropped. That is
[odlgroup/odl#359](https://github.com/odlgroup/odl/issues/359), open since
2016; on current ODL the CUDA path no longer has the problem and the CPU path
still does, so the two backends disagree without a warning. This repository
synthesizes the shift explicitly instead. A guard that had passed first time
would have produced a flat "robust to COR" curve and nobody would have
questioned it.

## Quick start

```bash
pip install torch numpy scipy h5py
pip install "odl==0.8.1"     # 1.0 is an API rewrite; tomo moved and numpy interop breaks
export LODOPAB_DIR=/path/to/lodopab

python guards/adjoint.py             # verify the operator before trusting anything (seconds)
python tests/test_mismatch_guards.py # every guard against a deliberately broken input
python harness/run_selfcalib.py      # closed-form COR self-calibration, no ground truth (minutes)
```

The adjoint check runs first for a reason: everything downstream is a claim
about an operator, and an operator whose adjoint is wrong produces numbers
that look entirely reasonable.

## Scope, honestly

The reconstruction results are at the **classical, untrained** end of the
problem, where the ceiling sits near 33.7 dB on the test split used here;
learned reconstructors on this benchmark report 35.4–36.3 on the challenge
split, and the LPD trained in `learned/` reaches 35.0 on 128 test images.
The learned side is used as a test bed for operator mismatch, not as a
contender on the leaderboard. What is demonstrated here is a reproduction
done to the bottom, a search loop that found a gain that survives a paired
test at n = 128, a mismatch study whose one negative gate is reported as
such, and a harness that caught the moment its own headline numbers were
too good.

Training the learned models needs the LoDoPaB validation split
(`LODOPAB_VAL_DIR`), dival 0.6.2 and a GPU for a few hours; the trained
checkpoints are not in the repository. Everything else runs from the test
files alone.

The methodology the loop follows is written up separately in
[breakthrough-harness](https://github.com/GuoCheng24/breakthrough-harness).

## License

MIT.
