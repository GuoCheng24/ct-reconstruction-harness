# Seven layers between a published number and your own

The official LoDoPaB-CT TV baseline is 33.36 dB. Our first honest attempt at
"the same method" scored 29.4. Every one of the seven differences below was
worth between 0.3 and 3 dB, none was stated in a sentence anyone would call
a description of the method, and the last one took eight hypotheses to find.

This file is the log. It is here because a reproduction that only reports its
final number teaches nothing, and because every layer is a place where a
different project would have stopped and reported an improvement over a
baseline it had not actually matched.

## Layer 1 — the optimizer is not the one in the equations

TV-regularized reconstruction is presented in textbooks as a splitting
method: ADMM, primal-dual, FISTA. The official baseline runs **Adam** on the
unconstrained objective. Not an approximation of a splitting method — plain
Adam, 5000 iterations.

This matters beyond convergence speed. Adam normalizes per-parameter, which
means it partially hides a wrong regularizer scale (see layer 4): a sweep
that would look catastrophic under gradient descent looks merely flat.

## Layer 2 — the loss is not MSE

The data term is the **Poisson negative log-likelihood** of the pre-log
counts, not the mean squared error of the post-log sinogram. In our
implementation the two agree to about 0.07 dB once you note that the Poisson
NLL is, to second order, an inverse-variance weighted MSE — which is exactly
why substituting a weighted least-squares data term reproduces it (we
measured this in round 2: `wls` lands 0.07 dB from `poisson`).

The consequence is not the 0.07 dB. It is that the regularization weight
that balances a *weighted* data term is nowhere near the one that balances an
unweighted one, and swapping the loss without moving the weight puts you
three decades away from the optimum.

## Layer 3 — the constants live in a JSON file, not the paper

γ = 20.556, 5000 iterations. These are in the framework's configuration
files, not in the text. Once the loss was right (layer 2), the published γ
transferred **exactly** — no retuning. That is the tell that you have finally
matched the recipe: their constants become optimal for your code.

Corollary worth stating: if you find yourself retuning a published
hyperparameter to make a published method work, you have not reproduced the
method. You have built a different one that happens to share its name.

## Layer 4 — the initialization is part of the method

Filtered back-projection with a **Hann filter at frequency scaling 0.1**, not
zeros, not a plain ramp FBP. On a non-convex-in-practice 5000-step run, the
initialization is not a detail; it selects which basin you optimize in.

## Layer 5 — no constraints

No non-negativity, no box constraint. Adding the physically obvious
constraint makes the number worse. We tried, because it seemed too obvious
to be wrong.

## Layer 6 — the metric convention

The official PSNR uses an **adaptive per-image data range** (max − min of the
ground truth), not a fixed range of 1.0. The same reconstruction scores about
3 dB apart under the two conventions. A table is only comparable under its
own convention, which is why this harness reports both, labelled.

This one is not a subtlety of tomography. It is arithmetic, and it is
responsible for more incomparable tables in the literature than any modeling
choice on this list.

## Layer 7 — the forward operator itself

The last 2.4% of residual mismatch survived all six layers above. Seven
hypotheses failed: detector spacing, angle convention, filter normalization,
interpolation order, ray-driven versus pixel-driven projection, scaling of
the post-log transform, and detector count.

The eighth was a **transpose**: applying our operator to the transposed
ground truth dropped residuals by factors of 6 to 485 depending on the image.
The per-sample spread was itself the clue — the error was coupled to how
asymmetric each image happened to be, which is why one nearly symmetric
sample had shown a reassuring 0.976 correlation and proved nothing.

**Rule extracted:** a validation sample that cannot distinguish your
hypothesis from its negation is not weak evidence. It is no evidence, and it
is worse than none because it feels like some.

---

# Four lessons that were not about this baseline

## The adjoint is the transpose of your implementation, not the inverse of your geometry

Our first back-projector implemented the geometrically obvious thing: rotate
by −θ and accumulate. It is not the adjoint. `grid_sample` is a *gather*; its
adjoint is a *scatter*, and the two differ wherever the sampling grid is not
measure-preserving. The dot-product test caught it at errors of 0.49 to 0.82.

The fix is to stop deriving the adjoint and let autodiff take the vector-Jacobian
product of the forward operator — exact for a linear map, by construction.

**And the part that is easy to miss:** a wrong adjoint passes a dot-product
test with error *exactly* 0.0 when the test inputs are structured (constant
images, symmetric phantoms). Structure in the test input can cancel exactly
the asymmetry the test is meant to detect. Random inputs, always.
`guards/adjoint.py` does this.

## A stability criterion of 1/L is sufficient, not necessary

We derived a step size bound of 1/L, observed divergence above it, and
believed the derivation. The true bound for the proximal-gradient iteration
here is 2/L: step 2.00 converges, 2.01 diverges to 4e21. A shrinking prox
raises it further still.

The cost of the wrong constant was a factor of two in every experiment that
used it, which is the kind of error that never announces itself.

## A flat hyperparameter sweep is not evidence of insensitivity

The regularization weight swept across four decades with no measurable
effect. The conclusion "this parameter does not matter" was wrong: every
tested value was on the same side of the force balance. One gradient-norm
evaluation at the starting point — ‖∇data‖ versus γ‖∇reg‖ — located the
balance three decades below the sweep floor.

Two things conspired: Adam's per-parameter normalization hid the absolute
scale, and a data term that averages over measurements was being balanced
against a regularizer that sums over pixels.

---

## A p-value of 4 × 10⁻³¹ that a change of random seed reverses

The mismatch family has one axis that is not a coordinate gauge: per-angle
jitter genuinely destroys measurement information. On that axis the
self-calibrated operator — which carries an estimated centre-of-rotation and
no jitter at all — scored *above* the oracle operator carrying the true
jitter: +0.16 dB, paired over 128 images, t = 11.9, p = 2 × 10⁻²², winning on
121 of them. Read as an unrolled network preferring the operator it was
trained on over the one that actually produced the data, that would have been
worth a paragraph of its own.

It is not true. The same measurement with seven further jitter realizations:

```
  seed    swap   selfcal   selfcal − swap   paired t          p    wins
    0    32.43     32.59        +0.159        +11.92   2 × 10⁻²²  121/128
    1    32.32     32.50        +0.184        +13.48   3 × 10⁻²⁶  120/128
    2    32.37     32.56        +0.191        +11.58   1 × 10⁻²¹  118/128
    3    32.55     32.33        −0.220         −7.94   9 × 10⁻¹³   30/128
    4    32.20     32.44        +0.248        +15.52   4 × 10⁻³¹  125/128
    5    32.25     32.40        +0.149         +9.96   1 × 10⁻¹⁷  105/128
    6    32.32     32.35        +0.032         +2.82   6 × 10⁻³    60/128
    7    32.51     32.48        −0.035         −3.34   1 × 10⁻³    38/128
```

Six point one way and two the other, and every one of them is significant.
The claim is about the *distribution* of jitter, so the unit of replication is
the jitter realization, not the image — all 128 images share a single draw.
Across the eight draws: +0.09 ± 0.05 dB, t = 1.62, **p = 0.15**.

Nothing about the paired test was wrong. It is a correct, very powerful test
of "given this particular jitter, does self-calibration win", and that is not
a question anyone has. The two p-values differ by thirty orders of magnitude
and only the weak one is answering the question that was asked.

The general form: when a perturbation is drawn once and applied to every
item, the items are not replicates of the claim. The cheapest defence is to
re-draw the perturbation before writing the sentence down, which costs one
more run and is the only thing that separates this paragraph from a result.
Per-image numbers for all eight draws are in
[`results/lpd_jitter_seed_sweep.json`](results/lpd_jitter_seed_sweep.json).

---

# What this cost, and what it bought

Seven layers, four separate lessons, and eight hypotheses on the last layer.
The result is a reproduction that matches the published number without any
tuning, which is the only state from which an improvement claim means
anything.

Once there, the search loop found `tgv2 ratio 0.3` on held-out data —
+0.70 ± 0.04 dB above the recipe we had just matched, paired on 128 images,
winning 125 of them. (An earlier version of this sentence quoted 34.51 dB and
"above the published DIP+TV 34.41"; that was the mean of the first 16 images,
which are 0.8 dB easier than the rest, and at n = 128 the winner sits at
33.71, below DIP+TV. The paired gain is what survived.) That gain is worth
reporting precisely because
everything above it was checked first.
