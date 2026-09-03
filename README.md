# inverse-problems-from-scratch

**Reproduce a published CT reconstruction baseline exactly, then beat it — and
show every step of both.**

Low-dose CT reconstruction, built up from the forward operator: a GPU Radon
transform with an autodiff adjoint, proximal and Adam solvers, a scoring
harness that owns the ground truth, and a search loop that proposes variants
and confirms them on held-out data.

```
                                 ours     published
  FBP (Hann, fs 0.1)             31.05      30.19
  TV  (Adam, Poisson NLL)        33.83      33.36     ← published gamma, untouched
  TGV ratio 0.3  (found by the loop)
                                 34.51      34.41     ← published DIP+TV, for context
```

LoDoPaB-CT test set, official metric convention, no tuning on the evaluation
files. The middle row is the one that took the work: getting a matched
reproduction is what makes the third row mean anything.

## Why this repository exists

Most reconstruction code either wraps a framework you cannot see inside, or
reports an improvement over a baseline it never actually matched. This does
neither. The forward operator is 116 lines you can read; the adjoint is
verified against its own transpose on random inputs; the harness refuses to
report a gain that does not survive on files no tuning has touched.

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
| [`guards/`](guards/) | The adjoint test — on **random** inputs, because structured ones pass a wrong adjoint with error exactly 0.0 |
| [`honesty/`](honesty/) | Null-space decomposition: how much of a reconstruction is determined by the data and how much is prior |
| [`crime/`](crime/) | Inverse-crime control — does the result depend on who generated the observations? |
| [`loop/`](loop/) | The search loop: propose variants with reasons, sweep, select on calibration, confirm on held-out |

Full numbers, per-image spread, and the two findings from round 2 are in
**[RESULTS.md](RESULTS.md)**.

## Two results worth knowing even if you never run this

**TGV with ratio below 1 beats the literature convention.** Second-order total
generalized variation is conventionally run with its two weights in a 2:1
ratio. Here the optimum is well under 1, by 0.68 dB on held-out data — enough
to pass the published DIP+TV number with an untrained method.

**The Poisson likelihood buys you its variance weighting and nothing else.**
Swapping in a weighted least-squares data term reproduces it to 0.07 dB. If
your pipeline carries a Poisson NLL for principle rather than for the
weighting, it is carrying the cost for none of the benefit.

## Quick start

```bash
pip install torch numpy scipy
pip install "odl==0.8.1"     # 1.0 is an API rewrite; tomo moved and numpy interop breaks
export LODOPAB_DIR=/path/to/lodopab

python guards/adjoint.py           # verify the operator before trusting anything (seconds)
python harness/run_tvadam_eval.py  # the matched baseline
```

The adjoint check runs first for a reason: everything downstream is a claim
about an operator, and an operator whose adjoint is wrong produces numbers
that look entirely reasonable.

## Scope, honestly

This is the **classical, untrained** end of the problem, and round 2 closed
it: the ceiling in this space is around 34.5 dB. Learned reconstructors on
this benchmark sit at 35.4–36.3, and that is where the frontier is. What is
demonstrated here is a reproduction done to the bottom, a search loop that
found a real gain, and a harness that would have caught it if the gain had
not been real.

The methodology the loop follows is written up separately in
[breakthrough-harness](https://github.com/GuoCheng24/breakthrough-harness).

## License

MIT.
