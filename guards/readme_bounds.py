"""Every number the documentation quotes is re-derived from the committed results.

The documentation and the results files drift apart silently: a table gets re-measured
and one of the three places that quote it is missed, or -- as happened here -- the prose
says which test file a table came from and the data says a different one. Nothing fails,
the repository just starts lying.

This guard re-reads `results/*.json` and checks, for every table and every bound in
README.md, RESULTS.md and DEBUGGING.md:

  * each cell of the mismatch tables, against the run it claims to come from;
  * each scalar bound (the cost of a 4-pixel shift, the self-calibration gap, the
    correlation of the consistency residuals, ...), recomputed from the per-axis rows;
  * the *file label*: the prose names a test file, and the run's `file` field must agree.

It is meant to be run in CI and needs no data, no GPU and no odl -- only the committed
JSON. Run `python guards/readme_bounds.py --self-test` to watch it fail on a
deliberately corrupted copy of the documentation.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILURES = []


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def flat(s):
    """Literal checks must not depend on where the prose happens to wrap."""
    return " ".join(s.split())


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'   ' + detail if detail and not ok else ''}")
    if not ok:
        FAILURES.append(f"{label}{'   ' + detail if detail else ''}")


def row(rows, axis, delta):
    for r in rows:
        if r["axis"] == axis and abs(r["delta"] - delta) < 1e-9:
            return r
    raise KeyError(f"{axis} {delta} not in results")


def fenced(doc, heading):
    """The first ``` block after a heading."""
    after = doc.split(heading, 1)[1]
    return after.split("```", 2)[1]


def md_table(doc, heading, stop=None):
    """Rows of the first pipe table after a heading, as lists of stripped cells."""
    after = doc.split(heading, 1)[1]
    if stop:
        after = after.split(stop, 1)[0]
    out = []
    for line in after.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        out.append(cells)
    return out


def num(cell):
    cell = cell.replace("**", "").replace("−", "-").replace("ĉ", "").strip()
    m = re.search(r"-?\d+\.?\d*", cell)
    return float(m.group()) if m else None


def main(readme, results_md, debugging):
    mm = load("lpd_mismatch_n128.json")
    primary = mm["official_recipe_n128"]
    rows = primary["rows"]
    base = row(rows, "none", 0.0)["naive"]
    comp = load("lpd_mismatch_composed_n128.json")
    seeds = load("lpd_jitter_seed_sweep.json")

    print("file labels")
    check("the learned table's run is on test file 000", primary["file"] == 0,
          f"the results file says file {primary['file']}")
    check("README says so", "test file 000" in flat(readme))
    check("RESULTS.md says so", "test file 000" in flat(results_md))
    check("the composed run is the same file", comp["file"] == primary["file"],
          f"{comp['file']} vs {primary['file']}")
    check("the seed sweep is the same file", seeds["file"] == primary["file"],
          f"{seeds['file']} vs {primary['file']}")

    print("README, the four-axis table")
    tbl = fenced(readme, "## What happens to a learned reconstructor when the operator is wrong")
    quoted = {}
    for line in tbl.splitlines():
        parts = line.split()
        if parts and parts[0] in ("cor", "rot", "det_scale", "ang_jitter"):
            # skip the axis, the delta and its unit; the delta may itself be a decimal
            tail = parts[3:] if len(parts) > 2 and parts[2] in ("px", "steps", "%") else parts[2:]
            quoted[parts[0]] = [float(p) for p in tail if re.fullmatch(r"\d+\.\d+", p)]
    for axis, delta in (("cor", 4.0), ("rot", 4.0), ("det_scale", 0.005), ("ang_jitter", 4.0)):
        r = row(rows, axis, delta)
        want = [round(r["naive"], 2), round(r["swap"], 2)]
        if axis == "cor":
            want.append(round(r["selfcal"], 2))
            want.append(round(base, 2))
        got = quoted.get(axis)
        check(f"{axis} row", got == want, f"README {got} vs results {want}")

    print("README, the scalar bounds")
    cor_rows = [r for r in rows if r["axis"] == "cor"]
    other = [r for r in rows if r["axis"] != "none"]
    import math
    pred = [20 * math.log10(max(max(r["hl_ratio"]), 1e-12)) for r in other]
    meas = [base - r["naive"] for r in other]
    mp, mm_ = sum(pred) / len(pred), sum(meas) / len(meas)
    sp = math.sqrt(sum((p - mp) ** 2 for p in pred))
    sm = math.sqrt(sum((m - mm_) ** 2 for m in meas))
    pearson = sum((p - mp) * (m - mm_) for p, m in zip(pred, meas)) / (sp * sm)
    facts = [
        (f"costs {base - row(rows, 'cor', 4.0)['naive']:.1f} dB", "cost of a 4-pixel shift"),
        (f"lands {max(abs(r['selfcal'] - r['swap']) for r in cor_rows):.3f} dB from the oracle",
         "self-calibration against the oracle"),
        (f"to {max(abs(r['d_hat_px'] - r['delta']) for r in cor_rows):.3f} pixel",
         "accuracy of the closed-form shift estimate"),
        (f"to {max(abs(r['d_hat_px'] - r['cor_px']) for r in comp['rows'] if 'd_hat_px' in r):.3f} pixel",
         "the same estimate with other axes wrong"),
        (f"(r = {pearson:.2f})", "correlation of the consistency residuals"),
        (f"{max(m - p for p, m in zip(pred, meas)):.1f} dB", "worst under-estimation on a gauge axis"),
    ]
    for literal, what in facts:
        check(what, literal in flat(readme), f"README does not contain {literal!r}")

    print("RESULTS.md, the bounds on the shift estimate")
    no_jit = [abs(r["d_hat_px"] - r["cor_px"]) for r in comp["rows"]
              if "d_hat_px" in r and r["jit_steps"] == 0]
    with_jit = [abs(r["d_hat_px"] - r["cor_px"]) for r in comp["rows"]
                if "d_hat_px" in r and r["jit_steps"] > 0]
    for value, what in ((max(no_jit), "the estimate with no jitter present"),
                        (max(with_jit), "the estimate with jitter present")):
        check(what, f"inside {value:.4f} pixel" in flat(results_md),
              f"RESULTS.md does not contain 'inside {value:.4f} pixel'")

    print("RESULTS.md, the two tables")
    for cells in md_table(results_md, "## Learned side: operator mismatch on a trained LPD"):
        axis = cells[0].strip("`").replace("—", "none")
        if axis not in ("none", "cor", "rot", "det_scale", "ang_jitter"):
            continue
        delta = 0.0 if axis == "none" else num(cells[1])
        if axis == "det_scale":
            delta = delta / 100.0
        r = row(rows, axis, delta)
        for col, key in ((2, "naive"), (3, "swap"), (4, "selfcal"), (5, "gauge_fixed")):
            if col >= len(cells) or not cells[col].strip() or key not in r:
                continue
            check(f"{axis} {cells[1]} {key}", abs(num(cells[col]) - r[key]) < 0.005,
                  f"documented {cells[col]} vs {r[key]:.4f}")
    for cells in md_table(results_md, "### Several axes wrong at once"):
        want = None
        for r in comp["rows"]:
            if r["case"] == cells[0].replace(" px", "").replace("cor ", "cor").replace(
                    " + rot ", "+rot").replace(" + jitter ", "+jit").replace(
                    " + scale ", "+scale").replace(" step", "").replace("s", "").replace(" ", ""):
                want = r
        if want is None:
            continue
        for col, key in ((1, "naive"), (2, "swap"), (3, "selfcal"), (4, "selfcal_gauge_fixed")):
            if col >= len(cells) or not cells[col].strip() or key not in want:
                continue
            check(f"composed {cells[0]} {key}", abs(num(cells[col]) - want[key]) < 0.005,
                  f"documented {cells[col]} vs {want[key]:.4f}")

    print("DEBUGGING.md, the seed sweep")
    a = seeds["across_seeds"]
    check("the across-seed p-value", f"p = {a['p']:.2f}" in debugging,
          f"expected 'p = {a['p']:.2f}'")
    check("the across-seed mean and standard error",
          f"+{a['mean_gap_db']:.2f} ± {a['se_db']:.2f} dB" in debugging,
          f"expected '+{a['mean_gap_db']:.2f} ± {a['se_db']:.2f} dB'")
    check("every seed is in the table",
          all(f"{r['gap']:+.3f}".replace("-", "−") in debugging for r in seeds["paired_over_images"]))
    check("the sign really does flip",
          any(r["gap"] < 0 for r in seeds["paired_over_images"])
          and any(r["gap"] > 0 for r in seeds["paired_over_images"]))


def read_docs():
    return ((ROOT / "README.md").read_text(),
            (ROOT / "RESULTS.md").read_text(),
            (ROOT / "DEBUGGING.md").read_text())


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        # Every guard must be seen failing on a deliberately broken input, *for the right
        # reason*. "Some check failed" is not enough: if the documentation is already
        # failing, an injected corruption that is silently missed still leaves a non-empty
        # failure list and the self-test passes while testing nothing. So the baseline
        # failures are taken first and each corruption must produce a failure that is
        # not in it.
        docs = read_docs()
        readme, results_md, debugging = docs
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            main(*docs)
        baseline = set(FAILURES)
        if baseline:
            print(f"note: {len(baseline)} check(s) already failing before the self-test; "
                  "each corruption must add one of its own")
        for what, bad in (("a table cell", (readme.replace("20.65", "21.16"), results_md, debugging)),
                          ("a file label", (readme.replace("test file 000", "test file 001"),
                                            results_md.replace("test file 000", "test file 001"),
                                            debugging)),
                          ("a scalar bound", (readme.replace("costs 14.3 dB", "costs 13.9 dB"),
                                              results_md, debugging)),
                          ("the seed sweep", (readme, results_md,
                                              debugging.replace("p = 0.49", "p = 0.0049")))):
            FAILURES.clear()
            print(f"\n--- self-test: corrupting {what} ---")
            with contextlib.redirect_stdout(io.StringIO()):
                main(*bad)
            fresh = [f for f in FAILURES if f not in baseline]
            if not fresh:
                sys.exit(f"self-test failed: corrupting {what} produced no new failure "
                         f"(caught {len(FAILURES)}, all of them pre-existing)")
            print(f"  -> caught: {fresh[0]}")
        print("\nself-test passed: every corruption produced a failure of its own\n")
        FAILURES.clear()

    print("--- checking the documentation against results/ ---")
    main(*read_docs())
    if FAILURES:
        sys.exit("\n" + "\n".join(f"documentation does not match the data: {f}" for f in FAILURES))
    print("\nthe documentation matches the data.")
