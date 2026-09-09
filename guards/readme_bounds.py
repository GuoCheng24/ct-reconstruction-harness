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


def classical(readme):
    """The headline table and the three results, against results/evaluation_n128.json."""
    ev = load("evaluation_n128.json")
    fbp_off = ev["fbp"]["official_odl_astra_operator"]
    fbp_ops = ev["fbp"]["ops_operator_calib_on_4_images"]
    tv, iso, tgv = (ev["tv_aniso_official_recipe"], ev["tv_iso"], ev["tgv2_ratio0.3_gamma28"])
    pub, pair = ev["published_challenge_set"], ev["paired"]
    wls = ev["poisson_vs_wls_tgv2_r0.5_g20_n32"]["paired_poisson_minus_wls"]

    print("README, the headline table")
    for run, label in ((fbp_off, "FBP, official operator"), (fbp_ops, "FBP, the operator in ops/"),
                       (tv, "TV"), (tgv, "TGV")):
        literal = f"{run['psnr_mean']:.2f} ± {run['psnr_se']:.2f}   {run['n']}"
        check(label, " ".join(literal.split()) in flat(readme),
              f"README does not contain {literal!r}")
    p = pair["tgv_minus_tv_aniso"]
    check("the paired gain",
          f"+{p['mean']:.2f} ± {p['se']:.2f}   t = {p['t']:.1f}   wins {p['wins']}/{p['n']}"
          .replace("   ", " ") in flat(readme))
    for value, label in ((pub["FBP"], "published FBP"), (pub["TV"], "published TV"),
                         (pub["DIP+TV"], "published DIP+TV")):
        check(label, f"{value:.2f}" in flat(readme), f"README does not contain {value:.2f}")

    print("README, the three results")
    check("the TGV gain, restated",
          f"+{p['mean']:.2f} ± {p['se']:.2f} dB over the matched" in flat(readme))
    check("winning on 125 of them", f"winning on {p['wins']} of them" in flat(readme))
    i = pair["tv_iso_minus_tv_aniso"]
    check("isotropic TV over anisotropic",
          f"is +{i['mean']:.2f} ± {i['se']:.2f} dB, paired, on {i['wins']} of {i['n']} images"
          in flat(readme))
    check("Poisson against weighted least squares",
          f"is {wls['mean']:.2f} ± {wls['se']:.2f} dB from Poisson, paired on {wls['n']} images, "
          f"p = {wls['p']:.2f}".replace("-0", "−0") in flat(readme))

    print("README, how much easier the first sixteen images are")
    for run, key, label in ((tv, "psnr_mean", "TV"), (tgv, "psnr_mean", "TGV")):
        first = run["first16_mean"]
        rest = (run["n"] * run[key] - 16 * first) / (run["n"] - 16)
        check(f"{label}, first sixteen against the other {run['n'] - 16}",
              f"{first - rest:.2f} dB for {label}" in flat(readme),
              f"README does not contain '{first - rest:.2f} dB for {label}'")
    rest = (fbp_off["n"] * fbp_off["psnr_mean"] - 16 * fbp_off["first16"]) / (fbp_off["n"] - 16)
    check(f"FBP, first sixteen against the other {fbp_off['n'] - 16}",
          f"by {fbp_off['first16'] - rest:.2f} dB for FBP against the remaining {fbp_off['n'] - 16}"
          in flat(readme))


def main(readme, results_md, debugging):
    classical(readme)
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

    print("the floor of the shift estimate")
    floor = load("estimator_floor.json")["files"]
    f0, f1 = floor["000"]["d_hat_px_mean"], floor["001"]["d_hat_px_mean"]
    check("README quotes the floor of file 000",
          f"reports {f0:.4f} px".replace("-", "\u2212") in flat(readme))
    check("README quotes the floor of file 001",
          f"gives {f1:.4f} px".replace("-", "\u2212") in flat(readme))
    check("README quotes the ratio between them", f"{f1 / f0:.2f} times larger" in flat(readme))
    check("RESULTS.md quotes both floors",
          f"reports {f0:.5f} px on file 000 and {f1:.5f} px on file 001"
          .replace("-", "\u2212") in flat(results_md))
    check("RESULTS.md quotes the ratio", f"factor of {f1 / f0:.2f}" in flat(results_md))
    # the decomposition table: the leftover at each shift, minus that file's own floor
    left = {}
    for key, tag, fl in (("official_recipe_n128", "000", f0),
                         ("official_recipe_n128_file001", "001", f1)):
        left[tag] = {r["delta"]: (r["d_hat_px"] - r["delta"]) - fl
                     for r in mm[key]["rows"] if r["axis"] == "cor"}
        for delta, v in left[tag].items():
            check(f"file {tag}, shift {delta} px, after removing the floor",
                  f"{v:.6f}".replace("-", "\u2212") in flat(results_md),
                  f"RESULTS.md does not contain {v:.6f}".replace("-", "\u2212"))
    small = max(abs(v) for t in left for d, v in left[t].items() if d <= 1.0)
    check("the claim that it is small out to a one-pixel shift", small < 4e-5,
          f"largest is {small:.2e} px")
    big = max(abs(left['000'][d] - left['001'][d]) / abs(left['000'][d]) for d in (2.0, 4.0))
    check("the claim that the two files agree to 6 per cent where it is measurable",
          f"{big:.0%}" == "6%", f"they agree to {big:.1%}")
    check("README says so", "agrees between the two files to 6 per cent" in flat(readme))

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
    words = {4: "three", 5: "four", 6: "five", 7: "six", 8: "seven", 9: "eight", 12: "eleven"}
    check("DEBUGGING.md says how many further draws were made",
          f"with {words[a['n_seeds']]} further jitter realizations" in flat(debugging),
          f"expected 'with {words[a['n_seeds']]} further jitter realizations'")
    gap = math.log10(a["p"] / min(r["p_t"] for r in seeds["paired_over_images"]))
    names = {26: "twenty-six", 29: "thirty", 30: "thirty", 31: "thirty-one"}
    check("the gap between the two p-values",
          f"differ by {names[round(gap)]} orders of magnitude" in flat(debugging),
          f"the two p-values differ by 10^{gap:.1f}")


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
                                              debugging.replace("p = 0.15", "p = 0.0015"))),
                          ("how many jitter draws there were",
                           (readme, results_md,
                            debugging.replace("seven further", "three further"))),
                          ("the headline table", (readme.replace("33.00 ± 0.33", "33.36 ± 0.33"),
                                                  results_md, debugging)),
                          ("the paired gain", (readme.replace("wins 125/128", "wins 128/128"),
                                               results_md, debugging)),
                          ("how easy the first sixteen are",
                           (readme.replace("0.95 dB for TV", "0.80 dB for TV"),
                            results_md, debugging)),
                          ("the floor of the shift estimate",
                           (readme.replace("reports −0.0012 px", "reports −0.0010 px"),
                            results_md, debugging)),
                          ("a cell of the decomposition table",
                           (readme, results_md.replace("−0.000984", "−0.000934"), debugging))):
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
