"""Break things on purpose to verify that every guard **really does fail** -- the iron rule that
a guard never seen to fail is not yet a guard.

Guards 1 and 2 each caught a real bug on their very first run on 2026-08-28 (a hand-copied
constant off by 3.1e-9; astra silently discarding the detector offset), so their ability to fail
has been confirmed against reality.
Guards 3 and 4 passed first time -- a guard that passes first time is **an untested guard**, so
this file builds counterexamples for them.

Run with: python tests/test_mismatch_guards.py   (guards 2/4 need a GPU)
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "harness"))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import mismatch as M                    # noqa: E402

PASS, FAIL = [], []


def expect_raise(name, fn):
    """fn must raise RuntimeError; if it does not, the guard is deaf."""
    try:
        fn()
    except RuntimeError as e:
        PASS.append((name, str(e)[:70]))
        return
    except Exception as e:
        FAIL.append((name, f"raised the wrong exception type {type(e).__name__}: {e}"))
        return
    FAIL.append((name, "**no exception raised** -- the guard is deaf"))


def expect_ok(name, fn):
    try:
        fn()
        PASS.append((name, "normal case passed"))
    except Exception as e:
        FAIL.append((name, f"normal case unexpectedly failed: {e}"))


# ----------------------------------------------------------- guard 1 ------
def break_guard1():
    """Quietly move the delta=0 geometry by half a pixel -- guard 1 must catch it."""
    real = M.det_half
    try:
        M.det_half = lambda: real() + 0.5 * M.det_cell()
        M.check_zero_reduces_to_official()
    finally:
        M.det_half = real


# ----------------------------------------------------------- guard 3 ------
def break_guard3_direction():
    """Reverse the shift direction (the most common bug) -- guard 3 must catch it."""
    real = M.shift_detector
    try:
        M.shift_detector = lambda y, p: real(y, -p)
        M.check_shift_is_exact()
    finally:
        M.shift_detector = real


def break_guard3_adjoint():
    """Write the adjoint as the "forward" map (i.e. forget the transpose) -- the adjoint check in guard 3 must catch it."""
    real = M.shift_detector_adjoint
    try:
        M.shift_detector_adjoint = lambda g, p: M.shift_detector(g, p)
        M.check_shift_is_exact()
    finally:
        M.shift_detector_adjoint = real


def break_guard3_halfpixel():
    """Adjoint off by half a pixel -- tests the sensitivity floor of the guard, not only coarse errors."""
    real = M.shift_detector_adjoint
    try:
        M.shift_detector_adjoint = lambda g, p: real(g, p + 0.5)
        M.check_shift_is_exact()
    finally:
        M.shift_detector_adjoint = real


# -------------------------------------------------------- guard 2 / 4 -----
def break_guard2(x, dev):
    """Make the operator ignore the mismatch parameters (simulating silent failures like "the parameter never reached the geometry")."""
    real = M.MismatchedOperator.__init__

    def deaf(self, rot=0.0, det_scale=1.0, cor=0.0, ang_jitter=0.0, **kw):
        return real(self, **kw)          # quietly drop every mismatch parameter
    try:
        M.MismatchedOperator.__init__ = deaf
        M.check_perturbation_is_alive(x, device=dev)
    finally:
        M.MismatchedOperator.__init__ = real


def break_guard4(x, dev):
    """Reverse the sign of the data-side perturbation -- catching this is the entire reason
    guard 4 exists. (The operator side and the data side are two independent implementations;
    a flipped sign is invisible to any single-sided self-check.)"""
    real = M.perturb_observation
    try:
        M.perturb_observation = lambda y, rot=0.0, det_scale=1.0, cor=0.0, \
            ang_jitter=0.0, seed=0: real(y, rot=-rot, det_scale=det_scale,
                                         cor=-cor, ang_jitter=ang_jitter, seed=seed)
        M.check_data_and_operator_agree(x, device=dev)
    finally:
        M.perturb_observation = real


def main():
    dev = "cuda:0" if torch.cuda.is_available() else None
    print("=" * 72)
    print("deliberate breakage → the guards must fail")
    print("=" * 72)

    # guard1 compares against the official ODL operator; without odl there is
    # nothing to compare to, so it is skipped rather than reported as a failure.
    try:
        import odl  # noqa: F401
        has_odl = True
    except ImportError:
        has_odl = False
    if has_odl:
        expect_raise("guard1: delta=0 geom off half px", break_guard1)
        expect_ok("guard1: passes when intact", M.check_zero_reduces_to_official)
    else:
        print("  [--] guard1 skipped: odl is not installed (pip install odl==0.8.1)")

    expect_raise("guard3: shift direction reversed", break_guard3_direction)
    expect_raise("guard3: adjoint not transposed", break_guard3_adjoint)
    expect_raise("guard3: adjoint off half pixel", break_guard3_halfpixel)
    expect_ok("guard3: passes when intact", M.check_shift_is_exact)

    # The GPU half needs one LoDoPaB file. Skip it when the data is not on this
    # machine rather than crashing: the CPU guards above are the ones that must
    # run everywhere, including in CI.
    import os
    # lodopab imports h5py at module level, which a clean environment need not
    # have; the CPU guards below must not depend on it.
    try:
        from lodopab import DATA
        have_data = os.path.exists(f"{DATA}/ground_truth_test_000.hdf5")
    except ImportError as exc:
        DATA, have_data = f"<unavailable: {exc}>", False
    if dev and not have_data:
        print()
        print(f"skipping the GPU guards: no LoDoPaB data at {DATA}")
        print("  (set LODOPAB_DIR, or see README for the download)")
    if dev and have_data:
        import h5py
        with h5py.File(f"{DATA}/ground_truth_test_000.hdf5", "r") as h:
            x = torch.tensor(np.array(h["data"][0]).copy(), dtype=torch.float32)
        expect_raise("guard2: operator ignores params", lambda: break_guard2(x, dev))
        expect_ok("guard2: passes when intact",
                  lambda: M.check_perturbation_is_alive(x, device=dev))
        expect_ok("guard4: passes when intact",
                  lambda: M.check_data_and_operator_agree(x, device=dev))
        expect_raise("guard4: data-side sign reversed",
                     lambda: break_guard4(x, dev))
    else:
        print("(no GPU, skipping guards 2/4)")

    print()
    for n, m in PASS:
        print(f"  [ok] {n:<34} {m}")
    for n, m in FAIL:
        print(f"  [!!] {n:<34} {m}")
    print(f"\n{len(PASS)} passed / {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
