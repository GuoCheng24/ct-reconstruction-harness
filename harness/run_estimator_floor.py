"""Where the closed-form self-calibration stops improving, and why it is not a calibration error.

The Helgason-Ludwig first moment recovers the detector offset to about a thousandth of a pixel
(harness/run_selfcalib.py, and the `cor` rows of the learned mismatch tables). It does not
recover it exactly, and the leftover has a shape worth naming: across a 16-fold range of true
shift -- 0.25, 0.5, 1, 2, 4 pixels -- the error is the *same number*, to within 10^-4 pixel.
An error that does not grow with the quantity being estimated is not an estimation error.

This script measures where it comes from. The estimator is run on the **unperturbed**
observations, where the true offset is zero, and what it reports there is compared with the
offset it leaves behind at every shift:

    file 000    d_hat on unperturbed data  -0.00120 px    leftover at every shift  -0.00121 px
    file 001    d_hat on unperturbed data  -0.00486 px    leftover at every shift  -0.00487 px

The two agree to 10^-5 pixel on both files, and the two files disagree with each other by a
factor of four. So the estimator is unbiased for the *shift*: its whole residual is a fixed
property of the image set it is run on, present before any mismatch is introduced, and it
sets a floor that no amount of calibration accuracy will go below. Choosing a different 128
images moves that floor further than changing the shift by a factor of sixteen does.

(The model M1(θ) = a cosθ + b sinθ + d·M0(θ) has the object's centre of mass in its own two
terms a, b, so in the continuum this floor should be zero. That it is not is a property of the
discretization, and this script does not claim to explain it -- only to measure it and to show
it is not a function of the shift.)

    python harness/run_estimator_floor.py            # needs LODOPAB_DIR, no GPU
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

import identifiability as ID
import mismatch as M
from lodopab import DATA

N = int(os.environ.get("FLOOR_N", "128"))


def main():
    geom = M.official_geometry()
    cell = M.det_cell()
    det = np.asarray(geom.det_partition.grid.coord_vectors[0], dtype=np.float64)
    angles = np.asarray(geom.angles, dtype=np.float64)

    out = {"_note": __doc__.strip().splitlines()[0], "n": N, "files": {}}
    print(f"{'file':>6}{'d_hat on unperturbed':>24}{'se':>10}")
    for f in (0, 1):
        with h5py.File(f"{DATA}/observation_test_{f:03d}.hdf5", "r") as h:
            y = np.array(h["data"][:N]).astype(np.float64)
        d = np.array([ID.estimate_cor(y[i], det, angles)["d_hat_m"] / cell for i in range(N)])
        out["files"][f"{f:03d}"] = {"d_hat_px_mean": float(d.mean()),
                                    "d_hat_px_se": float(d.std(ddof=1) / np.sqrt(N)),
                                    "d_hat_px": d.tolist()}
        print(f"{f:>6}{d.mean():>+24.5f}{d.std(ddof=1) / np.sqrt(N):>10.5f}")

    path = os.path.join(RESULTS, "estimator_floor.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwritten to {os.path.relpath(path, os.path.join(HERE, '..'))}")
    print("Compare with the `cor` rows of results/lpd_mismatch_n128.json: the leftover there,")
    print("at every shift from 0.25 to 4 pixels, is this number.")


if __name__ == "__main__":
    main()
