"""Identifiability and gauge analysis of the mismatch -- decides whether the "operator
mismatch" direction contains a real problem at all.

## Why this is the first experiment to run (rather than measuring degradation first)

For "learned reconstruction is fragile to operator mismatch" to hold, the mismatch itself must
be a **genuine loss of information**. But the four mismatch axes of parallel-beam CT are not
mathematically homogeneous:

  rot (global angle offset Δ)  -- reconstructing with the wrong angle = rotating the object
                                  **as a whole** by −Δ. No information is lost at all, only the
                                  coordinate frame changes. Computing PSNR in a fixed reference
                                  frame shows a huge "degradation", but that is a **metric
                                  artefact**, not a worse reconstruction. -> gauge freedom
  det_scale (scale 1+ε)        -- likewise approximately an isotropic scaling, again mostly a
                                  geometric gauge.
  cor (detector offset d)      -- the Helgason-Ludwig first-moment condition gives a
                                  **closed form**: for parallel beam the true sinogram satisfies
                                  M1(θ)=∫p(θ,s)s ds = a cosθ + b sinθ, while an offset d adds a
                                  constant d·M0 to M1. Fit M1(θ) against {cosθ, sinθ, 1} and the
                                  constant term / M0 is d -- milliseconds, no ground truth, no
                                  iteration. -> already solved by classical self-calibration
  ang_jitter (per-angle jitter)-- 1000 unknowns, which low-order moment conditions cannot pin
                                  down; and it is genuinely irregular sampling being used as if
                                  it were regular. -> the only axis that may be truly hard

Without doing this analysis first, a "mismatch robustness" paper would be attacking **a gauge
artefact plus an already-solved calibration problem** -- the numbers look good, but a single
reviewer remark ("COR is calibrated away by consistency conditions") punctures it.

References: Helgason-Ludwig consistency conditions; moment-based methods for CT geometry
self-calibration (Noo et al.).
"""
import math
import numpy as np


def moments(sino, det_coords):
    """Zeroth/first moments: M0(θ)=∫p ds, M1(θ)=∫p·s ds. sino is (n_ang, n_det)."""
    ds = float(det_coords[1] - det_coords[0])
    m0 = sino.sum(axis=1) * ds
    m1 = (sino * det_coords[None, :]).sum(axis=1) * ds
    return m0, m1


def estimate_cor(sino, det_coords, angles):
    """Closed-form estimate of the detector offset d (in m) from the HL first-moment condition.
    No ground truth and no iteration needed.

    Model: M1(θ) = a cosθ + b sinθ + d * M0(θ)
    M0(θ) is known for every θ, so solve least squares for (a, b, d).
    """
    m0, m1 = moments(sino, det_coords)
    A = np.stack([np.cos(angles), np.sin(angles), m0], axis=1)
    coef, *_ = np.linalg.lstsq(A, m1, rcond=None)
    resid = m1 - A @ coef
    return {"d_hat_m": float(coef[2]), "a": float(coef[0]), "b": float(coef[1]),
            "resid_rms": float(np.sqrt((resid ** 2).mean())),
            "m1_rms": float(np.sqrt((m1 ** 2).mean()))}


def hl_residual_spectrum(sino, det_coords, angles, nmax=6):
    """Helgason-Ludwig residuals order by order -- a far richer consistency test than the first
    moment alone.

    HL condition: M_n(θ)=∫p(θ,s)s^n ds must be a **homogeneous polynomial of degree n** in
    (cosθ, sinθ) (basis {cos^k θ · sin^{n-k} θ}, k=0..n). Fit least squares at each order and
    take the relative residual.

    Why it has to be read order by order (a lesson paid for on 2026-08-28): with n=1 alone the
    residual of angular jitter **goes down rather than up** (a factor of 0.82-0.88) and so looks
    "harmless", while it in fact costs 1.7 dB -- a false negative on the most dangerous axis.
    Reading across orders makes the structure immediately clear:
      gauge perturbations (rot / det_scale) give exactly 1.000 at **every order** (multiple
        confirmations of Proposition 1);
      a detector offset shows up only at **odd orders** (the d·n·M_{n-1} it adds to M_n has the
        opposite parity to the admissible harmonics);
      angular jitter shows up only at **even orders** (from n=2 on; reaching a factor of 9.6 at
        n=6 with a 16-angle step).
    The per-order residual ratios therefore form a fingerprint that **identifies which kind of
    mismatch** is present, rather than being a single alarm scalar.
    """
    p = np.asarray(sino, dtype=np.float64)
    ds = float(det_coords[1] - det_coords[0])
    c, s = np.cos(angles), np.sin(angles)
    out = []
    for n in range(nmax + 1):
        Mn = (p * (det_coords[None, :] ** n)).sum(axis=1) * ds
        B = np.stack([c ** k * s ** (n - k) for k in range(n + 1)], axis=1)
        coef, *_ = np.linalg.lstsq(B, Mn, rcond=None)
        r = Mn - B @ coef
        out.append(float(np.sqrt((r ** 2).mean())
                         / max(np.sqrt((Mn ** 2).mean()), 1e-30)))
    return np.array(out)


def consistency_residual(sino, det_coords, angles):
    """Consistency residual with no offset term -- purely "does this data look like something a
    parallel-beam object could have produced".

    Fit M1 with {cosθ, sinθ} only; a large residual = the data is incompatible with the assumed
    geometry. This is a **ground-truth-free** mismatch detection signal (genuinely computable at
    deployment time), used to judge which mismatches are "visible".
    """
    m0, m1 = moments(sino, det_coords)
    A = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    coef, *_ = np.linalg.lstsq(A, m1, rcond=None)
    r1 = m1 - A @ coef
    return {"m1_resid_rel": float(np.sqrt((r1 ** 2).mean())
                                  / max(np.sqrt((m1 ** 2).mean()), 1e-30)),
            "m0_var_rel": float(m0.std() / max(abs(m0.mean()), 1e-30))}
