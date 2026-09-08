"""Every bound quoted in README.md must hold against results/*.json.

Two numbers in the mismatch paragraph were wrong when this was written: the
closed-form COR estimate was quoted at 0.003 px, which was the classical TV
figure copied into the learned-side paragraph (the measured value there is
0.0059), and the oracle gap was quoted at 0.006 dB against a measured 0.0064,
i.e. a bound that its own data violated. Rounding a bound down is the failure
mode this catches.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
mm = json.loads((ROOT / "results/lpd_mismatch_n128.json").read_text())["official_recipe_n128"]
cor = [r for r in mm["rows"] if r["axis"] == "cor"]
measured = {
    "px":  max(abs(r["d_hat_px"] - r["delta"]) for r in cor),
    "db":  max(abs(r["swap"] - r["selfcal"]) for r in cor),
}
readme = (ROOT / "README.md").read_text(encoding="utf-8")
claims = {
    "px": float(re.search(r"within ([\d.]+) pixel", readme).group(1)),
    "db": float(re.search(r"([\d.]+) dB from the oracle", readme).group(1)),
}
bad = [f"README claims {k} <= {claims[k]} but the measured maximum is {measured[k]:.4f}"
       for k in claims if claims[k] < measured[k]]
for k in claims:
    print(f"  {k}: README {claims[k]}  >=  measured {measured[k]:.4f}  {'ok' if not any(k in b for b in bad) else 'FAIL'}")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("every quoted bound holds")
