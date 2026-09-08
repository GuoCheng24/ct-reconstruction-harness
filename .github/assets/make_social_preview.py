"""Generate the 1280x640 social preview card. Regenerate with:  python .github/assets/make_social_preview.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, ACCENT, BG = "#14181f", "#5b6472", "#0b6e4f", "#fbfaf7"
fig = plt.figure(figsize=(12.8, 6.4), dpi=100); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.035, 0.055), 0.93, 0.89, boxstyle="round,pad=0,rounding_size=0.02",
                            fc="white", ec="#e4e2dd", lw=1.4))
ax.text(0.075, 0.80, "ct-reconstruction-harness", fontsize=34, weight="bold", color=INK, va="center")
ax.text(0.075, 0.705, "Reproduce the LoDoPaB-CT baselines from scratch, then beat one with a paired test",
        fontsize=15.5, color=MUTED, va="center")
rows = [("FBP, official operator", "30.52", "n = 3553, published 30.19 on the challenge split"),
        ("TV-Adam, official recipe", "33.00", "n = 128, published 33.36, gamma untouched"),
        ("TGV (found by the loop)", "+0.70", "dB paired over the recipe, t = 16, wins 125/128")]
y = 0.545
for name, num, note in rows:
    ax.text(0.075, y, name, fontsize=15, color=INK, va="center")
    ax.text(0.455, y, num, fontsize=21, weight="bold", color=ACCENT, va="center", ha="right")
    ax.text(0.485, y, note, fontsize=12.5, color=MUTED, va="center")
    y -= 0.105
ax.plot([0.075, 0.925], [0.20, 0.20], color="#e4e2dd", lw=1.2)
ax.text(0.075, 0.135, "Every guard is run against a deliberately broken operator and must fail for the right reason.",
        fontsize=13, color=MUTED, va="center")
fig.savefig("/dev/stdout" if False else __file__.replace("make_social_preview.py", "social-preview.png"),
            facecolor=BG); print("wrote social-preview.png")
