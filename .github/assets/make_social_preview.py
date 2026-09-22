"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The finding is a paired comparison, and what makes a paired comparison convincing is not its mean
but how one-sided it is: 125 of 128 held-out images. So the card counts them out as tiles, which
reads as a shape at the 360 px a Slack unfurl gives a card, with the reproduction numbers above.

Everything is read from results/evaluation_n128.json, the file guards/readme_bounds.py already
holds the README to.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cardkit import SANS, card  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "results/evaluation_n128.json").read_text())
P = D["paired"]["tgv_minus_tv_aniso"]
tv = D["tv_aniso_official_recipe"]["psnr_mean"]
published = D["published_challenge_set"]["TV"]
wins, n, mean, se = P["wins"], P["n"], P["mean"], P["se"]


def chart(ax, accent):
    ax.text(0.78, 3.42, f"{tv:.2f} dB", fontsize=44, fontweight="bold", color="#17181a",
            family=SANS)
    ax.text(3.62, 3.42, f"reproducing the published {published:.2f}", fontsize=34,
            color="#55585c", family=SANS)
    ax.text(0.78, 2.62, f"+{mean:.2f} dB", fontsize=44, fontweight="bold", color="#1a7f37",
            family=SANS)
    ax.text(3.62, 2.62, f"paired gain, se {se:.2f}, found by the loop", fontsize=34,
            color="#55585c", family=SANS)

    # one tile per held-out image, so 125 of 128 is a shape and not a claim
    cols, x0, y0, s, gap = 32, 0.80, 2.18, 0.23, 0.05
    for i in range(n):
        r, c = divmod(i, cols)
        ax.add_patch(plt_rect(x0 + c * (s + gap), y0 - r * (s + gap), s, s,
                              "#1a7f37" if i < wins else "#cf222e"))
    ax.text(0.78, 0.98, f"{wins} of {n} held-out images improve", fontsize=34,
            fontweight="bold", color="#17181a", family=SANS)


def plt_rect(x, y, w, h, c):
    import matplotlib.pyplot as plt
    return plt.Rectangle((x, y), w, h, color=c, zorder=3)


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#1a7f37", badge="C",
    kicker="LOW-DOSE CT  ·  built from the forward operator up",
    headline="Reproduce the baseline, then beat it",
    evidence="LoDoPaB-CT test split, official metric convention",
    chart=chart,
    footer="github.com/GuoCheng24/ct-reconstruction-harness",
    headline_size=44,
)
print(f"written {pathlib.Path(out).name}  TV {tv:.2f} vs {published:.2f}, +{mean:.2f} dB, {wins}/{n}")
