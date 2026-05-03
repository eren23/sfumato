"""Figure 1 — Prefix-damage hierarchy across Track 1 LoRA versions.

Five conditions × three model variants (base, Track 1 v2 4/7-modules,
Track 1 v3 7/7-modules), grouped bars with Clopper-Pearson 95% CI
whiskers. Annotates the Q-0.5B / Q-1.5B *capacity-amplified* inversion
finding from `e2/RESULTS_TRACK1.md`. cmaj ceiling shown.

Numbers are pulled verbatim from `scripts/binom_ci.py` (HEADLINE list)
and the Track-1 results table; binomial CIs are recomputed here so the
figure does not require touching the source script.

Outputs:
  phase2/figures/fig1_prefix_hierarchy.png
  phase2/figures/fig1_prefix_hierarchy.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import PALETTE  # noqa: E402

plt.style.use(str(HERE / "sfumato.mplstyle"))

OUT_DIR = HERE
RNG_SEED = 0  # deterministic; nothing random here but pinned for hygiene


# ---------------------------------------------------------------------------
# Data — (k, n) per cell, sourced from scripts/binom_ci.py HEADLINE list.
# Order matches the 5-condition layout in e2/RESULTS_TRACK1.md.
# ---------------------------------------------------------------------------
N = 200

CONDS = [
    ("C2",        "no prefix"),
    ("C2hint",    "minimal hint"),
    ("C2empty",   "Plan: empty"),
    ("C3p Q-0.5B", "weak planner"),
    ("C3p Q-1.5B", "strong planner"),
]

# k for each (variant, cond)
DATA_K = {
    "base": [148, 136, 132, 128, 120],   # 74 / 68 / 66 / 64 / 60
    "v2":   [141, 147, 146, 120, 134],   # 70.5 / 73.5 / 73.0 / 60.0 / 67.0
    "v3":   [146, 147, 148, 130, 108],   # 73.0 / 73.5 / 74.0 / 65.0 / 54.0
}

# Reference: cmaj b=5 ceiling on test set (apples-to-apples baseline).
# Track-1 v3 cmaj on test = 79.5%; base test cmaj = 79.0%. We draw the
# base-cmaj ceiling as the meaningful reference (the others' cmaj numbers
# are footnoted in caption).
CMAJ_CEILING = 79.0   # %, base test cmaj b=5 (RESULTS_TRACK1.md)

VARIANT_LABELS = {
    "base": "base LLaDA-8B",
    "v2":   "Track 1 v2  (4/7 modules, 10M params)",
    "v3":   "Track 1 v3  (7/7 modules, 22M params)",
}


def cp_ci(k: int, n: int) -> tuple[float, float, float]:
    """Clopper-Pearson 95% CI; returns (pct, lo_pct, hi_pct)."""
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def draw() -> None:
    np.random.seed(RNG_SEED)

    pcts = {v: [cp_ci(k, N)[0] for k in DATA_K[v]] for v in DATA_K}
    cis = {v: [cp_ci(k, N) for k in DATA_K[v]] for v in DATA_K}

    x = np.arange(len(CONDS))
    bw = 0.26  # bar width

    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    variants = ["base", "v2", "v3"]
    colors = [PALETTE.base, PALETTE.v2, PALETTE.v3]
    offsets = [-bw, 0.0, +bw]
    for v, c, off in zip(variants, colors, offsets):
        ys = pcts[v]
        bars = ax.bar(
            x + off, ys, bw,
            color=c, edgecolor=PALETTE.ink, linewidth=0.5,
            label=VARIANT_LABELS[v],
            zorder=2,
        )
        # CI whiskers (Clopper-Pearson exact)
        lows = [pcts[v][i] - cis[v][i][1] for i in range(len(CONDS))]
        highs = [cis[v][i][2] - pcts[v][i] for i in range(len(CONDS))]
        ax.errorbar(
            x + off, ys, yerr=[lows, highs],
            fmt="none", ecolor=PALETTE.ink, elinewidth=0.8,
            capsize=2.5, capthick=0.8, zorder=3,
        )

    # Numeric labels above v3 bars (the headline column).
    for i, val in enumerate(pcts["v3"]):
        ax.text(
            x[i] + bw, val + 1.4, f"{val:.1f}",
            ha="center", va="bottom", fontsize=8.5,
            color=PALETTE.v3, fontweight="semibold", zorder=4,
        )

    # cmaj ceiling reference line
    ax.axhline(
        CMAJ_CEILING, color=PALETTE.ok, lw=0.9, ls=(0, (5, 3)),
        alpha=0.85, zorder=1,
    )
    ax.text(
        len(CONDS) - 0.55, CMAJ_CEILING + 0.4,
        f"base cmaj b=5 ceiling = {CMAJ_CEILING:.0f}%",
        ha="right", va="bottom", fontsize=8.5, color=PALETTE.ok,
    )

    # ---- Annotations: Q-0.5B / Q-1.5B capacity-amplified inversion ----
    # v2 → v3 deltas (pp): Q-0.5B = +5.0 (60.0 → 65.0), Q-1.5B = -13.0 (67.0 → 54.0)
    ax.annotate(
        "Q-0.5B  +5.0 pp at v3\n(planner-trust improves)",
        xy=(3 + bw, pcts["v3"][3] + 0.5),
        xytext=(2.85, 84.5),
        fontsize=8.5, color=PALETTE.ok,
        ha="center", va="bottom",
        arrowprops=dict(
            arrowstyle="-|>", color=PALETTE.ok, lw=0.8,
            shrinkA=2, shrinkB=2, mutation_scale=8,
        ),
    )
    ax.annotate(
        "Q-1.5B  −13.0 pp at v3\n(capacity-amplified\nregression)",
        xy=(4 + bw + 0.05, pcts["v3"][4]),
        xytext=(4.30, 84.5),
        fontsize=8.5, color=PALETTE.warn, fontweight="semibold",
        ha="left", va="bottom",
        arrowprops=dict(
            arrowstyle="-|>", color=PALETTE.warn, lw=0.8,
            shrinkA=2, shrinkB=2, mutation_scale=8,
            connectionstyle="arc3,rad=-0.25",
        ),
    )
    # Highlight the inversion zone with a pale background fill.
    ax.axvspan(2.55, 4.55, color=PALETTE.warn_soft, alpha=0.18, zorder=0)

    # ---- Axes / labels ----
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c[0]}\n{c[1]}" for c in CONDS],
        fontsize=9.5,
    )
    ax.set_ylabel("GSM8K-test accuracy  (%, N=200, 95% Clopper-Pearson CI)")
    ax.set_ylim(42, 92)
    ax.set_yticks(np.arange(45, 86, 5))
    ax.legend(
        loc="lower left", fontsize=8.5, ncol=1, frameon=False,
        bbox_to_anchor=(0.0, -0.32),
    )
    ax.set_title(
        "Figure 1   Prefix-damage hierarchy across Track 1 LoRA versions",
    )
    fig.text(
        0.0, 0.965,
        "Static prefixes (left): v3 closes the format-brittleness gap.   "
        "Content-rich plans (right): capacity amplifies a planner-trust axis "
        "in opposite directions for Q-0.5B and Q-1.5B.",
        fontsize=8.7, color=PALETTE.sub, ha="left", va="bottom",
        transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.945))
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig1_prefix_hierarchy.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
