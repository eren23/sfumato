"""Figure 2 — Branch-agreement distribution shift (diversity went UP).

Two-panel layout:
  (a) full 5-bin grouped distribution (5/5-same → 5/5-unique), base vs
      Track-1-v2+commit-v2, with Clopper-Pearson 95% CIs;
  (b) derived "diversity score" overlay — mean unique answers per
      problem, +13% bump (1.825 → 2.07) shown as an annotated horizontal
      bar pair.

Numbers come straight from `e2/RESULTS_TRACK1.md` §"Diversity-expansion
finding" and `scripts/binom_ci.py` HEADLINE list.

Outputs:
  phase2/figures/fig2_branch_agreement.png
  phase2/figures/fig2_branch_agreement.pdf
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
RNG_SEED = 0
N = 200

BINS = ["5/5\nsame", "4/5\nunique", "3/5\nunique", "2/5\nunique", "5/5\nunique"]

# k counts for each bin (RESULTS_TRACK1.md & binom_ci.py HEADLINE)
BASE_K = [103, 12, 23, 55, 7]   # 51.5, 6.0, 11.5, 27.5, 3.5  %
LORA_K = [95,  17, 36, 39, 13]  # 47.5, 8.5, 18.0, 19.5, 6.5  %

# Diversity score = mean unique-answers/problem (bin index k unique × freq)
# precomputed in RESULTS_TRACK1.md: 1.825 → 2.07 (+13%)
BASE_MEAN_UNIQUE = 1.825
LORA_MEAN_UNIQUE = 2.07

BASE_LABEL = "base LLaDA on test"
LORA_LABEL = "Track 1 v2 + commit v2"


def cp_ci(k: int, n: int) -> tuple[float, float, float]:
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def draw() -> None:
    np.random.seed(RNG_SEED)

    base_pcts = [cp_ci(k, N)[0] for k in BASE_K]
    lora_pcts = [cp_ci(k, N)[0] for k in LORA_K]
    base_cis = [cp_ci(k, N) for k in BASE_K]
    lora_cis = [cp_ci(k, N) for k in LORA_K]

    fig = plt.figure(figsize=(11.0, 5.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[3.0, 1.4], wspace=0.32,
        left=0.07, right=0.97, top=0.86, bottom=0.16,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): grouped 5-bin distribution ------------------------
    x = np.arange(len(BINS))
    bw = 0.36
    base_bars = ax.bar(
        x - bw / 2, base_pcts, bw,
        color=PALETTE.base, edgecolor=PALETTE.ink, linewidth=0.5,
        label=BASE_LABEL, zorder=2,
    )
    lora_bars = ax.bar(
        x + bw / 2, lora_pcts, bw,
        color=PALETTE.v3, edgecolor=PALETTE.ink, linewidth=0.5,
        label=LORA_LABEL, zorder=2,
    )

    for i in range(len(BINS)):
        # CI whiskers
        ax.errorbar(
            x[i] - bw / 2, base_pcts[i],
            yerr=[[base_pcts[i] - base_cis[i][1]], [base_cis[i][2] - base_pcts[i]]],
            fmt="none", ecolor=PALETTE.ink, elinewidth=0.7,
            capsize=2.5, capthick=0.7, zorder=3,
        )
        ax.errorbar(
            x[i] + bw / 2, lora_pcts[i],
            yerr=[[lora_pcts[i] - lora_cis[i][1]], [lora_cis[i][2] - lora_pcts[i]]],
            fmt="none", ecolor=PALETTE.ink, elinewidth=0.7,
            capsize=2.5, capthick=0.7, zorder=3,
        )
        # Numeric labels
        ax.text(
            x[i] - bw / 2, base_pcts[i] + 1.0, f"{base_pcts[i]:.1f}",
            ha="center", va="bottom", fontsize=8.0, color=PALETTE.sub,
        )
        ax.text(
            x[i] + bw / 2, lora_pcts[i] + 1.0, f"{lora_pcts[i]:.1f}",
            ha="center", va="bottom", fontsize=8.0, color=PALETTE.v3,
            fontweight="semibold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BINS, fontsize=9.5)
    ax.set_ylabel("% of problems  (N=200, b=5 t=0.7)")
    ax.set_xlabel("Branch-agreement bin  (unique-answer count out of 5 branches)")
    ax.set_ylim(0, 62)
    ax.set_title("(a)  Full distribution shift  —  mass migrates rightward")
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    # Direction arrow showing rightward mass shift (placed up at y=37,
    # high enough to not collide with bars or value labels)
    ax.annotate(
        "", xy=(4.0, 36.0), xytext=(1.0, 36.0),
        arrowprops=dict(
            arrowstyle="->", color=PALETTE.warn, lw=1.4,
            alpha=0.7, mutation_scale=16,
        ),
    )
    ax.text(
        2.5, 38.0, "diversity gain  (more answer modes per problem)",
        ha="center", va="bottom", fontsize=9, color=PALETTE.warn,
        style="italic",
    )

    # ---- Panel (b): diversity-score derived metric --------------------
    # Horizontal bar pair: mean unique answers per problem
    metrics = [BASE_MEAN_UNIQUE, LORA_MEAN_UNIQUE]
    labels = [BASE_LABEL, LORA_LABEL]
    colors = [PALETTE.base, PALETTE.v3]
    y_pos = np.array([1.0, 0.4])
    bars = ax2.barh(
        y_pos, metrics, height=0.32,
        color=colors, edgecolor=PALETTE.ink, linewidth=0.5,
        zorder=2,
    )
    for ypos, val, lab, col in zip(y_pos, metrics, labels, colors):
        ax2.text(
            val + 0.05, ypos, f"{val:.3f}",
            va="center", ha="left", fontsize=10,
            color=col, fontweight="semibold",
        )
        ax2.text(
            0.04, ypos + 0.22, lab,
            va="bottom", ha="left", fontsize=8.5, color=PALETTE.sub,
        )

    # Delta annotation
    delta_pct = (LORA_MEAN_UNIQUE - BASE_MEAN_UNIQUE) / BASE_MEAN_UNIQUE * 100
    ax2.annotate(
        "", xy=(LORA_MEAN_UNIQUE, 0.4), xytext=(BASE_MEAN_UNIQUE, 1.0),
        arrowprops=dict(
            arrowstyle="-|>", color=PALETTE.warn, lw=1.0,
            connectionstyle="arc3,rad=-0.18", mutation_scale=10,
        ),
    )
    ax2.text(
        2.5, 0.7, f"+{delta_pct:.0f}%",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=PALETTE.warn,
    )
    ax2.text(
        2.5, 0.55, "mean unique\nanswers / problem",
        ha="center", va="top", fontsize=8.5, color=PALETTE.warn,
    )

    ax2.set_xlim(0, 3.2)
    ax2.set_ylim(0.0, 1.5)
    ax2.set_yticks([])
    ax2.set_xticks([1, 2, 3])
    ax2.set_xlabel("unique answers / problem")
    ax2.set_title("(b)  Derived diversity score")
    # Reduce panel-b chrome to draw eye to the metric
    ax2.spines["bottom"].set_color(PALETTE.sub)
    ax2.grid(False)
    ax2.tick_params(axis="x", colors=PALETTE.sub)

    # ---- Title + subtitle (figure-level) ------------------------------
    fig.suptitle(
        "Figure 2   Branch-agreement distribution shift  —  Track 1 LoRA expanded "
        "sampling diversity rather than collapsing it",
        fontsize=11.5, x=0.07, ha="left", y=0.965,
    )
    fig.text(
        0.07, 0.92,
        "5/5-same drops from 51.5% to 47.5% (4 pp);  mass migrates into 4/5- and "
        "3/5-unique bins;  derived diversity score climbs +13%.",
        fontsize=9, color=PALETTE.sub, ha="left", va="bottom",
    )

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig2_branch_agreement.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
