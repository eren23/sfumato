"""Figure 4 (NEW) — c2c-vs-cmaj compositionality 2x2.

Rows  : base LLaDA  /  Track-1-v3 + commit-v3
Cols  : single-shot (c2c)  /  branch-vote b=5 (cmajc)

Four cell values + arrows showing where the compute-time (b=5 vote) and
param-time (commit-LoRA) consensus contributions land. The "double-dip
violation" headline from RESULTS_TRACK2.md (cmajc ≤ cmaj+1 was predicted;
observed +3 pp) is annotated as a positive composition finding.

Outputs:
  phase2/figures/fig4_compositionality.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import PALETTE  # noqa: E402

plt.style.use(str(HERE / "sfumato.mplstyle"))

OUT_DIR = HERE
RNG_SEED = 0
N = 200

# ---------------------------------------------------------------------------
# Cell data — k counts (binom_ci.HEADLINE) + RESULTS_TRACK2.md table
# ---------------------------------------------------------------------------
# Note: cmajc-v3 (Track 1 v3 + commit v3) was not run before pod preempt;
# we use the v2-commit cmajc number (164/200 = 82.0%) as the closest
# documented Track-1+commit b=5 figure, with a footnote in caption.
CELLS = {
    ("base",  "single"): {"k": 148, "label": "C2 base",                            "metric": "c2c"},    # 74.0%
    ("base",  "vote"):   {"k": 158, "label": "cmaj base",                          "metric": "cmaj"},   # 79.0%
    ("track", "single"): {"k": 158, "label": "Track 1 v3 + commit v3",             "metric": "c2c"},    # 79.0%
    ("track", "vote"):   {"k": 164, "label": "Track 1 v2 + commit v2  (*)",        "metric": "cmajc"},  # 82.0%
}

ROW_LABELS = ["base LLaDA-8B", "Track 1 v3  +  commit v3"]
COL_LABELS = ["single-shot  (b=1)", "branch-vote  (b=5, t=0.7)"]
ROW_KEYS = ["base", "track"]
COL_KEYS = ["single", "vote"]


def cp_ci(k: int, n: int) -> tuple[float, float, float]:
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def draw() -> None:
    np.random.seed(RNG_SEED)

    fig, ax = plt.subplots(figsize=(11.0, 6.6))

    # Cell coordinates: cells laid out on a unit grid; cell centres at
    # (col + 0.5, row + 0.5). row=0 is *bottom* (base) so the visual
    # "up" direction is the param-time axis.
    cell_w, cell_h = 1.0, 1.0
    pad = 0.06

    # Compute pcts, find max for normalising fill
    cell_data = {}
    for r_i, r in enumerate(ROW_KEYS):
        for c_i, c in enumerate(COL_KEYS):
            spec = CELLS[(r, c)]
            v, lo, hi = cp_ci(spec["k"], N)
            cell_data[(r_i, c_i)] = (v, lo, hi, spec)

    # Draw the 4 cells as rounded rectangles. Colour = cool ramp keyed
    # to accuracy bin so the eye reads "more blue = better".
    def shade_for(v):
        # <76 base, <80.5 v2, >=80.5 v3
        if v < 76:
            return PALETTE.base
        elif v < 80.5:
            return PALETTE.v2
        else:
            return PALETTE.v3

    for (r_i, c_i), (v, lo, hi, spec) in cell_data.items():
        x0, y0 = c_i + pad, r_i + pad
        w = cell_w - 2 * pad
        h = cell_h - 2 * pad
        col = shade_for(v)
        rect = Rectangle(
            (x0, y0), w, h,
            facecolor=col, alpha=0.18,
            edgecolor=col, linewidth=1.0, zorder=1,
        )
        ax.add_patch(rect)

        # Big number — anchored higher so cell text below has clear room.
        ax.text(
            c_i + 0.5, r_i + 0.66, f"{v:.1f}%",
            ha="center", va="center",
            fontsize=28, fontweight="bold", color=col, zorder=3,
        )
        # CI line (one line, fits in cell width)
        ax.text(
            c_i + 0.5, r_i + 0.40,
            f"95% CI [{lo:.1f}, {hi:.1f}]",
            ha="center", va="center", fontsize=8.5, color=PALETTE.sub, zorder=3,
        )
        ax.text(
            c_i + 0.5, r_i + 0.30,
            f"k={spec['k']} / N={N}",
            ha="center", va="center", fontsize=7.8, color=PALETTE.sub, zorder=3,
        )
        # Cell label — italic, anchored at bottom of cell.
        ax.text(
            c_i + 0.5, r_i + 0.16,
            spec["label"],
            ha="center", va="center", fontsize=8.3, color=PALETTE.ink,
            style="italic", zorder=3,
        )
        # Metric tag (top-left of cell)
        ax.text(
            x0 + 0.04, y0 + h - 0.04,
            spec["metric"], ha="left", va="top",
            fontsize=9, color=col, fontweight="semibold", zorder=3,
        )

    # ---- Arrows: compositionality story --------------------------------
    # Arrows live in the inter-cell gutters (the `pad` strips). Vertical
    # arrows are drawn in the gutter ABOVE each column-cell pair; horizontal
    # arrows are drawn in the gutter to the RIGHT of each row-cell pair.
    base_v = cell_data[(0, 0)][0]
    base_vote = cell_data[(0, 1)][0]
    track_v = cell_data[(1, 0)][0]
    track_vote = cell_data[(1, 1)][0]

    # Horizontal arrows (compute-time vote):  span the gutter at x=1.0,
    # mid-cell vertically. Label sits centred on the arrow itself in a
    # small white-bg chip — it never overlaps cell content because the
    # arrow lives in the inter-cell gutter.
    arr_kw = dict(arrowstyle="-|>", color=PALETTE.warn, lw=1.6,
                  mutation_scale=16, zorder=5)
    for r_i, delta in ((0, base_vote - base_v), (1, track_vote - track_v)):
        ax.add_patch(FancyArrowPatch(
            (1.0 - pad - 0.02, r_i + 0.50),
            (1.0 + pad + 0.02, r_i + 0.50),
            **arr_kw,
        ))
        ax.text(
            1.0, r_i + 0.50, f"+{delta:.1f} pp",
            ha="center", va="center", fontsize=8.5, color=PALETTE.warn,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor=PALETTE.warn, linewidth=0.6),
        )

    # Vertical arrows (param-time LoRA):  span the gutter at y=1.0,
    # mid-column horizontally.
    v_kw = dict(arrowstyle="-|>", color=PALETTE.v3, lw=1.6,
                mutation_scale=16, zorder=5)
    for c_i, delta in ((0, track_v - base_v), (1, track_vote - base_vote)):
        ax.add_patch(FancyArrowPatch(
            (c_i + 0.50, 1.0 - pad - 0.02),
            (c_i + 0.50, 1.0 + pad + 0.02),
            **v_kw,
        ))
        ax.text(
            c_i + 0.50, 1.0, f"+{delta:.1f} pp",
            ha="center", va="center", fontsize=8.5, color=PALETTE.v3,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor=PALETTE.v3, linewidth=0.6),
        )

    # Legend for the arrow colours (drawn under the grid). Use small
    # arrow patches instead of unicode glyphs so every font renders them.
    legend_y = -0.32
    ax.add_patch(FancyArrowPatch(
        (0.05, legend_y), (0.30, legend_y),
        arrowstyle="-|>", color=PALETTE.warn, lw=1.4,
        mutation_scale=12, zorder=6,
    ))
    ax.text(
        0.36, legend_y,
        "compute-time vote  (b=5 cmaj)",
        ha="left", va="center", fontsize=9, color=PALETTE.warn,
        fontweight="semibold",
    )
    ax.add_patch(FancyArrowPatch(
        (1.95, legend_y - 0.07), (1.95, legend_y + 0.07),
        arrowstyle="-|>", color=PALETTE.v3, lw=1.4,
        mutation_scale=12, zorder=6,
    ))
    ax.text(
        2.10, legend_y,
        "param-time commit + Track-1 LoRA",
        ha="left", va="center", fontsize=9, color=PALETTE.v3,
        fontweight="semibold",
    )

    # ---- Composition annotation: are the gains additive? --------------
    # Predicted "no double-dip" was cmajc <= cmaj + 1 pp (param-time
    # gain absorbs into compute-time). Observed: gains compose.
    additive_predicted = base_v + (base_vote - base_v) + (track_v - base_v)  # = base + col + row
    observed = track_vote
    # Right-side prose box with white background so it reads cleanly.
    prose = (
        "Composition reading\n"
        "\n"
        f"  predicted-additive  $\\approx$  {additive_predicted:.1f}%\n"
        f"  observed                  =  {observed:.1f}%\n"
        f"  delta                          =  {observed - additive_predicted:+.1f} pp\n"
        "\n"
        "Pre-reg ceiling:  cmajc $\\leq$ cmaj + 1 pp.\n"
        f"Observed: +{track_vote - base_vote:.1f} pp on top of cmaj —\n"
        "structural and consensus mechanisms\n"
        "do partially independent work."
    )
    ax.text(
        2.32, 1.0, prose,
        ha="left", va="center", fontsize=9.0, color=PALETTE.ink,
        bbox=dict(
            boxstyle="round,pad=0.55", facecolor="white",
            edgecolor=PALETTE.rule, linewidth=0.8,
        ),
        linespacing=1.3,
    )

    # ---- Row / col labels ----------------------------------------------
    for r_i, lab in enumerate(ROW_LABELS):
        ax.text(
            -0.06, r_i + 0.5, lab,
            ha="right", va="center", fontsize=10.5, color=PALETTE.ink,
            fontweight="semibold", rotation=90,
        )
    for c_i, lab in enumerate(COL_LABELS):
        ax.text(
            c_i + 0.5, 2.06, lab,
            ha="center", va="bottom", fontsize=10.5, color=PALETTE.ink,
            fontweight="semibold",
        )

    # Footnote
    ax.text(
        0.0, -0.18,
        "* cmajc cell uses Track 1 v2 + commit v2 (Track 1 v3 + commit v3 cmajc not "
        "run before pod preempt; documented in RESULTS_TRACK2 §Open follow-ups #1).",
        ha="left", va="center", fontsize=7.5, color=PALETTE.sub, style="italic",
    )

    # Tidy axes
    ax.set_xlim(-0.55, 5.10)
    ax.set_ylim(-0.55, 2.30)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.suptitle(
        "Figure 4   c2c-vs-cmaj compositionality",
        fontsize=12.5, x=0.04, ha="left", y=0.965, fontweight="semibold",
    )
    fig.text(
        0.04, 0.92,
        "Param-time (LoRA) and compute-time (b=5 vote) consensus mechanisms "
        "compose, not absorb.",
        fontsize=9.5, color=PALETTE.sub, ha="left", va="bottom",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig4_compositionality.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
