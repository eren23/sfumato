"""Figure 3 — c2c design-iteration disentangling.

Five-condition waterfall showing how the +6.0pp c2c lift decomposes:
{v1/v2 baseline, v3-LoRA-only, ABL_A (block-coverage alone), ABL_B
(full-response loss alone), v3-full (both)}. Each bar carries a
Clopper-Pearson 95% CI, an attribution annotation, and a hover tooltip
in the interactive HTML version with the three diagnosed failure modes
(late-block-only commit, answer-span-only loss, mechanism gating).

Outputs:
  phase2/figures/fig3_c2c_disentangling.png      (static fallback for paper)
  phase2/figures/fig3_c2c_disentangling.pdf      (vector paper version)
  phase2/figures/fig3_c2c_disentangling.html     (interactive Plotly)

Numbers:    e2/RESULTS_TRACK2.md §Disentangling ablations
            scripts/binom_ci.py HEADLINE list (Tab6 entries)
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

# ---------------------------------------------------------------------------
# Conditions — keyed left-to-right; k counts from binom_ci.HEADLINE.
# ---------------------------------------------------------------------------
CONDS = [
    {
        "key": "v1v2",
        "title": "v1/v2 baseline",
        "subtitle": "no commit  (or v1/v2 commit)",
        "k": 141,
        "color": PALETTE.base,
        "delta": None, "delta_caption": "baseline",
        "failure_modes": {
            "late-block-only commit": 1,
            "answer-span-only loss":  1,
            "mode collapse (v1)":     0,
        },
    },
    {
        "key": "v3_alone",
        "title": "v3 LoRA only",
        "subtitle": "no commit at all",
        "k": 146,
        "color": "#cbd5e1",  # slate-300 (lighter cool stop, between base and v2)
        "delta": +2.5, "delta_caption": "v3 LoRA capacity",
        "failure_modes": {
            "late-block-only commit": 0,
            "answer-span-only loss":  0,
            "no commit applied":      1,
        },
    },
    {
        "key": "abl_a",
        "title": "ABL_A",
        "subtitle": "v3 LoRA + commit-v2\n(n_blocks=3, answer-span loss)",
        "k": 154,
        "color": PALETTE.v2,
        "delta": +4.0, "delta_caption": "block coverage  (dominant driver)",
        "failure_modes": {
            "late-block-only commit": 0,
            "answer-span-only loss":  1,
            "no commit applied":      0,
        },
    },
    {
        "key": "abl_b",
        "title": "ABL_B",
        "subtitle": "v3 LoRA + commit-v3\n(n_blocks=1, full-response loss)",
        "k": 146,
        "color": PALETTE.warn,
        "delta": +0.0, "delta_caption": "full-loss without coverage  =  useless",
        "failure_modes": {
            "late-block-only commit": 1,
            "answer-span-only loss":  0,
            "no commit applied":      0,
        },
    },
    {
        "key": "v3_full",
        "title": "v3 full",
        "subtitle": "commit + n_blocks=3\n+ full-response loss",
        "k": 158,
        "color": PALETTE.v3,
        "delta": +6.0, "delta_caption": "combined  —  near 80% target",
        "failure_modes": {
            "late-block-only commit": 0,
            "answer-span-only loss":  0,
            "no commit applied":      0,
        },
    },
]

PRE_REG_TARGET = 80.0   # %
CMAJ_BASE_TEST = 79.0   # %, base test cmaj b=5


def cp_ci(k: int, n: int) -> tuple[float, float, float]:
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def draw_static() -> None:
    np.random.seed(RNG_SEED)

    pcts = [cp_ci(c["k"], N) for c in CONDS]   # (val, lo, hi)
    vals = [p[0] for p in pcts]
    los = [p[0] - p[1] for p in pcts]
    his = [p[2] - p[0] for p in pcts]

    x = np.arange(len(CONDS))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    bars = ax.bar(
        x, vals, width=0.62,
        color=[c["color"] for c in CONDS],
        edgecolor=PALETTE.ink, linewidth=0.5, zorder=2,
    )
    ax.errorbar(
        x, vals, yerr=[los, his],
        fmt="none", ecolor=PALETTE.ink, elinewidth=0.8,
        capsize=3, capthick=0.8, zorder=3,
    )

    # Headline value above each bar (offset above CI top so it doesn't
    # collide with whiskers). Below the value, render a delta chip in
    # the bar's own colour so the +pp story reads at-a-glance.
    for i, (val, c, hi_ext) in enumerate(zip(vals, CONDS, his)):
        y_top = val + hi_ext + 0.4
        ax.text(
            x[i], y_top, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=11.5, fontweight="bold",
            color=PALETTE.ink, zorder=4,
        )
        if c["delta"] is None:
            chip = "baseline"
            chip_color = PALETTE.sub
        else:
            sign = "+" if c["delta"] >= 0 else ""
            chip = f"{sign}{c['delta']:.1f} pp   {c['delta_caption']}"
            chip_color = c["color"]
        ax.text(
            x[i], y_top + 1.55, chip,
            ha="center", va="bottom", fontsize=8.3,
            color=chip_color, fontweight="semibold", zorder=4,
        )

    # Target + cmaj reference lines
    ax.axhline(
        PRE_REG_TARGET, color=PALETTE.warn, lw=0.9, ls=(0, (5, 3)),
        alpha=0.85, zorder=1,
    )
    ax.text(
        len(CONDS) - 0.55, PRE_REG_TARGET + 0.4,
        f"pre-reg target = {PRE_REG_TARGET:.0f}%",
        ha="right", va="bottom", fontsize=8.5, color=PALETTE.warn,
    )
    ax.axhline(
        CMAJ_BASE_TEST, color=PALETTE.ok, lw=0.9, ls=(0, (1, 2)),
        alpha=0.8, zorder=1,
    )
    ax.text(
        len(CONDS) - 1.55, CMAJ_BASE_TEST - 0.45,
        f"base cmaj b=5 on test = {CMAJ_BASE_TEST:.0f}%",
        ha="right", va="top", fontsize=8.5, color=PALETTE.ok,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c['title']}\n{c['subtitle']}" for c in CONDS],
        fontsize=8.5,
    )
    ax.set_ylabel("c2c accuracy  (%, GSM8K-test, N=200, 95% Clopper-Pearson CI)")
    ax.set_ylim(64, 91)
    ax.set_title("Figure 3   c2c design-iteration  —  block coverage drives the lift")
    fig.text(
        0.0, 0.95,
        "Full-response loss alone (ABL_B) recovers nothing; combined with multi-block "
        "commit it adds +2 pp on top of ABL_A.",
        fontsize=8.7, color=PALETTE.sub, ha="left", va="bottom",
        transform=fig.transFigure,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig3_c2c_disentangling.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def draw_html() -> None:
    """Interactive Plotly HTML with hover tooltips per condition."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotly is required for fig3 HTML") from e

    pcts = [cp_ci(c["k"], N) for c in CONDS]
    vals = [p[0] for p in pcts]
    los = [p[0] - p[1] for p in pcts]
    his = [p[2] - p[0] for p in pcts]

    customdata = []
    for c, (v, lo, hi) in zip(CONDS, pcts):
        fm_lines = "<br>".join(
            f"  - {name}: {'present' if flag else 'fixed'}"
            for name, flag in c["failure_modes"].items()
        )
        if c["delta"] is None:
            attribution = "baseline"
        else:
            sign = "+" if c["delta"] >= 0 else ""
            attribution = f"{sign}{c['delta']:.1f} pp  ({c['delta_caption']})"
        customdata.append([
            c["title"],
            c["subtitle"].replace("\n", "  "),
            f"{v:.1f}", f"{lo:.1f}", f"{hi:.1f}",
            attribution,
            fm_lines,
            c["k"], N,
        ])

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "<i>%{customdata[1]}</i><br>"
        "c2c = %{customdata[2]}%  "
        "(95% CP CI [%{customdata[3]}, %{customdata[4]}])<br>"
        "k = %{customdata[7]} / N = %{customdata[8]}<br>"
        "attribution: %{customdata[5]}<br><br>"
        "<b>failure modes:</b><br>%{customdata[6]}"
        "<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[c["title"] for c in CONDS],
            y=vals,
            marker_color=[c["color"] for c in CONDS],
            marker_line_color=PALETTE.ink, marker_line_width=0.6,
            error_y=dict(
                type="data", symmetric=False,
                array=his, arrayminus=los,
                color=PALETTE.ink, thickness=1.0, width=4,
            ),
            customdata=customdata,
            hovertemplate=hover,
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont=dict(size=12, color=PALETTE.ink, family="Inter, Helvetica, Arial, sans-serif"),
            showlegend=False,
        )
    )

    # Reference lines
    fig.add_hline(
        y=PRE_REG_TARGET, line=dict(color=PALETTE.warn, dash="dash", width=1),
        annotation_text=f"pre-reg target = {PRE_REG_TARGET:.0f}%",
        annotation_position="top right",
        annotation_font_color=PALETTE.warn, annotation_font_size=10,
    )
    fig.add_hline(
        y=CMAJ_BASE_TEST, line=dict(color=PALETTE.ok, dash="dot", width=1),
        annotation_text=f"base cmaj b=5 on test = {CMAJ_BASE_TEST:.0f}%",
        annotation_position="bottom right",
        annotation_font_color=PALETTE.ok, annotation_font_size=10,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Figure 3   c2c design-iteration  —  block coverage drives the lift"
                "<br>"
                "<sub style='color:#374151'>Hover any bar for per-condition CI and "
                "diagnosed failure modes.  Full-response loss without multi-block "
                "commit is useless (ABL_B).</sub>"
            ),
            x=0.0, xanchor="left", font=dict(size=15, color=PALETTE.ink, family="Inter, Helvetica, Arial, sans-serif"),
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=10, color=PALETTE.sub, family="Inter, Helvetica, Arial, sans-serif"),
            showgrid=False, linecolor=PALETTE.ink,
        ),
        yaxis=dict(
            title=dict(text="c2c accuracy (%)", font=dict(size=11, color=PALETTE.sub)),
            range=[64, 88],
            gridcolor=PALETTE.rule, zeroline=False, linecolor=PALETTE.ink,
            tickfont=dict(size=10, color=PALETTE.sub),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=20, t=110, b=70),
        height=540, width=1000,
        font=dict(family="Inter, Helvetica, Arial, sans-serif"),
    )

    out = OUT_DIR / "fig3_c2c_disentangling.html"
    fig.write_html(
        str(out),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=False, responsive=True),
    )
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    draw_static()
    draw_html()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
