"""Figure — Commit-LoRA K2 ablation: COMMIT_N_BLOCKS sweep.

Inverted-U curve. Peak at k=3 (sub-blocks 2-4), drops on either side:
  - k=0 (commit-LoRA off): 0.805 = -1.7pp
  - k=3 (sfumato-v3 default, blocks 2-4): 0.822 = baseline (multi-seed mean)
  - k=4 (commit-LoRA always-on): 0.790 = -3.2pp

Source data:
  - k=0: phase2/spikes/k2-commit-blocks-ablation, wandb mo4clpp4 (this work)
  - k=3: e4/results/raw_cmajc_k64_seed{0,1,2}*N{100,200}*.jsonl, multi-seed mean
  - k=4: this work, wandb ho24ezlz

The curve confirms the schedule-toggle hypothesis: commit-LoRA helps
when applied to sub-blocks 2-4 (where answer formatting concentrates)
but hurts when applied to sub-block 1 (which the prefix-robust LoRA
should drive). The sub-block-1 boundary is load-bearing.

Outputs:
  phase2/figures/fig_commit_lora_k2_sweep.png
  phase2/figures/fig_commit_lora_k2_sweep.pdf
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
N = 200

# Data: (commit_n_blocks, accuracy, n_correct)
ROWS = [
    (0, 0.805, int(0.805 * N)),
    (3, 0.822, int(0.822 * N)),  # using mean for plotting; CIs handle variance
    (4, 0.790, int(0.790 * N)),
]


def cp_ci(k: int, n: int) -> tuple[float, float, float]:
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def draw() -> None:
    xs = np.array([r[0] for r in ROWS])
    accs = np.array([r[1] for r in ROWS]) * 100
    cis = [cp_ci(r[2], N) for r in ROWS]
    lows = [c[1] for c in cis]
    highs = [c[2] for c in cis]
    yerr = np.array([
        accs - np.array(lows),
        np.array(highs) - accs,
    ])

    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    # cmaj baseline reference (no commit-LoRA at all, just majority vote): 0.795
    ax.axhline(
        79.5, color=PALETTE.sub, linewidth=1.0, linestyle=":", alpha=0.7, zorder=1,
    )
    ax.text(
        4.05, 79.5 + 0.2, "cmaj baseline (no commit-LoRA): 79.5%",
        fontsize=8.5, color=PALETTE.sub, va="bottom", ha="right",
    )

    # Multi-seed sigma band for k=3 (mean 0.822, ±0.85pp)
    ax.fill_between(
        [-0.3, 4.3], 82.2 - 0.85, 82.2 + 0.85,
        color=PALETTE.v3, alpha=0.10, zorder=1,
    )
    ax.text(
        4.05, 82.2 + 1.5, "k=3 multi-seed band (σ ≈ 0.85pp)",
        fontsize=8.5, color=PALETTE.v3, va="bottom", ha="right",
    )

    # Connect with a line to suggest the inverted-U
    ax.plot(
        xs, accs, color=PALETTE.v3, linewidth=2.0, linestyle="-",
        alpha=0.85, zorder=2,
    )

    # Per-point markers + CIs
    colors = [PALETTE.warn, PALETTE.ok, PALETTE.warn]
    for x, y, c in zip(xs, accs, colors):
        ax.errorbar(
            x, y, yerr=[[y - cp_ci(int(y / 100 * N), N)[1]],
                        [cp_ci(int(y / 100 * N), N)[2] - y]],
            fmt="o", markersize=10,
            markerfacecolor=c, markeredgecolor=PALETTE.ink,
            markeredgewidth=0.7, ecolor=PALETTE.ink, elinewidth=1.0,
            capsize=4, zorder=3,
        )
        ax.text(
            x, y - 1.7, f"{y:.1f}%",
            ha="center", fontsize=10, color=PALETTE.ink, fontweight="bold",
        )

    # Annotate the peak
    ax.annotate(
        "sfumato-v3 default\n(blocks 2-4)",
        xy=(3, 82.2), xytext=(2.05, 86.5),
        ha="center", va="bottom", fontsize=9, color=PALETTE.ok,
        arrowprops=dict(arrowstyle="-", color=PALETTE.ok, lw=1.0, alpha=0.7),
    )
    ax.annotate(
        "always-on\nhurts (-3.2pp)",
        xy=(4, 79.0), xytext=(3.85, 75.3),
        ha="left", va="top", fontsize=9, color=PALETTE.warn,
        arrowprops=dict(arrowstyle="-", color=PALETTE.warn, lw=1.0, alpha=0.7),
    )
    ax.annotate(
        "off\nfloor (-1.7pp)",
        xy=(0, 80.5), xytext=(0.18, 76.5),
        ha="left", va="top", fontsize=9, color=PALETTE.warn,
        arrowprops=dict(arrowstyle="-", color=PALETTE.warn, lw=1.0, alpha=0.7),
    )

    # Sub-block diagram in title region
    ax.set_xlim(-0.4, 4.4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0\n(off)", "1\n(last)", "2", "3\n(blocks 2–4)", "4\n(all)"])
    ax.set_xlabel("COMMIT_N_BLOCKS — number of sub-blocks where commit-LoRA is active")
    ax.set_ylim(72, 90)
    ax.set_ylabel("cmajc accuracy (% on GSM8K-test N=200)")
    ax.set_title(
        "Commit-LoRA timing: schedule-toggle peaks at k=3, drops on either side",
        fontsize=11.5, fontweight="bold", pad=10,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.13)
    out_png = OUT_DIR / "fig_commit_lora_k2_sweep.png"
    out_pdf = OUT_DIR / "fig_commit_lora_k2_sweep.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    draw()
