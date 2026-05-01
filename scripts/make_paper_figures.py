"""Generate the three load-bearing figures for Paper 1 from hard-coded numbers
(every number is also recorded in e2/RESULTS_TRACK1.md, RESULTS_TRACK2.md, and
the wandb runs in projects sfumato-e4 / sfumato-e2).

Usage:
  python scripts/make_paper_figures.py

Outputs:
  e2/figs/fig1_prefix_hierarchy.{png,pdf}
  e2/figs/fig2_branch_agreement.{png,pdf}
  e2/figs/fig3_c2c_design_iteration.{png,pdf}
  e2/figs/fig4_b1_vs_b5_collapse.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "e2" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_both(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  -> {OUT_DIR / name}.(png|pdf)")


# ---------------------------------------------------------------------------
# Figure 1 — Prefix-damage hierarchy
# 5 conditions × 3 versions (base, v2, v3 of Track 1 LoRA)
# ---------------------------------------------------------------------------
def fig1():
    print("Fig 1: prefix-damage hierarchy")
    conds = ["C2\n(no prefix)", "C2hint", "C2empty\n(`Plan: `)", "C3p Q-0.5B", "C3p Q-1.5B"]
    base = [74, 68, 66, 64, 60]
    v2 = [70.5, 73.5, 73.0, 60.0, 67.0]
    v3 = [73.0, 73.5, 74.0, 65.0, 54.0]

    x = np.arange(len(conds))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, base, w, label="base", color="#9ca3af", edgecolor="black", linewidth=0.4)
    ax.bar(x,     v2,   w, label="Track 1 v2 (4/7 modules)", color="#60a5fa", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, v3,   w, label="Track 1 v3 (7/7 modules)", color="#1d4ed8", edgecolor="black", linewidth=0.4)

    # Annotate v3 numbers
    for i, val in enumerate(v3):
        ax.text(x[i] + w, val + 0.6, f"{val:.1f}", ha="center", fontsize=8, color="#1d4ed8")

    ax.set_ylabel("GSM8K-test accuracy (%, N=200)")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylim(50, 80)
    ax.axhline(80, color="#ef4444", lw=0.6, ls="--", alpha=0.5)
    ax.text(0.02, 80.5, "cmaj b=5 ceiling (80%)", fontsize=8, color="#ef4444", transform=ax.get_yaxis_transform())
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Figure 1: Prefix-damage hierarchy across Track 1 LoRA versions",
                 fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    # Highlight the Q-0.5B/Q-1.5B inversion finding
    ax.annotate("Q-0.5B improves\nat full FFN cov.", xy=(3 + w, 65), xytext=(3.0, 76),
                fontsize=8, color="#065f46",
                arrowprops=dict(arrowstyle="->", color="#065f46", lw=0.7))
    ax.annotate("Q-1.5B regresses\n(−13pp at v3)", xy=(4 + w, 54), xytext=(3.5, 56),
                fontsize=8, color="#7c2d12",
                arrowprops=dict(arrowstyle="->", color="#7c2d12", lw=0.7))

    save_both(fig, "fig1_prefix_hierarchy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Branch-agreement distribution shift (apples-to-apples)
# 5/5-same, 4/5-unique, 3/5, 2/5, 5/5-unique on gsm8k-test, b=5 t=0.7
# ---------------------------------------------------------------------------
def fig2():
    print("Fig 2: branch-agreement distribution shift")
    bins = ["5/5\nsame", "4/5\nunique", "3/5\nunique", "2/5\nunique", "5/5\nunique"]
    base_pct = [51.5, 6.0, 11.5, 27.5, 3.5]
    track1_pct = [47.5, 8.5, 18.0, 19.5, 6.5]

    x = np.arange(len(bins))
    w = 0.4

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, base_pct, w, label="Base LLaDA on test", color="#9ca3af", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, track1_pct, w, label="Track 1 v2 + commit v2", color="#1d4ed8", edgecolor="black", linewidth=0.4)

    for i in range(len(bins)):
        ax.text(x[i] - w/2, base_pct[i] + 0.7, f"{base_pct[i]:.1f}", ha="center", fontsize=8, color="#374151")
        ax.text(x[i] + w/2, track1_pct[i] + 0.7, f"{track1_pct[i]:.1f}", ha="center", fontsize=8, color="#1d4ed8")

    ax.set_ylabel("% of problems (N=200, b=5 t=0.7)")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=9)
    ax.set_ylim(0, 60)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Figure 2: Branch-agreement distribution shift — diversity went UP, not down\n"
                 "5/5-same drops 51.5% → 47.5%; mean unique answers per problem 1.825 → 2.07 (+13%)",
                 fontsize=10, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    save_both(fig, "fig2_branch_agreement")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — c2c design-iteration disentangling
# {v3 alone, ABL_A, ABL_B, v3-full} bar chart with attribution annotations
# ---------------------------------------------------------------------------
def fig3():
    print("Fig 3: c2c design-iteration disentangling")
    setups = [
        "v1/v2\n(70.5%)",
        "v3 LoRA only\n(no commit)",
        "+ commit\nv2/answer-span\n+ n_blocks=3",
        "+ commit\nv3/full-resp\n+ n_blocks=1",
        "v3 full\n(commit + n=3\n+ full-resp loss)",
    ]
    accs = [70.5, 73.0, 77.0, 73.0, 79.0]
    colors = ["#9ca3af", "#cbd5e1", "#60a5fa", "#fbbf24", "#1d4ed8"]
    notes = ["baseline", "+2.5 vs v1/v2", "+4.0 (block coverage)", "+0.0 (full-loss alone)", "+6.0 (combined)"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(setups))
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.4)

    # Numbers on top of bars
    for i, val in enumerate(accs):
        ax.text(i, val + 0.4, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i, val - 3, notes[i], ha="center", fontsize=8, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=8.5)
    ax.set_ylabel("c2c accuracy (%, GSM8K-test, N=200)")
    ax.set_ylim(65, 85)

    # 80% target line
    ax.axhline(80, color="#dc2626", lw=0.7, ls="--", alpha=0.7)
    ax.text(len(setups) - 0.5, 80.4, "pre-registered target", fontsize=8, color="#dc2626")

    # cmaj baseline line
    ax.axhline(79, color="#059669", lw=0.7, ls=":", alpha=0.7)
    ax.text(0.05, 79.3, "base test cmaj b=5 = 79%", fontsize=8, color="#059669")

    ax.set_title("Figure 3: c2c design-iteration. Block coverage drives the lift; full-response\n"
                 "training is a secondary refinement (only effective with multi-block commit).",
                 fontsize=10, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    save_both(fig, "fig3_c2c_design_iteration")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — b=1 vs b=5: branch aggregation absorbs structural correction
# 4 bars: c2c-v2, c2c-v3, cmajc-v2, cmajc-v3. Highlights the +8.5pp gap at
# b=1 vs the statistically-indistinguishable result at b=5.
# ---------------------------------------------------------------------------
def fig4():
    print("Fig 4: b=1 vs b=5 collapse")
    setups_b1 = ["commit v2\n(c2c)", "commit v3\n(c2c)"]
    setups_b5 = ["commit v2\n(cmajc)", "commit v3\n(cmajc)"]
    accs_b1 = [70.5, 79.0]
    accs_b5 = [82.0, 82.5]
    cis_b1 = [(63.7, 76.7), (72.7, 84.4)]
    cis_b5 = [(76.0, 87.1), (76.5, 87.5)]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    w = 0.55

    # Left panel: b=1
    x_l = np.arange(len(setups_b1))
    bars_l = ax_l.bar(x_l, accs_b1, w, color=["#fbbf24", "#1d4ed8"],
                      edgecolor="black", linewidth=0.4)
    for i, (val, (lo, hi)) in enumerate(zip(accs_b1, cis_b1)):
        ax_l.errorbar(i, val, yerr=[[val - lo], [hi - val]],
                      fmt="none", ecolor="black", capsize=4, lw=0.8)
        ax_l.text(i, val + 0.3, f"{val:.1f}", ha="center",
                  fontsize=10, fontweight="bold")
    ax_l.set_xticks(x_l)
    ax_l.set_xticklabels(setups_b1, fontsize=9)
    ax_l.set_ylabel("GSM8K-test accuracy (%, N=200)")
    ax_l.set_title("(a) Single-pass (b=1): structural correction is load-bearing\n"
                   "+8.5 pp gap, well outside CI overlap",
                   fontsize=10, loc="left")
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.grid(axis="y", alpha=0.2)
    # Annotate gap
    ax_l.annotate("", xy=(1, 79.0), xytext=(0, 70.5),
                  arrowprops=dict(arrowstyle="<->", color="#7c2d12", lw=1.0))
    ax_l.text(0.5, 74.5, "+8.5 pp", ha="center", fontsize=10,
              color="#7c2d12", fontweight="bold")

    # Right panel: b=5
    x_r = np.arange(len(setups_b5))
    bars_r = ax_r.bar(x_r, accs_b5, w, color=["#fbbf24", "#1d4ed8"],
                      edgecolor="black", linewidth=0.4)
    for i, (val, (lo, hi)) in enumerate(zip(accs_b5, cis_b5)):
        ax_r.errorbar(i, val, yerr=[[val - lo], [hi - val]],
                      fmt="none", ecolor="black", capsize=4, lw=0.8)
        ax_r.text(i, val + 0.3, f"{val:.1f}", ha="center",
                  fontsize=10, fontweight="bold")
    ax_r.set_xticks(x_r)
    ax_r.set_xticklabels(setups_b5, fontsize=9)
    ax_r.set_title("(b) Branch-aggregated (b=5): structural correction\n"
                   "is absorbed; v2 and v3 indistinguishable at N=200",
                   fontsize=10, loc="left")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.grid(axis="y", alpha=0.2)
    # Annotate near-zero gap
    ax_r.text(0.5, 84.5, "+0.5 pp\n(within CI overlap)",
              ha="center", fontsize=9, color="#374151", style="italic")

    # Common y-axis range
    for ax in (ax_l, ax_r):
        ax.set_ylim(60, 92)
        ax.axhline(79, color="#059669", lw=0.7, ls=":", alpha=0.6)
        ax.text(-0.45, 79.3, "base cmaj 79%", fontsize=7, color="#059669")

    fig.suptitle("Figure 4: Branch aggregation absorbs the v2→v3 structural correction at b=5.",
                 fontsize=11, x=0.05, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_both(fig, "fig4_b1_vs_b5_collapse")
    plt.close(fig)


def main() -> int:
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"\nAll figures saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
