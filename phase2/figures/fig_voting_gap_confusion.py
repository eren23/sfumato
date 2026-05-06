"""Figure — Voting-rule gap confusion matrix on GSM8K-test N=200.

Per-problem 2x2 contingency: cmaj-correct/wrong x oracle-correct/wrong.
Substrate: e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl
(LLaDA-8B + prefix-robust-v3 LoRA, 5 branches, k=64, tau=0.7).

Numbers (computed from JSONL):
  cmaj correct       159 / 200 = 79.5%
  oracle correct     177 / 200 = 88.5%
  voting gap          18 / 200 =  9.0pp
  both right         159 (the easy cells)
  ORACLE-ONLY (gap)   18  <-- voting threw away a correct branch
  cmaj-only            0  (impossible by construction)
  both wrong          23

Outputs:
  phase2/figures/fig_voting_gap_confusion.png
  phase2/figures/fig_voting_gap_confusion.pdf
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import PALETTE  # noqa: E402

plt.style.use(str(HERE / "sfumato.mplstyle"))

OUT_DIR = HERE
JSONL = REPO_ROOT / "e4" / "results" / "raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl"

PAT_STRICT = re.compile(r"(?:####|Answer:)\s*(-?\$?\d[\d,]*\.?\d*)")
NUM_PAT = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract(s: str) -> str | None:
    if not s:
        return None
    m = PAT_STRICT.search(s)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    nums = NUM_PAT.findall(s)
    return nums[-1].replace(",", "").replace("$", "") if nums else None


def norm(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


def compute_cells() -> dict:
    cells = {"both_right": 0, "oracle_only": 0, "cmaj_only": 0, "both_wrong": 0}
    n = 0
    for line in JSONL.open():
        rec = json.loads(line)
        n += 1
        gold = norm(rec["gold"])
        cmaj_pred = norm(str(rec["pred"]))
        cmaj_ok = cmaj_pred == gold
        branches = [
            extract(rec["trace"].get(f"branch_{i}", "") or "") for i in range(5)
        ]
        branches = [norm(b) for b in branches if b is not None]
        oracle_ok = gold in branches
        if cmaj_ok and oracle_ok:
            cells["both_right"] += 1
        elif (not cmaj_ok) and oracle_ok:
            cells["oracle_only"] += 1
        elif cmaj_ok and not oracle_ok:
            cells["cmaj_only"] += 1
        else:
            cells["both_wrong"] += 1
    cells["n"] = n
    cells["cmaj_acc"] = (cells["both_right"] + cells["cmaj_only"]) / n
    cells["oracle_acc"] = (cells["both_right"] + cells["oracle_only"]) / n
    cells["gap_pp"] = (cells["oracle_only"] - cells["cmaj_only"]) / n * 100
    return cells


def draw() -> None:
    c = compute_cells()
    n = c["n"]

    # 2x2 matrix, rows = oracle (top: had-it / bottom: no-correct-branch),
    # cols = cmaj (left: voting-correct / right: voting-wrong).
    matrix = np.array(
        [
            [c["both_right"], c["oracle_only"]],   # oracle correct
            [c["cmaj_only"],  c["both_wrong"]],    # oracle wrong
        ]
    )
    cell_pct = matrix / n * 100

    fig = plt.figure(figsize=(9.0, 5.6))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.55, 1.0], wspace=0.30,
        left=0.06, right=0.97, top=0.85, bottom=0.13,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Panel (a): the 2x2 contingency -----------------------------
    cmap_colors = [
        # both-right (green-ish ok), oracle-only GAP (warn), cmaj-only impossible (gray), both-wrong (gray)
        [PALETTE.ok, PALETTE.warn],
        [PALETTE.base, PALETTE.sub],
    ]
    for r in range(2):
        for cc in range(2):
            face = cmap_colors[r][cc]
            edge = PALETTE.ink
            rect = plt.Rectangle(
                (cc, 1 - r), 1, 1,
                facecolor=face, edgecolor=edge, linewidth=1.0, zorder=1,
            )
            ax.add_patch(rect)
            count = matrix[r, cc]
            pct = cell_pct[r, cc]
            txt = f"{count}\n({pct:.1f}%)"
            color = "white" if face != PALETTE.base else PALETTE.ink
            ax.text(
                cc + 0.5, 1 - r + 0.55, txt,
                ha="center", va="center", color=color,
                fontsize=14, fontweight="bold", zorder=3,
            )
    # Highlight the GAP cell
    gap_rect = plt.Rectangle(
        (1, 1), 1, 1, facecolor="none",
        edgecolor=PALETTE.ink, linewidth=2.5, zorder=4,
        linestyle="--",
    )
    ax.add_patch(gap_rect)
    ax.annotate(
        "voting-rule gap\n(oracle had it, vote missed)",
        xy=(1.5, 1.5), xytext=(2.55, 1.65),
        ha="left", va="center", fontsize=10, color=PALETTE.warn,
        arrowprops=dict(arrowstyle="-", color=PALETTE.warn, lw=1.3),
    )

    ax.set_xlim(-0.05, 3.4)
    ax.set_ylim(-0.05, 2.10)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["cmaj correct", "cmaj wrong"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["oracle\nwrong", "oracle\ncorrect"], fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"GSM8K-test N={n}, LLaDA-8B + prefix-robust-v3 LoRA, b=5, τ=0.7",
        fontsize=10.5, pad=8,
    )

    # ---- Panel (b): summary bar -------------------------------------
    bars = ["cmaj", "oracle", "gap"]
    vals = [c["cmaj_acc"] * 100, c["oracle_acc"] * 100, c["gap_pp"]]
    colors = [PALETTE.base, PALETTE.ok, PALETTE.warn]
    bbars = ax2.bar(bars, vals, color=colors, edgecolor=PALETTE.ink, linewidth=0.5)
    for bb, v in zip(bbars, vals):
        ax2.text(
            bb.get_x() + bb.get_width() / 2, bb.get_height() + 1.1,
            f"{v:.1f}%" if v > 30 else f"{v:.1f}pp",
            ha="center", fontsize=10, color=PALETTE.ink, fontweight="bold",
        )
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("accuracy / gap (%)")
    ax2.set_title("Aggregate accuracy + gap", fontsize=10.5, pad=8)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.suptitle(
        "The voting-rule gap: oracle had a correct branch on 9 pp of problems voting threw away",
        fontsize=12, fontweight="bold", y=0.96,
    )

    out_png = OUT_DIR / "fig_voting_gap_confusion.png"
    out_pdf = OUT_DIR / "fig_voting_gap_confusion.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    print(f"cells: {c}")


if __name__ == "__main__":
    draw()
