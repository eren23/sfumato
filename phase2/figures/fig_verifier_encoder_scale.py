"""Figure — Verifier encoder-scale trend (8 architectures).

Per-branch supervised classifier, mean-pooled last-layer hidden states +
2-layer MLP, 5-fold CV split by problem on N=1750 labelled branches
from the v3 LoRA substrate (200 GSM8K-test problems × 5 branches × 175
records left after filtering for clean branch text).

Numbers from phase2/spikes/verifier-aggregation/RESULT.md
Night-1 ADDENDUM #4 (2026-05-03):

  encoder              params  verifier   delta vs cmaj 80.5
  TF-IDF + LR          250K     66.5%     -14.0pp  gap-closure -156%
  Qwen3-Embedding-4B    4B      68.5%     -12.0pp           -133%
  Qwen2.5-0.5B (chat)  500M     72.0%     -8.5pp            -94%
  Qwen3-Embedding-8B    8B      72.5%     -8.0pp            -89%
  Qwen2.5-Math-7B       7B      74.0%     -6.5pp            -72%
  Qwen3-8B (chat)       8B      75.0%     -5.5pp            -61%
  Qwen2.5-7B (chat)     7B      76.5%     -4.0pp            -44%

Story: chat encoders monotone-improve with scale, but math-tuned and
embedding-tuned variants UNDERPERFORM at the same scale. Bottleneck is
the supervised-classification objective, not feature quality.
"""

from __future__ import annotations

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

# (label, params, verifier_acc, family)
# family ∈ {"chat", "embedding", "math", "tf-idf"}
ROWS = [
    ("TF-IDF + LR",          0.00025, 66.5, "tf-idf"),
    ("Qwen2.5-0.5B (chat)",  0.5,     72.0, "chat"),
    ("Qwen3-Emb-4B",         4.0,     68.5, "embedding"),
    ("Qwen2.5-Math-7B",      7.0,     74.0, "math"),
    ("Qwen2.5-7B (chat)",    7.0,     76.5, "chat"),
    ("Qwen3-Emb-8B",         8.0,     72.5, "embedding"),
    ("Qwen3-8B (chat)",      8.0,     75.0, "chat"),
]
CMAJ_BASELINE = 80.5  # mean across folds
ORACLE_CEILING = 89.5  # mean across folds


def draw() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    # Reference horizontal lines
    ax.axhline(
        CMAJ_BASELINE, color=PALETTE.ink, linewidth=1.2, linestyle="-",
        alpha=0.55, zorder=1,
    )
    ax.axhline(
        ORACLE_CEILING, color=PALETTE.ok, linewidth=1.2, linestyle="--",
        alpha=0.85, zorder=1,
    )
    ax.text(
        0.18, CMAJ_BASELINE + 0.4, f"cmaj baseline ({CMAJ_BASELINE:.1f}%)",
        fontsize=9, color=PALETTE.ink, alpha=0.8,
    )
    ax.text(
        0.18, ORACLE_CEILING + 0.4, f"oracle ceiling ({ORACLE_CEILING:.1f}%)",
        fontsize=9, color=PALETTE.ok,
    )

    # Plot points by family
    family_color = {
        "tf-idf": PALETTE.sub,
        "chat": PALETTE.v3,
        "embedding": PALETTE.warn,
        "math": "#a16207",  # darker amber for math (Tailwind amber-700-ish)
    }
    family_marker = {
        "tf-idf": "x",
        "chat": "o",
        "embedding": "s",
        "math": "D",
    }
    family_label = {
        "tf-idf": "TF-IDF",
        "chat": "chat-LM (chat)",
        "embedding": "embedding-tuned",
        "math": "math-tuned",
    }

    seen_labels = set()
    for label, params, acc, fam in ROWS:
        marker = family_marker[fam]
        color = family_color[fam]
        family_text = family_label[fam] if fam not in seen_labels else None
        seen_labels.add(fam)
        ax.scatter(
            params, acc, s=120, marker=marker,
            facecolors=color, edgecolors=PALETTE.ink, linewidths=0.7,
            zorder=4, label=family_text,
        )
        # offset label
        dx, dy = 1.18, 0.7
        ha = "left"
        if "Qwen3-8B (chat)" in label:
            dx, dy = 0.92, -1.25
            ha = "left"
        if "Qwen2.5-7B (chat)" in label:
            dx, dy = 1.1, 0.9
        if "Qwen2.5-Math-7B" in label:
            dx, dy = 1.1, -1.5
        if "Qwen3-Emb-8B" in label:
            dx, dy = 1.18, -1.4
        if "Qwen3-Emb-4B" in label:
            dx, dy = 1.18, -1.4
        ax.annotate(
            label, xy=(params, acc), xytext=(params * dx, acc + dy),
            ha=ha, va="center", fontsize=9, color=PALETTE.ink,
        )

    # Connect chat encoders to show the monotone trend
    chat_pts = sorted(
        [(p, a) for (_l, p, a, fam) in ROWS if fam == "chat"], key=lambda x: x[0]
    )
    cps = np.array(chat_pts)
    ax.plot(
        cps[:, 0], cps[:, 1], color=PALETTE.v3, linewidth=1.5,
        alpha=0.55, zorder=2, linestyle=":",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.0001, 30)
    ax.set_xlabel("Verifier encoder parameter count (B, log scale)")
    ax.set_ylim(60, 95)
    ax.set_ylabel("Verifier accuracy (% on 5-fold CV, N=1750 branches)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_title(
        "Verifier-encoder scaling: chat-LMs monotone, but embedding/math tunings hurt",
        fontsize=11.5, fontweight="bold", pad=10,
    )

    # Subtitle: gap-closure annotation
    fig.text(
        0.5, 0.91,
        "gap-closure: TF-IDF −156%  →  Qwen2.5-0.5B −94%  →  Qwen2.5-7B (chat) −44%   "
        "(all 8 architectures UNDER cmaj baseline)",
        ha="center", fontsize=9.5, color=PALETTE.sub,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_png = OUT_DIR / "fig_verifier_encoder_scale.png"
    out_pdf = OUT_DIR / "fig_verifier_encoder_scale.pdf"
    plt.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.13)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    draw()
