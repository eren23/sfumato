"""Real-data matplotlib visualizations from e4/results/raw_*.jsonl.

Produces three PNG figures:
  1. accuracy_vs_flops.png  — log-FLOPs scatter, all conditions, annotated
  2. branch_agreement.png   — cmaj b=5 disagreement histogram + correct/wrong
  3. correctness_heatmap.png — problem × condition correctness matrix

Usage:
    python scripts/plot_real_data.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "e4" / "results"
FIGS = RESULTS / "figs"
FIGS.mkdir(exist_ok=True)


def load_cells() -> dict[tuple, dict]:
    """Load all raw_*.jsonl, group by (cond, k, ar, branches), keep cells with n>=10."""
    cells: dict[tuple, dict] = {}
    for raw in sorted(RESULTS.glob("raw_*.jsonl")):
        rows = []
        for line in raw.open():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(rows) < 10:
            continue
        r0 = rows[0]
        key = (
            r0["condition"],
            r0.get("k_steps", 0),
            r0.get("ar_model", "unknown"),
            r0.get("branches", 1),
        )
        # If we have multiple files for the same key (e.g. seed variants),
        # store seeds separately.
        seed = r0.get("seed", 0)
        full_key = key + (seed,)
        cells[full_key] = {
            "condition": r0["condition"],
            "k_steps": r0.get("k_steps", 0),
            "ar_model": r0.get("ar_model", "unknown"),
            "branches": r0.get("branches", 1),
            "seed": seed,
            "n": len(rows),
            "n_correct": sum(int(r["correct"]) for r in rows),
            "accuracy": sum(int(r["correct"]) for r in rows) / len(rows),
            "mean_flops": sum(r["flops"] for r in rows) / len(rows),
            "rows": rows,
            "source": raw.name,
        }
    return cells


# Color palette: distinct hue per condition family.
COLORS = {
    "c1": "#888888",            # gray — pure AR baseline
    "c2": "#1f77b4",            # blue — pure diffusion (winner among singles)
    "c3": "#d62728",            # red — text-handoff hybrid (loses)
    "c3p": "#d62728",           # same as c3
    "c4": "#8b0000",            # dark red — multi-round (loses worse)
    "c2hint": "#ff9896",        # light red — generic prefix
    "c2empty": "#ff7f0e",       # orange — empty prefix
    "crev": "#9467bd",          # purple — reverse hybrid
    "cmerge": "#bcbd22",        # olive — branch+merge (mid)
    "cmaj": "#2ca02c",          # green — branch+vote (winner)
}

LABELS = {
    "c1": "C1: Qwen-0.5B AR",
    "c2": "C2: LLaDA-8B alone",
    "c3": "C3: AR plan→diff→AR final",
    "c3p": "C3p: AR plan→diff (no final)",
    "c4": "C4: multi-round hybrid",
    "c2hint": "C2hint: generic prefix",
    "c2empty": "C2empty: bare 'Plan: ' prefix",
    "crev": "crev: LLaDA→Qwen finalize",
    "cmerge": "cmerge: branches→Qwen merge",
    "cmaj": "cmaj: branches→majority vote",
}


def fig_accuracy_vs_flops(cells: dict[tuple, dict]) -> Path:
    """Scatter all cells on log-FLOPs vs accuracy. Connect k-sweeps and
    branch-sweeps with lines for the same model+condition family."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Group cells for line plotting:
    # (cond, ar_model, branches) groups vary by k.
    # (cond, ar_model, k) groups vary by branches.
    by_kgroup: dict[tuple, list[dict]] = defaultdict(list)
    by_bgroup: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells.values():
        if c["seed"] != 0:
            continue
        by_kgroup[(c["condition"], c["ar_model"], c["branches"])].append(c)
        by_bgroup[(c["condition"], c["ar_model"], c["k_steps"])].append(c)

    drawn_labels = set()
    # Plot k-sweep lines.
    for key, group in by_kgroup.items():
        cond = key[0]
        if len(group) < 2:
            continue
        group.sort(key=lambda x: x["k_steps"])
        xs = [c["mean_flops"] for c in group]
        ys = [c["accuracy"] * 100 for c in group]
        color = COLORS.get(cond, "#444")
        ax.plot(xs, ys, "-", color=color, alpha=0.4, linewidth=1)

    # Plot branch-sweep lines (cmaj b ∈ {3, 5, ...}).
    for key, group in by_bgroup.items():
        cond = key[0]
        if cond not in {"cmaj", "cmerge"} or len(group) < 2:
            continue
        group.sort(key=lambda x: x["branches"])
        xs = [c["mean_flops"] for c in group]
        ys = [c["accuracy"] * 100 for c in group]
        color = COLORS.get(cond, "#444")
        ax.plot(xs, ys, "--", color=color, alpha=0.5, linewidth=1.5)

    # Plot points.
    for c in cells.values():
        cond = c["condition"]
        color = COLORS.get(cond, "#444")
        # Different markers for different "ar_model" (planner) variants.
        marker = "o" if "0.5B" in c["ar_model"] else "s"
        size = 120 if cond in {"cmaj"} else 70
        if cond == "c2" and c["seed"] != 0:
            # mark non-seed-0 c2 with an x to show variance.
            marker = "x"
            size = 50
        label = LABELS.get(cond, cond)
        if c["branches"] > 1:
            label = f"{label} (b={c['branches']})"
        if "1.5B" in c["ar_model"] and "Plan" not in label:
            label = f"{label}, plan=Qwen-1.5B"
        if label in drawn_labels:
            label = None
        else:
            drawn_labels.add(label)
        ax.scatter(
            c["mean_flops"],
            c["accuracy"] * 100,
            color=color,
            marker=marker,
            s=size,
            edgecolors="black" if cond in {"c2", "cmaj"} else "none",
            linewidths=1.0 if cond in {"c2", "cmaj"} else 0,
            label=label,
            zorder=3 if cond in {"c2", "cmaj"} else 2,
        )

    # Annotate winners + key callouts.
    for c in cells.values():
        if c["seed"] != 0:
            continue
        if c["condition"] == "cmaj" and c["branches"] == 5:
            ax.annotate(
                f"  cmaj b=5 → {c['accuracy']*100:.0f}%\n  +6 pp at 5× FLOPs",
                xy=(c["mean_flops"], c["accuracy"] * 100),
                xytext=(15, -5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#2ca02c",
            )
        elif c["condition"] == "c2" and c["k_steps"] == 64:
            ax.annotate(
                f"  C2 ceiling (74%, saturates at k=64)",
                xy=(c["mean_flops"], c["accuracy"] * 100),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=9,
                color="#1f77b4",
            )
        elif c["condition"] == "c4" and c["k_steps"] == 64:
            ax.annotate(
                f"  worst hybrid (54%)",
                xy=(c["mean_flops"], c["accuracy"] * 100),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=9,
                color="#8b0000",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Mean FLOPs per problem")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Sfumato E4 — accuracy vs FLOPs on GSM8K dev200 (k=64 unless k-swept)",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(20, 90)

    # Custom legend.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.95)

    out = FIGS / "accuracy_vs_flops.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_branch_agreement(cells: dict[tuple, dict]) -> Path | None:
    """Histogram of inter-branch disagreement for cmaj b=5."""
    cmaj_b5 = next(
        (
            c
            for c in cells.values()
            if c["condition"] == "cmaj" and c["branches"] == 5
        ),
        None,
    )
    if not cmaj_b5:
        return None
    rows = cmaj_b5["rows"]
    # For each problem, count unique non-empty branch answers.
    n_unique = []
    by_cat = {"all_agree_correct": 0, "all_agree_wrong": 0,
              "majority_correct": 0, "majority_wrong": 0,
              "tie_correct": 0, "tie_wrong": 0}
    for r in rows:
        votes = r.get("trace", {}).get("votes", "")
        parts = [p.strip() for p in votes.split("|") if p.strip()]
        unique = set(parts)
        n_unique.append(len(unique))
        gold = r["gold"].strip()
        correct = bool(r["correct"])
        if len(unique) == 1:
            by_cat["all_agree_correct" if correct else "all_agree_wrong"] += 1
        elif len(unique) <= 2:
            by_cat["majority_correct" if correct else "majority_wrong"] += 1
        else:
            by_cat["tie_correct" if correct else "tie_wrong"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: histogram of n_unique answers
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax1.hist(
        n_unique,
        bins=bins,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.85,
    )
    ax1.set_xlabel("# unique answers across 5 branches")
    ax1.set_ylabel("# problems")
    ax1.set_title(f"cmaj b=5: branch agreement on N={len(rows)} problems")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.grid(True, alpha=0.3)
    for x in [1, 2, 3, 4, 5]:
        n = sum(1 for u in n_unique if u == x)
        if n:
            ax1.text(x, n + 0.5, str(n), ha="center", fontsize=9)

    # Right: stacked bar — agreement category × correct/wrong
    cats = ["all_agree", "majority (2 distinct)", "spread (≥3 distinct)"]
    correct_counts = [
        by_cat["all_agree_correct"],
        by_cat["majority_correct"],
        by_cat["tie_correct"],
    ]
    wrong_counts = [
        by_cat["all_agree_wrong"],
        by_cat["majority_wrong"],
        by_cat["tie_wrong"],
    ]
    x = np.arange(len(cats))
    ax2.bar(x, correct_counts, color="#2ca02c", label="correct")
    ax2.bar(
        x,
        wrong_counts,
        bottom=correct_counts,
        color="#d62728",
        alpha=0.7,
        label="wrong",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats)
    ax2.set_ylabel("# problems")
    ax2.set_title("cmaj b=5: agreement-class × outcome")
    ax2.legend(loc="upper right")
    for i, (c, w) in enumerate(zip(correct_counts, wrong_counts)):
        if c + w:
            ax2.text(
                i,
                c + w + 0.5,
                f"{c}/{c+w}",
                ha="center",
                fontsize=9,
            )
    ax2.grid(True, axis="y", alpha=0.3)

    out = FIGS / "branch_agreement.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_correctness_heatmap(cells: dict[tuple, dict]) -> Path:
    """Heatmap: problem × condition matrix, green=correct, red=wrong."""
    # Limit to N=50 single-seed cells, k=64 (where applicable).
    selected = []
    for c in cells.values():
        if c["seed"] != 0 or c["n"] != 50:
            continue
        if c["condition"] in {"c1"} and c["k_steps"] != 64:
            continue
        if c["condition"] in {"c2", "c3", "c3p", "c4", "crev"} and c["k_steps"] != 64:
            continue
        if c["condition"] in {"c2hint", "c2empty"} and c["k_steps"] != 64:
            continue
        if c["condition"] in {"cmaj", "cmerge"}:
            pass
        if c["condition"] == "c3p" and "1.5B" in c["ar_model"]:
            continue
        selected.append(c)

    # Order columns by accuracy (descending).
    selected.sort(key=lambda c: -c["accuracy"])

    n_problems = max(c["n"] for c in selected)
    matrix = np.full((n_problems, len(selected)), np.nan)
    col_labels = []
    for j, c in enumerate(selected):
        rows_by_idx = {r["idx"]: r for r in c["rows"]}
        for i in range(n_problems):
            r = rows_by_idx.get(i)
            matrix[i, j] = float(r["correct"]) if r else np.nan
        suffix = ""
        if c["branches"] > 1:
            suffix = f"\nb={c['branches']}"
        elif c["k_steps"] > 0 and c["condition"] != "c1":
            suffix = f"\nk={c['k_steps']}"
        col_labels.append(f"{c['condition']}{suffix}\n{c['accuracy']*100:.0f}%")

    fig, ax = plt.subplots(figsize=(max(6, len(selected) * 0.7), 11))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)
    ax.set_xlabel("Condition (sorted by acc)")
    ax.set_ylabel("Problem index (GSM8K dev200, 0-49)")
    ax.set_title("Per-problem correctness across conditions\n(green = correct, red = wrong)")
    ax.set_yticks(np.arange(0, n_problems, 5))
    ax.set_yticklabels(np.arange(0, n_problems, 5))

    # Add accuracy annotation row at the bottom.
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="correct (1) / wrong (0)")
    out = FIGS / "correctness_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    cells = load_cells()
    print(f"loaded {len(cells)} cells (n>=10)")
    out1 = fig_accuracy_vs_flops(cells)
    print(f"  -> {out1}")
    out2 = fig_branch_agreement(cells)
    if out2:
        print(f"  -> {out2}")
    out3 = fig_correctness_heatmap(cells)
    print(f"  -> {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
