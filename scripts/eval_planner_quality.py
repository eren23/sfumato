"""E2 — prefix×planner quality matrix, before vs after Track 1.

Sweeps every (prefix_tier, model_state) combination on a fixed slice of
GSM8K-test and reports per-cell accuracy + mean FLOPs.

Prefix tiers map to runner conditions:
    none           → CONDITION=c2          (no prefix)
    hint           → CONDITION=c2hint      (generic CoT hint)
    empty          → CONDITION=c2empty     ("Plan: " w/ no content)
    weak_qwen05    → CONDITION=c3p, AR_MODEL=Qwen/Qwen2.5-0.5B-Instruct
    medium_qwen15  → CONDITION=c3p, AR_MODEL=Qwen/Qwen2.5-1.5B-Instruct
    strong_qwen7b  → CONDITION=c3p, AR_MODEL=Qwen/Qwen2.5-7B-Instruct
    oracle         → SKIPPED by default (see CAVEATS).

Model states:
    base           → unset LORA_PATH
    track1_lora    → LORA_PATH=<--track1_path>

CAVEATS / oracle limitation:
    The "oracle" tier requires injecting a hand-crafted gold-quality plan as
    the prefix, but the runner has no PLAN_OVERRIDE hook today and writing
    one would be invasive. We deliberately skip oracle here; it can be
    bolted on later by:
      (a) adding PLAN_OVERRIDE / PROMPT_OVERRIDE to e4/runner.py, or
      (b) pre-building the full augmented prompts and running a c2 variant
          with PROMPT_OVERRIDE.
    This script implements 6 of the 7 tiers.

Outputs:
    e2/results/planner_quality.csv         (prefix_tier, model_state, accuracy, mean_flops, n)
    e2/results/planner_quality_heatmap.png (matplotlib heatmap)

Usage:
    python scripts/eval_planner_quality.py --n_problems 200
    python scripts/eval_planner_quality.py --lora_disabled  # base only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── prefix tier table ─────────────────────────────────────────────────────
# Each tier maps to (CONDITION, optional AR_MODEL override). None for AR_MODEL
# means "use the runner's default" (which is Qwen2.5-0.5B-Instruct, fine for
# c2/c2hint/c2empty since they don't actually call the AR planner).
@dataclass(frozen=True)
class Tier:
    name: str
    condition: str
    ar_model: str | None = None
    skip: bool = False
    skip_reason: str = ""


TIERS: list[Tier] = [
    Tier("none", "c2"),
    Tier("hint", "c2hint"),
    Tier("empty", "c2empty"),
    Tier("weak_qwen05", "c3p", "Qwen/Qwen2.5-0.5B-Instruct"),
    Tier("medium_qwen15", "c3p", "Qwen/Qwen2.5-1.5B-Instruct"),
    Tier("strong_qwen7b", "c3p", "Qwen/Qwen2.5-7B-Instruct"),
    Tier(
        "oracle",
        "c3p",
        skip=True,
        skip_reason=(
            "oracle prefix injection requires a PROMPT/PLAN_OVERRIDE hook in "
            "e4/runner.py that does not exist yet; see header CAVEATS."
        ),
    ),
]


def run_cell(
    tier: Tier,
    model_state: str,
    n_problems: int,
    seed: int,
    track1_path: str,
    k_steps: int,
    extra_env: dict[str, str] | None = None,
) -> tuple[float, float, int]:
    """Run one runner.py invocation; return (accuracy, mean_flops, n)."""
    env = os.environ.copy()
    env["CONDITION"] = tier.condition
    env["N_PROBLEMS"] = str(n_problems)
    env["SEED"] = str(seed)
    env["K_STEPS"] = str(k_steps)
    if tier.ar_model:
        env["AR_MODEL"] = tier.ar_model
    if model_state == "track1_lora":
        env["LORA_PATH"] = track1_path
    else:
        env.pop("LORA_PATH", None)
    # Always clear the commit adapter — this script measures Track 1 alone.
    env.pop("COMMIT_LORA_PATH", None)
    if extra_env:
        env.update(extra_env)

    print(
        f"[planner_quality] tier={tier.name} state={model_state} "
        f"cond={tier.condition} ar={tier.ar_model or '(default)'}",
        flush=True,
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "e4" / "runner.py")],
        env=env,
        cwd=REPO_ROOT,
        check=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner.py failed for tier={tier.name} state={model_state}"
        )

    # Read the jsonl the runner just wrote and compute accuracy/mean_flops.
    out_path = (
        REPO_ROOT
        / "e4"
        / "results"
        / f"raw_{tier.condition}_k{k_steps}_seed{seed}.jsonl"
    )
    if not out_path.exists():
        raise FileNotFoundError(out_path)

    rows: list[dict] = []
    with out_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return 0.0, 0.0, 0
    n = len(rows)
    acc = sum(int(bool(r.get("correct"))) for r in rows) / n
    mean_flops = sum(float(r.get("flops", 0.0)) for r in rows) / n
    return acc, mean_flops, n


def write_csv(out_csv: Path, results: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prefix_tier",
                "model_state",
                "condition",
                "ar_model",
                "accuracy",
                "mean_flops",
                "n",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def write_heatmap(out_png: Path, results: list[dict]) -> None:
    """Render an accuracy heatmap, prefix_tier × model_state."""
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        print(f"[planner_quality] matplotlib unavailable, skipping heatmap: {exc}")
        return

    tiers = [r["prefix_tier"] for r in results if r["model_state"] == "base"]
    if not tiers:
        # fall back to whatever model_state we have
        tiers = sorted({r["prefix_tier"] for r in results})
    states_present = sorted({r["model_state"] for r in results})

    grid = np.full((len(tiers), len(states_present)), np.nan)
    for i, t in enumerate(tiers):
        for j, s in enumerate(states_present):
            for r in results:
                if r["prefix_tier"] == t and r["model_state"] == s:
                    grid[i, j] = r["accuracy"]
                    break

    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(states_present), 0.55 * len(tiers) + 1.5))
    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(states_present)))
    ax.set_xticklabels(states_present)
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers)
    ax.set_xlabel("model_state")
    ax.set_ylabel("prefix_tier")
    ax.set_title("E2 prefix × model_state accuracy")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="white", fontsize=9)
            else:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < 0.55 else "black",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[planner_quality] heatmap -> {out_png.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_problems", type=int, default=200)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=REPO_ROOT / "e2" / "results" / "planner_quality.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--track1_path",
        type=str,
        default="eren23/sfumato-llada-prefix-robust",
    )
    parser.add_argument(
        "--lora_disabled",
        action="store_true",
        help="Skip the track1_lora rows (only run the 'base' column).",
    )
    parser.add_argument(
        "--k_steps",
        type=int,
        default=64,
        help="Diffusion steps. Defaults to 64 (matched to E4 sweeps).",
    )
    args = parser.parse_args()

    states = ["base"] if args.lora_disabled else ["base", "track1_lora"]

    results: list[dict] = []
    for tier in TIERS:
        if tier.skip:
            print(
                f"[planner_quality] SKIP tier={tier.name} ({tier.skip_reason})",
                flush=True,
            )
            continue
        for state in states:
            try:
                acc, mean_flops, n = run_cell(
                    tier=tier,
                    model_state=state,
                    n_problems=args.n_problems,
                    seed=args.seed,
                    track1_path=args.track1_path,
                    k_steps=args.k_steps,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[planner_quality] FAIL tier={tier.name} state={state}: {exc}",
                    flush=True,
                )
                acc, mean_flops, n = float("nan"), float("nan"), 0
            row = {
                "prefix_tier": tier.name,
                "model_state": state,
                "condition": tier.condition,
                "ar_model": tier.ar_model or "",
                "accuracy": acc,
                "mean_flops": mean_flops,
                "n": n,
            }
            results.append(row)
            print(
                f"[planner_quality]   acc={acc:.4f} mean_flops={mean_flops:.3e} n={n}",
                flush=True,
            )

    write_csv(args.out_csv, results)
    print(f"[planner_quality] csv -> {args.out_csv.relative_to(REPO_ROOT)}")

    heatmap_path = args.out_csv.with_name("planner_quality_heatmap.png")
    write_heatmap(heatmap_path, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
