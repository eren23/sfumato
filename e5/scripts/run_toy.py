"""T0 — top-level driver that trains all 4 variants and evaluates them.

Designed to be the single Crucible `run_project` entry point.

Sequentially runs:
  1. composite   (60M backbone, joint AR+diff loss)
  2. ar_only     (60M backbone, AR loss only)        — B1
  3. diff_only   (60M backbone, DIFF loss only)      — B2
  4. paired      (two 30M models trained separately) — B3

Then evaluates each on GSM8K-dev N=50:
  - composite → ar mode + paired mode (same model used twice)
  - ar_only   → ar mode
  - diff_only → diff mode
  - paired    → paired_separate mode (two distinct ckpts)

Writes a unified `e5/results/toy_composite_summary.json` and `.md`.

Env knobs (defaults tuned for a single A40 GPU-day):
  MAX_STEPS=3000          # ~30 min per 60M variant, ~15 min per 30M variant
  N_EVAL=50               # ~5 min per eval mode
  SEED=0
  BATCH_SIZE=8
  BLOCK_SIZE=256
  LR=6e-4
  D_MODEL=512  N_LAYERS=8  N_HEADS=8
  D_MODEL_SMALL=384  N_LAYERS_SMALL=6
  SMOKE=0                 # if 1: MAX_STEPS=100, N_EVAL=5 (validates pipeline)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_gsm8k_train_tokens  # noqa: E402
from e5.evaluate import run_one as eval_one  # noqa: E402


def env_int(k: str, d: int) -> int:
    return int(os.environ.get(k, str(d)))


def env_float(k: str, d: float) -> float:
    return float(os.environ.get(k, str(d)))


def main() -> None:
    smoke = env_int("SMOKE", 0) == 1
    max_steps = env_int("MAX_STEPS", 100 if smoke else 3000)
    n_eval = env_int("N_EVAL", 5 if smoke else 50)
    seed = env_int("SEED", 0)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    peak_lr = env_float("LR", 6e-4)
    d_model = env_int("D_MODEL", 512)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    d_model_small = env_int("D_MODEL_SMALL", 384)
    n_layers_small = env_int("N_LAYERS_SMALL", 6)

    base_out = REPO_ROOT / "e5" / "results" / f"toy_composite_seed{seed}{'_smoke' if smoke else ''}"
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"=== T0 driver: smoke={smoke} max_steps={max_steps} n_eval={n_eval} seed={seed} ===")
    print(f"=== output dir: {base_out} ===")

    print(f"\nloading GSM8K-train tokens (gpt2, cot)…")
    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"  {len(tokens):,} tokens loaded")

    train_results: dict[str, dict] = {}
    ckpts: dict[str, Path] = {}

    # 1. composite
    print(f"\n=== [1/4] training composite ===")
    composite_dir = base_out / "composite"
    composite_summary = train_one(
        variant="composite",
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        out_dir=composite_dir,
        seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
        peak_lr=peak_lr, eval_every=10**9, n_eval=n_eval, tokens=tokens,
    )
    train_results["composite"] = composite_summary
    ckpts["composite"] = composite_dir / "model.pt"

    # 2. ar_only (B1)
    print(f"\n=== [2/4] training ar_only (B1) ===")
    ar_dir = base_out / "ar_only"
    ar_summary = train_one(
        variant="ar_only",
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        out_dir=ar_dir,
        seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
        peak_lr=peak_lr, eval_every=10**9, n_eval=n_eval, tokens=tokens,
    )
    train_results["ar_only"] = ar_summary
    ckpts["ar_only"] = ar_dir / "model.pt"

    # 3. diff_only (B2)
    print(f"\n=== [3/4] training diff_only (B2) ===")
    diff_dir = base_out / "diff_only"
    diff_summary = train_one(
        variant="diff_only",
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        out_dir=diff_dir,
        seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
        peak_lr=peak_lr, eval_every=10**9, n_eval=n_eval, tokens=tokens,
    )
    train_results["diff_only"] = diff_summary
    ckpts["diff_only"] = diff_dir / "model.pt"

    # 4. paired (B3): two 30M sub-models
    print(f"\n=== [4/4] training paired (B3): two {d_model_small}d × {n_layers_small}L sub-models ===")
    paired_dir = base_out / "paired"
    paired_dir.mkdir(exist_ok=True)
    paired_ar_dir = paired_dir / "ar_only"
    paired_diff_dir = paired_dir / "diff_only"
    paired_ar_summary = train_one(
        variant="ar_only",
        d_model=d_model_small, n_layers=n_layers_small, n_heads=max(1, n_heads // 2),
        out_dir=paired_ar_dir,
        seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
        peak_lr=peak_lr, eval_every=10**9, n_eval=n_eval, tokens=tokens,
    )
    paired_diff_summary = train_one(
        variant="diff_only",
        d_model=d_model_small, n_layers=n_layers_small, n_heads=max(1, n_heads // 2),
        out_dir=paired_diff_dir,
        seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
        peak_lr=peak_lr, eval_every=10**9, n_eval=n_eval, tokens=tokens,
    )
    train_results["paired"] = {
        "ar_sub": paired_ar_summary,
        "diff_sub": paired_diff_summary,
        "n_params_total": paired_ar_summary["n_params"] + paired_diff_summary["n_params"],
    }
    ckpts["paired_ar"] = paired_ar_dir / "model.pt"
    ckpts["paired_diff"] = paired_diff_dir / "model.pt"

    print(f"\n=== EVALUATION ===")
    eval_results: dict[str, dict] = {}

    # composite: ar mode + paired mode
    print(f"\n[eval] composite (mode=ar)")
    r = eval_one(ckpts["composite"], "ar", n_eval=n_eval)
    eval_results["composite_ar"] = {k: v for k, v in r.items() if k != "results"}
    (base_out / "composite_ar_results.json").write_text(json.dumps(r, indent=2))

    print(f"\n[eval] composite (mode=paired, self-paired)")
    r = eval_one(ckpts["composite"], "paired", n_eval=n_eval)
    eval_results["composite_paired"] = {k: v for k, v in r.items() if k != "results"}
    (base_out / "composite_paired_results.json").write_text(json.dumps(r, indent=2))

    # B1 ar_only: ar mode
    print(f"\n[eval] ar_only / B1 (mode=ar)")
    r = eval_one(ckpts["ar_only"], "ar", n_eval=n_eval)
    eval_results["ar_only"] = {k: v for k, v in r.items() if k != "results"}
    (base_out / "ar_only_results.json").write_text(json.dumps(r, indent=2))

    # B2 diff_only: diff mode
    print(f"\n[eval] diff_only / B2 (mode=diff)")
    r = eval_one(ckpts["diff_only"], "diff", n_eval=n_eval)
    eval_results["diff_only"] = {k: v for k, v in r.items() if k != "results"}
    (base_out / "diff_only_results.json").write_text(json.dumps(r, indent=2))

    # B3 paired_separate
    print(f"\n[eval] paired_separate / B3 (mode=paired_separate)")
    r = eval_one(None, "paired_separate", n_eval=n_eval, ckpt_ar=ckpts["paired_ar"], ckpt_diff=ckpts["paired_diff"])
    eval_results["paired_separate"] = {k: v for k, v in r.items() if k != "results"}
    (base_out / "paired_separate_results.json").write_text(json.dumps(r, indent=2))

    # Verdict per the plan's decision table
    composite_acc = eval_results["composite_paired"]["accuracy"]
    b3_acc = eval_results["paired_separate"]["accuracy"]
    delta_pp = (composite_acc - b3_acc) * 100

    if delta_pp >= 5:
        verdict = "strong_positive"
    elif delta_pp >= 0:
        verdict = "within_noise_positive"
    elif delta_pp >= -3:
        verdict = "inconclusive"
    else:
        verdict = "empirical_negative"

    summary = {
        "smoke": bool(smoke),
        "max_steps": max_steps,
        "n_eval": n_eval,
        "seed": seed,
        "block_size": block_size,
        "batch_size": batch_size,
        "peak_lr": peak_lr,
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "d_model_small": d_model_small, "n_layers_small": n_layers_small,
        "train": train_results,
        "eval": eval_results,
        "headline": {
            "composite_paired_acc": composite_acc,
            "b3_paired_separate_acc": b3_acc,
            "delta_pp": round(delta_pp, 2),
            "verdict": verdict,
        },
    }
    (base_out / "toy_composite_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown summary
    band_text = {
        "strong_positive": "≥+5pp lift. Justify scale-up to BD3-LM-variant ($500-1500).",
        "within_noise_positive": "Positive within noise. Multi-seed (+$200) before scaling.",
        "inconclusive": "Within ±3pp. Reread Transfusion/BD3 ablations for scale.",
        "empirical_negative": "Composite < B3 − 3pp. Toy-scale negative for the vision; redesign or scale-up.",
    }
    md = f"""# T0 — toy composite AR+diffusion training, seed={seed}{' (SMOKE)' if smoke else ''}

## Headline

|  | accuracy | n_params |
|--|----------|----------|
| **Composite (paired mode)** | **{composite_acc*100:.1f}%** | {train_results['composite']['n_params']/1e6:.1f}M |
| **B3 paired_separate** | **{b3_acc*100:.1f}%** | {train_results['paired']['n_params_total']/1e6:.1f}M (paired) |
| **Δ (composite − B3)** | **{delta_pp:+.2f}pp** | |

**Verdict: `{verdict}`** — {band_text[verdict]}

## All-mode accuracy

| variant | inference mode | acc |
|---------|---------------|-----|
| composite | ar | {eval_results['composite_ar']['accuracy']*100:.1f}% |
| composite | paired (self) | {eval_results['composite_paired']['accuracy']*100:.1f}% |
| B1 ar_only | ar | {eval_results['ar_only']['accuracy']*100:.1f}% |
| B2 diff_only | diff | {eval_results['diff_only']['accuracy']*100:.1f}% |
| B3 paired_separate | paired (cross-model) | {eval_results['paired_separate']['accuracy']*100:.1f}% |

## Training

| variant | wall_s | final AR loss | final DIFF loss |
|---------|--------|---------------|------------------|
| composite | {train_results['composite']['wall_s']:.0f}s | {train_results['composite']['final_ar_loss']:.3f} | {train_results['composite']['final_diff_loss']:.3f} |
| ar_only (B1) | {train_results['ar_only']['wall_s']:.0f}s | {train_results['ar_only']['final_ar_loss']:.3f} | n/a |
| diff_only (B2) | {train_results['diff_only']['wall_s']:.0f}s | n/a | {train_results['diff_only']['final_diff_loss']:.3f} |
| B3 ar_sub | {train_results['paired']['ar_sub']['wall_s']:.0f}s | {train_results['paired']['ar_sub']['final_ar_loss']:.3f} | n/a |
| B3 diff_sub | {train_results['paired']['diff_sub']['wall_s']:.0f}s | n/a | {train_results['paired']['diff_sub']['final_diff_loss']:.3f} |

## Caveats

- Scale: 60M model on ~4M GSM8K tokens (~3 epochs). Published composite-training wins appear at 400M+; toy-scale absence of effect is *not* a clean falsification of the vision.
- Single seed: no error bars. Multi-seed before any strong claim.
- B3 inference uses AR-first-then-diff-fill at fixed K=64/64 split. Composite's "paired" mode uses the same model for both halves — that's the central comparison.
"""
    (base_out / "toy_composite_summary.md").write_text(md)

    print(f"\n=== DONE ===")
    print(f"Headline: composite[paired] = {composite_acc*100:.1f}% vs B3[paired_separate] = {b3_acc*100:.1f}% ({delta_pp:+.2f}pp, {verdict})")
    print(f"wrote {base_out/'toy_composite_summary.json'}")
    print(f"wrote {base_out/'toy_composite_summary.md'}")


if __name__ == "__main__":
    main()
