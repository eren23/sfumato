"""E3a — probe-5 multi-seed expansion to n=8.

The Phase D probe-5 result was n=3 with composite switch_64_32 in {0, 8, 6} —
a single lucky seed carried much of the mean. To upgrade from workshop-grade
to TMLR-grade we need a tighter CI. This script trains fresh seeds and runs
probe-5 on each.

Strategy: one pod runs the full pipeline end-to-end (train both variants per
seed, then probe-5 the composite checkpoint). No checkpoint transfer needed.

For each seed:
  1. Train 200M composite for MAX_STEPS (default 3000) on GSM8K-train
  2. Train 200M ar_only for MAX_STEPS on GSM8K-train
  3. Run probe-5 (ar_only, mode_switch_96_32, mode_switch_64_32, paired_64_64)
     on the composite ckpt.
  4. Run probe-5 modes on the ar_only ckpt for the matched baseline.
  5. Save per-seed JSON; aggregate at end.

Env:
  SEEDS=70,71,72,73,74    (default: 5 fresh seeds; combined with prior 3 = n=8)
  N_EVAL=50
  MAX_STEPS=3000
  OUT=e5/results/e3a_probe5_n8

Output: e5/results/e3a_probe5_n8/summary.json with per-seed × per-mode
accuracies for both composite and ar_only variants.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_gsm8k_train_tokens, load_gsm8k_dev_questions  # noqa: E402
from e5.scripts.probe5_mode_switch import (  # noqa: E402
    gen_ar, gen_mode_switch, gen_paired, eval_mode,
)
from e5.model_composite import CompositeConfig, CompositeLM  # noqa: E402


def env_int(k, d): return int(os.environ.get(k, str(d)))


def load_ckpt(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    m = CompositeLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m


def probe5_one_ckpt(ckpt_path: Path, problems, tok, device: str) -> dict:
    model = load_ckpt(ckpt_path, device)
    out = {}
    t0 = time.time()

    r = eval_mode(model, problems, lambda m, pr: gen_ar(m, pr, max_new=128), tok)
    out["ar_only"] = {k: v for k, v in r.items() if k != "results"}

    r = eval_mode(model, problems, lambda m, pr: gen_mode_switch(m, pr, k_ar=96, revise_len=32), tok)
    out["mode_switch_96_32"] = {k: v for k, v in r.items() if k != "results"}

    r = eval_mode(model, problems, lambda m, pr: gen_mode_switch(m, pr, k_ar=64, revise_len=32), tok)
    out["mode_switch_64_32"] = {k: v for k, v in r.items() if k != "results"}

    r = eval_mode(model, problems, lambda m, pr: gen_paired(m, pr, k_ar=64, k_diff=64), tok)
    out["paired_64_64"] = {k: v for k, v in r.items() if k != "results"}

    out["wall_s_total"] = round(time.time() - t0, 1)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    seeds_str = os.environ.get("SEEDS", "70,71,72,73,74")
    seeds = [int(s) for s in seeds_str.split(",") if s.strip()]
    max_steps = env_int("MAX_STEPS", 3000)
    n_eval = env_int("N_EVAL", 50)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    # 200M defaults
    d_model = env_int("D_MODEL", 896)
    n_layers = env_int("N_LAYERS", 12)
    n_heads = env_int("N_HEADS", 16)

    base_out = REPO_ROOT / "e5" / "results" / os.environ.get("OUT_NAME", "e3a_probe5_n8")
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={seeds} max_steps={max_steps}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"loaded {len(tokens):,} train tokens")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_eval)
    print(f"loaded {len(problems)} dev problems")

    all_rows = []
    for seed in seeds:
        print(f"\n========== seed={seed} ==========")
        seed_out = base_out / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)

        # Train both variants
        for variant in ("composite", "ar_only"):
            out_dir = seed_out / variant
            if (out_dir / "model.pt").exists():
                print(f"[seed={seed} {variant}] ckpt exists, skipping train")
                continue
            print(f"\n[seed={seed} {variant}] training...")
            train_one(
                variant=variant,
                d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                out_dir=out_dir,
                seed=seed, max_steps=max_steps,
                batch_size=batch_size, block_size=block_size,
                peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=tokens,
            )

        # Probe-5 both ckpts
        row = {"seed": seed, "composite": {}, "ar_only": {}}
        for variant in ("composite", "ar_only"):
            ckpt = seed_out / variant / "model.pt"
            if not ckpt.exists():
                print(f"WARN: missing {ckpt}, skip probe5")
                continue
            print(f"\n[seed={seed} {variant}] probe-5...")
            res = probe5_one_ckpt(ckpt, problems, tok, device)
            row[variant] = res
            for mode, m in res.items():
                if isinstance(m, dict) and "accuracy" in m:
                    print(f"  {mode}: acc={m['accuracy']*100:.1f}% ({m['n_correct']}/{m['n']})")

        (seed_out / "probe5.json").write_text(json.dumps(row, indent=2))
        all_rows.append(row)

        # Save aggregate after each seed
        (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    # Final aggregate
    print("\n========== FINAL SUMMARY ==========")
    for variant in ("composite", "ar_only"):
        modes = ("ar_only", "mode_switch_96_32", "mode_switch_64_32", "paired_64_64")
        print(f"\n{variant}:")
        for mode in modes:
            accs = [r[variant][mode]["accuracy"] for r in all_rows
                    if variant in r and mode in r[variant] and "accuracy" in r[variant][mode]]
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            sem = std / (len(accs) ** 0.5)
            print(f"  {mode}: mean={mean*100:.2f}% sem={sem*100:.2f}% n={len(accs)} per-seed={[f'{a*100:.0f}' for a in accs]}")

    (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {base_out/'summary.json'}")


if __name__ == "__main__":
    main()
