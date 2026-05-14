"""E3b — D2 multi-scale compute-matched control.

Phase-D's D2 was at 200M only: pure-diff trained 6k steps still loses to
composite-3k by 0.69 diff-NLL. To claim scale invariance for the paper,
we need the same control at 60M, 120M, 300M.

This script trains pure-diff at 6k steps (2× composite's 3k) at the three
target scales × 3 seeds each, then scores diff-NLL on a held-out chunk of
GSM8K answers using the same probe-1 protocol.

Existing composite-3k diff-NLL (from T0_PROBES_FINAL.md):
  60M  → 5.74
  120M → 5.68
  300M → 5.35

Decision rule per scale:
  pure-diff-6k diff-NLL ≥ composite-3k + 0.2  → composite wins (joint-training advantage)
  pure-diff-6k ≈ composite-3k (within 0.1)    → collapse to "more gradient signal" framing
  pure-diff-6k < composite-3k − 0.2            → joint training is not the explanation

Env:
  SCALES=60M,120M,300M     (default; map to {d, L, H} below)
  SEEDS=80,81,82           (default)
  MAX_STEPS=6000
  N_PROBES=100             (held-out GSM8K problems for diff-NLL)
  OUT_NAME=e3b_multiscale_d2
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_gsm8k_train_tokens  # noqa: E402
from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402


SCALE_CONFIGS = {
    "60M":  dict(d_model=512, n_layers=8,  n_heads=8),
    "120M": dict(d_model=640, n_layers=10, n_heads=10),
    "200M": dict(d_model=896, n_layers=12, n_heads=16),
    "250M": dict(d_model=960, n_layers=12, n_heads=16),
    "300M": dict(d_model=1024, n_layers=14, n_heads=16),
}


def env_int(k, d): return int(os.environ.get(k, str(d)))


def load_gsm8k_probe_problems(n=100, offset=7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(offset, min(offset + n, len(ds))):
        row = ds[idx]
        p = tok.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
        a = tok.encode(" " + row["answer"], add_special_tokens=False)
        out.append((p, a))
    return out


@torch.no_grad()
def score_diff_nll(model, problems, device, max_len=256, mask_ratio=0.5, seed=0):
    """Re-implement probe-1 (mask 50% of answer region, diff-mode NLL on masked positions)."""
    total_nll = 0.0
    total_count = 0
    rng = np.random.RandomState(seed)
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        full = prompt_ids + answer_ids
        ans_start = len(prompt_ids)
        ans_len = len(answer_ids)
        n_mask = max(1, int(ans_len * mask_ratio))
        positions = rng.choice(ans_len, size=n_mask, replace=False) + ans_start
        idx = torch.tensor([full], dtype=torch.long, device=device)
        masked = idx.clone()
        for p in positions:
            masked[0, p] = MASK_TOKEN_ID
        logits = model(masked, mode="diff")
        pred = logits[0, positions].float()
        targets = torch.tensor([full[p] for p in positions], dtype=torch.long, device=device)
        log_probs = torch.log_softmax(pred, dim=-1)
        nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    avg = total_nll / max(1, total_count)
    return {"avg_nll": round(avg, 4), "perplexity": round(math.exp(avg), 2),
            "n_tokens": total_count, "mask_ratio": mask_ratio}


def main():
    scales = os.environ.get("SCALES", "60M,120M,300M").split(",")
    seeds = [int(s) for s in os.environ.get("SEEDS", "80,81,82").split(",")]
    max_steps = env_int("MAX_STEPS", 6000)
    n_probe = env_int("N_PROBES", 100)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)

    out_name = os.environ.get("OUT_NAME", "e3b_multiscale_d2")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} scales={scales} seeds={seeds} max_steps={max_steps}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    problems = load_gsm8k_probe_problems(n=n_probe)
    print(f"loaded {len(tokens):,} train tokens, {len(problems)} probe problems")

    all_rows = []
    for scale in scales:
        scale = scale.strip()
        if scale not in SCALE_CONFIGS:
            print(f"WARN: unknown scale {scale!r}; skip")
            continue
        cfg_dict = SCALE_CONFIGS[scale]
        scale_out = base_out / scale
        scale_out.mkdir(parents=True, exist_ok=True)
        print(f"\n========== {scale}  cfg={cfg_dict} ==========")
        for seed in seeds:
            run_out = scale_out / f"diff_only_seed{seed}_6k"
            ckpt = run_out / "model.pt"
            if not ckpt.exists():
                print(f"\n[{scale} seed={seed}] training pure-diff 6k...")
                t0 = time.time()
                train_one(
                    variant="diff_only",
                    d_model=cfg_dict["d_model"],
                    n_layers=cfg_dict["n_layers"],
                    n_heads=cfg_dict["n_heads"],
                    out_dir=run_out,
                    seed=seed, max_steps=max_steps,
                    batch_size=batch_size, block_size=block_size,
                    peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=tokens,
                )
                print(f"  train wall_s={round(time.time()-t0, 1)}")
            else:
                print(f"[{scale} seed={seed}] ckpt exists, skip train")

            # Score
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            model_cfg = CompositeConfig(**ck["config"])
            model = CompositeLM(model_cfg).to(device)
            model.load_state_dict(ck["state_dict"])
            model.train(False)
            r = score_diff_nll(model, problems, device, max_len=block_size, mask_ratio=0.5, seed=seed)
            print(f"  diff-NLL = {r['avg_nll']} (n_tok={r['n_tokens']})")
            row = {"scale": scale, "seed": seed, "steps": max_steps, "variant": "diff_only_6k", **r}
            all_rows.append(row)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    # Final aggregate per scale
    print("\n========== FINAL AGGREGATE ==========")
    by_scale = {}
    for r in all_rows:
        by_scale.setdefault(r["scale"], []).append(r["avg_nll"])
    for scale, nlls in by_scale.items():
        m = sum(nlls) / len(nlls)
        s = (sum((x - m) ** 2 for x in nlls) / len(nlls)) ** 0.5
        print(f"  {scale}: pure-diff-6k diff-NLL mean={m:.3f} std={s:.3f} n={len(nlls)}  (per-seed={[round(x,3) for x in nlls]})")

    print(f"\nwrote {base_out/'summary.json'}")


if __name__ == "__main__":
    main()
