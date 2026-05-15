"""F6 — B3 paired-separate baseline at 500M total params.

User-requested test: the truly param-matched comparison for composite at 500M.

Composite-3k at 500M uses one 538M backbone for both AR and diff modes.
B3 paired-separate uses TWO models, each with half the param budget:
  ar_only-250M    × 3 seeds  (d=1024, L=16, H=16, ~265M params)
  diff_only-250M  × 3 seeds  (d=1024, L=16, H=16, ~265M params)

Total B3 budget = 2 × 265M = 530M (matched to composite's 538M).

Compare:
  - composite-3k AR-NLL (from F4) vs ar_only-250M AR-NLL
  - composite-3k diff-NLL (from F4) vs diff_only-250M diff-NLL

If composite beats B3 on diff axis: composite is strictly better when
you want both capabilities. If composite loses to B3 on AR axis: as
expected — but composite still owns diff with the single-backbone
advantage.

Env:
  SEEDS=180,181,182
  D_MODEL=1024 N_LAYERS=16 N_HEADS=16  (per sub-model)
  MAX_STEPS=3000
  OUT_NAME=f6_b3_paired_500m
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
def score_ar_nll(model, problems, device, max_len=256):
    total = 0.0; cnt = 0
    for prompt_ids, ans_ids in problems:
        if len(prompt_ids) + len(ans_ids) + 1 > max_len: continue
        full = prompt_ids + ans_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt_ids)
        pred = logits[0, ans_start - 1 : ans_start - 1 + len(ans_ids), :].float()
        target = torch.tensor(ans_ids, dtype=torch.long, device=device)
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total += float(nlls.sum()); cnt += int(nlls.size)
    return round(total / max(1, cnt), 4)


@torch.no_grad()
def score_diff_nll(model, problems, device, max_len=256, mask_ratio=0.5, seed=0):
    total = 0.0; cnt = 0
    rng = np.random.RandomState(seed)
    for prompt_ids, ans_ids in problems:
        if len(prompt_ids) + len(ans_ids) + 1 > max_len: continue
        full = prompt_ids + ans_ids
        ans_start = len(prompt_ids)
        ans_len = len(ans_ids)
        n_mask = max(1, int(ans_len * mask_ratio))
        positions = rng.choice(ans_len, size=n_mask, replace=False) + ans_start
        idx = torch.tensor([full], dtype=torch.long, device=device)
        masked = idx.clone()
        for p in positions:
            masked[0, p] = MASK_TOKEN_ID
        logits = model(masked, mode="diff")
        pred = logits[0, positions].float()
        target = torch.tensor([full[p] for p in positions], dtype=torch.long, device=device)
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total += float(nlls.sum()); cnt += int(nlls.size)
    return round(total / max(1, cnt), 4)


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "180,181,182").split(",")]
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    d_model = env_int("D_MODEL", 1024)
    n_layers = env_int("N_LAYERS", 16)
    n_heads = env_int("N_HEADS", 16)

    out_name = os.environ.get("OUT_NAME", "f6_b3_paired_500m")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seeds={seeds} sub-arch d={d_model} L={n_layers} H={n_heads}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    probes = load_gsm8k_probe_problems()
    print(f"loaded {len(tokens):,} train tokens, {len(probes)} probe problems")

    all_rows = []
    for variant in ("ar_only", "diff_only"):
        for seed in seeds:
            run_out = base_out / variant / f"seed{seed}"
            ckpt = run_out / "model.pt"
            if not ckpt.exists():
                print(f"\n[B3 sub-model {variant} seed={seed}] training...")
                t0 = time.time()
                train_one(
                    variant=variant,
                    d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    out_dir=run_out,
                    seed=seed, max_steps=max_steps,
                    batch_size=batch_size, block_size=block_size,
                    peak_lr=6e-4, eval_every=10**9, n_eval=0, tokens=tokens,
                )
                print(f"  train wall_s={round(time.time()-t0, 1)}")

            ck = torch.load(ckpt, map_location=device, weights_only=False)
            mcfg = CompositeConfig(**ck["config"])
            model = CompositeLM(mcfg).to(device)
            model.load_state_dict(ck["state_dict"])
            model.train(False)
            n_params = sum(p.numel() for p in model.parameters())
            ar_nll = score_ar_nll(model, probes, device, max_len=block_size) if variant == "ar_only" else None
            diff_nll = score_diff_nll(model, probes, device, max_len=block_size, seed=seed) if variant == "diff_only" else None
            row = {"variant": variant, "seed": seed, "steps": max_steps,
                   "n_params": n_params, "ar_nll": ar_nll, "diff_nll": diff_nll}
            all_rows.append(row)
            print(f"  n_params={n_params/1e6:.1f}M  AR-NLL={ar_nll}  diff-NLL={diff_nll}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    print("\n========== B3 PAIRED-SEPARATE AT 500M PARAM-MATCHED ==========")
    by_v = {}
    for r in all_rows: by_v.setdefault(r["variant"], []).append(r)
    if "ar_only" in by_v:
        rs = by_v["ar_only"]
        nlls = [r["ar_nll"] for r in rs]
        ar_m = sum(nlls)/len(nlls)
        per_params = rs[0]["n_params"]/1e6
        print(f"  B3 ar_only ({per_params:.0f}M each): AR-NLL mean={ar_m:.3f}  n={len(rs)}")
    if "diff_only" in by_v:
        rs = by_v["diff_only"]
        nlls = [r["diff_nll"] for r in rs]
        diff_m = sum(nlls)/len(nlls)
        per_params = rs[0]["n_params"]/1e6
        print(f"  B3 diff_only ({per_params:.0f}M each): diff-NLL mean={diff_m:.3f}  n={len(rs)}")
    print("\n  Compare to F4 composite-3k at 500M (~538M backbone, shared).")


if __name__ == "__main__":
    main()
