"""F3 — AR-only compute-matched control (mirror of E3b).

E3b showed pure-diff-6k still loses to composite-3k on the diffusion
axis. The symmetric question: does pure-AR-6k beat composite-3k on the
AR axis? If yes, the AR tax we observe is fully explained by training
compute; if no, the AR tax has the same "structural" character as the
diff win.

Train AR-only at 6k steps (2x composite's 3k) at three scales:
60M / 120M / 200M, 3 seeds each. Score AR-NLL on held-out chunk.

Compare against existing composite-3k AR-NLL (from T0_PROBES_FINAL.md).

Env:
  SCALES=60M,120M,200M
  SEEDS=130,131,132
  MAX_STEPS=6000
  OUT_NAME=f3_ar_compute_control
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
from e5.model_composite import CompositeConfig, CompositeLM  # noqa: E402


SCALE_CONFIGS = {
    "60M":  dict(d_model=512, n_layers=8,  n_heads=8),
    "120M": dict(d_model=640, n_layers=10, n_heads=10),
    "200M": dict(d_model=896, n_layers=12, n_heads=16),
    "300M": dict(d_model=1024, n_layers=14, n_heads=16),
}

# Existing composite-3k AR-NLL from T0_PROBES_FINAL.md (Phase D / overnight).
COMPOSITE_3K_AR_NLL = {"60M": None, "120M": None, "200M": 2.80, "300M": None}


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


def main():
    scales = os.environ.get("SCALES", "60M,120M,200M").split(",")
    seeds = [int(x) for x in os.environ.get("SEEDS", "130,131,132").split(",")]
    max_steps = env_int("MAX_STEPS", 6000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)

    out_name = os.environ.get("OUT_NAME", "f3_ar_compute_control")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} scales={scales} seeds={seeds} max_steps={max_steps}")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    problems = load_gsm8k_probe_problems()
    print(f"loaded {len(tokens):,} train tokens, {len(problems)} probe problems")

    all_rows = []
    for scale in scales:
        scale = scale.strip()
        if scale not in SCALE_CONFIGS:
            print(f"skip unknown scale {scale}")
            continue
        cfg = SCALE_CONFIGS[scale]
        scale_out = base_out / scale
        scale_out.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_out = scale_out / f"ar_only_seed{seed}_6k"
            ckpt = run_out / "model.pt"
            if not ckpt.exists():
                print(f"\n[{scale} seed={seed}] training AR-only 6k...")
                t0 = time.time()
                train_one(
                    variant="ar_only",
                    d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
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
            ar_nll = score_ar_nll(model, problems, device, max_len=block_size)
            row = {"scale": scale, "seed": seed, "steps": max_steps, "ar_nll_6k": ar_nll}
            all_rows.append(row)
            print(f"  AR-NLL = {ar_nll}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            (base_out / "summary.json").write_text(json.dumps(all_rows, indent=2))

    print("\n========== FINAL AGGREGATE ==========")
    by = {}
    for r in all_rows:
        by.setdefault(r["scale"], []).append(r["ar_nll_6k"])
    for scale, nlls in by.items():
        m = sum(nlls) / len(nlls)
        s = math.sqrt(sum((x - m) ** 2 for x in nlls) / len(nlls))
        comp = COMPOSITE_3K_AR_NLL.get(scale)
        lead = (comp - m) if comp is not None else None
        lead_str = f" composite-3k advantage = {lead:+.3f} NLL" if lead is not None else ""
        print(f"  {scale}: AR-only-6k AR-NLL mean={m:.3f} std={s:.3f} n={len(nlls)}{lead_str}")


if __name__ == "__main__":
    main()
