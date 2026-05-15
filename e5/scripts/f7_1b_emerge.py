"""F7 — does math emerge at 1B params on ~2B FineWeb-Edu tokens?

Runs end-to-end on a single A100 80GB pod:
  1. Stream + tokenize FineWeb-Edu (sample-10BT) to ~2B tokens cached to disk
  2. Train composite-1B for ~30k steps (BS=32 × T=1024 ≈ 1B tokens seen)
  3. (Optional) Train ar_only-1B same recipe for baseline
  4. Score AR-NLL on held-out GSM8K + generate sample completions

Arch: d=1536, L=24, H=16  →  ~1.0B params
Recipe: BS=8 × T=1024, peak_lr=3e-4 (lower than toy default), cosine to 10%

ENV:
  VARIANTS=composite,ar_only   (default: train both)
  MAX_STEPS=30000
  N_TARGET_TOKENS=2000000000
  D_MODEL=1536  N_LAYERS=24  N_HEADS=16
  BATCH_SIZE=8  BLOCK_SIZE=1024
  PEAK_LR=3e-4
  SEED=200

Output:
  e5/results/f7_1b_emerge/<variant>/model.pt + summary.json + samples.md
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.train import train_one  # noqa: E402
from e5.data import load_fineweb_tokens, load_gsm8k_dev_questions  # noqa: E402
from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.scripts.probe5_mode_switch import gen_ar, gen_mode_switch, gen_paired  # noqa: E402


def env_int(k, d): return int(os.environ.get(k, str(d)))
def env_float(k, d): return float(os.environ.get(k, str(d)))


ANSWER_PAT = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")
NUM_PAT = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text):
    m = ANSWER_PAT.search(text)
    if m: return m.group(1).replace(",", "").replace("$", "")
    nums = NUM_PAT.findall(text)
    return nums[-1].replace(",", "").replace("$", "") if nums else None


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
def score_ar_nll(model, problems, device, max_len=1024):
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


def generate_samples(model, dev_problems, tok, device, mode_fns, n=5):
    lines = []
    for p in dev_problems[:n]:
        q = p["question"]
        gold = p["gold"]
        prompt = p["prompt_tokens"]
        lines.append(f"\n### Q (idx={p['idx']})\n\n> {q.strip()}\n\n**Gold:** `{gold}`\n")
        for mode_name, fn in mode_fns.items():
            try:
                ids = fn(model, prompt)
            except Exception as e:
                lines.append(f"\n**{mode_name}** — ERROR: {e}\n")
                continue
            text = tok.decode([t for t in ids if t < 50257], skip_special_tokens=True)
            pred = extract_answer(text)
            correct = "✅" if pred == gold else "❌"
            lines.append(f"\n**{mode_name}** {correct} pred=`{pred}`\n\n```\n{text[:800].strip()}\n```\n")
    return "\n".join(lines)


def main():
    variants = os.environ.get("VARIANTS", "composite,ar_only").split(",")
    max_steps = env_int("MAX_STEPS", 30000)
    n_target_tokens = env_int("N_TARGET_TOKENS", 2_000_000_000)
    d_model = env_int("D_MODEL", 1536)
    n_layers = env_int("N_LAYERS", 24)
    n_heads = env_int("N_HEADS", 16)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 1024)
    peak_lr = env_float("PEAK_LR", 3e-4)
    seed = env_int("SEED", 200)

    out_name = os.environ.get("OUT_NAME", "f7_1b_emerge")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== F7 1B emerge ===")
    print(f"device={device} variants={variants} d={d_model} L={n_layers} H={n_heads}")
    print(f"BS={batch_size} T={block_size} steps={max_steps} target_tokens={n_target_tokens:,}")

    # Init a "driver-level" wandb run so the dashboard shows the run
    # immediately, even before tokenization/training starts. Each variant
    # gets its own wandb run inside train_one(); this is just for
    # "I'm alive, fetching FineWeb" visibility.
    driver_wb = None
    try:
        if os.environ.get("WANDB_API_KEY"):
            import wandb as wb
            driver_wb = wb.init(
                project=os.environ.get("WANDB_PROJECT", "sfumato-e5"),
                name=f"{out_name}-driver",
                group=os.environ.get("WANDB_GROUP", out_name),
                job_type="driver",
                config={"out_name": out_name, "n_target_tokens": n_target_tokens,
                        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                        "variants": variants, "max_steps": max_steps,
                        "batch_size": batch_size, "block_size": block_size,
                        "peak_lr": peak_lr, "seed": seed},
                reinit=True,
            )
            print(f"[wandb-driver] init OK: {driver_wb.url}", flush=True)
    except Exception as e:
        print(f"[wandb-driver] init failed: {e!s:.200}", flush=True)

    # Stream + cache tokens. This uses the existing load_fineweb_tokens cache.
    print(f"\nFetching ~{n_target_tokens/1e9:.1f}B FineWeb-Edu tokens (cached on pod)...")
    t0 = time.time()
    if driver_wb is not None:
        try: driver_wb.log({"phase": 0, "phase_name": "tokenizing"})
        except Exception: pass
    tokens = load_fineweb_tokens(n_tokens=n_target_tokens, seed=1337 + seed)
    fetch_s = time.time() - t0
    print(f"  got {len(tokens):,} tokens in {fetch_s:.1f}s")
    if driver_wb is not None:
        try:
            driver_wb.summary["fetch_wall_s"] = fetch_s
            driver_wb.summary["n_tokens_fetched"] = len(tokens)
            driver_wb.log({"phase": 1, "phase_name": "fetched"})
        except Exception: pass

    # Eval setup
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    gsm_probes = load_gsm8k_probe_problems(n=100)
    gsm_dev = load_gsm8k_dev_questions(n=10)

    for variant in variants:
        variant = variant.strip()
        run_out = base_out / variant
        ckpt = run_out / "model.pt"
        if not ckpt.exists():
            print(f"\n=== Training {variant}-1B (seed={seed}, {max_steps} steps) ===")
            t0 = time.time()
            train_one(
                variant=variant,
                d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                out_dir=run_out,
                seed=seed, max_steps=max_steps,
                batch_size=batch_size, block_size=block_size,
                peak_lr=peak_lr, eval_every=10**9, n_eval=0, tokens=tokens,
            )
            print(f"  train wall_s={time.time()-t0:.1f}")
        else:
            print(f"[{variant}] ckpt exists, skip train")

        # Score + sample
        print(f"\n=== Scoring + sampling {variant} ===")
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = CompositeConfig(**ck["config"])
        model = CompositeLM(cfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.train(False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  n_params={n_params/1e6:.1f}M")

        ar_nll = score_ar_nll(model, gsm_probes, device, max_len=block_size)
        print(f"  GSM8K AR-NLL: {ar_nll}")

        if variant == "composite":
            mode_fns = {
                "composite[ar_only]":    lambda m, p: gen_ar(m, p, max_new=200),
                "composite[mode_switch]": lambda m, p: gen_mode_switch(m, p, k_ar=160, revise_len=40),
                "composite[paired]":     lambda m, p: gen_paired(m, p, k_ar=100, k_diff=100),
            }
        else:
            mode_fns = {
                "ar_only[greedy]": lambda m, p: gen_ar(m, p, max_new=200),
            }

        samples_md = generate_samples(model, gsm_dev, tok, device, mode_fns, n=10)
        (run_out / "samples.md").write_text(f"# F7 {variant}-1B samples (seed={seed})\n\n"
                                            f"Trained on ~{len(tokens)/1e9:.2f}B FineWeb-Edu tokens, "
                                            f"{max_steps} steps, BS={batch_size}×T={block_size}.\n\n"
                                            f"**GSM8K-held-out AR-NLL: {ar_nll}**\n"
                                            + samples_md)
        (run_out / "score.json").write_text(json.dumps({
            "variant": variant, "n_params": n_params, "ar_nll": ar_nll,
            "seed": seed, "max_steps": max_steps, "block_size": block_size,
        }, indent=2))
        print(f"  wrote {run_out/'samples.md'}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
