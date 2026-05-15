"""F8 — authentic Parameter-Golf-scale composite vs ar_only on FineWeb-Edu.

OpenAI Model Craft Challenge "Parameter Golf" constraints:
  - 16 MB artifact (model + everything packaged)
  - 10 minute training on 8×H100
  - Evaluated by bits-per-byte (BPB) on FineWeb validation set
    (tokenizer-agnostic — what matters is bytes, not tokens)

We can't replicate the 8×H100/10min compute envelope on a single pod, but
we can match its FLOPs (≈4.7e18) on a single A40 in ~8 hours and pick a
model that fits the 16 MB cap.

Arch: d=128, L=8, H=8, GPT-2 vocab (50258 incl. MASK) → 8.09M params
                                                        15.4 MB bf16 ✓

Recipe: BS=64 × T=512, peak_lr=6e-4, ~30k steps → ≈1B tokens seen.
Same FineWeb-Edu stream as F7 (lets F7 vs F8 be a clean scale ablation).

ENV:
  VARIANTS=composite,ar_only
  MAX_STEPS=30000
  N_TARGET_TOKENS=1500000000
  D_MODEL=128  N_LAYERS=8  N_HEADS=8
  BATCH_SIZE=64  BLOCK_SIZE=512
  PEAK_LR=6e-4
  SEED=210

Eval:
  - bits-per-byte on a 1M-byte FineWeb-Edu held-out slice (BPB, the official
    param-golf metric, tokenizer-agnostic)
  - GSM8K AR-NLL (for cross-substrate consistency with F7)
  - sample generations on a few GSM8K problems
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
from e5.model_composite import CompositeConfig, CompositeLM  # noqa: E402
from e5.scripts.probe5_mode_switch import gen_ar  # noqa: E402


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


def fetch_fineweb_val_texts(n_texts=200, offset=12000):
    """Fetch raw FineWeb-Edu texts (not tokenized) for BPB scoring.

    Returns a list of raw strings, used to compute total NLL / total bytes
    independent of any tokenization choice.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < offset:
            continue
        t = ex.get("text", "")
        if t and len(t) > 200:
            out.append(t)
        if len(out) >= n_texts:
            break
    return out


@torch.no_grad()
def score_bpb_on_texts(model, texts, tok, device, max_len=512):
    """Bits-per-byte on raw English text.

    BPB = (sum NLL in nats / ln(2)) / (sum bytes).
    Tokenizer-agnostic by construction.
    """
    total_nats = 0.0
    total_bytes = 0
    n_tokens = 0
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        ids = ids[:max_len]
        if len(ids) < 2:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        pred = logits[0, :-1].float()
        target = idx[0, 1:]
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nats += float(nlls.sum())
        # Decode the predicted-tokens slice to bytes to normalize per byte
        # The first token (which is conditioned-on, not predicted) is excluded.
        decoded = tok.decode(ids[1:], skip_special_tokens=True)
        total_bytes += len(decoded.encode("utf-8"))
        n_tokens += len(ids) - 1
    bpb = (total_nats / math.log(2)) / max(1, total_bytes)
    return {"bpb": round(bpb, 4), "total_bytes": total_bytes, "n_tokens": n_tokens,
            "avg_nll_per_token": round(total_nats / max(1, n_tokens), 4),
            "n_texts": len(texts)}


@torch.no_grad()
def score_ar_nll(model, problems, device, max_len=512):
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


def generate_samples(model, dev_problems, tok, device, label, n=5):
    lines = []
    for p in dev_problems[:n]:
        q = p["question"]; gold = p["gold"]; prompt = p["prompt_tokens"]
        ids = gen_ar(model, prompt, max_new=200)
        text = tok.decode([t for t in ids if t < 50257], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = "✅" if pred == gold else "❌"
        lines.append(f"\n### Q (idx={p['idx']})\n\n> {q.strip()}\n\n**Gold:** `{gold}`  **{label}:** {correct} pred=`{pred}`\n\n```\n{text[:600].strip()}\n```\n")
    return "\n".join(lines)


def main():
    variants = os.environ.get("VARIANTS", "composite,ar_only").split(",")
    max_steps = env_int("MAX_STEPS", 30000)
    n_target_tokens = env_int("N_TARGET_TOKENS", 1_500_000_000)
    d_model = env_int("D_MODEL", 128)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    batch_size = env_int("BATCH_SIZE", 64)
    block_size = env_int("BLOCK_SIZE", 512)
    peak_lr = env_float("PEAK_LR", 6e-4)
    seed = env_int("SEED", 210)

    out_name = os.environ.get("OUT_NAME", "f8_paramgolf_small")
    base_out = REPO_ROOT / "e5" / "results" / out_name
    base_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== F8 param-golf-scale ===")
    print(f"device={device} variants={variants} d={d_model} L={n_layers} H={n_heads}")
    print(f"BS={batch_size} T={block_size} steps={max_steps} target_tokens={n_target_tokens:,}")

    # Up-front wandb.init so the run shows immediately in the dashboard
    # while we tokenize. train_one() will later open per-variant runs.
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

    # Pre-fetch tokens on this pod.
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

    print("\nFetching FineWeb val texts for BPB scoring (independent slice, offset 12000)...")
    val_texts = fetch_fineweb_val_texts(n_texts=200, offset=12000)
    print(f"  got {len(val_texts)} val texts ({sum(len(t.encode('utf-8')) for t in val_texts)} bytes total)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    gsm_probes = load_gsm8k_probe_problems(n=100)
    gsm_dev = load_gsm8k_dev_questions(n=10)

    for variant in variants:
        variant = variant.strip()
        run_out = base_out / variant
        ckpt = run_out / "model.pt"
        if not ckpt.exists():
            print(f"\n=== Training {variant}-8M (seed={seed}, {max_steps} steps) ===")
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

        print(f"\n=== Scoring + sampling {variant} ===")
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = CompositeConfig(**ck["config"])
        model = CompositeLM(cfg).to(device)
        model.load_state_dict(ck["state_dict"])
        model.train(False)
        n_params = sum(p.numel() for p in model.parameters())
        artifact_mb = n_params * 2 / (1024 * 1024)
        print(f"  n_params={n_params/1e6:.2f}M  artifact={artifact_mb:.1f} MB bf16  (param-golf cap: 16 MB)")

        bpb = score_bpb_on_texts(model, val_texts, tok, device, max_len=block_size)
        ar_nll = score_ar_nll(model, gsm_probes, device, max_len=block_size)
        print(f"  FineWeb-val BPB: {bpb['bpb']}  (avg_nll/tok={bpb['avg_nll_per_token']})")
        print(f"  GSM8K AR-NLL: {ar_nll}")

        samples_md = generate_samples(model, gsm_dev, tok, device, f"{variant}[greedy]", n=10)
        (run_out / "samples.md").write_text(
            f"# F8 param-golf {variant} samples (seed={seed})\n\n"
            f"**Arch:** d={d_model} L={n_layers} H={n_heads} = {n_params/1e6:.2f}M params, "
            f"{artifact_mb:.1f} MB bf16 (param-golf 16 MB cap: "
            f"{'✓' if artifact_mb <= 16 else '✗'})\n\n"
            f"**FineWeb-val BPB:** {bpb['bpb']} bits/byte  "
            f"(avg_nll/token={bpb['avg_nll_per_token']}, on {bpb['n_texts']} texts, "
            f"{bpb['total_bytes']} bytes)\n\n"
            f"**GSM8K-held-out AR-NLL:** {ar_nll}\n\n"
            + samples_md
        )
        (run_out / "score.json").write_text(json.dumps({
            "variant": variant, "n_params": n_params, "artifact_mb": artifact_mb,
            "fineweb_bpb": bpb["bpb"], "fineweb_avg_nll_per_tok": bpb["avg_nll_per_token"],
            "gsm8k_ar_nll": ar_nll, "seed": seed, "max_steps": max_steps,
        }, indent=2))
        print(f"  wrote {run_out/'samples.md'}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
