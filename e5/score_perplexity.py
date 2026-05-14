"""Tier-1 perplexity scorer for e5 composite-training checkpoints.

Why this exists: GSM8K-dev accuracy at toy scale is at the noise floor (0-6%,
1-3 problems correct out of 50). The binomial SD ≈ 2pp dominates the
composite-vs-B3 architectural delta, regardless of model size and training
steps tested (60M-300M, 3k-10k steps). A continuous metric — average per-
token NLL on a held-out chunk of GSM8K-train CoT tokens — lets us detect
~0.5pp-scale differences that GSM8K-dev accuracy can't.

Method:
  - Pick a held-out chunk of GSM8K-train (problems whose idx is NOT in
    e4/data/gsm8k_dev_200.json's indices). We use GSM8K-train indices
    7000-7100 (last 100 of the 7473-problem split); these are not in the
    dev set and were not in the cached training stream's typical sliding
    window position, so they're approximately held out.
  - Tokenize each problem's question + gold answer (CoT) with GPT-2 BPE.
  - For each checkpoint, run in AR mode: model(idx, mode='ar') gives
    logits at every position. We compute the cross-entropy between the
    logits at position t and idx[t+1] (next-token prediction), but only
    over the answer-region tokens (questions don't count toward the
    score, since they're "the prompt").
  - Average per-token NLL across all answer-region tokens across all 100
    held-out problems. Lower is better.

For each variant directory under e5/results/, score:
  - composite/model.pt (AR mode)
  - ar_only/model.pt
  - diff_only/model.pt (AR-mode perplexity for direct comparison, even though
    it was trained on the diffusion loss; this lets us see how badly diff_only
    fits the AR distribution it never trained against)
  - paired/ar_only/model.pt (B3's AR sub)
  - paired/diff_only/model.pt (B3's diff sub, AR-mode perplexity)

For T1 (ARFlowLM) checkpoints, same structure but with flow_only instead of
diff_only.

The headline metric is:
  Δ_NLL = composite_NLL - B3_ar_only_NLL
which is the *AR-half-of-composite* perplexity gain from joint training,
holding the B3 architecture fixed (B3's AR sub is trained on AR loss only).
A NEGATIVE Δ_NLL means the composite's joint training improved the AR head's
modelling of the gold answer distribution; i.e. composite wins.

Note: this is NOT a "composite paired generation" perplexity, which would
require a more complex pipeline (AR for first half, then sample from diff
for second half). We use AR-mode-NLL across both halves as a clean,
deterministic, continuous proxy.

Usage:
  python e5/score_perplexity.py                  # score all e5/results/*
  CKPT=e5/results/t0_3k_seed0/composite/model.pt python e5/score_perplexity.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Lazy-load the model classes (one of two, depending on which the ckpt is).
def _load_composite(path: Path, device: str):
    from e5.model_composite import CompositeConfig, CompositeLM
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    m = CompositeLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "composite"


def _load_arflow(path: Path, device: str):
    from e5.model_arflow import ARFlowConfig, ARFlowLM
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ARFlowConfig(**ck["config"])
    m = ARFlowLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "arflow"


def load_checkpoint(path: Path, device: str):
    """Auto-detect ckpt type. CompositeLM ckpts have a `vocab_size` of 50258
    (GPT2 + MASK token); ARFlowLM ckpts have 50257 (GPT2 only). Use that as
    a discriminator."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vsize = ck["config"].get("vocab_size", 50258)
    if vsize == 50258:
        return _load_composite(path, device)
    return _load_arflow(path, device)


def get_holdout_chunks(n_problems: int = 100, indices_offset: int = 7000):
    """Tokenize n_problems from GSM8K-train starting at indices_offset.
    Returns a list of (prompt_tokens, answer_tokens) tuples. The answer
    tokens are the gold CoT after the "Answer:" marker."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(indices_offset, min(indices_offset + n_problems, len(ds))):
        row = ds[idx]
        prompt_text = f"Question: {row['question']}\nAnswer:"
        answer_text = " " + row["answer"]  # leading space for proper BPE merge
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        out.append((prompt_ids, answer_ids))
    return out


@torch.no_grad()
def score_ar_nll(model, problems, device: str, max_len: int = 256) -> dict:
    """Average per-token NLL over the answer-region tokens of each problem.
    Run model in AR mode. Skip problems whose total length exceeds max_len.

    Returns {avg_nll, total_tokens, n_problems_scored, n_skipped}.
    """
    total_nll = 0.0
    total_tokens = 0
    n_scored = 0
    n_skipped = 0
    eot_id = 50256
    # Some checkpoints have vocab_size=50258 (with MASK token at 50257).
    # The held-out tokens use GPT-2 ids 0..50256, all valid for either.
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            n_skipped += 1
            continue
        full = prompt_ids + answer_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")  # (1, T, V)
        # We score positions in the answer region: predicting answer_ids[t]
        # from positions [0..len(prompt)+t-1]. The next-token logits at
        # position (len(prompt)+t-1) target answer_ids[t]. We use ALL answer
        # positions including the first (predicting answer_ids[0] from the
        # last prompt position).
        ans_start = len(prompt_ids)
        # logits[0, ans_start-1 : ans_start-1+len(answer_ids), :] predict
        # full[ans_start : ans_start+len(answer_ids)].
        pred_logits = logits[0, ans_start - 1 : ans_start - 1 + len(answer_ids), :].float()
        target_ids = torch.tensor(answer_ids, dtype=torch.long, device=device)
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        nlls = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total_nll += float(nlls.sum().item())
        total_tokens += int(nlls.numel())
        n_scored += 1
    avg = total_nll / max(1, total_tokens)
    return {
        "avg_nll": round(avg, 6),
        "perplexity": round(math.exp(avg), 4),
        "total_tokens": total_tokens,
        "n_problems_scored": n_scored,
        "n_skipped": n_skipped,
    }


def score_one_directory(result_dir: Path, problems, device: str) -> dict:
    """Score every model.pt under result_dir (recursive). Returns a flat dict
    of subpath → score."""
    out = {}
    for ckpt in sorted(result_dir.rglob("model.pt")):
        rel = ckpt.relative_to(result_dir)
        try:
            model, cfg, kind = load_checkpoint(ckpt, device)
        except Exception as e:
            out[str(rel)] = {"error": f"load_failed: {e!s:.200}"}
            continue
        try:
            score = score_ar_nll(model, problems, device=device, max_len=cfg.block_size)
            score["kind"] = kind
            score["n_params"] = sum(p.numel() for p in model.parameters())
            out[str(rel)] = score
        except Exception as e:
            out[str(rel)] = {"error": f"score_failed: {e!s:.200}"}
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    return out


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device={device}")
    n_holdout = int(os.environ.get("N_HOLDOUT", "100"))
    print(f"loading {n_holdout} held-out GSM8K-train problems (idx 7000+)...")
    problems = get_holdout_chunks(n_problems=n_holdout)
    print(f"loaded {len(problems)} problems")
    avg_lens = sum(len(p) + len(a) for p, a in problems) / len(problems)
    print(f"avg total length: {avg_lens:.0f} tokens")

    results_root = REPO_ROOT / "e5" / "results"
    out: dict[str, dict] = {}

    # Optional single-ckpt mode for debugging
    if os.environ.get("CKPT"):
        ckpt = Path(os.environ["CKPT"])
        model, cfg, kind = load_checkpoint(ckpt, device)
        score = score_ar_nll(model, problems, device=device, max_len=cfg.block_size)
        score["kind"] = kind
        print(json.dumps(score, indent=2))
        return

    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.rglob("model.pt")):
            continue
        print(f"\n--- {d.name} ---")
        out[d.name] = score_one_directory(d, problems, device)
        for k, v in out[d.name].items():
            if "error" in v:
                print(f"  {k}: ERROR {v['error']}")
            else:
                print(f"  {k}: avg_nll={v['avg_nll']:.4f} ppl={v['perplexity']:.2f} ({v['n_problems_scored']}p {v['total_tokens']}tok)")

    summary_path = results_root / "perplexity_raw.json"
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
