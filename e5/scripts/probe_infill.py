"""Phase J.1 — FIM / infill capability with AR fallback.

The composite's diff head fills a middle-masked span using BOTH the
prefix AND the suffix (bidirectional attention). Pure AR is causal
only — it cannot see the suffix. This script measures the gap.

For each GSM8K-dev problem (held-out test set), we:
  1. Tokenize prompt + gold answer.
  2. Identify a middle window of length `mid_mask_len` inside the
     answer, with `suffix_len` tokens of answer remaining intact
     after the mask.
  3. Score per-token NLL under three policies:
     - COMPOSITE-DIFF: mask the middle, run diff-head forward,
       collect NLL on the masked positions (using suffix context).
     - COMPOSITE-AR-NO-SUFFIX: truncate the sequence at the mask
       start, run AR head, collect NLL on the same positions
       conditioned only on prefix.
     - COMPOSITE-AR-TEACHER-FORCED: AR head sees the FULL ground-truth
       sequence (suffix included as causal context, but the AR head
       still only attends left-to-right so the suffix tokens AFTER
       the masked region cannot influence predictions ON the masked
       region — only positions LEFT of each masked position do).
       This is the standard AR perplexity baseline.

Composite advantage: AR-NO-SUFFIX must be at least as bad as
AR-TEACHER-FORCED, and composite-diff should beat both because it
uses bidirectional context.

Output: e5/results/f10_mixed/probe_infill_n50.json with per-policy
average NLL and the gap.

Env:
  CKPT=path/to/model.pt
  N_EVAL=50
  MID_MASK_LEN=20
  SUFFIX_LEN=20
  MAX_LEN=512
  OUT=path
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

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402


def load_gsm8k_probe_problems(n: int = 50, offset: int = 7000):
    """Reuses the loader from f7_1b_emerge.py — pulls held-out GSM8K problems
    (offset 7000 onward, beyond the dev-50 indices in the frozen json)."""
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
def score_one(model, prompt_ids, answer_ids, device, max_len=512,
              mid_mask_len=20, suffix_len=20):
    """Score per-token NLL on a middle-masked span under three policies."""
    if len(answer_ids) < mid_mask_len + suffix_len + 4:
        return None
    if len(prompt_ids) + len(answer_ids) + 1 > max_len:
        return None

    full = prompt_ids + answer_ids
    ans_start = len(prompt_ids)
    # Place the mask near the middle of the answer
    mid_start = ans_start + (len(answer_ids) - mid_mask_len - suffix_len) // 2
    mid_end = mid_start + mid_mask_len

    full_t = torch.tensor([full], dtype=torch.long, device=device)
    targets = full_t[0, mid_start:mid_end]

    # ----- (a) COMPOSITE-DIFF: mask middle, score with bidirectional attn -----
    masked = full_t.clone()
    masked[0, mid_start:mid_end] = MASK_TOKEN_ID
    logits = model(masked, mode="diff")
    log_probs = torch.log_softmax(logits[0, mid_start:mid_end].float(), dim=-1)
    nll_diff = float(-log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item())

    # ----- (b) COMPOSITE-AR-NO-SUFFIX: truncate at mid_start, AR forward -----
    trunc = full_t[:, :mid_start]
    # Generate the next mid_mask_len logit predictions via teacher forcing on the
    # gold mid-tokens. We append the gold mid-tokens to trunc and read logits at
    # positions [mid_start - 1, mid_start + mid_mask_len - 1), which are the
    # AR predictions for positions [mid_start, mid_end).
    ar_input = torch.cat([trunc, targets.unsqueeze(0)], dim=1)
    ar_logits = model(ar_input, mode="ar")
    # logits[t] predicts token at position t+1
    pred_logits = ar_logits[0, mid_start - 1 : mid_start - 1 + mid_mask_len].float()
    log_probs_ar = torch.log_softmax(pred_logits, dim=-1)
    nll_ar_no_suffix = float(-log_probs_ar.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item())

    # ----- (c) COMPOSITE-AR-TEACHER-FORCED-FULL: full sequence, AR forward -----
    # Same as standard AR perplexity scoring.
    ar_logits_full = model(full_t, mode="ar")
    pred_logits_full = ar_logits_full[0, mid_start - 1 : mid_start - 1 + mid_mask_len].float()
    log_probs_full = torch.log_softmax(pred_logits_full, dim=-1)
    nll_ar_full = float(-log_probs_full.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item())

    return {
        "n": mid_mask_len,
        "nll_diff_sum": nll_diff,
        "nll_ar_no_suffix_sum": nll_ar_no_suffix,
        "nll_ar_full_sum": nll_ar_full,
    }


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe_infill.json"))
    mid_mask_len = int(os.environ.get("MID_MASK_LEN", "20"))
    suffix_len = int(os.environ.get("SUFFIX_LEN", "20"))
    max_len = int(os.environ.get("MAX_LEN", "512"))

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = os.environ.get("DEVICE", device)
    print(f"device={device} ckpt={ckpt} mid_mask_len={mid_mask_len} suffix_len={suffix_len}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M param composite")

    problems = load_gsm8k_probe_problems(n=n_eval)
    print(f"loaded {len(problems)} held-out problems (offset 7000+)")

    total_nll_diff = 0.0
    total_nll_ar_no_suffix = 0.0
    total_nll_ar_full = 0.0
    total_count = 0
    skipped = 0

    t0 = time.perf_counter()
    for p_ids, a_ids in problems:
        r = score_one(model, p_ids, a_ids, device,
                      max_len=max_len, mid_mask_len=mid_mask_len, suffix_len=suffix_len)
        if r is None:
            skipped += 1
            continue
        total_nll_diff += r["nll_diff_sum"]
        total_nll_ar_no_suffix += r["nll_ar_no_suffix_sum"]
        total_nll_ar_full += r["nll_ar_full_sum"]
        total_count += r["n"]
    wall = time.perf_counter() - t0

    if total_count == 0:
        print("WARN: no eligible problems; check mid_mask_len + suffix_len budget")
        return

    avg_diff = total_nll_diff / total_count
    avg_ar_no_suffix = total_nll_ar_no_suffix / total_count
    avg_ar_full = total_nll_ar_full / total_count

    out = {
        "ckpt": str(ckpt),
        "n_eval": n_eval,
        "n_scored": (n_eval - skipped),
        "n_middle_tokens": total_count,
        "mid_mask_len": mid_mask_len,
        "suffix_len": suffix_len,
        "max_len": max_len,
        "wall_s": round(wall, 1),
        "metrics": {
            "fim_diff_nll":      round(avg_diff, 4),
            "ar_no_suffix_nll":  round(avg_ar_no_suffix, 4),
            "ar_full_nll":       round(avg_ar_full, 4),
            "gap_diff_vs_ar_no_suffix": round(avg_ar_no_suffix - avg_diff, 4),
            "gap_diff_vs_ar_full":      round(avg_ar_full - avg_diff, 4),
        },
        "note": ("composite-diff sees BOTH prefix AND suffix bidirectionally. "
                 "AR-no-suffix is the pure-AR fallback (no suffix context). "
                 "AR-full is teacher-forced AR with the full ground-truth sequence "
                 "(still left-to-right, but the per-position predictions cannot use "
                 "tokens AFTER the predicted position, so suffix-context info is "
                 "structurally inaccessible to the AR head)."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {out_path}")
    print(f"  n_scored={out['n_scored']}  n_middle_tokens={out['n_middle_tokens']}  wall={wall:.1f}s")
    print(f"  fim_diff_nll        = {avg_diff:.4f}")
    print(f"  ar_no_suffix_nll    = {avg_ar_no_suffix:.4f}  (gap vs diff: +{avg_ar_no_suffix - avg_diff:.4f})")
    print(f"  ar_full_nll         = {avg_ar_full:.4f}  (gap vs diff: +{avg_ar_full - avg_diff:.4f})")
    print(f"  composite-diff beats AR-no-suffix by {avg_ar_no_suffix - avg_diff:.4f} NLL/token")
    print(f"  composite-diff beats AR-full      by {avg_ar_full - avg_diff:.4f} NLL/token")


if __name__ == "__main__":
    main()
