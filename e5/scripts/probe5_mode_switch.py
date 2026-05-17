"""Probe #5 — mode-switching inference: composite generates first K tokens
in AR mode, then re-masks the last J tokens and runs diff-mode denoising
to "revise". Compare GSM8K-dev accuracy vs pure-AR-only generation from
the same composite model AND vs ar_only baseline.

This is the actual inference recipe from the original PLAN.md vision
("a learned mode-router at every sub-block boundary chooses among
extend-AR, diffuse-current-block, re-diffuse-last-K-blocks, ..."). We
never tested it.

We pick a small set of composite checkpoints that scored well on GSM8K
overnight (200M seeds 50, 52, 53 hit +4 to +6 pp on the original metric).

Three inference modes per checkpoint:
  ar_only       greedy AR decode for max_new tokens
  mode_switch   AR for first 96 tokens, re-mask last 32, diff-revise
  paired        composite_paired (T0's evaluate.py paired mode):
                AR first 64, diff fill next 64

Headline: did mode_switch beat ar_only on GSM8K-dev N=50?

Env: CKPT=path/to/composite/model.pt N_EVAL=50 OUT=path
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.data import load_gsm8k_dev_questions  # noqa: E402

ANSWER_PAT = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")
NUM_PAT = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    m = ANSWER_PAT.search(text)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    nums = NUM_PAT.findall(text)
    return nums[-1].replace(",", "").replace("$", "") if nums else None


@torch.no_grad()
def gen_ar(model, prompt, max_new=128, eot=50256, temperature=0.0,
           top_p=None, top_k=None, repetition_penalty=1.0,
           no_repeat_ngram_size=0):
    """AR decoder with anti-repetition mechanisms.

    Defaults (temperature=0.0, no penalties) reproduce greedy. For
    undertrained models prefer:
      temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3
    """
    device = next(model.parameters()).device
    bs = model.cfg.block_size
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    gen_start = len(prompt)
    for _ in range(max_new):
        ctx = idx[:, -bs:]
        logits = model(ctx, mode="ar")[:, -1, :].float()

        # Repetition penalty: divide logits of tokens already in sequence by penalty.
        if repetition_penalty != 1.0:
            seen = idx[0].tolist()
            seen_set = set(seen)
            for tok in seen_set:
                if logits[0, tok] > 0:
                    logits[0, tok] = logits[0, tok] / repetition_penalty
                else:
                    logits[0, tok] = logits[0, tok] * repetition_penalty

        # No-repeat-ngram: block any (n-1)-gram followed by a token that would
        # complete an n-gram already in the sequence.
        if no_repeat_ngram_size > 0:
            seen = idx[0].tolist()
            n = no_repeat_ngram_size
            if len(seen) >= n - 1:
                tail = tuple(seen[-(n - 1):])
                banned = set()
                for i in range(len(seen) - n + 1):
                    if tuple(seen[i:i + n - 1]) == tail:
                        banned.add(seen[i + n - 1])
                for tok in banned:
                    logits[0, tok] = float("-inf")

        # Sample or argmax
        if temperature <= 0:
            nxt = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / temperature
            # top-k filter
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[..., -1, None]] = float("-inf")
            # top-p filter
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # keep tokens where cum_prob <= top_p (always keep top-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits.scatter_(-1, sorted_idx, sorted_logits.masked_fill(remove, float("-inf")))
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                nxt = int(torch.argmax(logits, dim=-1).item())
            else:
                nxt = int(torch.multinomial(probs, num_samples=1).item())
        if nxt == eot:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return idx[0].tolist()[gen_start:]


@torch.no_grad()
def diff_revise(model, full_ids: list[int], revise_start: int, revise_end: int, n_steps: int = 16):
    """Re-mask positions [revise_start, revise_end) and apply iterative
    diff denoising. Returns the revised token id list."""
    device = next(model.parameters()).device
    idx = torch.tensor([full_ids], dtype=torch.long, device=device)
    # Mask the revise region
    idx[0, revise_start:revise_end] = MASK_TOKEN_ID
    for step in range(n_steps):
        logits = model(idx, mode="diff")
        region = idx[0, revise_start:revise_end]
        masked = (region == MASK_TOKEN_ID)
        if not masked.any():
            break
        probs = torch.softmax(logits[0, revise_start:revise_end].float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        n_masked = int(masked.sum().item())
        n_to_unmask = max(1, int(n_masked * (step + 1) / n_steps) - ((revise_end - revise_start) - n_masked))
        n_to_unmask = min(n_to_unmask, n_masked)
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked, k=n_to_unmask)
        new_region = region.clone()
        new_region[top_idx] = pred[top_idx]
        idx[0, revise_start:revise_end] = new_region
    # Force-decode any remaining masks to eos
    region = idx[0, revise_start:revise_end].tolist()
    region = [t if t != MASK_TOKEN_ID else 50256 for t in region]
    return full_ids[:revise_start] + region + full_ids[revise_end:]


@torch.no_grad()
def gen_mode_switch(model, prompt, k_ar=96, revise_len=32, n_diff_steps=16):
    """AR for first k_ar tokens, then re-mask last revise_len and diff-revise.
    Total generated tokens = k_ar (the revise overlaps with the last 32 ARs)."""
    ar_part = gen_ar(model, prompt, max_new=k_ar)
    full = prompt + ar_part
    if len(ar_part) < revise_len:
        return ar_part
    revise_end = len(full)
    revise_start = revise_end - revise_len
    refined = diff_revise(model, full, revise_start, revise_end, n_steps=n_diff_steps)
    return refined[len(prompt):]


@torch.no_grad()
def gen_paired(model, prompt, k_ar=64, k_diff=64, n_diff_steps=16):
    """T0's paired mode for reference: AR first k_ar, then diff-fill next k_diff."""
    device = next(model.parameters()).device
    cfg = model.cfg
    ar_part = gen_ar(model, prompt, max_new=k_ar)
    extended = prompt + ar_part
    if len(extended) + k_diff > cfg.block_size:
        extended = extended[-(cfg.block_size - k_diff):]
    idx = torch.tensor([extended + [MASK_TOKEN_ID] * k_diff], dtype=torch.long, device=device)
    gen_start = len(extended)
    for step in range(n_diff_steps):
        logits = model(idx, mode="diff")
        region = idx[0, gen_start:gen_start + k_diff]
        masked = (region == MASK_TOKEN_ID)
        if not masked.any():
            break
        probs = torch.softmax(logits[0, gen_start:gen_start + k_diff].float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        n_masked = int(masked.sum().item())
        n_to_unmask = max(1, int(n_masked * (step + 1) / n_diff_steps) - (k_diff - n_masked))
        n_to_unmask = min(n_to_unmask, n_masked)
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked, k=n_to_unmask)
        new_region = region.clone()
        new_region[top_idx] = pred[top_idx]
        idx[0, gen_start:gen_start + k_diff] = new_region
    diff_part = idx[0, gen_start:gen_start + k_diff].tolist()
    diff_part = [t if t != MASK_TOKEN_ID else 50256 for t in diff_part]
    return ar_part + diff_part


def eval_mode(model, problems, mode_fn, tok) -> dict:
    results = []
    t0 = time.time()
    for i, p in enumerate(problems):
        gen = mode_fn(model, p["prompt_tokens"])
        text = tok.decode([t for t in gen if t < 50257], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred is not None and pred == p["gold"])
        results.append({"idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": correct, "text": text})
    n_corr = sum(r["correct"] for r in results)
    return {"n": len(results), "n_correct": n_corr, "accuracy": round(n_corr / max(1, len(results)), 4),
            "wall_s": round(time.time() - t0, 1), "results": results}


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe5_results.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} ckpt={ckpt}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M param composite")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_eval)

    out = {}
    print("\n[ar_only]")
    r = eval_mode(model, problems, lambda m, pr: gen_ar(m, pr, max_new=128), tok)
    out["ar_only"] = {k: v for k, v in r.items() if k != "results"}
    print(f"  acc={r['accuracy']*100:.1f}%")

    print("\n[mode_switch] AR 96 + diff-revise last 32")
    r = eval_mode(model, problems, lambda m, pr: gen_mode_switch(m, pr, k_ar=96, revise_len=32), tok)
    out["mode_switch_96_32"] = {k: v for k, v in r.items() if k != "results"}
    print(f"  acc={r['accuracy']*100:.1f}%")

    print("\n[mode_switch] AR 64 + diff-revise last 32")
    r = eval_mode(model, problems, lambda m, pr: gen_mode_switch(m, pr, k_ar=64, revise_len=32), tok)
    out["mode_switch_64_32"] = {k: v for k, v in r.items() if k != "results"}
    print(f"  acc={r['accuracy']*100:.1f}%")

    print("\n[paired] AR 64 + diff-fill 64 (T0 baseline)")
    r = eval_mode(model, problems, lambda m, pr: gen_paired(m, pr, k_ar=64, k_diff=64), tok)
    out["paired_64_64"] = {k: v for k, v in r.items() if k != "results"}
    print(f"  acc={r['accuracy']*100:.1f}%")

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(f"Summary: ar={out['ar_only']['accuracy']*100:.1f}% switch={out['mode_switch_96_32']['accuracy']*100:.1f}% paired={out['paired_64_64']['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
