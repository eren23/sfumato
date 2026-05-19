"""Phase J.2 — revision quality via per-position NLL on gold continuation.

For each GSM8K-dev problem, we run three generation policies. We then
score per-token NLL of the GENERATED tokens against the gold answer,
position by position, to see whether mode-switch revision helps any
positions vs pure AR.

Policies:
  P_ar_128         gen_ar(model, prompt, max_new=128)
  P_ms_96_32       gen_mode_switch(model, prompt, k_ar=96, revise_len=32, ...)
  P_ms_64_32       gen_mode_switch(model, prompt, k_ar=64, revise_len=32, ...)

For each policy we compute, for each gold token g_t and generated
token x_t at position t in [0..127):
  - per-token agreement (x_t == g_t)
  - per-token AR-head NLL of the GOLD token g_t conditioned on
    [prompt + x_{0..t-1}] (teacher-forced AR perplexity of gold
    given the model's own first-t outputs)

The per-position curve of (NLL_ar_128 − NLL_ms_64_32) reveals where
revision helped, if anywhere.

Output: e5/results/f10_mixed/probe_revision_nll_n50.json with per-policy
average NLL and per-position arrays.

Env:
  CKPT=path/to/model.pt
  N_EVAL=50
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
from e5.scripts.probe5_mode_switch import gen_ar, gen_mode_switch  # noqa: E402


def load_problems(n: int = 50, offset: int = 7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(offset, min(offset + n, len(ds))):
        row = ds[idx]
        p = tok.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
        a = tok.encode(" " + row["answer"], add_special_tokens=False)
        out.append({"idx": idx, "prompt_tokens": p, "gold_answer_tokens": a})
    return out


@torch.no_grad()
def per_position_nll(model, prompt: list[int], generated: list[int],
                     gold: list[int], device: str, max_pos: int = 128):
    """Teacher-force the model on [prompt + generated[:t-1]] and score the
    gold token at position t. Returns list of per-position NLLs of length
    min(len(generated), len(gold), max_pos)."""
    n = min(len(generated), len(gold), max_pos)
    if n == 0:
        return []
    # We feed [prompt + generated[:n-1] + gold[n-1]] all at once and read
    # logits at positions corresponding to each gold token. Actually we want
    # the AR head to score P(gold_t | prompt + generated[:t]) for each t in [0..n).
    # We feed [prompt + generated[:n]] and read logits at the appropriate
    # positions for predicting gold[t], but the model's logit at position p
    # predicts token at p+1. So logits at position (len(prompt) + t - 1)
    # predict position (len(prompt) + t), which we score against gold[t].
    nlls = []
    for t in range(n):
        ctx = prompt + generated[:t]  # everything up to and including the
        # most recent generated token (or just prompt if t==0)
        if len(ctx) == 0:
            continue
        idx = torch.tensor([ctx[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = float(-log_probs[gold[t]].item())
        nlls.append(nll)
    return nlls


def main():
    ckpt = Path(os.environ["CKPT"])
    n_eval = int(os.environ.get("N_EVAL", "50"))
    out_path = Path(os.environ.get("OUT", "probe_revision_nll.json"))

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = os.environ.get("DEVICE", device)
    print(f"device={device} ckpt={ckpt}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.train(False)
    print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.1f}M param composite")

    problems = load_problems(n=n_eval)
    print(f"loaded {len(problems)} held-out problems")

    ar_kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
    diff_kw = dict(diff_temperature=0.8, diff_top_p=0.9, diff_repetition_penalty=1.15)

    policies = {
        "ar_128":      lambda m, p: gen_ar(m, p, max_new=128, **ar_kw),
        "ms_96_32":    lambda m, p: gen_mode_switch(m, p, k_ar=96, revise_len=32, **ar_kw, **diff_kw),
        "ms_64_32":    lambda m, p: gen_mode_switch(m, p, k_ar=64, revise_len=32, **ar_kw, **diff_kw),
    }

    out = {
        "ckpt": str(ckpt),
        "n_eval": n_eval,
        "decode": {"ar": ar_kw, "diff": diff_kw},
        "policies": {},
    }

    # Aggregate per-position arrays
    max_pos = 128

    t_start = time.perf_counter()
    for name, fn in policies.items():
        print(f"\n[{name}]")
        per_problem_nlls = []  # list of (length-N) arrays
        agreements = []
        for i, p in enumerate(problems):
            prompt = p["prompt_tokens"]
            gold = p["gold_answer_tokens"]
            gen = fn(model, prompt)
            # Per-token agreement
            n_match = sum(1 for k in range(min(len(gen), len(gold))) if gen[k] == gold[k])
            n_compare = min(len(gen), len(gold))
            agreements.append((n_match, n_compare))
            # Per-position NLL of gold under teacher-forced (prompt + generated[:t]) AR
            nlls = per_position_nll(model, prompt, gen, gold, device, max_pos=max_pos)
            per_problem_nlls.append(nlls)
            if (i + 1) % 10 == 0:
                print(f"  scored {i+1}/{len(problems)}")

        # Aggregate per-position mean
        pos_mean = [0.0] * max_pos
        pos_count = [0] * max_pos
        for nlls in per_problem_nlls:
            for t, v in enumerate(nlls):
                pos_mean[t] += v
                pos_count[t] += 1
        per_pos = [round(pos_mean[t] / pos_count[t], 4) if pos_count[t] > 0 else None
                   for t in range(max_pos)]

        total_match = sum(m for m, _ in agreements)
        total_compare = sum(c for _, c in agreements)
        mean_nll_all = sum(v for nlls in per_problem_nlls for v in nlls) / max(1, sum(len(n) for n in per_problem_nlls))

        # Slice means: 0-64, 64-96, 96-128
        def slice_mean(start, end):
            s = sum(per_pos[t] for t in range(start, end) if per_pos[t] is not None)
            n = sum(1 for t in range(start, end) if per_pos[t] is not None)
            return round(s / max(1, n), 4) if n else None

        summary = {
            "n_problems": len(problems),
            "agreement_pct": round(100 * total_match / max(1, total_compare), 2),
            "mean_nll_all_pos": round(mean_nll_all, 4),
            "mean_nll_0_64": slice_mean(0, 64),
            "mean_nll_64_96": slice_mean(64, 96),
            "mean_nll_96_128": slice_mean(96, 128),
            "per_pos_mean_nll": per_pos,
        }
        out["policies"][name] = summary
        print(f"  agreement={summary['agreement_pct']:.2f}%  mean_nll={summary['mean_nll_all_pos']:.4f}  "
              f"0-64={summary['mean_nll_0_64']}  64-96={summary['mean_nll_64_96']}  96-128={summary['mean_nll_96_128']}")

    wall = time.perf_counter() - t_start
    out["wall_s"] = round(wall, 1)

    # Compute deltas: ar_128 - ms_*
    base = out["policies"]["ar_128"]
    for name in ("ms_96_32", "ms_64_32"):
        p = out["policies"][name]
        deltas = {}
        for k in ("mean_nll_all_pos", "mean_nll_0_64", "mean_nll_64_96", "mean_nll_96_128"):
            if base[k] is not None and p[k] is not None:
                deltas[k] = round(base[k] - p[k], 4)
        p["delta_vs_ar_128"] = deltas

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {out_path}  wall={wall:.1f}s")
    print("Summary:")
    print(f"  {'policy':14s} {'agreement':>10s} {'all_nll':>10s} {'0-64':>10s} {'64-96':>10s} {'96-128':>10s}")
    for name, p in out["policies"].items():
        print(f"  {name:14s} {p['agreement_pct']:>9.2f}% {p['mean_nll_all_pos']:>10.4f}  "
              f"{p['mean_nll_0_64']:>10.4f}  {p['mean_nll_64_96']:>10.4f}  {p['mean_nll_96_128']:>10.4f}")


if __name__ == "__main__":
    main()
