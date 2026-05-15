"""Generate natural-language samples from trained 200M checkpoints.

Loads composite + ar_only + diff_only at the same seed, generates 4 modes
on a handful of GSM8K-dev problems, and saves a side-by-side markdown.

Modes:
  ar_only       — pure AR greedy decode, 128 tokens
  mode_switch   — AR 96 then re-mask last 32, diff-revise
  paired        — AR 64 then diff-fill next 64
  diff_only     — pure diff sample (with random mask init), 128 tokens

Env:
  SEED=50              which seed's checkpoints to use
  N_PROBLEMS=5         how many dev problems to sample
  OUT=e5/results/samples_seed{SEED}.md
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
from e5.scripts.probe5_mode_switch import gen_ar, gen_mode_switch, gen_paired  # noqa: E402

ANSWER_PAT = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")
NUM_PAT = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    m = ANSWER_PAT.search(text)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    nums = NUM_PAT.findall(text)
    return nums[-1].replace(",", "").replace("$", "") if nums else None


@torch.no_grad()
def gen_diff_only(model, prompt, n_new=128, n_steps=16):
    """Pure diff generation: AR encodes prompt context-free? Actually we just
    use ar+diff to seed from full-mask after prompt and denoise."""
    device = next(model.parameters()).device
    bs = model.cfg.block_size
    pad = max(0, n_new - (bs - len(prompt)))
    if pad > 0:
        prompt = prompt[pad:]  # truncate prompt if needed
    full = prompt + [MASK_TOKEN_ID] * n_new
    idx = torch.tensor([full], dtype=torch.long, device=device)
    gen_start = len(prompt)
    for step in range(n_steps):
        logits = model(idx, mode="diff")
        region = idx[0, gen_start:]
        masked = (region == MASK_TOKEN_ID)
        if not masked.any():
            break
        probs = torch.softmax(logits[0, gen_start:].float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        n_masked = int(masked.sum().item())
        n_to_unmask = max(1, int(n_masked * (step + 1) / n_steps) - (n_new - n_masked))
        n_to_unmask = min(n_to_unmask, n_masked)
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked, k=n_to_unmask)
        new_region = region.clone()
        new_region[top_idx] = pred[top_idx]
        idx[0, gen_start:] = new_region
    out = idx[0, gen_start:].tolist()
    return [t if t != MASK_TOKEN_ID else 50256 for t in out]


def load_model(seed: int, variant: str, device: str):
    ckpt_path = REPO_ROOT / "e5" / "results" / f"t0_200M_seed{seed}" / variant / "model.pt"
    if not ckpt_path.exists():
        return None
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    m = CompositeLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m


def main():
    seed = int(os.environ.get("SEED", "50"))
    n_problems = int(os.environ.get("N_PROBLEMS", "5"))
    out_path = Path(os.environ.get("OUT", f"e5/results/samples_seed{seed}.md"))
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"device={device} seed={seed} n_problems={n_problems}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_problems)

    lines = [f"# Generation samples — 200M seed={seed} (overnight checkpoints)",
             "",
             "Modes compared per problem:",
             "",
             "- `composite[ar_only]` — composite ckpt, greedy AR decode 128 tokens",
             "- `composite[mode_switch]` — AR 96 + diff-revise last 32 (16 steps)",
             "- `composite[paired]` — AR 64 + diff-fill next 64 (16 steps)",
             "- `composite[diff_only]` — pure diff from full-mask suffix (16 steps)",
             "- `ar_only ckpt` — pure-AR-trained model, greedy 128 tokens",
             "- `diff_only ckpt` — pure-diff-trained model, mask-fill 128 tokens",
             ""]

    for variant in ("composite", "ar_only", "diff_only"):
        model = load_model(seed, variant, device)
        if model is None:
            print(f"WARN: {variant} ckpt missing")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        print(f"loaded {variant} ({n_params/1e6:.1f}M params)")

        for i, p in enumerate(problems):
            q = p["question"]
            gold = p["gold"]
            prompt = p["prompt_tokens"]

            results = {}
            if variant == "composite":
                t0 = time.time()
                results["ar_only"] = gen_ar(model, prompt, max_new=128)
                results["mode_switch"] = gen_mode_switch(model, prompt, k_ar=96, revise_len=32)
                results["paired"] = gen_paired(model, prompt, k_ar=64, k_diff=64)
                results["diff_only"] = gen_diff_only(model, prompt, n_new=128)
                wall = time.time() - t0
            elif variant == "ar_only":
                t0 = time.time()
                results["ar_ckpt_greedy"] = gen_ar(model, prompt, max_new=128)
                wall = time.time() - t0
            else:  # diff_only
                t0 = time.time()
                results["diff_ckpt"] = gen_diff_only(model, prompt, n_new=128)
                wall = time.time() - t0
            print(f"  problem {i}: {variant} {wall:.1f}s")

            if i == 0 or variant == "composite":  # write problem header on first variant per problem
                lines.append(f"\n---\n## Problem {p['idx']}\n")
                lines.append(f"**Question:**\n```\n{q.strip()}\n```\n")
                lines.append(f"**Gold answer:** `{gold}`\n")

            for mode, ids in results.items():
                text = tok.decode([t for t in ids if t < 50257], skip_special_tokens=True)
                pred = extract_answer(text)
                correct = "✅" if pred == gold else "❌"
                lines.append(f"**{variant}[{mode}]** {correct} pred=`{pred}`")
                lines.append(f"```\n{text[:600].strip()}\n```\n")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
