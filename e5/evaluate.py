"""T0 — GSM8K-dev evaluation across modes.

For each trained checkpoint, compute accuracy on the first N=50 problems
of `e4/data/gsm8k_dev_200.json` in multiple inference modes:

  - ar: greedy AR decode for max_new tokens
  - diff: iterative mask-denoise (start: prompt + max_new masks)
  - paired: AR generates first K tokens, then diff fills the rest
  - paired_separate: same as `paired` but with two distinct checkpoints

The plan's decision band compares `composite[paired]` vs `B3[paired_separate]`.

Usage:
  CKPT=e5/results/.../model.pt MODE=ar N_EVAL=50 python e5/evaluate.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    if nums:
        return nums[-1].replace(",", "").replace("$", "")
    return None


def load_model_from_ckpt(ckpt_path: Path, device: str) -> CompositeLM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ckpt["config"])
    model = CompositeLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.train(False)
    return model


@torch.no_grad()
def generate_ar(
    model: CompositeLM,
    prompt_ids: list[int],
    max_new: int = 128,
    eot_id: int = 50256,
    temperature: float = 0.0,
) -> list[int]:
    device = next(model.parameters()).device
    block_size = model.cfg.block_size
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = idx[:, -block_size:]
        logits = model(ctx, mode="ar")[:, -1, :]
        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        if next_id == eot_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return idx[0].tolist()[len(prompt_ids):]


@torch.no_grad()
def generate_diff(
    model: CompositeLM,
    prompt_ids: list[int],
    max_new: int = 128,
    n_steps: int = 32,
    mask_token_id: int = MASK_TOKEN_ID,
) -> list[int]:
    """Iterative mask-denoising. Start with prompt_ids + [MASK]*max_new.
    At each step, predict logits for masked positions, replace the
    highest-confidence fraction with argmax tokens.
    """
    device = next(model.parameters()).device
    block_size = model.cfg.block_size
    if len(prompt_ids) + max_new > block_size:
        prompt_ids = prompt_ids[-(block_size - max_new):]

    idx = torch.tensor(
        [prompt_ids + [mask_token_id] * max_new],
        dtype=torch.long, device=device,
    )
    gen_start = len(prompt_ids)
    for step in range(n_steps):
        logits = model(idx, mode="diff")
        gen_logits = logits[0, gen_start:gen_start + max_new]
        gen_idx = idx[0, gen_start:gen_start + max_new]
        masked = (gen_idx == mask_token_id)
        if not masked.any():
            break
        probs = torch.softmax(gen_logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        n_masked = int(masked.sum().item())
        n_to_unmask = max(1, int(n_masked * (step + 1) / n_steps) - (max_new - n_masked))
        n_to_unmask = min(n_to_unmask, n_masked)
        conf_masked = torch.where(masked, conf, torch.full_like(conf, -1.0))
        _, top_idx = torch.topk(conf_masked, k=n_to_unmask)
        new_gen = gen_idx.clone()
        new_gen[top_idx] = pred[top_idx]
        idx[0, gen_start:gen_start + max_new] = new_gen
    final_gen = idx[0, gen_start:gen_start + max_new].tolist()
    final_gen = [t if t != mask_token_id else 50256 for t in final_gen]
    return final_gen


@torch.no_grad()
def generate_paired(
    ar_model: CompositeLM,
    diff_model: CompositeLM,
    prompt_ids: list[int],
    k_ar: int = 64,
    k_diff: int = 64,
    n_diff_steps: int = 24,
) -> list[int]:
    """AR-then-diff pipeline. For composite, ar_model == diff_model. For
    B3, two separate trained models."""
    ar_part = generate_ar(ar_model, prompt_ids, max_new=k_ar)
    extended = prompt_ids + ar_part
    diff_part = generate_diff(diff_model, extended, max_new=k_diff, n_steps=n_diff_steps)
    return ar_part + diff_part


def decode(tokens: list[int], tokenizer) -> str:
    return tokenizer.decode([t for t in tokens if t < 50257], skip_special_tokens=True)


def run_one(
    ckpt_path: Path | None,
    mode: str,
    n_eval: int = 50,
    max_new: int = 128,
    n_diff_steps: int = 32,
    ckpt_ar: Path | None = None,
    ckpt_diff: Path | None = None,
) -> dict:
    """mode ∈ {'ar', 'diff', 'paired', 'paired_separate'}."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "paired_separate":
        ar_model = load_model_from_ckpt(ckpt_ar, device)
        diff_model = load_model_from_ckpt(ckpt_diff, device)
    else:
        model = load_model_from_ckpt(ckpt_path, device)

    problems = load_gsm8k_dev_questions(n=n_eval)
    results = []
    t0 = time.time()

    for i, p in enumerate(problems):
        prompt = p["prompt_tokens"]
        if mode == "ar":
            gen = generate_ar(model, prompt, max_new=max_new)
        elif mode == "diff":
            gen = generate_diff(model, prompt, max_new=max_new, n_steps=n_diff_steps)
        elif mode == "paired":
            gen = generate_paired(model, model, prompt, k_ar=max_new // 2, k_diff=max_new // 2, n_diff_steps=n_diff_steps)
        elif mode == "paired_separate":
            gen = generate_paired(ar_model, diff_model, prompt, k_ar=max_new // 2, k_diff=max_new // 2, n_diff_steps=n_diff_steps)
        else:
            raise ValueError(mode)
        text = decode(gen, tok)
        pred = extract_answer(text)
        correct = int(pred is not None and pred == p["gold"])
        results.append({"idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": correct, "text": text})
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{mode}] {i+1}/{n_eval} acc={acc:.3f} ({el:.0f}s)", flush=True)

    acc = sum(r["correct"] for r in results) / max(1, len(results))
    return {
        "mode": mode,
        "n": len(results),
        "n_correct": sum(r["correct"] for r in results),
        "accuracy": round(acc, 4),
        "wall_s": round(time.time() - t0, 1),
        "results": results,
    }


def main():
    ckpt = os.environ.get("CKPT")
    mode = os.environ.get("MODE", "ar")
    n_eval = int(os.environ.get("N_EVAL", "50"))
    max_new = int(os.environ.get("MAX_NEW", "128"))
    n_diff_steps = int(os.environ.get("N_DIFF_STEPS", "32"))
    out_path = os.environ.get("OUT")

    if mode == "paired_separate":
        ckpt_ar = Path(os.environ["CKPT_AR"])
        ckpt_diff = Path(os.environ["CKPT_DIFF"])
        out = run_one(None, mode, n_eval=n_eval, max_new=max_new, n_diff_steps=n_diff_steps, ckpt_ar=ckpt_ar, ckpt_diff=ckpt_diff)
    else:
        out = run_one(Path(ckpt), mode, n_eval=n_eval, max_new=max_new, n_diff_steps=n_diff_steps)
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}: acc={out['accuracy']*100:.1f}% ({out['n_correct']}/{out['n']})")
    else:
        print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
