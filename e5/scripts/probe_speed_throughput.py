"""Phase J.0 — speed / parallel-decode throughput on F10 final.

Measures wall-clock per generation and tokens-per-second for several
inference modes on the same composite ckpt. The point: diff-mode
fills K tokens in N forward passes (default N=16) while AR-mode needs
K forwards. With no KV-cache (our case), this should translate
directly to a K/N speedup in wall-clock.

Configs:
  composite_ar_128                gen_ar(model, prompt, 128)
  composite_diff_revise_128_16    diff_revise on full-mask suffix, 16 steps
  composite_paired_64_64_16       gen_paired(k_ar=64, k_diff=64, n_steps=16)
  composite_paired_32_96_16       gen_paired(k_ar=32, k_diff=96, n_steps=16)
  composite_paired_16_112_16      gen_paired(k_ar=16, k_diff=112, n_steps=16)

Output: e5/results/f10_mixed/probe_speed_n20.json with mean/SD wall_s
and tokens/sec per config across N_PROBLEMS x N_TRIALS calls.

Env:
  CKPT=path/to/model.pt
  N_PROBLEMS=20  N_TRIALS=5
  OUT=path
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID  # noqa: E402
from e5.data import load_gsm8k_dev_questions  # noqa: E402
from e5.scripts.probe5_mode_switch import gen_ar, diff_revise, gen_paired  # noqa: E402


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


@torch.no_grad()
def time_call(fn, *args, device: str = "cpu", warmup: int = 1):
    """Warm up once, then time a single call. Returns (out, wall_s)."""
    for _ in range(warmup):
        _ = fn(*args)
    _sync(device)
    t0 = time.perf_counter()
    out = fn(*args)
    _sync(device)
    return out, time.perf_counter() - t0


@torch.no_grad()
def diff_fill_full(model, prompt: list[int], n_new: int, n_steps: int,
                   diff_kw: dict):
    """Composite diff-only mode: append n_new MASK tokens, denoise in n_steps."""
    device = next(model.parameters()).device
    cfg = model.cfg
    extended = list(prompt)
    if len(extended) + n_new > cfg.block_size:
        extended = extended[-(cfg.block_size - n_new):]
    revise_start = len(extended)
    revise_end = revise_start + n_new
    full = extended + [MASK_TOKEN_ID] * n_new
    refined = diff_revise(model, full, revise_start, revise_end,
                          n_steps=n_steps, **diff_kw)
    return refined[revise_start:]


def main():
    ckpt = Path(os.environ["CKPT"])
    n_problems = int(os.environ.get("N_PROBLEMS", "20"))
    n_trials = int(os.environ.get("N_TRIALS", "5"))
    out_path = Path(os.environ.get("OUT", "probe_speed.json"))

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {n_params/1e6:.1f}M param composite")

    problems = load_gsm8k_dev_questions(n=n_problems)
    print(f"loaded {len(problems)} prompts; mean prompt len = {sum(len(p['prompt_tokens']) for p in problems)/len(problems):.1f}")

    ar_kw = dict(temperature=0.8, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
    diff_kw = dict(diff_temperature=0.8, diff_top_p=0.9, diff_repetition_penalty=1.15)

    configs = {
        "composite_ar_128": (
            lambda m, p: gen_ar(m, p, max_new=128, **ar_kw),
            128, "AR head, 128 tokens, 128 forwards",
        ),
        "composite_diff_revise_128_16": (
            lambda m, p: diff_fill_full(m, p, 128, 16, diff_kw),
            128, "diff head, 128 tokens, 16 forwards",
        ),
        "composite_paired_64_64_16": (
            lambda m, p: gen_paired(m, p, k_ar=64, k_diff=64, n_diff_steps=16, **ar_kw, **diff_kw),
            64 + 64, "AR(64) + diff(64, 16 steps) = 80 forwards",
        ),
        "composite_paired_32_96_16": (
            lambda m, p: gen_paired(m, p, k_ar=32, k_diff=96, n_diff_steps=16, **ar_kw, **diff_kw),
            32 + 96, "AR(32) + diff(96, 16 steps) = 48 forwards",
        ),
        "composite_paired_16_112_16": (
            lambda m, p: gen_paired(m, p, k_ar=16, k_diff=112, n_diff_steps=16, **ar_kw, **diff_kw),
            16 + 112, "AR(16) + diff(112, 16 steps) = 32 forwards",
        ),
    }

    out = {
        "ckpt": str(ckpt),
        "device": device,
        "n_params": n_params,
        "n_problems": n_problems,
        "n_trials": n_trials,
        "decode": {"ar": ar_kw, "diff": diff_kw},
        "configs": {},
    }

    for name, (fn, target_tokens, note) in configs.items():
        print(f"\n[{name}] {note}")
        wall_samples = []
        tok_samples = []
        # Warm once outside loop to JIT/MPS-cache
        _ = fn(model, problems[0]["prompt_tokens"])
        _sync(device)
        for t in range(n_trials):
            for p in problems:
                prompt = p["prompt_tokens"]
                _sync(device)
                t0 = time.perf_counter()
                gen = fn(model, prompt)
                _sync(device)
                wall = time.perf_counter() - t0
                wall_samples.append(wall)
                tok_samples.append(len(gen))
        mean_wall = statistics.mean(wall_samples)
        sd_wall = statistics.stdev(wall_samples) if len(wall_samples) > 1 else 0.0
        mean_tok = statistics.mean(tok_samples)
        tps = mean_tok / mean_wall if mean_wall > 0 else float("nan")
        print(f"  mean wall_s = {mean_wall:.3f} ± {sd_wall:.3f}  "
              f"mean tokens = {mean_tok:.1f}  tokens/sec = {tps:.1f}")
        out["configs"][name] = {
            "note": note,
            "n_samples": len(wall_samples),
            "mean_wall_s": round(mean_wall, 4),
            "sd_wall_s": round(sd_wall, 4),
            "mean_tokens": round(mean_tok, 2),
            "tokens_per_sec": round(tps, 2),
            "target_tokens": target_tokens,
        }

    # Compute relative speedup vs composite_ar_128
    baseline = out["configs"]["composite_ar_128"]["tokens_per_sec"]
    for name, cfg in out["configs"].items():
        cfg["speedup_vs_ar"] = round(cfg["tokens_per_sec"] / max(baseline, 1e-9), 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}\nSummary (tokens/sec, speedup vs AR-128):")
    for name, cfg in out["configs"].items():
        print(f"  {name:34s} {cfg['tokens_per_sec']:8.1f}  ×{cfg['speedup_vs_ar']:5.2f}")


if __name__ == "__main__":
    main()
