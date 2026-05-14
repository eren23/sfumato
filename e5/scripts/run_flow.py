"""T0b — toy continuous-flow LM driver: train, ODE-sample, GSM8K-dev eval.

Single Crucible entry point for the parallel flow run. Trains one FlowLM
and evaluates it on GSM8K-dev N=50 via ODE sampling.

Env knobs (defaults match T0's train recipe for fair comparison):
  SEED=0
  MAX_STEPS=3000
  BATCH_SIZE=8
  BLOCK_SIZE=256
  LR=6e-4
  D_MODEL=512 N_LAYERS=8 N_HEADS=8
  COND_PROMPT_LEN=64       # how many tokens of the GSM8K-dev prompt go as clean conditioning
  N_EVAL=50
  MAX_NEW=128              # tokens to generate per problem
  N_ODE_STEPS=32           # ODE Euler steps at inference
  SMOKE=0                  # if 1: MAX_STEPS=100 N_EVAL=5 N_ODE_STEPS=8

Outputs:
  e5/results/toy_flow_seed{SEED}{_smoke}/model.pt
  e5/results/toy_flow_seed{SEED}{_smoke}/train_log.jsonl
  e5/results/toy_flow_seed{SEED}{_smoke}/eval_results.json
  e5/results/toy_flow_seed{SEED}{_smoke}/summary.md
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_flow import FlowConfig, FlowLM  # noqa: E402
from e5.data import load_gsm8k_train_tokens, load_gsm8k_dev_questions  # noqa: E402
from e5.train_flow import train_flow_one  # noqa: E402


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


def env_int(k: str, d: int) -> int:
    return int(os.environ.get(k, str(d)))


def env_float(k: str, d: float) -> float:
    return float(os.environ.get(k, str(d)))


@torch.no_grad()
def ode_sample(
    model: FlowLM,
    prompt_ids: list[int],
    max_new: int,
    n_ode_steps: int = 32,
    device: str = "cuda",
) -> list[int]:
    """Integrate dz/dt = -v_θ(z, t) from t=1 (noise) to t=0 (clean).

    With training convention z_t = (1-t)·z_clean + t·z_noise and target
    velocity v* = z_clean - z_noise, we have dz_t/dt = -v* (because
    z increases noise as t→1). So at inference we step:
        z_{t - dt} = z_t + dt · v_θ(z_t, t)   (Euler step toward clean)
    """
    cfg = model.cfg
    D = cfg.d_model
    total = len(prompt_ids) + max_new
    assert total <= cfg.block_size, f"context {total} > block_size {cfg.block_size}"
    B = 1
    z = torch.randn(B, max_new, D, device=device)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    dt = 1.0 / n_ode_steps
    for step in range(n_ode_steps):
        t_val = 1.0 - step * dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v = model(z, t, prompt_ids=prompt_t)  # (B, max_new, D)
        z = z + dt * v.float()  # move toward clean

    tokens = model.decode_to_tokens(z)
    return tokens[0].tolist()


def main():
    smoke = env_int("SMOKE", 0) == 1
    seed = env_int("SEED", 0)
    max_steps = env_int("MAX_STEPS", 100 if smoke else 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    peak_lr = env_float("LR", 6e-4)
    d_model = env_int("D_MODEL", 512)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    cond_prompt_len = env_int("COND_PROMPT_LEN", 64)
    n_eval = env_int("N_EVAL", 5 if smoke else 50)
    max_new = env_int("MAX_NEW", 128)
    n_ode_steps = env_int("N_ODE_STEPS", 8 if smoke else 32)

    out_dir = REPO_ROOT / "e5" / "results" / f"toy_flow_seed{seed}{'_smoke' if smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== T0b flow driver: smoke={smoke} max_steps={max_steps} n_eval={n_eval} seed={seed} ===")
    print(f"=== output: {out_dir} ===")

    print(f"\nloading GSM8K-train tokens (gpt2, cot)…")
    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"  {len(tokens):,} tokens loaded")

    print(f"\n=== training flow model ===")
    train_summary = train_flow_one(
        out_dir=out_dir,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        block_size=block_size,
        peak_lr=peak_lr,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        cond_prompt_len=cond_prompt_len,
        tokens=tokens,
    )

    print(f"\n=== evaluation: GSM8K-dev N={n_eval}, ODE steps={n_ode_steps}, max_new={max_new} ===")

    # Load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    cfg = FlowConfig(**ckpt["config"])
    model = FlowLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.train(False)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_eval)

    results = []
    t_eval0 = time.time()
    for i, p in enumerate(problems):
        prompt = p["prompt_tokens"]
        # Ensure prompt fits.
        if len(prompt) + max_new > cfg.block_size:
            prompt = prompt[-(cfg.block_size - max_new):]
        gen_ids = ode_sample(model, prompt, max_new=max_new, n_ode_steps=n_ode_steps, device=device)
        text = tok.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred is not None and pred == p["gold"])
        results.append({"idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": correct, "text": text})
        if (i + 1) % 10 == 0:
            el = time.time() - t_eval0
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [flow_eval] {i+1}/{n_eval} acc={acc:.3f} ({el:.0f}s)", flush=True)

    acc = sum(r["correct"] for r in results) / max(1, len(results))
    eval_summary = {
        "mode": "flow_ode",
        "n": len(results),
        "n_correct": sum(r["correct"] for r in results),
        "accuracy": round(acc, 4),
        "wall_s": round(time.time() - t_eval0, 1),
        "n_ode_steps": n_ode_steps,
        "max_new": max_new,
        "cond_prompt_len": cond_prompt_len,
    }
    (out_dir / "eval_results.json").write_text(json.dumps({**eval_summary, "results": results}, indent=2))

    summary = {
        "smoke": bool(smoke),
        "train": train_summary,
        "eval": eval_summary,
    }
    (out_dir / "toy_flow_summary.json").write_text(json.dumps(summary, indent=2))

    md = f"""# T0b — toy continuous-flow LM, seed={seed}{' (SMOKE)' if smoke else ''}

## Headline

| metric | value |
|---|---|
| GSM8K-dev accuracy | **{acc*100:.1f}%** ({sum(r["correct"] for r in results)}/{len(results)}) |
| Final flow MSE loss | {train_summary['final_loss']:.4f} |
| Params | {train_summary['n_params']/1e6:.1f}M |
| Train wall | {train_summary['wall_s']:.0f}s |
| Eval wall | {eval_summary['wall_s']:.0f}s |
| ODE steps | {n_ode_steps} |
| Cond prompt len | {cond_prompt_len} |

## Comparison points

- T0 diff_only baseline (discrete mask diffusion at same scale): _filled by T0 run_
- T0 composite (joint AR+discrete-diff): _filled by T0 run_

If `flow ≥ T0.diff_only + 3pp` → continuous-flow design point worth pursuing for the composite redesign.

## Caveats

- 60M @ ~4M GSM8K tokens, 3000 steps, single seed. Same caveats as T0.
- ODE sampling is Euler-only with {n_ode_steps} steps; higher-order (Heun) or more steps could lift acc but isn't tested here.
- Decoding is nearest-neighbour in the embedding table; ELF uses a "shared-weight network" for this — our toy uses cosine similarity, which is a strictly weaker decoder.
"""
    (out_dir / "toy_flow_summary.md").write_text(md)

    print(f"\n=== DONE ===")
    print(f"flow GSM8K-dev acc: {acc*100:.1f}% ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"wrote {out_dir/'toy_flow_summary.json'}")
    print(f"wrote {out_dir/'toy_flow_summary.md'}")


if __name__ == "__main__":
    main()
