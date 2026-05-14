"""T1 — driver: train ARFlowLM 4 variants and evaluate.

Mirrors run_toy.py's structure but with model_arflow.py (AR + continuous-flow
composite) replacing the discrete-mask composite.

Variants:
  composite   60M (or D_MODEL configured), shared backbone, both heads
  ar_only     same arch, AR loss only
  flow_only   same arch, flow loss only
  paired_sep  two smaller models (30M each), one AR-only one flow-only

Eval modes per the plan's decision table:
  composite_ar     : greedy AR from prompt
  composite_paired : AR first half, then composite-flow ODE second half
  ar_only          : greedy AR from prompt
  flow_only        : composite-flow ODE from prompt + max_new masks
  paired_separate  : AR-model AR + flow-model ODE

Headline comparison: composite_paired vs paired_separate.
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

from e5.model_arflow import ARFlowConfig, ARFlowLM  # noqa: E402
from e5.train_arflow import train_arflow_one  # noqa: E402
from e5.data import load_gsm8k_train_tokens, load_gsm8k_dev_questions  # noqa: E402


ANSWER_PAT = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")
NUM_PAT = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    m = ANSWER_PAT.search(text)
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    nums = NUM_PAT.findall(text)
    return nums[-1].replace(",", "").replace("$", "") if nums else None


def env_int(k, d): return int(os.environ.get(k, str(d)))
def env_float(k, d): return float(os.environ.get(k, str(d)))


def load_ckpt(ckpt_path: Path, device: str) -> ARFlowLM:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ARFlowConfig(**ck["config"])
    m = ARFlowLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m


@torch.no_grad()
def gen_ar(model: ARFlowLM, prompt: list[int], max_new: int = 128, eot: int = 50256) -> list[int]:
    device = next(model.parameters()).device
    bs = model.cfg.block_size
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = idx[:, -bs:]
        logits = model(ctx, mode="ar")[:, -1, :]
        nxt = int(torch.argmax(logits, dim=-1).item())
        if nxt == eot:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return idx[0].tolist()[len(prompt):]


@torch.no_grad()
def gen_flow_ode(
    model: ARFlowLM,
    prompt: list[int],
    max_new: int = 128,
    n_steps: int = 32,
) -> list[int]:
    """Integrate dz/dt = v_θ(z_t, t) from t=1 (noise) to t=0 (clean)."""
    device = next(model.parameters()).device
    cfg = model.cfg
    if len(prompt) + max_new > cfg.block_size:
        prompt = prompt[-(cfg.block_size - max_new):]
    B = 1
    z = torch.randn(B, max_new, cfg.d_model, device=device)
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    dt = 1.0 / n_steps
    for s in range(n_steps):
        t_val = 1.0 - s * dt
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)
        v = model(z, mode="flow", t=t, prompt_ids=prompt_t)
        z = z + dt * v.float()
    return model.decode_to_tokens(z)[0].tolist()


@torch.no_grad()
def gen_paired(
    ar_model: ARFlowLM,
    flow_model: ARFlowLM,
    prompt: list[int],
    k_ar: int = 64,
    k_flow: int = 64,
    n_ode: int = 24,
) -> list[int]:
    ar_part = gen_ar(ar_model, prompt, max_new=k_ar)
    extended = prompt + ar_part
    flow_part = gen_flow_ode(flow_model, extended, max_new=k_flow, n_steps=n_ode)
    return ar_part + flow_part


def eval_problems(gen_fn, problems, tokenizer) -> dict:
    results = []
    t0 = time.time()
    for i, p in enumerate(problems):
        gen = gen_fn(p["prompt_tokens"])
        # Filter out vocab-extension tokens (none here, but defensive).
        text = tokenizer.decode([t for t in gen if t < 50257], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred is not None and pred == p["gold"])
        results.append({"idx": p["idx"], "gold": p["gold"], "pred": pred, "correct": correct, "text": text})
        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  eval {i+1}/{len(problems)} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    acc = sum(r["correct"] for r in results) / max(1, len(results))
    return {"n": len(results), "n_correct": sum(r["correct"] for r in results), "accuracy": round(acc, 4),
            "wall_s": round(time.time() - t0, 1), "results": results}


def main():
    smoke = env_int("SMOKE", 0) == 1
    max_steps = env_int("MAX_STEPS", 100 if smoke else 3000)
    n_eval = env_int("N_EVAL", 5 if smoke else 50)
    seed = env_int("SEED", 0)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    peak_lr = env_float("LR", 6e-4)
    d_model = env_int("D_MODEL", 512)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    d_model_small = env_int("D_MODEL_SMALL", 384)
    n_layers_small = env_int("N_LAYERS_SMALL", 6)
    cond_prompt_len = env_int("COND_PROMPT_LEN", 64)
    max_new = env_int("MAX_NEW", 128)
    n_ode = env_int("N_ODE_STEPS", 8 if smoke else 32)

    base_out = REPO_ROOT / "e5" / "results" / f"t1_arflow_seed{seed}{'_smoke' if smoke else ''}"
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"=== T1 ARFlow driver: smoke={smoke} max_steps={max_steps} n_eval={n_eval} seed={seed} ===")
    print(f"=== output: {base_out} ===")

    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"loaded {len(tokens):,} tokens")

    train_results = {}
    ckpts = {}

    print(f"\n=== [1/4] composite ===")
    d = base_out / "composite"
    s = train_arflow_one("composite", d_model, n_layers, n_heads, d,
                         seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                         peak_lr=peak_lr, cond_prompt_len=cond_prompt_len, tokens=tokens)
    train_results["composite"] = s
    ckpts["composite"] = d / "model.pt"

    print(f"\n=== [2/4] ar_only ===")
    d = base_out / "ar_only"
    s = train_arflow_one("ar_only", d_model, n_layers, n_heads, d,
                         seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                         peak_lr=peak_lr, cond_prompt_len=cond_prompt_len, tokens=tokens)
    train_results["ar_only"] = s
    ckpts["ar_only"] = d / "model.pt"

    print(f"\n=== [3/4] flow_only ===")
    d = base_out / "flow_only"
    s = train_arflow_one("flow_only", d_model, n_layers, n_heads, d,
                         seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                         peak_lr=peak_lr, cond_prompt_len=cond_prompt_len, tokens=tokens)
    train_results["flow_only"] = s
    ckpts["flow_only"] = d / "model.pt"

    print(f"\n=== [4/4] paired_separate (B3): two {d_model_small}d × {n_layers_small}L sub-models ===")
    paired_dir = base_out / "paired"
    paired_dir.mkdir(exist_ok=True)
    s_ar = train_arflow_one("ar_only", d_model_small, n_layers_small, max(1, n_heads // 2), paired_dir / "ar_only",
                            seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                            peak_lr=peak_lr, cond_prompt_len=cond_prompt_len, tokens=tokens)
    s_fl = train_arflow_one("flow_only", d_model_small, n_layers_small, max(1, n_heads // 2), paired_dir / "flow_only",
                            seed=seed, max_steps=max_steps, batch_size=batch_size, block_size=block_size,
                            peak_lr=peak_lr, cond_prompt_len=cond_prompt_len, tokens=tokens)
    train_results["paired"] = {"ar_sub": s_ar, "flow_sub": s_fl,
                                "n_params_total": s_ar["n_params"] + s_fl["n_params"]}
    ckpts["paired_ar"] = paired_dir / "ar_only" / "model.pt"
    ckpts["paired_flow"] = paired_dir / "flow_only" / "model.pt"

    print(f"\n=== EVAL N={n_eval}, ODE steps={n_ode}, max_new={max_new} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    problems = load_gsm8k_dev_questions(n=n_eval)

    eval_results = {}

    # composite_ar
    print(f"\n[eval] composite_ar")
    m = load_ckpt(ckpts["composite"], device)
    eval_results["composite_ar"] = eval_problems(lambda pr: gen_ar(m, pr, max_new=max_new), problems, tok)
    eval_results["composite_ar"] = {k: v for k, v in eval_results["composite_ar"].items() if k != "results"}

    # composite_paired (self-paired: same model for both halves)
    print(f"\n[eval] composite_paired (self)")
    eval_results["composite_paired"] = eval_problems(
        lambda pr: gen_paired(m, m, pr, k_ar=max_new // 2, k_flow=max_new // 2, n_ode=n_ode),
        problems, tok)
    eval_results["composite_paired"] = {k: v for k, v in eval_results["composite_paired"].items() if k != "results"}
    del m

    # B1 ar_only
    print(f"\n[eval] ar_only (B1)")
    m = load_ckpt(ckpts["ar_only"], device)
    eval_results["ar_only"] = eval_problems(lambda pr: gen_ar(m, pr, max_new=max_new), problems, tok)
    eval_results["ar_only"] = {k: v for k, v in eval_results["ar_only"].items() if k != "results"}
    del m

    # B2 flow_only
    print(f"\n[eval] flow_only (B2)")
    m = load_ckpt(ckpts["flow_only"], device)
    eval_results["flow_only"] = eval_problems(lambda pr: gen_flow_ode(m, pr, max_new=max_new, n_steps=n_ode),
                                              problems, tok)
    eval_results["flow_only"] = {k: v for k, v in eval_results["flow_only"].items() if k != "results"}
    del m

    # B3 paired_separate
    print(f"\n[eval] paired_separate (B3)")
    ar_m = load_ckpt(ckpts["paired_ar"], device)
    fl_m = load_ckpt(ckpts["paired_flow"], device)
    eval_results["paired_separate"] = eval_problems(
        lambda pr: gen_paired(ar_m, fl_m, pr, k_ar=max_new // 2, k_flow=max_new // 2, n_ode=n_ode),
        problems, tok)
    eval_results["paired_separate"] = {k: v for k, v in eval_results["paired_separate"].items() if k != "results"}
    del ar_m, fl_m

    comp = eval_results["composite_paired"]["accuracy"]
    b3 = eval_results["paired_separate"]["accuracy"]
    delta_pp = (comp - b3) * 100
    if delta_pp >= 5: verdict = "strong_positive"
    elif delta_pp >= 0: verdict = "within_noise_positive"
    elif delta_pp >= -3: verdict = "inconclusive"
    else: verdict = "empirical_negative"

    summary = {
        "smoke": bool(smoke), "max_steps": max_steps, "n_eval": n_eval, "seed": seed,
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "d_model_small": d_model_small, "n_layers_small": n_layers_small,
        "cond_prompt_len": cond_prompt_len, "n_ode_steps": n_ode,
        "train": train_results, "eval": eval_results,
        "headline": {"composite_paired_acc": comp, "b3_paired_separate_acc": b3,
                     "delta_pp": round(delta_pp, 2), "verdict": verdict},
    }
    (base_out / "t1_arflow_summary.json").write_text(json.dumps(summary, indent=2))

    md = f"""# T1 — toy AR + continuous-flow composite, seed={seed}{' (SMOKE)' if smoke else ''}

## Headline

| | accuracy | n_params |
|--|---|---|
| **composite (paired)** | **{comp*100:.1f}%** | {train_results['composite']['n_params']/1e6:.1f}M |
| **B3 paired_separate** | **{b3*100:.1f}%** | {train_results['paired']['n_params_total']/1e6:.1f}M |
| **Δ** | **{delta_pp:+.2f}pp** | |

Verdict: **{verdict}**

## All modes

| variant | mode | acc |
|---|---|---|
| composite | ar | {eval_results['composite_ar']['accuracy']*100:.1f}% |
| composite | paired (self) | {comp*100:.1f}% |
| ar_only | ar | {eval_results['ar_only']['accuracy']*100:.1f}% |
| flow_only | flow | {eval_results['flow_only']['accuracy']*100:.1f}% |
| paired_separate | paired (cross) | {b3*100:.1f}% |

## Caveats

- 60M-class model on ~4M GSM8K tokens, 3k steps, single seed.
- Continuous-flow decoding via cosine-NN lookup (weaker than ELF's shared-weight decoder).
- ODE: Euler with {n_ode} steps. Higher-order or more steps may lift acc.
"""
    (base_out / "t1_arflow_summary.md").write_text(md)
    print(f"\n=== DONE ===")
    print(f"composite[paired]={comp*100:.1f}% vs B3={b3*100:.1f}% ({delta_pp:+.2f}pp, {verdict})")


if __name__ == "__main__":
    main()
