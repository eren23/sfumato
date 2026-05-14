"""T0b — training loop for the toy continuous-flow LM.

Matches T0's discrete training recipe in MAX_STEPS / BATCH_SIZE / LR so
direct comparison vs `diff_only` is fair on tokens, optimizer steps, and
wall-clock.

Env knobs:
  SEED=0
  MAX_STEPS=3000
  BATCH_SIZE=8
  BLOCK_SIZE=256
  LR=6e-4
  D_MODEL=512 N_LAYERS=8 N_HEADS=8
  COND_PROMPT_LEN=64           # how many leading tokens of each window are
                                # treated as a clean conditioning prefix (the
                                # model attends to but does not predict
                                # velocity for). 0 = unconditional.
  OUT_DIR=e5/results/toy_flow_seed{SEED}
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from e5.model_flow import FlowConfig, FlowLM, flow_loss  # noqa: E402
from e5.data import load_gsm8k_train_tokens, GSM8KStreamingDataset  # noqa: E402


def env_int(k: str, d: int) -> int:
    return int(os.environ.get(k, str(d)))


def env_float(k: str, d: float) -> float:
    return float(os.environ.get(k, str(d)))


def env_str(k: str, d: str) -> str:
    return os.environ.get(k, d)


def get_lr(step: int, max_steps: int, peak_lr: float, warmup: int = 200, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def train_flow_one(
    out_dir: Path,
    *,
    seed: int,
    max_steps: int,
    batch_size: int,
    block_size: int,
    peak_lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    cond_prompt_len: int,
    tokens: np.ndarray,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = FlowConfig(
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
    )
    model = FlowLM(cfg).to(device)
    n_params = model.num_params()
    print(f"[flow] model: d={d_model} L={n_layers} H={n_heads} → {n_params/1e6:.1f}M params, cond_prompt_len={cond_prompt_len}")

    ds = GSM8KStreamingDataset(tokens, block_size=block_size, length=batch_size * max_steps, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    summary_path = out_dir / "summary.json"
    log_fh = open(log_path, "w")

    history: list[dict] = []
    t0 = time.time()
    last_loss = float("nan")

    for step in range(max_steps):
        try:
            window = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            window = next(loader_iter)
        window = window.to(device, non_blocking=True)
        # `window` shape: (B, block_size + 1). We split into prompt + gen.
        full = window[:, :block_size]
        if cond_prompt_len > 0:
            prompt_ids = full[:, :cond_prompt_len].contiguous()
            clean_ids = full[:, cond_prompt_len:].contiguous()
        else:
            prompt_ids = None
            clean_ids = full

        lr = get_lr(step, max_steps, peak_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            loss = flow_loss(model, clean_ids, prompt_ids=prompt_ids)
        last_loss = float(loss.detach())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0 or step == max_steps - 1:
            rec = {"step": step, "loss": last_loss, "lr": lr, "wallclock_s": round(time.time() - t0, 2)}
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            history.append(rec)
            if step % 200 == 0:
                print(f"[flow] step {step:5d}/{max_steps} loss={last_loss:.4f} lr={lr:.2e} wall={rec['wallclock_s']:.0f}s", flush=True)

    log_fh.close()
    wall_s = time.time() - t0

    ckpt_path = out_dir / "model.pt"
    torch.save({
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "variant": "flow_only",
        "n_params": n_params,
        "max_steps": max_steps,
        "cond_prompt_len": cond_prompt_len,
    }, ckpt_path)

    summary = {
        "variant": "flow_only",
        "n_params": n_params,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "block_size": block_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "peak_lr": peak_lr,
        "seed": seed,
        "cond_prompt_len": cond_prompt_len,
        "wall_s": round(wall_s, 1),
        "final_loss": last_loss,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[flow] done. {wall_s:.0f}s. wrote {ckpt_path} and {summary_path}")
    return summary


def main():
    seed = env_int("SEED", 0)
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    peak_lr = env_float("LR", 6e-4)
    d_model = env_int("D_MODEL", 512)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    cond_prompt_len = env_int("COND_PROMPT_LEN", 64)

    print(f"loading GSM8K-train tokens (gpt2, cot)…")
    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"  {len(tokens):,} tokens loaded")

    out_dir = Path(env_str("OUT_DIR", f"e5/results/toy_flow_seed{seed}"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    train_flow_one(
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


if __name__ == "__main__":
    main()
