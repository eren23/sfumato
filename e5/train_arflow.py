"""T1 — training loop for ARFlowLM.

Variants (set via VARIANT env):
  composite   : α: 1.0 → 0.5 linear over training; alternate AR/flow batches
  ar_only     : α = 1.0 always
  flow_only   : α = 0.0 always
  paired_sep  : two smaller models trained separately (one AR, one flow)

Env knobs match T0's recipe so direct comparison is fair.
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

from e5.model_arflow import ARFlowConfig, ARFlowLM, ar_loss, flow_loss  # noqa: E402
from e5.data import GSM8KStreamingDataset, make_ar_batch  # noqa: E402


def get_lr(step: int, max_steps: int, peak_lr: float, warmup: int = 200, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def alpha_schedule(step: int, max_steps: int, variant: str) -> float:
    if variant == "ar_only":
        return 1.0
    if variant == "flow_only":
        return 0.0
    if variant == "composite":
        progress = min(max(step / max(1, max_steps), 0.0), 1.0)
        return 1.0 - 0.5 * progress
    raise ValueError(variant)


def train_arflow_one(
    variant: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    out_dir: Path,
    *,
    seed: int,
    max_steps: int,
    batch_size: int,
    block_size: int,
    peak_lr: float,
    cond_prompt_len: int,
    tokens: np.ndarray,
) -> dict:
    """Train one ARFlowLM under the given variant. cond_prompt_len controls
    the prompt prefix length for the flow path (the AR path ignores it).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = ARFlowConfig(
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
    )
    model = ARFlowLM(cfg).to(device)
    n_params = model.num_params()
    print(f"[{variant}] arflow: d={d_model} L={n_layers} H={n_heads} → {n_params/1e6:.1f}M params, cond_len={cond_prompt_len}")

    ds = GSM8KStreamingDataset(tokens, block_size=block_size, length=batch_size * max_steps, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_dir / "train_log.jsonl", "w")

    t0 = time.time()
    rng = np.random.default_rng(seed + 12345)
    last_ar = float("nan")
    last_flow = float("nan")

    for step in range(max_steps):
        try:
            window = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            window = next(loader_iter)
        window = window.to(device, non_blocking=True)  # (B, block_size+1)

        alpha = alpha_schedule(step, max_steps, variant)
        if variant == "ar_only":
            mode = "ar"
        elif variant == "flow_only":
            mode = "flow"
        else:
            mode = "ar" if rng.random() < alpha else "flow"

        lr = get_lr(step, max_steps, peak_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            if mode == "ar":
                idx, targets = make_ar_batch(window)
                logits = model(idx, mode="ar")
                loss = ar_loss(logits, targets)
                last_ar = float(loss.detach())
            else:
                full = window[:, :block_size]
                if cond_prompt_len > 0:
                    prompt_ids = full[:, :cond_prompt_len].contiguous()
                    clean_ids = full[:, cond_prompt_len:].contiguous()
                else:
                    prompt_ids = None
                    clean_ids = full
                loss = flow_loss(model, clean_ids, prompt_ids=prompt_ids)
                last_flow = float(loss.detach())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0 or step == max_steps - 1:
            rec = {"step": step, "mode": mode, "loss": float(loss.detach()),
                   "alpha": alpha, "lr": lr, "wallclock_s": round(time.time() - t0, 2)}
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            if step % 200 == 0:
                print(f"[{variant}] step {step:5d}/{max_steps} mode={mode:4s} loss={float(loss.detach()):.4f} α={alpha:.2f} lr={lr:.2e} wall={rec['wallclock_s']:.0f}s", flush=True)

    log_fh.close()
    wall_s = time.time() - t0

    ckpt = out_dir / "model.pt"
    torch.save({
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "variant": variant,
        "n_params": n_params,
        "cond_prompt_len": cond_prompt_len,
        "max_steps": max_steps,
    }, ckpt)

    summary = {
        "variant": variant,
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
        "final_ar_loss": last_ar,
        "final_flow_loss": last_flow,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{variant}] done. {wall_s:.0f}s. wrote {ckpt}")
    return summary
