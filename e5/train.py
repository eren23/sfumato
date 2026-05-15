"""T0 — joint training loop for composite AR + diffusion.

Env-driven variants:
  VARIANT=composite        → α: 1.0 → 0.5 over training; mix AR & DIFF batches
  VARIANT=ar_only          → α = 1.0; AR-only (B1 baseline)
  VARIANT=diff_only        → α = 0.0; DIFF-only (B2 baseline)
  VARIANT=paired_separate  → run two smaller models sequentially: AR then DIFF (B3)

Other env knobs:
  SEED=0
  MAX_STEPS=3000                    # ~3 epochs on the ~4M-token GSM8K-train stream at BS=8 BLOCK=256
  BATCH_SIZE=8
  BLOCK_SIZE=256
  LR=6e-4
  D_MODEL=512  N_LAYERS=8  N_HEADS=8   # composite / ar_only / diff_only
  D_MODEL_SMALL=384  N_LAYERS_SMALL=6  # B3 each (~30M)
  EVAL_EVERY=500
  N_EVAL=50
  OUT_DIR=e5/results/toy_composite_seed{SEED}_{VARIANT}
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

from e5.model_composite import (  # noqa: E402
    CompositeConfig,
    CompositeLM,
    MASK_TOKEN_ID,
    ar_loss,
    diff_loss,
)
from e5.data import (  # noqa: E402
    load_gsm8k_train_tokens,
    GSM8KStreamingDataset,
    make_ar_batch,
    make_diff_batch,
)


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


def alpha_schedule(step: int, max_steps: int, variant: str) -> float:
    """Returns the AR-loss weight α ∈ [0, 1] at this step.
      composite: 1.0 → 0.5 linear over training (warm-start AR, then mix)
      ar_only: 1.0 always
      diff_only: 0.0 always
      composite_fixed_<value>: constant α = <value>/100 (e.g. composite_fixed_30 → 0.30)

    Honored env override: ALPHA_OVERRIDE=0.3 forces α to that value across
    all composite-class variants (overrides the schedule).
    """
    env_override = os.environ.get("ALPHA_OVERRIDE")
    if variant == "ar_only":
        return 1.0
    if variant == "diff_only":
        return 0.0
    if variant.startswith("composite_fixed_"):
        try:
            pct = int(variant.split("_")[-1])
            return pct / 100.0
        except ValueError:
            pass
    if env_override is not None and variant in ("composite",) or variant.startswith("composite"):
        if env_override is not None:
            try:
                return float(env_override)
            except ValueError:
                pass
    if variant == "composite":
        progress = min(max(step / max(1, max_steps), 0.0), 1.0)
        return 1.0 - 0.5 * progress
    raise ValueError(f"unknown variant {variant!r}")


def train_one(
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
    eval_every: int,
    n_eval: int,
    tokens: np.ndarray,
    val_problems: list | None = None,
    sample_prompts: list | None = None,
    val_every: int = 1000,
    sample_every: int = 2000,
    tokenizer_for_samples=None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = CompositeConfig(
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
    )
    model = CompositeLM(cfg).to(device)
    n_params = model.num_params()
    print(f"[{variant}] model: d={d_model} L={n_layers} H={n_heads} → {n_params/1e6:.1f}M params")

    ds = GSM8KStreamingDataset(tokens, block_size=block_size, length=batch_size * max_steps, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2, drop_last=True)
    loader_iter = iter(loader)

    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    summary_path = out_dir / "summary.json"
    log_fh = open(log_path, "w")

    # W&B logging — skip silently if WANDB_API_KEY missing or wandb not installed.
    wb = None
    try:
        if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "disabled":
            import wandb as wb_lib  # noqa
            wb = wb_lib.init(
                project=os.environ.get("WANDB_PROJECT", "sfumato-e5"),
                name=os.environ.get("WANDB_RUN_NAME", f"{out_dir.parent.name}-{variant}-seed{seed}"),
                group=os.environ.get("WANDB_GROUP", out_dir.parent.name),
                config={
                    "variant": variant, "d_model": d_model, "n_layers": n_layers,
                    "n_heads": n_heads, "max_steps": max_steps, "batch_size": batch_size,
                    "block_size": block_size, "peak_lr": peak_lr, "seed": seed,
                    "n_params": n_params, "n_train_tokens": len(tokens),
                },
                reinit=True,
            )
            print(f"[wandb] init OK: {wb.url if wb else 'no url'}", flush=True)
    except Exception as e:
        print(f"[wandb] init failed: {e!s:.200}", flush=True)
        wb = None

    history = {"steps": [], "ar_loss": [], "diff_loss": [], "alpha": [], "lr": [], "wallclock_s": []}
    t0 = time.time()
    rng = np.random.default_rng(seed + 12345)

    last_ar = float("nan")
    last_diff = float("nan")

    for step in range(max_steps):
        try:
            window = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            window = next(loader_iter)
        window = window.to(device, non_blocking=True)

        alpha = alpha_schedule(step, max_steps, variant)
        # Per-step mode coin flip weighted by α. We compute exactly one
        # mode's loss per step (cheaper than two forwards). The schedule
        # determines distribution over steps.
        if variant == "ar_only":
            mode = "ar"
        elif variant == "diff_only":
            mode = "diff"
        else:
            mode = "ar" if rng.random() < alpha else "diff"

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
                idx_masked, idx_orig, masked = make_diff_batch(window, mask_token_id=MASK_TOKEN_ID)
                logits = model(idx_masked, mode="diff")
                loss = diff_loss(logits, idx_orig, masked)
                last_diff = float(loss.detach())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0 or step == max_steps - 1:
            rec = {
                "step": step,
                "mode": mode,
                "loss": float(loss.detach()),
                "alpha": alpha,
                "lr": lr,
                "wallclock_s": round(time.time() - t0, 2),
            }
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            history["steps"].append(step)
            history["ar_loss"].append(last_ar)
            history["diff_loss"].append(last_diff)
            history["alpha"].append(alpha)
            history["lr"].append(lr)
            history["wallclock_s"].append(rec["wallclock_s"])
            if wb is not None:
                try:
                    wb.log({
                        "step": step, "mode": mode, "loss": rec["loss"],
                        "ar_loss_last": last_ar, "diff_loss_last": last_diff,
                        "alpha": alpha, "lr": lr, "wallclock_s": rec["wallclock_s"],
                    }, step=step)
                except Exception:
                    pass
            if step % 200 == 0:
                print(f"[{variant}] step {step:5d}/{max_steps} mode={mode:4s} loss={float(loss.detach()):.4f} α={alpha:.2f} lr={lr:.2e} wall={rec['wallclock_s']:.0f}s", flush=True)

        # ---- Val NLL on held-out problems ----
        if val_problems and (step + 1) % val_every == 0 and wb is not None:
            try:
                model.train(False)
                with torch.no_grad():
                    total_nll = 0.0; cnt = 0
                    for prompt_ids, ans_ids in val_problems[:50]:
                        if len(prompt_ids) + len(ans_ids) + 1 > block_size: continue
                        full = prompt_ids + ans_ids
                        v_idx = torch.tensor([full], dtype=torch.long, device=device)
                        v_logits = model(v_idx, mode="ar").float()
                        ans_start = len(prompt_ids)
                        v_pred = v_logits[0, ans_start - 1 : ans_start - 1 + len(ans_ids), :]
                        v_target = torch.tensor(ans_ids, dtype=torch.long, device=device)
                        v_logp = torch.log_softmax(v_pred, dim=-1)
                        v_nll = -v_logp.gather(1, v_target.unsqueeze(1)).squeeze(1).sum().item()
                        total_nll += v_nll; cnt += len(ans_ids)
                    val_nll = total_nll / max(1, cnt)
                wb.log({"val_ar_nll": val_nll, "val_ar_perplexity": math.exp(val_nll)}, step=step)
                print(f"[{variant}] step {step:5d} val_ar_nll={val_nll:.4f}", flush=True)
            except Exception as e:
                print(f"[val warn] {e!s:.150}", flush=True)
            finally:
                model.train(True)

        # ---- Sample text generation logged to wandb ----
        if sample_prompts and (step + 1) % sample_every == 0 and wb is not None and tokenizer_for_samples is not None:
            try:
                model.train(False)
                rows = []
                with torch.no_grad():
                    for sp in sample_prompts[:3]:
                        ids = list(sp.get("prompt_tokens", sp) if isinstance(sp, dict) else sp)
                        gen = list(ids)
                        for _ in range(64):
                            ctx = torch.tensor([gen[-block_size:]], dtype=torch.long, device=device)
                            logits = model(ctx, mode="ar")[:, -1, :]
                            nxt = int(torch.argmax(logits, dim=-1).item())
                            if nxt == 50256: break
                            gen.append(nxt)
                        cont = tokenizer_for_samples.decode([t for t in gen[len(ids):] if t < 50257], skip_special_tokens=True)
                        prompt_text = sp["question"] if isinstance(sp, dict) and "question" in sp else "(prompt)"
                        rows.append([step, prompt_text[:120], cont[:300]])
                tbl = wb_lib.Table(columns=["step", "prompt", "completion"], data=rows)
                wb.log({"samples": tbl}, step=step)
                print(f"[{variant}] step {step:5d} logged {len(rows)} samples to wandb", flush=True)
            except Exception as e:
                print(f"[sample warn] {e!s:.150}", flush=True)
            finally:
                model.train(True)

    log_fh.close()
    wall_s = time.time() - t0

    # Save checkpoint (state dict only) for eval
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "variant": variant,
        "n_params": n_params,
        "max_steps": max_steps,
    }, ckpt_path)

    if wb is not None:
        try:
            wb.summary["wall_s"] = wall_s
            wb.summary["final_ar_loss"] = last_ar
            wb.summary["final_diff_loss"] = last_diff
            wb.finish()
        except Exception:
            pass

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
        "wall_s": round(wall_s, 1),
        "final_ar_loss": last_ar,
        "final_diff_loss": last_diff,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[{variant}] done. {wall_s:.0f}s. wrote {ckpt_path} and {summary_path}")
    return summary


def main():
    variant = env_str("VARIANT", "composite")
    seed = env_int("SEED", 0)
    max_steps = env_int("MAX_STEPS", 3000)
    batch_size = env_int("BATCH_SIZE", 8)
    block_size = env_int("BLOCK_SIZE", 256)
    peak_lr = env_float("LR", 6e-4)
    d_model = env_int("D_MODEL", 512)
    n_layers = env_int("N_LAYERS", 8)
    n_heads = env_int("N_HEADS", 8)
    d_model_small = env_int("D_MODEL_SMALL", 384)
    n_layers_small = env_int("N_LAYERS_SMALL", 6)
    eval_every = env_int("EVAL_EVERY", 500)
    n_eval = env_int("N_EVAL", 50)

    print(f"loading GSM8K-train tokens (gpt2, cot)…")
    tokens = load_gsm8k_train_tokens(include_reasoning=True)
    print(f"  {len(tokens):,} tokens loaded")

    base_out = Path(env_str("OUT_DIR", f"e5/results/toy_composite_seed{seed}_{variant}"))
    if not base_out.is_absolute():
        base_out = REPO_ROOT / base_out

    if variant == "paired_separate":
        # B3: train two smaller (≈30M each) models — one AR-only, one diff-only.
        # Each gets max_steps steps so total compute matches the single-model variants.
        sub_dirs = []
        for sub_variant in ("ar_only", "diff_only"):
            sub_dir = base_out / sub_variant
            print(f"\n=== B3 paired_separate: training {sub_variant} ({d_model_small}d × {n_layers_small}L) ===")
            train_one(
                variant=sub_variant,
                d_model=d_model_small,
                n_layers=n_layers_small,
                n_heads=max(1, n_heads // 2),
                out_dir=sub_dir,
                seed=seed,
                max_steps=max_steps,
                batch_size=batch_size,
                block_size=block_size,
                peak_lr=peak_lr,
                eval_every=eval_every,
                n_eval=n_eval,
                tokens=tokens,
            )
            sub_dirs.append(str(sub_dir))
        (base_out / "meta.json").write_text(json.dumps({
            "variant": "paired_separate",
            "sub_dirs": sub_dirs,
            "d_model_each": d_model_small,
            "n_layers_each": n_layers_small,
        }, indent=2))
        return

    train_one(
        variant=variant,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        out_dir=base_out,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        block_size=block_size,
        peak_lr=peak_lr,
        eval_every=eval_every,
        n_eval=n_eval,
        tokens=tokens,
    )


if __name__ == "__main__":
    main()
