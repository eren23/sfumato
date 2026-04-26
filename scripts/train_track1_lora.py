"""Track 1 LoRA: prefix-robustness fine-tune for LLaDA-8B-Instruct.

Trains a LoRA adapter on `eren23/sfumato-prefix-robust-gsm8k` (or the local
parquet fallback at `e2/data/prefix_robust_dataset.parquet`) so that LLaDA's
GSM8K accuracy stops collapsing under different plan-prefix formats.

Crucible BYO-trainer contract (env-var driven, prints stdout markers):

    step:{step}/{total} train_loss:{loss}
    step:{step}/{total} val_loss:{loss} val_bpb:{tokens_seen}
    Serialized model {path} {bytes}

LLaDA training objective (from ML-GSAI/SMDM pretrain/train_mdm.py):
sample t ~ U(0,1), p_mask = (1-eps)*t + eps, mask response tokens with prob
p_mask, swap them for MASK_ID=126336, and CE on the masked positions only,
re-weighted by 1/p_mask. Prompt tokens are never masked.

Pinned: transformers==4.46.3 (LLaDA breaks on transformers 5.x).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants from LLaDA modeling code
# ----------------------------------------------------------------------------
MASK_ID = 126336  # LLaDA <mask> token id
EPS = 1e-3


# ----------------------------------------------------------------------------
# Env helpers (mirrors examples/huggingface_finetune/train.py style)
# ----------------------------------------------------------------------------
def env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_NAME = env("MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct")
DATASET_REPO = env("HF_DATASET_ID", "eren23/sfumato-prefix-robust-gsm8k")
LOCAL_PARQUET = env(
    "LOCAL_PARQUET",
    str(Path(__file__).resolve().parents[1] / "e2" / "data" / "prefix_robust_dataset.parquet"),
)

LORA_R = env_int("LORA_R", 16)
LORA_ALPHA = env_int("LORA_ALPHA", 32)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)
LR = env_float("LR", 1e-4)
EPOCHS = env_int("EPOCHS", 3)
WARMUP_STEPS = env_int("WARMUP_STEPS", 500)
BATCH_SIZE = env_int("BATCH_SIZE", 1)
GRAD_ACCUM = env_int("GRAD_ACCUM", 4)
SEED = env_int("SEED", 42)
MAX_LENGTH = env_int("MAX_LENGTH", 512)
LOG_INTERVAL = env_int("LOG_INTERVAL", 50)
EVAL_INTERVAL = env_int("EVAL_INTERVAL", 500)
EVAL_BATCHES = env_int("EVAL_BATCHES", 32)

OUTPUT_REPO = env("HF_OUTPUT_REPO", "eren23/sfumato-llada-prefix-robust")
SAVE_DIR = Path(env("SAVE_DIR", "/tmp/track1_lora"))
RESUME_FROM = env("RESUME_FROM", None)
PUSH = env_bool("PUSH_TO_HUB", True)
WANDB_RUN_NAME = env("WANDB_RUN_NAME", "track1-prefix-robust")

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ----------------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------------
def build_tokenize_fn(tokenizer, max_length: int):
    """Return a fn that maps a dataset row -> {input_ids, prompt_len}.

    Schema expectations (dataset created by another script):
      - "prompt": question + prefix (the part the model conditions on)
      - "response": ground-truth answer (the part to denoise)

    Falls back to ("question" + "prefix", "answer") if those keys are present.
    Pads to fixed `max_length`. Truncates from the right.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # LLaDA tokenizer typically lacks an explicit pad token; reuse eos.
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id

    def _resolve(row):
        if "prompt" in row and "response" in row:
            return row["prompt"], row["response"]
        if "question" in row and "answer" in row:
            prefix = row.get("prefix", "") or ""
            return row["question"] + prefix, row["answer"]
        raise KeyError(
            f"row needs (prompt,response) or (question,answer); keys={list(row.keys())}"
        )

    def tokenize_fn(row):
        prompt_text, response_text = _resolve(row)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        # Append EOS so the model learns to terminate.
        eos = tokenizer.eos_token_id
        if eos is not None and (not response_ids or response_ids[-1] != eos):
            response_ids = response_ids + [eos]

        # Truncate / pad
        if len(prompt_ids) >= max_length - 1:
            # Pathological — keep only the last bit of the prompt and one response slot.
            prompt_ids = prompt_ids[: max_length - 1]
            response_ids = response_ids[:1]
        avail = max_length - len(prompt_ids)
        response_ids = response_ids[:avail]
        ids = prompt_ids + response_ids
        prompt_len = len(prompt_ids)
        # Right-pad
        pad_n = max_length - len(ids)
        if pad_n > 0:
            ids = ids + [pad_id] * pad_n
        return {
            "input_ids": ids,
            "prompt_len": prompt_len,
        }

    return tokenize_fn


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def compute_llada_loss(
    model,
    batch,
    *,
    p_mask_low: float = EPS,
    p_mask_high: float = 1.0 - EPS,
    span_start_key: str = "prompt_len",
    span_end=None,
):
    """Per-example masked-diffusion CE (LLaDA objective).

    Args:
      batch: dict with `input_ids` (B,L) and `prompt_len` (B,) on device.
      p_mask_low / p_mask_high: range for the per-example mask probability.
        Track 1 uses U(eps, 1-eps); Track 2 narrows this.
      span_start_key: name of the per-example tensor giving the response start.
      span_end: optional per-example end tensor (used by Track 2 for the
        answer-only span). If None, runs to L (full sequence length).
    """
    import torch
    import torch.nn.functional as F

    input_ids = batch["input_ids"]
    spans_start = batch[span_start_key]
    B, L = input_ids.shape
    device = input_ids.device

    # We rebuild a noisy batch tensor and per-example mask for a single fwd pass.
    noisy = input_ids.clone()
    target_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    p_mask_per = torch.zeros(B, device=device)
    for i in range(B):
        t = torch.rand(1, device=device).item()
        p = (p_mask_high - p_mask_low) * t + p_mask_low
        p_mask_per[i] = p
        s = int(spans_start[i].item())
        e = int(span_end[i].item()) if span_end is not None else L
        if e <= s:
            continue
        m = torch.rand(e - s, device=device) < p
        target_mask[i, s:e] = m
        noisy[i, s:e][m] = MASK_ID

    # Forward pass
    out = model(noisy)
    logits = out.logits if hasattr(out, "logits") else out[0]

    # Loss: only on masked positions, weighted by 1/p_mask, normalized
    # by batch_size and seq_len (matches SMDM ref).
    if not target_mask.any():
        zero = torch.zeros((), device=device, dtype=logits.dtype, requires_grad=True)
        return zero, p_mask_per.mean().item()

    # Compute per-position CE only at masked positions.
    flat_logits = logits[target_mask]  # (M, V)
    flat_targets = input_ids[target_mask]  # (M,)
    ce = F.cross_entropy(flat_logits.float(), flat_targets, reduction="none")
    # Weight each token by 1/p_mask of its example.
    rows = target_mask.nonzero(as_tuple=False)[:, 0]  # (M,)
    weights = 1.0 / p_mask_per[rows]
    loss = (ce * weights).sum() / B / L
    return loss, p_mask_per.mean().item()


# ----------------------------------------------------------------------------
# Validation pass (don't call it `eval` — that triggers spurious linters).
# ----------------------------------------------------------------------------
def run_validation(model, loader, max_batches: int) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = compute_llada_loss(model, batch)
            losses.append(float(loss.detach()))
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main() -> int:
    try:
        import torch
        import torch.nn.functional as F  # noqa: F401  (used by compute_llada_loss)
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 2

    # Reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA detected — this script targets RTX 4090.", file=sys.stderr)

    # ------------------------------------------------------------------ wandb
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb

            _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "sfumato"),
                name=WANDB_RUN_NAME,
                config={
                    "model": MODEL_NAME,
                    "dataset": DATASET_REPO,
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lr": LR,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "grad_accum": GRAD_ACCUM,
                    "max_length": MAX_LENGTH,
                    "track": "track1-prefix-robust",
                    "seed": SEED,
                },
            )
            wandb = _wandb
        except Exception as exc:
            print(f"wandb init failed: {exc}", file=sys.stderr)
            wandb = None

    # ------------------------------------------------------------------ model
    print(f"Loading base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    # Activation checkpointing — required to fit LLaDA-8B + LoRA in 24 GB.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as exc:
            print(f"gradient_checkpointing_enable failed: {exc}", file=sys.stderr)
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    # ------------------------------------------------------------------- lora
    if RESUME_FROM:
        print(f"Resuming LoRA adapter from {RESUME_FROM}", flush=True)
        model = PeftModel.from_pretrained(model, RESUME_FROM, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------- dataset
    train_ds = None
    val_ds = None
    try:
        print(f"Loading dataset: {DATASET_REPO}", flush=True)
        ds = load_dataset(DATASET_REPO)
        train_ds = ds["train"]
        val_ds = ds.get("validation") or ds.get("test")
    except Exception as exc:
        print(f"HF load failed ({exc}); falling back to {LOCAL_PARQUET}", flush=True)
        if not Path(LOCAL_PARQUET).exists():
            print(f"ERROR: local parquet missing: {LOCAL_PARQUET}", file=sys.stderr)
            return 2
        full = Dataset.from_parquet(LOCAL_PARQUET)
        # Hold out 5% for val (deterministic via seed).
        split = full.train_test_split(test_size=0.05, seed=SEED)
        train_ds = split["train"]
        val_ds = split["test"]

    tokenize_fn = build_tokenize_fn(tokenizer, MAX_LENGTH)
    print(f"Tokenizing train ({len(train_ds)} rows)...", flush=True)
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    if val_ds is not None:
        print(f"Tokenizing val ({len(val_ds)} rows)...", flush=True)
        val_ds = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names)

    train_ds.set_format(type="torch", columns=["input_ids", "prompt_len"])
    if val_ds is not None:
        val_ds.set_format(type="torch", columns=["input_ids", "prompt_len"])

    # ----------------------------------------------------------------- loader
    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
            "prompt_len": torch.tensor(
                [int(b["prompt_len"]) for b in batch], dtype=torch.long, device=device
            ),
        }

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        if val_ds is not None
        else None
    )

    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    print(
        f"Training: epochs={EPOCHS} steps/epoch={steps_per_epoch} total={total_steps}",
        flush=True,
    )

    # -------------------------------------------------------------- optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=True,
    )

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # -------------------------------------------------------------- main loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    global_step = 0
    tokens_seen = 0
    start_time = time.monotonic()

    for epoch in range(EPOCHS):
        model.train()
        accum_loss = 0.0
        accum_p = 0.0
        optim.zero_grad(set_to_none=True)

        for micro_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, p_mask_avg = compute_llada_loss(model, batch)
            (loss / GRAD_ACCUM).backward()
            accum_loss += float(loss.detach()) / GRAD_ACCUM
            accum_p += p_mask_avg / GRAD_ACCUM
            tokens_seen += int(batch["input_ids"].numel())

            if (micro_idx + 1) % GRAD_ACCUM == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % LOG_INTERVAL == 0 or global_step == 1:
                    print(
                        f"step:{global_step}/{total_steps} train_loss:{accum_loss:.4f}",
                        flush=True,
                    )
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": accum_loss,
                            "train/p_mask_avg": accum_p,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": float(grad_norm),
                            "train/tokens_seen": tokens_seen,
                            "epoch": epoch + (micro_idx + 1) / max(1, len(train_loader)),
                        },
                        step=global_step,
                    )
                accum_loss = 0.0
                accum_p = 0.0

                # ------------------------------------------------------ validation
                if (
                    val_loader is not None
                    and EVAL_INTERVAL > 0
                    and global_step % EVAL_INTERVAL == 0
                ):
                    val_loss = run_validation(model, val_loader, EVAL_BATCHES)
                    print(
                        f"step:{global_step}/{total_steps} val_loss:{val_loss:.4f} val_bpb:{tokens_seen}",
                        flush=True,
                    )
                    if wandb is not None:
                        wandb.log({"val/loss": val_loss}, step=global_step)
                    model.train()

        # -------------------------------------------------------- end-of-epoch save
        epoch_dir = SAVE_DIR / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        print(f"[epoch {epoch + 1}] saved adapter -> {epoch_dir}", flush=True)

    # ------------------------------------------------------------------- final validation
    if val_loader is not None:
        val_loss = run_validation(model, val_loader, EVAL_BATCHES)
        print(
            f"step:{global_step}/{total_steps} val_loss:{val_loss:.4f} val_bpb:{tokens_seen}",
            flush=True,
        )
        if wandb is not None:
            wandb.log({"val/loss_final": val_loss}, step=global_step)

    # --------------------------------------------------------------- final save
    model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    total_bytes = sum(p.stat().st_size for p in SAVE_DIR.rglob("*") if p.is_file())
    print(f"Serialized model {SAVE_DIR} {total_bytes} bytes", flush=True)

    # ----------------------------------------------------------------- push
    if PUSH:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            print("PUSH_TO_HUB=true but HF_TOKEN not set; skipping push.", file=sys.stderr)
        else:
            try:
                model.push_to_hub(
                    OUTPUT_REPO,
                    token=token,
                    private=True,
                    commit_message=f"Track 1 LoRA seed={SEED}",
                )
                tokenizer.push_to_hub(OUTPUT_REPO, token=token, private=True)
                print(f"pushed adapter -> {OUTPUT_REPO}", flush=True)
            except Exception as exc:
                print(f"push_to_hub failed: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - start_time
    print(f"done in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
