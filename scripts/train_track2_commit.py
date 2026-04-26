"""Track 2 LoRA: consensus-distillation fine-tune for LLaDA-8B-Instruct.

Trains a SMALLER LoRA (FFN of last 8 layers only) on cmaj b=5 majority-vote
outputs that disagree with greedy. Dataset: `eren23/sfumato-consensus-gsm8k`.

Why a smaller, FFN-only adapter? The hypothesis is that consensus / commitment
is encoded near the output side of the network — patching only the late FFN
keeps the adapter cheap (~1-2M params) and avoids broad behavioral drift.

Differences vs Track 1:
  - LORA_R=8, LORA_ALPHA=16
  - target_modules = FFN only ("gate_proj", "up_proj", "down_proj")
  - layers_to_transform = [24..31]   (last 8 of LLaDA-8B's 32 layers)
  - Mask probability narrowed: U(0.3, 0.9)  ("consensus matters when many
    tokens are uncertain")
  - Loss is computed on the **answer span only** — delimited by "Answer:" or
    "#### " inside `majority_answer`. We concatenate (question + a randomly
    chosen branch text) as the prompt and train to denoise the trailing
    majority-answer tokens.

Crucible BYO-trainer contract (env-var driven, prints stdout markers):

    step:{step}/{total} train_loss:{loss}
    step:{step}/{total} val_loss:{loss} val_bpb:{tokens_seen}
    Serialized model {path} {bytes}

Pinned: transformers==4.46.3 (LLaDA breaks on transformers 5.x).
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

# Reuse the loss + helpers from Track 1.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_track1_lora import (  # type: ignore  # noqa: E402
    EPS,
    MASK_ID,
    compute_llada_loss,
    env,
    env_bool,
    env_float,
    env_int,
    run_validation,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_NAME = env("MODEL_NAME", "GSAI-ML/LLaDA-8B-Instruct")
DATASET_REPO = env("HF_DATASET_ID", "eren23/sfumato-consensus-gsm8k")
LOCAL_PARQUET = env(
    "LOCAL_PARQUET",
    str(Path(__file__).resolve().parents[1] / "e2" / "data" / "consensus_dataset.parquet"),
)

LORA_R = env_int("LORA_R", 8)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)
LR = env_float("LR", 5e-5)
EPOCHS = env_int("EPOCHS", 2)
WARMUP_STEPS = env_int("WARMUP_STEPS", 200)
BATCH_SIZE = env_int("BATCH_SIZE", 2)
GRAD_ACCUM = env_int("GRAD_ACCUM", 2)
SEED = env_int("SEED", 42)
MAX_LENGTH = env_int("MAX_LENGTH", 512)
LOG_INTERVAL = env_int("LOG_INTERVAL", 50)
EVAL_INTERVAL = env_int("EVAL_INTERVAL", 500)
EVAL_BATCHES = env_int("EVAL_BATCHES", 32)

P_MASK_LOW = env_float("P_MASK_LOW", 0.3)
P_MASK_HIGH = env_float("P_MASK_HIGH", 0.9)

OUTPUT_REPO = env("HF_OUTPUT_REPO", "eren23/sfumato-llada-commit")
SAVE_DIR = Path(env("SAVE_DIR", "/tmp/track2_commit"))
RESUME_FROM = env("RESUME_FROM", None)
PUSH = env_bool("PUSH_TO_HUB", True)
WANDB_RUN_NAME = env("WANDB_RUN_NAME", "track2-commit")

# FFN-only late-layer adaptation
LORA_TARGETS = ["gate_proj", "up_proj", "down_proj"]
LORA_LAYERS_TO_TRANSFORM = list(range(24, 32))


# ----------------------------------------------------------------------------
# Dataset shaping (consensus dataset → answer-only-span supervision)
# ----------------------------------------------------------------------------
ANSWER_DELIMS = ("Answer:", "#### ")


def _find_answer_start(text: str):
    """Return char offset of the first answer-delim hit, or None."""
    best = None
    for d in ANSWER_DELIMS:
        i = text.find(d)
        if i >= 0 and (best is None or i < best):
            best = i
    return best


def build_track2_tokenize_fn(tokenizer, max_length: int, rng: random.Random):
    """Tokenize a consensus row into (input_ids, prompt_len, answer_end).

    Schema (created by another script, eren23/sfumato-consensus-gsm8k):
      - "question": GSM8K question
      - "branches": list[str] (b=5) of full branch generations
      - "majority_answer": str — the consensus answer text (containing one of
        ANSWER_DELIMS)

    We build:
      prompt    = question + "\n" + (random branch text up to its answer mark)
      response  = majority_answer (only the answer-delim-onward portion is
                  marked as the supervised span; tokens before that delim are
                  considered extra context, not denoised)

    Output keys:
      input_ids   : list[int] padded to max_length
      prompt_len  : int — start of the answer-only span
      answer_end  : int — end (exclusive) of the answer-only span
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id
    eos = tokenizer.eos_token_id

    def tokenize_fn(row):
        question = row.get("question", "") or ""
        majority = row.get("majority_answer", "") or ""
        branches = row.get("branches") or []
        branch = rng.choice(branches) if branches else ""

        # Locate answer span inside majority_answer (fall back to whole text)
        a_off = _find_answer_start(majority)
        if a_off is None:
            answer_text = majority
        else:
            answer_text = majority[a_off:]

        # Prompt = question + branch context (lets the model condition on the
        # noisy branch when committing to consensus)
        prompt_text = question
        if branch:
            prompt_text = prompt_text + "\n\nBranch:\n" + branch
        prompt_text = prompt_text + "\n\nAnswer:\n"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if eos is not None and (not answer_ids or answer_ids[-1] != eos):
            answer_ids = answer_ids + [eos]

        if len(prompt_ids) >= max_length - 1:
            prompt_ids = prompt_ids[: max_length - 1]
            answer_ids = answer_ids[:1]
        avail = max_length - len(prompt_ids)
        answer_ids = answer_ids[:avail]

        ids = prompt_ids + answer_ids
        prompt_len = len(prompt_ids)
        answer_end = len(ids)
        pad_n = max_length - len(ids)
        if pad_n > 0:
            ids = ids + [pad_id] * pad_n

        return {
            "input_ids": ids,
            "prompt_len": prompt_len,
            "answer_end": answer_end,
        }

    return tokenize_fn


# ----------------------------------------------------------------------------
# Validation pass (re-spans answer-only)
# ----------------------------------------------------------------------------
def run_validation_track2(model, loader, max_batches: int) -> float:
    import torch

    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, _ = compute_llada_loss(
                    model,
                    batch,
                    p_mask_low=P_MASK_LOW,
                    p_mask_high=P_MASK_HIGH,
                    span_end=batch["answer_end"],
                )
            losses.append(float(loss.detach()))
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    try:
        import torch
        import torch.nn.functional as F  # noqa: F401
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        return 2

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA detected — this script targets RTX 4090.", file=sys.stderr)

    # ---------------------------------------------------------------- wandb
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
                    "p_mask_range": [P_MASK_LOW, P_MASK_HIGH],
                    "lora_targets": LORA_TARGETS,
                    "lora_layers": LORA_LAYERS_TO_TRANSFORM,
                    "track": "track2-commit",
                    "seed": SEED,
                },
            )
            wandb = _wandb
        except Exception as exc:
            print(f"wandb init failed: {exc}", file=sys.stderr)
            wandb = None

    # ---------------------------------------------------------------- model
    print(f"Loading base model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
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

    # ----------------------------------------------------------------- lora
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
            layers_to_transform=LORA_LAYERS_TO_TRANSFORM,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # expect ~1-2M params

    # -------------------------------------------------------------- dataset
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
        split = full.train_test_split(test_size=0.05, seed=SEED)
        train_ds = split["train"]
        val_ds = split["test"]

    tokenize_fn = build_track2_tokenize_fn(tokenizer, MAX_LENGTH, rng)
    print(f"Tokenizing train ({len(train_ds)} rows)...", flush=True)
    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    if val_ds is not None:
        print(f"Tokenizing val ({len(val_ds)} rows)...", flush=True)
        val_ds = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names)

    cols = ["input_ids", "prompt_len", "answer_end"]
    train_ds.set_format(type="torch", columns=cols)
    if val_ds is not None:
        val_ds.set_format(type="torch", columns=cols)

    # --------------------------------------------------------------- loader
    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
            "prompt_len": torch.tensor(
                [int(b["prompt_len"]) for b in batch], dtype=torch.long, device=device
            ),
            "answer_end": torch.tensor(
                [int(b["answer_end"]) for b in batch], dtype=torch.long, device=device
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

    # ----------------------------------------------------------- optimizer
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

    # ----------------------------------------------------------- main loop
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
                loss, p_mask_avg = compute_llada_loss(
                    model,
                    batch,
                    p_mask_low=P_MASK_LOW,
                    p_mask_high=P_MASK_HIGH,
                    span_end=batch["answer_end"],
                )
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

                if (
                    val_loader is not None
                    and EVAL_INTERVAL > 0
                    and global_step % EVAL_INTERVAL == 0
                ):
                    val_loss = run_validation_track2(model, val_loader, EVAL_BATCHES)
                    print(
                        f"step:{global_step}/{total_steps} val_loss:{val_loss:.4f} val_bpb:{tokens_seen}",
                        flush=True,
                    )
                    if wandb is not None:
                        wandb.log({"val/loss": val_loss}, step=global_step)
                    model.train()

        epoch_dir = SAVE_DIR / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_dir))
        print(f"[epoch {epoch + 1}] saved adapter -> {epoch_dir}", flush=True)

    # -------------------------------------------------- final validation
    if val_loader is not None:
        val_loss = run_validation_track2(model, val_loader, EVAL_BATCHES)
        print(
            f"step:{global_step}/{total_steps} val_loss:{val_loss:.4f} val_bpb:{tokens_seen}",
            flush=True,
        )
        if wandb is not None:
            wandb.log({"val/loss_final": val_loss}, step=global_step)

    # --------------------------------------------------------- final save
    model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    total_bytes = sum(p.stat().st_size for p in SAVE_DIR.rglob("*") if p.is_file())
    print(f"Serialized model {SAVE_DIR} {total_bytes} bytes", flush=True)

    # --------------------------------------------------------------- push
    if PUSH:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if token is None:
            print("PUSH_TO_HUB=true but no HF token in env; skipping push.", file=sys.stderr)
        else:
            try:
                model.push_to_hub(
                    OUTPUT_REPO,
                    token=token,
                    private=True,
                    commit_message=f"Track 2 LoRA seed={SEED}",
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
