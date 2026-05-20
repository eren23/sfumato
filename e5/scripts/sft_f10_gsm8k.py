"""Phase L — SFT F10 on GSM8K-train Q/A pairs.

F10 was trained on 95% FineWeb-Edu + 5% GSM8K mixed-in raw tokens; the
GSM8K signal was a token-stream concatenation, never a supervised
prompt->target pair. This script does proper SFT:
  - prompt = "Question: {q}\nAnswer: "
  - target = full step-by-step reasoning + "#### N"
  - AR cross-entropy ONLY on target positions (prompt positions masked
    with ignore_index=-100)

Expected: GSM8K-dev free-run accuracy jumps from ~4% (F10 base) to
~10-20% after 1-2 epochs.

Uses the AR head only. Diff head is unchanged. Resumable checkpoint
saved every 500 steps + at end.

ENV:
  CKPT=path/to/f10/slim.pt
  OUT_DIR=path/to/output/dir
  N_EPOCHS=2
  BATCH=2  (Mac MPS limit at 305M; pod can go higher)
  LR=3e-5
  MAX_LEN=384  (longer GSM8K answers truncated)
  SAVE_EVERY=500
  SEED=2026
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from e5.model_composite import CompositeLM, CompositeConfig


def load_composite_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CompositeConfig(**{k: cfg_dict[k] for k in CompositeConfig.__dataclass_fields__ if k in cfg_dict})
    model = CompositeLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, cfg


def build_sft_examples(tok, max_len: int) -> list[dict]:
    """Tokenize GSM8K-train into (input_ids, prompt_len) pairs."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    examples = []
    for row in ds:
        q = row["question"].strip()
        a = row["answer"].strip()
        prompt = f"Question: {q}\nAnswer: "
        full = prompt + a
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        full_ids = tok.encode(full, add_special_tokens=False)
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]
        if len(prompt_ids) >= len(full_ids):
            continue  # answer truncated to zero
        examples.append({
            "input_ids": full_ids,
            "prompt_len": len(prompt_ids),
        })
    return examples


def make_batch(examples: list[dict], indices: list[int], pad_id: int, device: str):
    """Collate a batch with right-padded ids and label masks (-100 outside answer)."""
    batch = [examples[i] for i in indices]
    max_len = max(len(ex["input_ids"]) for ex in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        plen = ex["prompt_len"]
        # Standard AR shift: target at position t is ids[t+1].
        # Predict answer tokens (ids[plen..L-1]) FROM positions (plen-1..L-2).
        # So labels[plen-1 .. L-2] = ids[plen .. L-1].
        labels[i, plen - 1 : L - 1] = torch.tensor(ids[plen:L], dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def main():
    ckpt_path = Path(os.environ["CKPT"])
    out_dir = Path(os.environ["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = int(os.environ.get("N_EPOCHS", "2"))
    batch_size = int(os.environ.get("BATCH", "2"))
    lr = float(os.environ.get("LR", "3e-5"))
    max_len = int(os.environ.get("MAX_LEN", "384"))
    save_every = int(os.environ.get("SAVE_EVERY", "500"))
    seed = int(os.environ.get("SEED", "2026"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[sft] device={device}  ckpt={ckpt_path}  out={out_dir}")

    print("[sft] loading F10 ...")
    model, cfg = load_composite_from_ckpt(ckpt_path, device=device)
    model.train(True)
    print(f"[sft] model: {model.num_params()/1e6:.1f}M params, block_size={cfg.block_size}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    pad_id = tok.eos_token_id or 50256

    print("[sft] tokenising GSM8K-train Q/A pairs ...")
    examples = build_sft_examples(tok, max_len=max_len)
    print(f"[sft] {len(examples)} examples, avg len {np.mean([len(e['input_ids']) for e in examples]):.0f}")

    n_steps_per_epoch = len(examples) // batch_size
    total_steps = n_epochs * n_steps_per_epoch
    print(f"[sft] total_steps={total_steps}  batch={batch_size}  lr={lr}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                               betas=(0.9, 0.95))
    log_rows = []
    step = 0
    t_start = time.time()
    for epoch in range(n_epochs):
        order = np.random.permutation(len(examples))
        for i in range(0, len(examples) - batch_size + 1, batch_size):
            indices = list(order[i : i + batch_size])
            input_ids, labels = make_batch(examples, indices, pad_id, device)
            T = input_ids.shape[1]
            if T > cfg.block_size:
                input_ids = input_ids[:, : cfg.block_size]
                labels = labels[:, : cfg.block_size]
            logits = model(input_ids, mode="ar")  # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 25 == 0:
                wall = time.time() - t_start
                row = {"step": step, "epoch": epoch, "loss": float(loss.item()),
                       "wall_s": wall}
                log_rows.append(row)
                print(f"[{step:5d}/{total_steps}] ep={epoch} loss={row['loss']:.4f} "
                      f"wall={wall/60:.1f}m")
            if step > 0 and step % save_every == 0:
                ckpt_out = out_dir / f"model_step{step}.pt"
                torch.save({
                    "config": cfg.__dict__,
                    "state_dict": model.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "variant": "sft_gsm8k",
                    "n_params": model.num_params(),
                }, ckpt_out)
                print(f"[sft] saved {ckpt_out.name}")
            step += 1

    final_out = out_dir / "model_final.pt"
    torch.save({
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "step": step,
        "epoch": n_epochs,
        "variant": "sft_gsm8k",
        "n_params": model.num_params(),
    }, final_out)
    print(f"[sft] saved final {final_out.name}")
    # Optional HF push
    if os.environ.get("HF_PUSH_REPO") and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
            sub = os.environ.get("HF_PUSH_SUBDIR", out_dir.name)
            api.upload_file(path_or_fileobj=str(final_out),
                            path_in_repo=f"{sub}/model_final.pt",
                            repo_id=os.environ["HF_PUSH_REPO"], repo_type="model")
            print(f"[sft] pushed to HF: {os.environ['HF_PUSH_REPO']}/{sub}/")
        except Exception as e:
            print(f"[sft] hf-push failed: {e!s:.200}")
    log_path = out_dir / "train_log.jsonl"
    with open(log_path, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[sft] done. total wall = {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
