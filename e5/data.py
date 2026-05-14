"""T0 — GSM8K-train data pipeline.

Strategy:
  1. Tokenize all of GSM8K-train (7473 problems) with GPT-2 BPE.
  2. Concatenate everything into one long token stream, separated by an
     EOT token (50256 in GPT-2). Total ~3-5M tokens depending on CoT
     length.
  3. Sample fixed-length windows of block_size (256) for training. Each
     window is presented either as an AR target (shifted next-token
     targets) or a DIFF target (masked, predict masked) per the α
     schedule, set at batch construction time.

GSM8K-train answers contain step-by-step reasoning followed by "#### N".
We keep the full reasoning text for both modes — the model should learn
to AR-extend reasoning AND to fill in masked spans of reasoning.

Implementation note: no pre-defined train/val split here. The plan uses
the existing `e4/data/gsm8k_dev_200.json` indices (first N=50 of the
test split) as the held-out evaluation; GSM8K's own train and test
splits are disjoint, so leakage is impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

GPT2_EOT = 50256
DEFAULT_TOKENIZER = "gpt2"


def load_gsm8k_train_tokens(
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Path | None = None,
    include_reasoning: bool = True,
) -> np.ndarray:
    """Tokenize all of gsm8k-train into one long uint16 numpy array.

    `include_reasoning` keeps the full step-by-step answer (with "####"
    final marker). If False, drop everything after the first "####".
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sfumato_e5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"gsm8k_train_{tokenizer_name}_{'cot' if include_reasoning else 'qa'}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.eos_token_id is None:
        tok.eos_token_id = GPT2_EOT

    chunks: list[np.ndarray] = []
    for row in ds:
        q = row["question"]
        a = row["answer"]
        if not include_reasoning and "####" in a:
            a = "#### " + a.split("####")[-1].strip()
        text = f"Question: {q}\nAnswer: {a}"
        ids = tok.encode(text, add_special_tokens=False)
        chunks.append(np.array(ids, dtype=np.uint16))
        chunks.append(np.array([GPT2_EOT], dtype=np.uint16))

    full = np.concatenate(chunks)
    np.save(cache_path, full)
    return full


class GSM8KStreamingDataset(torch.utils.data.Dataset):
    """Returns random `block_size`-length windows from the token stream.

    `length` defines an artificial epoch — number of windows per epoch.
    With block_size=256 and a token stream of ~4M tokens, there are
    ~16k unique non-overlapping windows; we oversample randomly.
    """

    def __init__(self, tokens: np.ndarray, block_size: int = 256, length: int | None = None, seed: int = 0):
        self.tokens = tokens
        self.block_size = block_size
        self.length = length or max(1, (len(tokens) - block_size - 1) // 64)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        n = len(self.tokens)
        start = self.rng.integers(0, n - self.block_size - 1)
        window = self.tokens[start : start + self.block_size + 1].astype(np.int64)
        return torch.from_numpy(window)  # (block_size + 1,) — last token is the AR target


def make_ar_batch(window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """AR mode: predict token t+1 from tokens [0..t].
    window: (B, T+1). Returns (idx, targets), both (B, T).
    """
    idx = window[:, :-1].contiguous()
    targets = window[:, 1:].contiguous()
    return idx, targets


def make_diff_batch(window: torch.Tensor, mask_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DIFF mode: drop the last position, mask a random fraction of the
    remaining T positions, return (idx_masked, idx_original, masked_bool).
    """
    from e5.model_composite import sample_mask_ratios, apply_mask
    idx_original = window[:, :-1].contiguous()  # (B, T)
    B, T = idx_original.shape
    ratios = sample_mask_ratios(B, device=idx_original.device, mode="uniform")
    idx_masked, masked = apply_mask(idx_original, ratios, mask_token_id=mask_token_id)
    return idx_masked, idx_original, masked


def load_gsm8k_dev_questions(
    indices_json: Path = Path("e4/data/gsm8k_dev_200.json"),
    n: int = 50,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[dict]:
    """Load the GSM8K test problems referenced by the frozen indices file.

    Returns a list of {idx, question, gold, question_tokens (list[int])}.
    Used by the eval loop in `e5.eval`.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    spec = json.loads((repo_root / indices_json).read_text())
    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = []
    for idx in spec["indices"][:n]:
        row = ds[idx]
        ans = row["answer"]
        gold = ans.split("####")[-1].strip().replace(",", "") if "####" in ans else ans.strip()
        text = f"Question: {row['question']}\nAnswer:"
        toks = tok.encode(text, add_special_tokens=False)
        out.append({"idx": idx, "question": row["question"], "gold": gold, "prompt_tokens": toks})
    return out
