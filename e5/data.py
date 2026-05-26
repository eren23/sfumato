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


def make_diff_batch(
    window: torch.Tensor,
    mask_token_id: int,
    mask_mode: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DIFF mode: drop the last position, mask a fraction of the remaining
    T positions, return (idx_masked, idx_original, masked_bool).

    mask_mode dispatches to one of three primitives in `model_composite`:
      "uniform" / "midrange"       -> apply_mask          (Bernoulli per-token,
                                                          default MDLM behaviour)
      "span_uniform" /
      "span_midrange"              -> apply_span_mask     (T5/SpanBERT-style
                                                          geometric spans; data
                                                          lever D2)
      "anti_ar_uniform" /
      "anti_ar_midrange"           -> apply_position_biased_mask
                                                          (suffix-weighted mask;
                                                          data lever D1)

    Default "uniform" preserves the original behaviour, so existing training
    drivers are unaffected.
    """
    from e5.model_composite import (
        sample_mask_ratios,
        apply_mask,
        apply_span_mask,
        apply_position_biased_mask,
    )
    idx_original = window[:, :-1].contiguous()  # (B, T)
    B, T = idx_original.shape
    ratio_mode = "midrange" if mask_mode.endswith("midrange") else "uniform"
    ratios = sample_mask_ratios(B, device=idx_original.device, mode=ratio_mode)
    if mask_mode.startswith("span_"):
        idx_masked, masked = apply_span_mask(
            idx_original, ratios, mask_token_id=mask_token_id
        )
    elif mask_mode.startswith("anti_ar_"):
        idx_masked, masked = apply_position_biased_mask(
            idx_original, ratios, mask_token_id=mask_token_id
        )
    else:
        idx_masked, masked = apply_mask(
            idx_original, ratios, mask_token_id=mask_token_id
        )
    return idx_masked, idx_original, masked


def load_fineweb_tokens(
    n_tokens: int = 50_000_000,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Path | None = None,
    seed: int = 1337,
) -> np.ndarray:
    """Stream the FineWeb-Edu sample from HF, tokenize, return first `n_tokens`.

    Cached under ~/.cache/sfumato_e5/fineweb_<tokenizer>_<n>.npy.

    Parameter-Golf substrate: this is the FineWeb-based corpus used by the
    OpenAI Parameter Golf LM competition (willdepueoai/parameter-golf). We
    tokenize with GPT-2 BPE for compatibility with the existing CompositeLM
    vocab (50258 = 50257 + MASK).
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sfumato_e5"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"fineweb_{tokenizer_name}_{n_tokens}.npy"
    if cache_path.exists():
        arr = np.load(cache_path)
        if len(arr) >= n_tokens:
            return arr[:n_tokens]
        # else fall through to extend

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.eos_token_id is None:
        tok.eos_token_id = GPT2_EOT

    # Stream FineWeb-Edu (10B sample) — broad clean English web text.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    chunks: list[np.ndarray] = []
    total = 0
    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        chunks.append(np.array(ids, dtype=np.uint16))
        chunks.append(np.array([GPT2_EOT], dtype=np.uint16))
        total += len(ids) + 1
        if total >= n_tokens:
            break

    full = np.concatenate(chunks)[:n_tokens]
    np.save(cache_path, full)
    return full


def load_gsm8k_dev_questions(
    indices_json: Path = Path("e4/data/gsm8k_dev_200.json"),
    n: int = 50,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    prompt_format: str = "qa",
    fewshot_k: int = 2,
) -> list[dict]:
    """Load GSM8K test problems referenced by the frozen indices file.

    prompt_format:
      "qa"     — "Question: ...\\nAnswer:"  (default, original behavior)
      "prose"  — doc-style preamble, no QA markers (matches FineWeb-Edu
                 training distribution; model just continues prose)
      "fewshot" — prepend `fewshot_k` worked examples from GSM8K-train
                 in raw "Question:...Answer:...#### N\\n\\n" format

    Returns list of {idx, question, gold, prompt_tokens, prompt_text}.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    spec = json.loads((repo_root / indices_json).read_text())
    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    fewshot_prefix = ""
    if prompt_format == "fewshot":
        train_ds = load_dataset("gsm8k", "main", split="train")
        examples = []
        for i in range(fewshot_k):
            ex = train_ds[i]
            examples.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n")
        fewshot_prefix = "\n".join(examples) + "\n"

    out = []
    for idx in spec["indices"][:n]:
        row = ds[idx]
        ans = row["answer"]
        gold = ans.split("####")[-1].strip().replace(",", "") if "####" in ans else ans.strip()
        if prompt_format == "qa":
            text = f"Question: {row['question']}\nAnswer:"
        elif prompt_format == "prose":
            text = (f"A recent math problem reads: \"{row['question']}\" "
                    f"The reasoning goes as follows.")
        elif prompt_format == "fewshot":
            text = fewshot_prefix + f"Question: {row['question']}\nAnswer:"
        else:
            raise ValueError(f"unknown prompt_format={prompt_format!r}")
        toks = tok.encode(text, add_special_tokens=False)
        out.append({"idx": idx, "question": row["question"], "gold": gold,
                    "prompt_tokens": toks, "prompt_text": text,
                    "prompt_format": prompt_format})
    return out


def load_mixed_tokens(
    fineweb_tokens_target: int = 2_850_000_000,
    gsm8k_repeats: int = 20,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Path | None = None,
    seed: int = 1337,
) -> np.ndarray:
    """Mix raw FineWeb-Edu prose (~95%) with formatted GSM8K Q/A (~5%).

    Concatenates `gsm8k_repeats` copies of the full GSM8K-train (~4M
    tokens × 20 reps = ~80M tokens of Q/A signal) shuffled into the
    FineWeb stream. Each GSM8K example is formatted as
    "Question: ...\\nAnswer: ...\\n#### N\\n<EOT>" so the model learns
    the canonical Q/A structure.
    """
    fineweb = load_fineweb_tokens(n_tokens=fineweb_tokens_target,
                                  tokenizer_name=tokenizer_name,
                                  cache_dir=cache_dir, seed=seed)

    cache_dir = cache_dir or Path.home() / ".cache" / "sfumato_e5"
    qa_cache = cache_dir / f"gsm8k_qa_{tokenizer_name}_x{gsm8k_repeats}.npy"
    if qa_cache.exists():
        qa = np.load(qa_cache)
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        ds = load_dataset("gsm8k", "main", split="train")
        chunks = []
        for _ in range(gsm8k_repeats):
            for row in ds:
                text = f"Question: {row['question']}\nAnswer: {row['answer']}\n"
                ids = tok.encode(text, add_special_tokens=False)
                chunks.append(np.array(ids, dtype=np.uint16))
                chunks.append(np.array([GPT2_EOT], dtype=np.uint16))
        qa = np.concatenate(chunks)
        np.save(qa_cache, qa)

    # Interleave: shuffle indices and insert QA chunks at random positions.
    # Simpler: concatenate qa to fineweb then shuffle whole-document blocks.
    # For our training (random-window sampler), simple concat suffices.
    print(f"  fineweb tokens: {len(fineweb):,}  qa tokens: {len(qa):,}")
    return np.concatenate([fineweb, qa])
