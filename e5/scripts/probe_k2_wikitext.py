"""Phase M.3c — K.2 on WikiText-2 (non-math substrate generalisation test).

Reuses the K.2 (per-token diff-draft + AR-refill) machinery from
probe_diff_draft_ar_refill.py, but feeds WikiText-2-raw prose instead
of GSM8K Q/A pairs. Tests whether the K.2 NLL improvement we measured
on math substrates generalises to general English prose.

For each WikiText-2 chunk (≥ 192 tokens):
  - prompt = first 64 tokens
  - gold continuation = next 128 tokens
  - K.2: AR(64) then diff-fill 64 then percentile-rank refill
  - Score per-token NLL of K.2 output against gold

ENV: CKPT=...  N_EVAL=50  OUT=...  K_AR=64  K_DIFF=64  N_DIFF_STEPS=16
     CONF_THRESHOLD=50.0  DEVICE=mps|cpu
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the K.2 implementation
import e5.scripts.probe_diff_draft_ar_refill as k2_mod


def load_wikitext_chunks(n: int = 50, prefix_len: int = 64, gold_len: int = 128):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    target_len = prefix_len + gold_len
    chunks = []
    for row in ds:
        t = (row["text"] or "").strip()
        if len(t) < 800:  # rough char filter; need enough tokens
            continue
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) < target_len:
            continue
        prompt = ids[:prefix_len]
        gold = ids[prefix_len:prefix_len + gold_len]
        chunks.append({
            "idx": len(chunks),
            "prompt_tokens": prompt,
            "gold_answer_tokens": gold,
        })
        if len(chunks) >= n:
            break
    return chunks, tok


def main():
    # Monkey-patch the GSM8K loader with the wikitext loader before calling
    # the K.2 main routine. The interface is identical (returns list of
    # {prompt_tokens, gold_answer_tokens} plus tokenizer).
    k2_mod.load_gsm8k_probe_problems = load_wikitext_chunks
    k2_mod.main()


if __name__ == "__main__":
    main()
