"""Cheap diagnostic: does commit-LoRA actually change anything at decode time?

Generate 10 fixed GSM8K problems at temp=0 (deterministic) under three settings:
  A) base + Track 1 LoRA only          (apply_commit=False, no commit-LoRA loaded)
  B) base + Track 1 + commit-LoRA OFF  (apply_commit=False, commit-LoRA loaded but disabled)
  C) base + Track 1 + commit-LoRA ON   (apply_commit=True)

Compare decoded text + extracted answer:
  - A == B → confirms commit-LoRA load doesn't accidentally affect base behavior
  - B != C → commit-LoRA IS shifting tokens at decode time
  - B == C → commit-LoRA is a learned-identity / no-op at inference

Tells us whether v3 (commit on more blocks, full-response loss) is worth running
or whether the commit-adapter design is fundamentally broken.

Usage on pod:
  python scripts/commit_effect_diagnostic.py --n 10 \
      --lora_path eren23/sfumato-llada-prefix-robust-v2 \
      --commit_path eren23/sfumato-llada-commit-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _difflen(a: str, b: str) -> int:
    """First char index where a, b diverge; -1 if identical."""
    if a == b:
        return -1
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k_steps", type=int, default=64)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--commit_path", type=str, required=True)
    parser.add_argument("--diff_model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    args = parser.parse_args()

    from datasets import load_dataset  # type: ignore

    from e4 import diff_llada, grade  # type: ignore

    ds = load_dataset("gsm8k", "main", split="test")

    # Setting B & C: load Track 1 + commit-LoRA, then toggle apply_commit.
    print(f"[diag] loading: lora={args.lora_path} commit={args.commit_path}", flush=True)
    model_bc = diff_llada.load(
        args.diff_model,
        mock=False,
        lora_path=args.lora_path,
        commit_lora_path=args.commit_path,
    )

    rows = []
    for i in range(args.n):
        ex = ds[i]
        question = ex["question"]
        gold = ex["answer"].rsplit("####", 1)[-1].strip()

        text_off, _ = model_bc.denoise_block(
            prompt=question, k_steps=args.k_steps, seed=0, temperature=0.0,
            apply_commit=False,
        )
        ans_off = grade.extract_answer(text_off)

        text_on, _ = model_bc.denoise_block(
            prompt=question, k_steps=args.k_steps, seed=0, temperature=0.0,
            apply_commit=True,
        )
        ans_on = grade.extract_answer(text_on)

        diverge = _difflen(text_off, text_on)
        rows.append({
            "idx": i,
            "gold": gold,
            "ans_off": ans_off,
            "ans_on": ans_on,
            "answer_changed": ans_off != ans_on,
            "text_changed": text_off != text_on,
            "diverge_at_char": diverge,
            "len_off": len(text_off),
            "len_on": len(text_on),
        })
        marker = ""
        if not rows[-1]["text_changed"]:
            marker = " [IDENTICAL]"
        elif rows[-1]["answer_changed"]:
            marker = " [ANSWER FLIPPED]"
        else:
            marker = f" [text differs at char {diverge}, same answer]"
        print(
            f"  prob={i:2d} gold={gold:>8s} off={ans_off:>10s} on={ans_on:>10s}{marker}",
            flush=True,
        )

    # Aggregate
    text_changed = sum(1 for r in rows if r["text_changed"])
    answer_changed = sum(1 for r in rows if r["answer_changed"])
    print(f"\n[diag] of {len(rows)} problems:")
    print(f"  text changed (commit altered output at all): {text_changed}/{len(rows)}")
    print(f"  answer changed                             : {answer_changed}/{len(rows)}")
    if text_changed == 0:
        verdict = "FULL NO-OP — commit-LoRA learned identity. v3 won't help; redesign needed."
    elif answer_changed == 0:
        verdict = "PARTIAL — commit shifts text but never flips answer. Capacity/late-block-only limited; v3 worth running."
    else:
        verdict = "ACTIVE — commit flips answers in some cases. The aggregate c2c=70.5% means net effect is zero/negative; investigate which cases improve vs degrade."
    print(f"  verdict: {verdict}")

    out = Path(REPO_ROOT) / "e2" / "data" / "commit_effect_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": {
        "n": len(rows),
        "text_changed": text_changed,
        "answer_changed": answer_changed,
        "verdict": verdict,
    }}, indent=2))
    print(f"  saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
