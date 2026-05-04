"""GSM8K answer extraction + grading.

GSM8K gold answers come after `####`. Predictions can be free-form; we extract
the last number-like token and compare to the gold integer.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")
# Final-answer marker patterns LLaDA-Instruct emits at the end of CoT:
#   "#### 18"     (GSM8K-style)
#   "Answer: 18"  (sfumato system prompt)
_FINAL_ANS = re.compile(r"(?:####|Answer:)\s*(-?\$?\d[\d,]*\.?\d*)")


def extract_answer(text: str) -> str:
    """Pull the last numeric span from `text`. Returns "" if none found."""
    if not text:
        return ""
    matches = _NUMBER.findall(text)
    if not matches:
        return ""
    cleaned = matches[-1].replace("$", "").replace(",", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned


def extract_final_answer(text: str) -> str:
    """Pull the number ONLY from a final-answer marker (`#### N` / `Answer: N`).

    Returns "" when the marker hasn't been emitted yet. Used by ESC quorum
    on partial CoT so mid-reasoning numbers ("16 - 3 = 13") don't trigger
    false-quorum on a non-answer digit.
    """
    if not text:
        return ""
    matches = _FINAL_ANS.findall(text)
    if not matches:
        return ""
    cleaned = matches[-1].replace("$", "").replace(",", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned


def is_correct(pred: str, gold: str) -> bool:
    """Strict numeric equality after normalization."""
    p = extract_answer(pred)
    g = extract_answer(gold) or gold.strip()
    if not p or not g:
        return False
    try:
        return float(p) == float(g)
    except ValueError:
        return p.strip() == g.strip()
