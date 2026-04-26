"""GSM8K answer extraction + grading.

GSM8K gold answers come after `####`. Predictions can be free-form; we extract
the last number-like token and compare to the gold integer.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


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
