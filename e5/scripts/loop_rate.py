"""Compute degeneracy metrics on probe5 results_sample text.

A generation is "loopy" if any of:
- exact token-level 3-gram repeats ≥ 4 times consecutively, OR
- exact 3-word phrase repeats ≥ 3 times consecutively, OR
- single word repeats ≥ 5 times consecutively.

Usage:
  python e5/scripts/loop_rate.py PATH/probe5_*.json [more.json ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


WORD = re.compile(r"\b\w+\b")


def word_runs(text: str) -> int:
    """Longest run of consecutive identical words."""
    words = WORD.findall(text.lower())
    if len(words) < 2:
        return 1
    best = run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def phrase_runs(text: str, k: int = 3) -> int:
    """Longest run of consecutive identical k-word phrases."""
    words = WORD.findall(text.lower())
    if len(words) < 2 * k:
        return 1
    grams = [tuple(words[i : i + k]) for i in range(len(words) - k + 1)]
    best = run = 1
    for i in range(k, len(grams)):
        if grams[i] == grams[i - k]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def is_loopy(text: str) -> bool:
    return word_runs(text) >= 5 or phrase_runs(text, 3) >= 3


def score_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {"path": str(path), "modes": {}}
    for mode, payload in data.items():
        if not isinstance(payload, dict) or "results_sample" not in payload:
            continue
        results = payload["results_sample"]
        loopy = sum(1 for r in results if is_loopy(r.get("text", "")))
        max_word = max((word_runs(r.get("text", "")) for r in results), default=0)
        max_phr = max((phrase_runs(r.get("text", ""), 3) for r in results), default=0)
        out["modes"][mode] = {
            "n_sampled": len(results),
            "n_loopy": loopy,
            "loop_rate": round(loopy / max(1, len(results)), 3),
            "max_word_run": max_word,
            "max_phrase_run": max_phr,
            "accuracy": payload.get("accuracy", None),
        }
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        result = score_file(Path(arg))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
