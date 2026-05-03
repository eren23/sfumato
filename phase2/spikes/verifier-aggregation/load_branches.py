"""Load per-branch records from raw_cmaj jsonl files.

Returns (problem_id, branch_idx, branch_text, gold_answer, branch_extracted_answer, correct, tau).
"""
from __future__ import annotations
import json
import os
import pathlib
import re
from typing import Iterator, NamedTuple

# RESULTS_DIR is env-overridable; defaults try local-mac first then pod path.
_DEFAULT = "/Users/eren/Documents/AI/sfumato/e4/results"
if not pathlib.Path(_DEFAULT).exists():
    _DEFAULT = "/workspace/sfumato/e4/results"
RESULTS_DIR = pathlib.Path(os.environ.get("SFUMATO_RESULTS_DIR", _DEFAULT))

# Phase-1 cmaj jsonls (already on disk)
PHASE1_JSONLS = {
    0.3: RESULTS_DIR / "raw_cmaj_k64_seed0_b5_t0.3.jsonl",
    0.7: RESULTS_DIR / "raw_cmaj_k64_seed0_b5.jsonl",
    1.0: RESULTS_DIR / "raw_cmaj_k64_seed0_b5_t1.0.jsonl",
}


class BranchRow(NamedTuple):
    problem_id: str
    branch_idx: int
    branch_text: str
    gold: str
    extracted: str
    correct: bool
    tau: float
    source: str  # filename for provenance


_ANSWER_RE = re.compile(r"Answer\s*[:=]\s*([\-\+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"([\-\+]?\d[\d,]*\.?\d*)")


def extract_answer(text: str) -> str:
    """Mirror runner.py's grade.py answer extraction (numeric, last-match fallback)."""
    if not text:
        return ""
    # Prefer "Answer: ..." pattern
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    # Fallback: last number in text
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return ""


def parse_votes(votes_str: str) -> list[str]:
    return [v.strip() for v in votes_str.split("|")]


def load_jsonl(path: pathlib.Path, tau_override: float | None = None) -> Iterator[BranchRow]:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tau = tau_override if tau_override is not None else rec.get("temperature", 0.7)
        gold = str(rec["gold"]).strip()
        trace = rec.get("trace", {})
        votes = parse_votes(trace.get("votes", ""))
        for i in range(5):
            key = f"branch_{i}"
            if key not in trace:
                break
            text = trace[key]
            extracted = votes[i] if i < len(votes) else extract_answer(text)
            yield BranchRow(
                problem_id=str(rec["id"]),
                branch_idx=i,
                branch_text=text,
                gold=gold,
                extracted=extracted,
                correct=(extracted == gold),
                tau=float(tau),
                source=path.name,
            )


def load_phase1() -> list[BranchRow]:
    """Load all Phase-1 branches across the 3 τ files."""
    rows = []
    for tau, path in PHASE1_JSONLS.items():
        rows.extend(load_jsonl(path, tau_override=tau))
    return rows


def load_substrate(substrate_dir: pathlib.Path = RESULTS_DIR) -> list[BranchRow]:
    """Load Phase-2 D3.5 substrate harvest jsonls (if they exist)."""
    rows = []
    candidates = [
        # Night-1 substrate: full N=200 cmaj b=5 τ=0.7 with v3 LoRA
        ("raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl", 0.7),
        # Earlier candidates (left for back-compat)
        ("raw_cmaj_k64_seed0_b5_t0.7_N100.jsonl", 0.7),
        ("raw_cmaj_k64_seed0_b5_t1.0_N100.jsonl", 1.0),
    ]
    for name, tau in candidates:
        p = substrate_dir / name
        if p.exists():
            rows.extend(load_jsonl(p, tau_override=tau))
    return rows


if __name__ == "__main__":
    p1 = load_phase1()
    sub = load_substrate()
    print(f"Phase-1 branches: {len(p1)} ({len(set(r.problem_id for r in p1))} unique problems)")
    print(f"Phase-2 substrate branches: {len(sub)} ({len(set(r.problem_id for r in sub))} unique problems)")
    print(f"Total: {len(p1) + len(sub)} branches")
    if p1:
        print(f"\nSample row: {p1[0]._replace(branch_text=p1[0].branch_text[:80] + '...')}")
        n_correct = sum(1 for r in p1 if r.correct)
        print(f"Phase-1 per-branch correctness rate: {n_correct}/{len(p1)} = {n_correct/len(p1):.2%}")
