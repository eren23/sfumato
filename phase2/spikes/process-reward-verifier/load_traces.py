"""T1.B.3: load per-step trace JSONLs + main outcome JSONL, produce
(features_14d, correct) tuples per (problem, branch) for the PRM head.

Substrate: cmajc N=100 BATCHED=0 BRANCHES=5 TRACE_STEPS=1 harvest from
T1.B.2. Sidecar JSONLs at e4/results/traces/<run_name>/branch_<b>_idx_<i>.jsonl
have one row per sub-block (4 rows per file). Main outcome JSONL at
e4/results/raw_cmajc_k64_seed0.jsonl has 1 row per problem with
trace.branch_<b> text, votes, winner, correct.

Per the amended PRE_REG (logit_shift_norm dropped due to 24GB pod), the
14-dim feature vector per (problem, branch) is:
- entropy_mean[0..3]                 (4 dims)
- entropy_max[0..3]                  (4 dims)
- commit_lora_active_fraction        (1 dim)
- mean_entropy / std_entropy          (2 dims)
- max_entropy_subblock_argmax (norm)  (1 dim)
- final_block_entropy_mean            (1 dim)
- ratio_block3_to_block0_entropy      (1 dim)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO))
from e4 import grade  # noqa: E402


def load_sidecars(traces_dir: Path) -> dict[tuple[int, int], list[dict]]:
    """Read all branch_<b>_idx_<i>.jsonl files in traces_dir.

    Returns dict keyed (branch_idx, problem_idx) → list of sub-block records
    (sorted by sub_block).
    """
    out: dict[tuple[int, int], list[dict]] = {}
    for path in sorted(traces_dir.glob("branch_*_idx_*.jsonl")):
        # Filename like branch_3_idx_42.jsonl.
        stem = path.stem  # "branch_3_idx_42"
        parts = stem.split("_")
        b = int(parts[1])
        i = int(parts[3])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: r["sub_block"])
        out[(b, i)] = rows
    return out


def load_outcome(outcome_path: Path) -> dict[int, dict]:
    """Read raw_cmajc_*.jsonl. Returns dict keyed by `idx` → row.

    Row contains `trace.branch_0..4` (text), `pred`, `gold`, `correct`,
    plus `trace.votes` like "13 | 13 | 12 | 13 | 13" so we can recover
    per-branch extracted answers.
    """
    out: dict[int, dict] = {}
    for line in outcome_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["idx"]] = rec
    return out


def per_branch_correct(outcome: dict, gold: str) -> dict[int, bool]:
    """Pull per-branch extracted answers from `trace.votes` (pipe-delim)
    and compare to gold. Returns {branch_idx: bool}.

    Note: extracted answers in `trace.votes` are already grade.extract_answer
    strings; we match on string equality after normalization (matching
    e4/grade.is_correct's normalize step is overkill — the run already
    extracted using extract_answer, so equality after strip suffices for
    the voting reranker test).
    """
    votes = outcome.get("trace", {}).get("votes", "")
    if not votes:
        return {}
    answers = [a.strip() for a in votes.split(" | ")]
    out: dict[int, bool] = {}
    for b, a in enumerate(answers):
        out[b] = grade.is_correct(a, gold)
    return out


def featurize_branch(sub_blocks: list[dict]) -> np.ndarray:
    """Aggregate per-sub-block records into a 14-dim feature vector."""
    n_blocks = 4
    em = [r.get("entropy_mean") for r in sub_blocks]
    ex = [r.get("entropy_max") for r in sub_blocks]
    ca = [bool(r.get("commit_lora_active", False)) for r in sub_blocks]
    # Pad / fill defensively (some sub-blocks may be missing if generation
    # short-circuited). Use 0.0 for missing entropy.
    em_padded = [(em[s] if s < len(em) and em[s] is not None else 0.0) for s in range(n_blocks)]
    ex_padded = [(ex[s] if s < len(ex) and ex[s] is not None else 0.0) for s in range(n_blocks)]
    ca_padded = [(ca[s] if s < len(ca) else False) for s in range(n_blocks)]

    feats: list[float] = []
    feats.extend(em_padded)                              # 4 dims
    feats.extend(ex_padded)                              # 4 dims
    feats.append(sum(ca_padded) / n_blocks)              # commit_lora_active_fraction
    feats.append(float(np.mean(em_padded)))              # mean_entropy
    feats.append(float(np.std(em_padded)))               # std_entropy
    feats.append(float(np.argmax(em_padded)) / n_blocks) # max_entropy_subblock_argmax (normalized)
    feats.append(em_padded[-1])                          # final_block_entropy_mean
    denom = em_padded[0] if em_padded[0] > 1e-6 else 1e-6
    feats.append(em_padded[-1] / denom)                  # ratio_block3_to_block0
    return np.asarray(feats, dtype=np.float32)


def build_dataset(traces_dir: Path, outcome_path: Path) -> dict:
    """Join sidecars with outcome → return dict with arrays:

    - X: (N_problems * 5, 14) float32
    - y: (N_problems * 5,) int (correct flag)
    - problem_idx: (N_problems * 5,) int
    - branch_idx: (N_problems * 5,) int
    - votes: list of per-branch extracted answer strings (for rerank eval)
    - winner: per-problem majority-vote answer (cmajc baseline)
    - gold: per-problem gold answer
    """
    sidecars = load_sidecars(traces_dir)
    outcomes = load_outcome(outcome_path)

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    pidx: list[int] = []
    bidx: list[int] = []
    votes_per_problem: dict[int, list[str]] = {}
    winner_per_problem: dict[int, str] = {}
    gold_per_problem: dict[int, str] = {}

    for i, outcome in outcomes.items():
        gold = outcome["gold"]
        per_b = per_branch_correct(outcome, gold)
        votes = [a.strip() for a in outcome.get("trace", {}).get("votes", "").split(" | ")]
        votes_per_problem[i] = votes
        winner_per_problem[i] = outcome.get("trace", {}).get("winner", "")
        gold_per_problem[i] = gold
        for b in range(5):
            sub_blocks = sidecars.get((b, i))
            if sub_blocks is None:
                continue
            X_rows.append(featurize_branch(sub_blocks))
            y_rows.append(int(per_b.get(b, False)))
            pidx.append(i)
            bidx.append(b)

    return {
        "X": np.stack(X_rows) if X_rows else np.zeros((0, 14), dtype=np.float32),
        "y": np.asarray(y_rows, dtype=np.int64),
        "problem_idx": np.asarray(pidx, dtype=np.int64),
        "branch_idx": np.asarray(bidx, dtype=np.int64),
        "votes": votes_per_problem,
        "winner": winner_per_problem,
        "gold": gold_per_problem,
    }


def main() -> None:
    traces_dir = REPO / "e4" / "results" / "traces" / "cmajc-prm-N100-seed0-trace"
    outcome_path = REPO / "e4" / "results" / "raw_cmajc_k64_seed0_prm.jsonl"
    if not traces_dir.exists() or not outcome_path.exists():
        print(f"missing inputs: traces_dir={traces_dir.exists()} outcome={outcome_path.exists()}")
        return
    ds = build_dataset(traces_dir, outcome_path)
    print(f"X shape: {ds['X'].shape}")
    print(f"y mean (per-branch correct rate): {ds['y'].mean():.4f}")
    print(f"problems: {len(set(ds['problem_idx'].tolist()))}")
    print(f"branches per problem (modal): "
          f"{int(np.median(np.bincount(ds['problem_idx'])))}")


if __name__ == "__main__":
    main()
