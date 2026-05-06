"""T1.B-redux: extended feature loader including logit_shift_norm.

Adds 5 new dims on top of T1.B's 14-d feature vector:
- logit_shift_norm[1..3]      (3 dims, skip block 0 — adapter toggle
                                fires between blocks 0 and 1)
- mean_logit_shift            (1 dim)
- argmax_logit_shift_subblock (1 dim, normalized 0..1)

Total 19-d feature vector. NaN values (block 0 always has no shift, or
when shadow forward fails) are filled with 0.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase2" / "spikes" / "process-reward-verifier"))

from load_traces import (  # noqa: E402  reuse loader plumbing
    build_dataset as _build_dataset_v1,
    load_sidecars,
    load_outcome,
    per_branch_correct,
)


def featurize_branch_v2(sub_blocks: list[dict]) -> np.ndarray:
    """Same as v1 (14 dims) plus 5 logit_shift_norm dims → 19 total."""
    n_blocks = 4
    em = [r.get("entropy_mean") for r in sub_blocks]
    ex = [r.get("entropy_max") for r in sub_blocks]
    ca = [bool(r.get("commit_lora_active", False)) for r in sub_blocks]
    ls = [r.get("logit_shift_norm") for r in sub_blocks]

    em_padded = [(em[s] if s < len(em) and em[s] is not None else 0.0) for s in range(n_blocks)]
    ex_padded = [(ex[s] if s < len(ex) and ex[s] is not None else 0.0) for s in range(n_blocks)]
    ca_padded = [(ca[s] if s < len(ca) else False) for s in range(n_blocks)]
    ls_padded = [(ls[s] if s < len(ls) and ls[s] is not None else 0.0) for s in range(n_blocks)]

    feats: list[float] = []
    feats.extend(em_padded)                                              # 4 dims (entropy_mean per block)
    feats.extend(ex_padded)                                              # 4 dims (entropy_max per block)
    feats.append(sum(ca_padded) / n_blocks)                              # 1 dim (commit_lora_fraction)
    feats.append(float(np.mean(em_padded)))                              # 1 dim (mean_entropy)
    feats.append(float(np.std(em_padded)))                               # 1 dim (std_entropy)
    feats.append(float(np.argmax(em_padded)) / n_blocks)                 # 1 dim (argmax-entropy subblock, norm)
    feats.append(em_padded[-1])                                          # 1 dim (final-block entropy)
    denom = em_padded[0] if em_padded[0] > 1e-6 else 1e-6
    feats.append(em_padded[-1] / denom)                                  # 1 dim (block3/block0 ratio)
    # NEW logit_shift_norm dims (5 total).
    feats.extend(ls_padded[1:])                                          # 3 dims (logit_shift block 1, 2, 3)
    feats.append(float(np.mean(ls_padded[1:])) if any(ls_padded[1:]) else 0.0)  # 1 dim (mean_logit_shift)
    nz_idx = float(np.argmax(ls_padded[1:])) / max(n_blocks - 1, 1) if any(ls_padded[1:]) else 0.0
    feats.append(nz_idx)                                                 # 1 dim (argmax_logit_shift block, norm)
    return np.asarray(feats, dtype=np.float32)


def build_dataset_v2(traces_dir: Path, outcome_path: Path) -> dict:
    """v2 dataset with 19-d features."""
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
            X_rows.append(featurize_branch_v2(sub_blocks))
            y_rows.append(int(per_b.get(b, False)))
            pidx.append(i)
            bidx.append(b)

    return {
        "X": np.stack(X_rows) if X_rows else np.zeros((0, 19), dtype=np.float32),
        "y": np.asarray(y_rows, dtype=np.int64),
        "problem_idx": np.asarray(pidx, dtype=np.int64),
        "branch_idx": np.asarray(bidx, dtype=np.int64),
        "votes": votes_per_problem,
        "winner": winner_per_problem,
        "gold": gold_per_problem,
    }


def main() -> None:
    traces_dir = REPO / "e4" / "results" / "traces" / "cmajc-prm-v2-N100-seed0-shifted"
    outcome_path = REPO / "e4" / "results" / "raw_cmajc_k64_seed0_prm_v2.jsonl"
    if not traces_dir.exists() or not outcome_path.exists():
        print(f"missing inputs: traces={traces_dir.exists()} outcome={outcome_path.exists()}")
        return
    ds = build_dataset_v2(traces_dir, outcome_path)
    print(f"X shape: {ds['X'].shape}")
    print(f"y mean: {ds['y'].mean():.4f}")
    print(f"problems: {len(set(ds['problem_idx'].tolist()))}")
    # Sanity: how many records have non-zero logit_shift?
    nonzero_ls = (ds["X"][:, 14:17] != 0).any(axis=1).sum()
    print(f"records with non-zero logit_shift_norm: {int(nonzero_ls)}/{len(ds['X'])}")


if __name__ == "__main__":
    main()
