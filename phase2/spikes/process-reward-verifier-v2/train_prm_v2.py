"""T1.B-redux: train PRM head on 19-d features (T1.B v1 features +
logit_shift_norm). Same MLP arch (32→16→1), same 5-fold CV split by
problem.

Usage: python3 train_prm_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "phase2" / "spikes" / "process-reward-verifier"))
sys.path.insert(0, str(REPO / "phase2" / "spikes" / "process-reward-verifier-v2"))

from load_traces_v2 import build_dataset_v2  # noqa: E402
from train_prm import cv_folds, fit_mlp  # noqa: E402  reuse v1 trainer
from e4 import grade  # noqa: E402


def evaluate_v2(ds: dict, n_folds: int = 5, seed: int = 0) -> dict:
    X = ds["X"]
    y = ds["y"]
    pid = ds["problem_idx"]
    bid = ds["branch_idx"]
    votes = ds["votes"]
    cmajc_winner = ds["winner"]
    gold = ds["gold"]

    prm_correct = []
    cmajc_correct = []
    oracle_correct = []

    for fold_idx, train_mask, val_mask in cv_folds(pid, n_folds=n_folds, seed=seed):
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va = X[val_mask]
        scores = fit_mlp(X_tr, y_tr, X_va, seed=seed + fold_idx)

        val_pids = pid[val_mask]
        val_bids = bid[val_mask]
        for p in np.unique(val_pids):
            mask = val_pids == p
            if not mask.any():
                continue
            sc = scores[mask]
            br = val_bids[mask]
            best_local = int(np.argmax(sc))
            best_branch = int(br[best_local])
            prm_pred = votes[int(p)][best_branch] if best_branch < len(votes[int(p)]) else ""
            g = gold[int(p)]
            prm_correct.append(grade.is_correct(prm_pred, g))
            cmajc_correct.append(grade.is_correct(cmajc_winner[int(p)], g))
            any_b_correct = any(grade.is_correct(v, g) for v in votes[int(p)] if v)
            oracle_correct.append(any_b_correct)

    return {
        "prm_acc": float(np.mean(prm_correct)),
        "cmajc_acc": float(np.mean(cmajc_correct)),
        "oracle_acc": float(np.mean(oracle_correct)),
        "n_problems": len(prm_correct),
    }


def main() -> None:
    traces_dir = REPO / "e4" / "results" / "traces" / "cmajc-prm-v2-N100-seed0-shifted"
    outcome_path = REPO / "e4" / "results" / "raw_cmajc_k64_seed0_prm_v2.jsonl"
    ds = build_dataset_v2(traces_dir, outcome_path)
    print(f"v2 dataset: X={ds['X'].shape} y_mean={ds['y'].mean():.4f}")

    res = evaluate_v2(ds, n_folds=5, seed=0)
    print(f"\nCross-validated 5-fold (seed=0):")
    print(f"  cmajc-vote   acc: {res['cmajc_acc']:.4f}")
    print(f"  PRM-rerank   acc: {res['prm_acc']:.4f}")
    print(f"  oracle       acc: {res['oracle_acc']:.4f}")
    print(f"  N problems: {res['n_problems']}")

    delta = res["prm_acc"] - res["cmajc_acc"]
    gap = res["oracle_acc"] - res["cmajc_acc"]
    closure = (delta / gap) if gap > 1e-9 else 0.0
    print(f"\n  Δ(PRM - cmajc) = {delta:+.4f} ({delta * 100:+.2f}pp)")
    print(f"  voting-rule gap = {gap:.4f} ({gap * 100:.2f}pp)")
    print(f"  gap closure: {closure * 100:.1f}%")
    if delta >= 0.06:
        print("\n  PRE-REG VERDICT: WIN (≥+6pp lift, ≥75% closure)")
    elif delta >= 0.02:
        print("\n  PRE-REG VERDICT: PARTIAL (+2pp ≤ Δ < +6pp)")
    else:
        print("\n  PRE-REG VERDICT: LOSS-CONFIRMED (Δ < +2pp; even logit_shift doesn't help)")


if __name__ == "__main__":
    main()
