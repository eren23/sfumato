"""T1.B.3: train PRM head on per-step features. 5-fold CV split BY
PROBLEM (so reranker never sees branches from a problem in its training
fold) and then rerank held-out problems.

Pre-reg WIN: PRM-rerank acc - cmajc-vote acc ≥ +6pp on N=100.

MLP: 14 → 32 → 16 → 1 sigmoid. Trained per-fold with BCE loss on the
per-branch correct flag.

Reranker: for each held-out problem, score all branches, pick the
argmax-PRM branch's extracted answer. Compare to:
- cmajc-vote: the original `trace.winner` (5-branch majority vote with
  commit-LoRA on)
- oracle: any-branch-correct (upper bound)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from load_traces import build_dataset
import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from e4 import grade  # noqa: E402

def cv_folds(problem_ids: np.ndarray, n_folds: int = 5, seed: int = 0):
    """5-fold split by unique problem id."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(problem_ids)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)
    for fold_idx, val_problems in enumerate(folds):
        val_mask = np.isin(problem_ids, val_problems)
        yield fold_idx, ~val_mask, val_mask


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
            seed: int = 0, epochs: int = 60, lr: float = 0.05) -> np.ndarray:
    """Lightweight numpy-only MLP. CPU-friendly. Returns P(correct) on X_val.

    Architecture: 14 → 32 (relu) → 16 (relu) → 1 sigmoid.
    Optimizer: Adam-like with momentum.
    """
    rng = np.random.default_rng(seed)
    in_dim = X_train.shape[1]

    def he(in_, out_):
        return rng.normal(0, np.sqrt(2.0 / in_), size=(in_, out_)).astype(np.float32)

    W1 = he(in_dim, 32); b1 = np.zeros(32, dtype=np.float32)
    W2 = he(32, 16);     b2 = np.zeros(16, dtype=np.float32)
    W3 = he(16, 1);      b3 = np.zeros(1, dtype=np.float32)

    # Normalize features (per-fold).
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    Xt = (X_train - mu) / sd
    Xv = (X_val - mu) / sd
    yt = y_train.astype(np.float32).reshape(-1, 1)

    def relu(x): return np.maximum(0, x)
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    # Adam state.
    params = [W1, b1, W2, b2, W3, b3]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n = Xt.shape[0]
    batch = min(64, n)
    for epoch in range(epochs):
        # Mini-batch shuffle.
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            xb = Xt[idx[s:s + batch]]
            yb = yt[idx[s:s + batch]]
            # Forward.
            z1 = xb @ W1 + b1; a1 = relu(z1)
            z2 = a1 @ W2 + b2; a2 = relu(z2)
            z3 = a2 @ W3 + b3; p = sig(z3)
            # BCE grad.
            dz3 = (p - yb) / max(xb.shape[0], 1)
            dW3 = a2.T @ dz3; db3 = dz3.sum(0)
            da2 = dz3 @ W3.T
            dz2 = da2 * (z2 > 0)
            dW2 = a1.T @ dz2; db2 = dz2.sum(0)
            da1 = dz2 @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = xb.T @ dz1; db1 = dz1.sum(0)
            grads = [dW1, db1, dW2, db2, dW3, db3]
            t = epoch * (n // batch + 1) + (s // batch) + 1
            for k, (p_, g_) in enumerate(zip(params, grads)):
                m_state[k] = beta1 * m_state[k] + (1 - beta1) * g_
                v_state[k] = beta2 * v_state[k] + (1 - beta2) * g_ * g_
                m_hat = m_state[k] / (1 - beta1 ** t)
                v_hat = v_state[k] / (1 - beta2 ** t)
                p_ -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Inference on val.
    z1 = Xv @ W1 + b1; a1 = relu(z1)
    z2 = a1 @ W2 + b2; a2 = relu(z2)
    z3 = a2 @ W3 + b3; p = sig(z3)
    return p.flatten()


def evaluate(ds: dict, n_folds: int = 5, seed: int = 0) -> dict:
    """5-fold CV: train PRM on train problems, score val problems' branches,
    pick argmax-PRM branch as predicted answer. Compare vs cmajc-vote and
    oracle."""
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
    fold_acc: list[dict] = []

    for fold_idx, train_mask, val_mask in cv_folds(pid, n_folds=n_folds, seed=seed):
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va = X[val_mask]
        scores = fit_mlp(X_tr, y_tr, X_va, seed=seed + fold_idx)

        # Group by problem.
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
            any_b_correct = any(
                grade.is_correct(v, g) for v in votes[int(p)] if v
            )
            oracle_correct.append(any_b_correct)
        fold_acc.append({
            "fold": fold_idx,
            "n_problems": int(np.unique(val_pids).size),
            "prm_acc": float(np.mean(prm_correct[-int(np.unique(val_pids).size):])),
            "cmajc_acc": float(np.mean(cmajc_correct[-int(np.unique(val_pids).size):])),
            "oracle_acc": float(np.mean(oracle_correct[-int(np.unique(val_pids).size):])),
        })

    return {
        "prm_acc": float(np.mean(prm_correct)),
        "cmajc_acc": float(np.mean(cmajc_correct)),
        "oracle_acc": float(np.mean(oracle_correct)),
        "n_problems": len(prm_correct),
        "fold_acc": fold_acc,
    }


def main() -> None:
    traces_dir = REPO / "e4" / "results" / "traces" / "cmajc-prm-N100-seed0-trace"
    outcome_path = REPO / "e4" / "results" / "raw_cmajc_k64_seed0_prm.jsonl"
    ds = build_dataset(traces_dir, outcome_path)
    print(f"dataset: X={ds['X'].shape} y_mean={ds['y'].mean():.4f}")

    res = evaluate(ds, n_folds=5, seed=0)
    print(f"\nCross-validated 5-fold (seed=0):")
    print(f"  cmajc-vote   acc: {res['cmajc_acc']:.4f}")
    print(f"  PRM-rerank   acc: {res['prm_acc']:.4f}")
    print(f"  oracle       acc: {res['oracle_acc']:.4f}")
    print(f"  N problems: {res['n_problems']}")

    delta = res["prm_acc"] - res["cmajc_acc"]
    gap = res["oracle_acc"] - res["cmajc_acc"]
    closure = (delta / gap) if gap > 1e-9 else 0.0
    print(f"\n  Δ(PRM - cmajc) = {delta:+.4f} ({delta * 100:+.2f}pp)")
    print(f"  voting-rule gap (oracle - cmajc) = {gap:.4f} ({gap * 100:.2f}pp)")
    print(f"  gap closure: {closure * 100:.1f}%")
    if delta >= 0.06:
        print("\n  PRE-REG VERDICT: WIN (≥+6pp lift, ≥75% closure)")
    elif delta >= 0.02:
        print("\n  PRE-REG VERDICT: PARTIAL (+2pp ≤ Δ < +6pp)")
    else:
        print("\n  PRE-REG VERDICT: LOSS (Δ < +2pp)")


if __name__ == "__main__":
    main()
