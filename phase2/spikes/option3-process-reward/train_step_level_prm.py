"""Step-level PRM verifier — actual per-step rewards, not aggregate features.

The night-2 process-MLP collapsed step features to 13 trajectory aggregates.
This trains a per-step model: given each StepState's full features (entropy,
top-k, commit-LoRA, logit-shift) + position-in-trajectory, predict step
"correctness contribution". Branch score = sum of per-step rewards.

Architecture: small MLP that takes (per-step feature vec ⊕ block_idx_onehot ⊕
trajectory_global_summary) → scalar reward. Train: each step's label = the
final branch correctness (weak supervision).

Pre-reg: same WIN/LOSS thresholds as option-3.

Reads: rich_substrate_n200.jsonl (or RICH_PATH env var).
Writes: phase2/spikes/option3-process-reward/option3_step_prm_results.json
"""
from __future__ import annotations
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RICH_PATH = pathlib.Path(os.environ.get("RICH_PATH",
    str(REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate_n200.jsonl")))
OUT_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/option3_step_prm_results.json"

sys.path.insert(0, str(REPO_ROOT / "phase2/spikes/option3-process-reward"))
from train_process_verifier import cv_split  # noqa: E402

# Feature size per step (must match _step_features below)
STEP_FEAT_DIM = 14
N_BLOCKS_MAX = 4  # GSM8K substrate uses 4 sub-blocks


def _step_features(record: dict, block_idx: int) -> np.ndarray:
    ent = np.array(record.get("entropy", []), dtype=np.float32)
    top_k = record.get("top_k_logits", [])
    top1 = np.array([row[0][1] if len(row) >= 1 else 0.0 for row in top_k], dtype=np.float32)
    top2 = np.array([row[1][1] if len(row) >= 2 else 1e-6 for row in top_k], dtype=np.float32)
    margin = top1 / np.maximum(top2, 1e-6)
    n = max(1, len(ent))
    return np.array([
        ent.mean() if len(ent) else 0.0,
        ent.std() if len(ent) else 0.0,
        ent.max() if len(ent) else 0.0,
        ent.min() if len(ent) else 0.0,
        top1.mean() if len(top1) else 0.0,
        top1.std() if len(top1) else 0.0,
        margin.mean() if len(margin) else 0.0,
        margin.max() if len(margin) else 0.0,
        float(record.get("commit_lora_active", False)),
        float(record.get("logit_shift_norm") or 0.0),
        float(record.get("wallclock_ms", 0)) / 1000.0,
        float(len(record.get("tokens_committed", []) or []) if isinstance(record.get("tokens_committed"), list) else (record.get("tokens_committed") or 0)),
        float(block_idx) / N_BLOCKS_MAX,
        float(n) / 32.0,
    ], dtype=np.float32)


def _branch_step_features(records: list[dict]) -> np.ndarray:
    """Returns (n_blocks, STEP_FEAT_DIM) — variable-length, padded later."""
    if not records: return np.zeros((1, STEP_FEAT_DIM), dtype=np.float32)
    return np.stack([_step_features(r, r.get("sub_block", i))
                     for i, r in enumerate(records)])


def evaluate_fold(rows, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Build per-step training set: each step is one example, label = branch correct
    X_train_steps, y_train_steps = [], []
    for i in train_idx:
        feats = _branch_step_features(rows[i]["records"])
        label = int(rows[i]["correct"])
        for f in feats:
            X_train_steps.append(f); y_train_steps.append(label)
    X_train = np.stack(X_train_steps)
    y_train = np.array(y_train_steps)
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return {"error": "degenerate train labels"}
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y_train)
    # Branch score = mean of per-step P(correct) (could also try min/sum)
    test_by_pid = defaultdict(list)
    for i in test_idx:
        feats = _branch_step_features(rows[i]["records"])
        step_probs = clf.predict_proba(scaler.transform(feats))[:, 1]
        # Aggregate to one branch-level reward — try mean (also store min for ablation)
        branch_score_mean = float(step_probs.mean())
        test_by_pid[rows[i]["problem_id"]].append((rows[i], branch_score_mean))
    cmaj_correct = ver_correct = oracle_correct = recovers = n = 0
    for pid, items in test_by_pid.items():
        n += 1
        gold = items[0][0]["gold"]
        votes = [r["extracted"] for r, _ in items]
        cmaj_pick = Counter(votes).most_common(1)[0][0]
        cmaj_hit = (cmaj_pick == gold)
        if cmaj_hit: cmaj_correct += 1
        if any(v == gold for v in votes): oracle_correct += 1
        ver_pick = max(items, key=lambda x: x[1])[0]["extracted"]
        if ver_pick == gold: ver_correct += 1
        if not cmaj_hit and ver_pick == gold: recovers += 1
    return {
        "n_problems": n, "n_train_steps": len(X_train_steps),
        "cmaj_acc": cmaj_correct/n, "verifier_acc": ver_correct/n, "oracle_acc": oracle_correct/n,
        "verifier_recovers": recovers,
        "delta_pp": (ver_correct - cmaj_correct)/n*100,
        "feature_importance": [float(c) for c in clf.coef_[0]],
    }


def main():
    if not RICH_PATH.exists():
        raise FileNotFoundError(RICH_PATH)
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    print(f"[step-prm] loaded {len(rows)} branches from {RICH_PATH.name}", flush=True)
    folds = cv_split(rows, k=5, seed=0)
    fold_results = []
    for i, (tr, te) in enumerate(folds):
        print(f"--- fold {i} ---", flush=True)
        r = evaluate_fold(rows, tr, te)
        fold_results.append(r)
        if "error" in r: print(f"  fold {i}: SKIP {r['error']}"); continue
        print(f"  fold {i}: cmaj={r['cmaj_acc']:.1%} verifier={r['verifier_acc']:.1%} "
              f"(d={r['delta_pp']:+.1f}pp) oracle={r['oracle_acc']:.1%}", flush=True)
    valid = [r for r in fold_results if "error" not in r]
    mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
    mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
    mean_oracle = float(np.mean([r["oracle_acc"] for r in valid]))
    dpp = (mean_ver - mean_cmaj) * 100
    print(f"\n[step-prm] === MEAN: cmaj={mean_cmaj:.1%} verifier={mean_ver:.1%} oracle={mean_oracle:.1%} ===")
    if mean_ver >= 0.89: dec = "WIN-DECISIVE"
    elif mean_ver >= 0.87: dec = "WIN-STRONG"
    elif mean_ver >= 0.83: dec = "WIN-MINOR"
    elif mean_ver >= mean_cmaj - 0.01: dec = "INCONCLUSIVE"
    else: dec = "LOSS"
    print(f"[step-prm] dpp={dpp:+.2f}  >>> DECISION: {dec} <<<")
    OUT_PATH.write_text(json.dumps({
        "rich_path": str(RICH_PATH), "n_branches": len(rows),
        "n_problems": len({r["problem_id"] for r in rows}),
        "fold_results": fold_results,
        "mean_cmaj": mean_cmaj, "mean_verifier": mean_ver, "mean_oracle": mean_oracle,
        "delta_pp_vs_cmaj": dpp, "decision": dec,
    }, indent=2))
    print(f"[step-prm] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
