"""Option-3 process-reward verifier on the rich substrate.

Trains a small MLP over per-step trajectory features (mean/max/std of entropy,
commit-LoRA-active fraction, total-mass logit-shift, etc.) → P(correct).
5-fold CV split by problem_id.

Reads phase2/spikes/option3-process-reward/rich_substrate.jsonl (one record per
(problem, branch) with full per-step `records` list).

Pre-reg threshold (binding): mean verifier accuracy ≥ cmaj baseline + 5pp on
held-out folds. Same architecture comparison as option-2 train_verifier_option2.py:
WIN-DECISIVE / WIN-STRONG / WIN-MINOR / INCONCLUSIVE / LOSS thresholds.
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np

try:
    from scipy.stats import beta
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

import os as _os
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RICH_PATH = pathlib.Path(_os.environ.get("RICH_PATH",
    str(REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate.jsonl")))
OUT_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/option3_results.json"


def cp_ci(k, n, alpha=0.05):
    if n == 0 or not HAVE_SCIPY: return 0.0, 1.0
    low = float(beta.ppf(alpha/2, k, n-k+1)) if k > 0 else 0.0
    high = float(beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.0
    return low, high


def per_branch_features(records: list[dict]) -> np.ndarray:
    """Reduce a list of per-step StepState records to a fixed-length feature vec.

    Features (all per-trajectory aggregates over 4 sub-blocks):
      - sub_block count
      - mean entropy (per-position averaged across positions, then across blocks)
      - max entropy
      - std entropy
      - mean entropy of top-50% high-entropy positions (proxy for "uncertain spans")
      - mean entropy of top-50% low-entropy positions (proxy for "confident spans")
      - commit_lora_active fraction (across blocks)
      - mean/max logit_shift_norm (when commit-LoRA active)
      - mean wallclock_ms (compute time per block)
      - top-1 prob mean (averaged across all positions across blocks)
      - top-1 prob std
      - top-2 prob ratio mean (top-1 / top-2 confidence margin)
    """
    if not records: return np.zeros(13, dtype=np.float32)
    n_blocks = len(records)
    all_ent = np.concatenate([np.array(r.get("entropy", []), dtype=np.float32) for r in records])
    if len(all_ent) == 0: all_ent = np.array([0.0])
    sorted_ent = np.sort(all_ent)
    half = len(sorted_ent) // 2
    high_half = sorted_ent[half:]
    low_half = sorted_ent[:half] if half > 0 else sorted_ent
    commit_active = [bool(r.get("commit_lora_active", False)) for r in records]
    logit_shifts = [r.get("logit_shift_norm") for r in records if r.get("logit_shift_norm") is not None]
    walls = [int(r.get("wallclock_ms", 0)) for r in records]
    # top-1 / top-2 probs from top_k_logits
    top1_probs, top2_probs = [], []
    for r in records:
        for row in r.get("top_k_logits", []):
            if len(row) >= 1: top1_probs.append(float(row[0][1]))
            if len(row) >= 2: top2_probs.append(float(row[1][1]))
    top1 = np.array(top1_probs) if top1_probs else np.array([0.0])
    top2 = np.array(top2_probs) if top2_probs else np.array([1e-6])
    margin = top1 / np.maximum(top2, 1e-6)
    return np.array([
        float(n_blocks),
        float(all_ent.mean()),
        float(all_ent.max()),
        float(all_ent.std()),
        float(high_half.mean()) if len(high_half) else 0.0,
        float(low_half.mean()) if len(low_half) else 0.0,
        float(np.mean(commit_active)),
        float(np.mean(logit_shifts)) if logit_shifts else 0.0,
        float(np.max(logit_shifts)) if logit_shifts else 0.0,
        float(np.mean(walls)) / 1000.0,  # to seconds
        float(top1.mean()),
        float(top1.std()),
        float(margin.mean()),
    ], dtype=np.float32)


def load_rich() -> list[dict]:
    if not RICH_PATH.exists():
        raise FileNotFoundError(f"{RICH_PATH} doesn't exist; run make_rich_substrate.py first")
    return [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]


def evaluate_fold(rows, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_train = np.stack([per_branch_features(rows[i]["records"]) for i in train_idx])
    y_train = np.array([int(rows[i]["correct"]) for i in train_idx])
    X_test = np.stack([per_branch_features(rows[i]["records"]) for i in test_idx])
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return {"error": "degenerate train labels"}
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train); Xte = scaler.transform(X_test)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(Xtr, y_train)
    test_probs = clf.predict_proba(Xte)[:, 1]
    by_problem = defaultdict(list)
    for j, i in enumerate(test_idx):
        by_problem[rows[i]["problem_id"]].append((rows[i], test_probs[j]))
    cmaj_correct = ver_correct = oracle_correct = recovers = n = 0
    for pid, items in by_problem.items():
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
        "n_problems": n, "n_train_branches": len(train_idx), "n_test_branches": len(test_idx),
        "cmaj_acc": cmaj_correct/n, "verifier_acc": ver_correct/n, "oracle_acc": oracle_correct/n,
        "verifier_recovers": recovers,
        "delta_pp": (ver_correct - cmaj_correct)/n*100,
        "gap_closure": (ver_correct - cmaj_correct) / max(1e-9, oracle_correct - cmaj_correct),
        "feature_importance": [float(c) for c in clf.coef_[0]],
    }


def cv_split(rows, k=5, seed=0):
    rng = np.random.default_rng(seed)
    pids = sorted({r["problem_id"] for r in rows})
    rng.shuffle(pids)
    fold_size = len(pids) // k
    by_pid_idx = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid_idx[r["problem_id"]].append(i)
    folds = []
    for f in range(k):
        s, e = f*fold_size, (f+1)*fold_size if f < k-1 else len(pids)
        test_pids = set(pids[s:e])
        ti, tri = [], []
        for pid, idxs in by_pid_idx.items():
            (ti if pid in test_pids else tri).extend(idxs)
        folds.append((np.array(tri), np.array(ti)))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = load_rich()
    print(f"[opt3] {len(rows)} (problem, branch) records across "
          f"{len({r['problem_id'] for r in rows})} problems", flush=True)
    folds = cv_split(rows, k=args.cv, seed=args.seed)
    fold_results = []
    for i, (tr, te) in enumerate(folds):
        res = evaluate_fold(rows, tr, te)
        fold_results.append(res)
        if "error" in res:
            print(f"  fold {i}: SKIP ({res['error']})"); continue
        print(f"  fold {i}: cmaj={res['cmaj_acc']:.1%} verifier={res['verifier_acc']:.1%} "
              f"(d={res['delta_pp']:+.1f}pp gap-closure={res['gap_closure']:.1%}) "
              f"oracle={res['oracle_acc']:.1%} recovers={res['verifier_recovers']}", flush=True)

    valid = [r for r in fold_results if "error" not in r]
    mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
    mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
    mean_oracle = float(np.mean([r["oracle_acc"] for r in valid]))
    print(f"\n[opt3] === MEAN: cmaj={mean_cmaj:.1%} verifier={mean_ver:.1%} oracle={mean_oracle:.1%} ===")
    print(f"[opt3] dpp_vs_cmaj = {(mean_ver-mean_cmaj)*100:+.2f}")
    print(f"[opt3] gap-closure = {(mean_ver-mean_cmaj)/max(1e-9, mean_oracle-mean_cmaj):.1%}")

    # Pre-reg decision (binding): cmaj+5pp threshold
    target_pp = 5.0
    decision = "WIN" if (mean_ver - mean_cmaj) * 100 >= target_pp else \
               "PARITY" if abs((mean_ver - mean_cmaj) * 100) < 1.0 else \
               "LOSS"
    print(f"\n[opt3] >>> DECISION: {decision} (Δpp={(mean_ver-mean_cmaj)*100:+.2f}, target ≥+{target_pp}) <<<")

    # Feature importance summary
    if valid and "feature_importance" in valid[0]:
        feat_names = ["n_blocks","mean_ent","max_ent","std_ent","high_ent","low_ent",
                      "commit_active","mean_shift","max_shift","mean_wall_s",
                      "mean_top1","std_top1","margin"]
        avg_imp = np.mean([r["feature_importance"] for r in valid], axis=0)
        order = np.argsort(np.abs(avg_imp))[::-1]
        print("\n[opt3] feature importance (mean |coef| across folds):")
        for i in order:
            print(f"  {feat_names[i]:15s} coef={avg_imp[i]:+.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_branches": len(rows),
        "n_problems": len({r["problem_id"] for r in rows}),
        "cv_folds": args.cv,
        "fold_results": fold_results,
        "mean_cmaj": mean_cmaj, "mean_verifier": mean_ver, "mean_oracle": mean_oracle,
        "delta_pp_vs_cmaj": (mean_ver - mean_cmaj) * 100,
        "gap_closure": (mean_ver - mean_cmaj) / max(1e-9, mean_oracle - mean_cmaj),
        "decision": decision,
    }, indent=2))
    print(f"\n[opt3] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
