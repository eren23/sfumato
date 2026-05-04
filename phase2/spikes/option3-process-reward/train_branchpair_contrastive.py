"""Branch-pair contrastive verifier — siblings within a problem, pair-ranked.

The night-2 process-MLP/embedding verifiers were all LOSS because they predict
P(correct) for each branch independently. But the actual decision is *which of
THIS problem's siblings is best*. Sibling pairs share the problem prompt, so
they cancel out the prompt-difficulty confound that absolute-label models can't.

This trains a pair-ranking model: for each (problem, branch_correct,
branch_wrong) pair we have, predict which is better. At test time, score each
test branch independently against a reference, pick argmax score.

Pre-reg: same WIN/LOSS thresholds as option-3 (verifier ≥ cmaj + 5pp).

Reads: rich_substrate_n200.jsonl (or RICH_PATH env var).
Writes: phase2/spikes/option3-process-reward/option3_pair_results.json
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
OUT_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/option3_pair_results.json"

sys.path.insert(0, str(REPO_ROOT / "phase2/spikes/option3-process-reward"))
from train_process_verifier import per_branch_features, cv_split  # noqa: E402


def make_pairs(rows_by_problem):
    """For each problem with ≥1 correct + ≥1 wrong branch, emit (winner_feat, loser_feat) pairs."""
    pairs = []
    for pid, items in rows_by_problem.items():
        correct = [r for r in items if r["correct"]]
        wrong = [r for r in items if not r["correct"]]
        for c in correct:
            for w in wrong:
                pairs.append((per_branch_features(c["records"]),
                              per_branch_features(w["records"])))
    return pairs


def evaluate_fold(rows, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    train_by_pid = defaultdict(list)
    for r in train_rows: train_by_pid[r["problem_id"]].append(r)
    test_by_pid = defaultdict(list)
    for r in test_rows: test_by_pid[r["problem_id"]].append(r)

    pairs = make_pairs(train_by_pid)
    if len(pairs) < 5:
        return {"error": f"too few train pairs ({len(pairs)})"}
    # Pair-features: (winner - loser) labeled +1; (loser - winner) labeled -1.
    Xa = np.stack([w - l for w, l in pairs])
    Xb = -Xa
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
    # Score = sigmoid(dot(coef, scaled_branch_features)). Per-branch absolute score.
    cmaj_correct = ver_correct = oracle_correct = recovers = n = 0
    for pid, items in test_by_pid.items():
        n += 1
        gold = items[0]["gold"]
        feats = np.stack([per_branch_features(r["records"]) for r in items])
        scores = clf.decision_function(scaler.transform(feats))
        votes = [r["extracted"] for r in items]
        cmaj_pick = Counter(votes).most_common(1)[0][0]
        cmaj_hit = (cmaj_pick == gold)
        if cmaj_hit: cmaj_correct += 1
        if any(v == gold for v in votes): oracle_correct += 1
        ver_pick = items[int(np.argmax(scores))]["extracted"]
        if ver_pick == gold: ver_correct += 1
        if not cmaj_hit and ver_pick == gold: recovers += 1
    return {
        "n_problems": n, "n_train_pairs": len(pairs),
        "cmaj_acc": cmaj_correct/n, "verifier_acc": ver_correct/n, "oracle_acc": oracle_correct/n,
        "verifier_recovers": recovers,
        "delta_pp": (ver_correct - cmaj_correct)/n*100,
        "gap_closure": (ver_correct - cmaj_correct) / max(1e-9, oracle_correct - cmaj_correct),
        "feature_importance": [float(c) for c in clf.coef_[0]],
    }


def main():
    if not RICH_PATH.exists():
        raise FileNotFoundError(RICH_PATH)
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    print(f"[pair] loaded {len(rows)} branches from {RICH_PATH.name}", flush=True)
    folds = cv_split(rows, k=5, seed=0)
    fold_results = []
    for i, (tr, te) in enumerate(folds):
        print(f"--- fold {i} ---", flush=True)
        r = evaluate_fold(rows, tr, te)
        fold_results.append(r)
        if "error" in r: print(f"  fold {i}: SKIP {r['error']}"); continue
        print(f"  fold {i}: cmaj={r['cmaj_acc']:.1%} verifier={r['verifier_acc']:.1%} "
              f"(d={r['delta_pp']:+.1f}pp) oracle={r['oracle_acc']:.1%} pairs={r['n_train_pairs']}",
              flush=True)
    valid = [r for r in fold_results if "error" not in r]
    mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
    mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
    mean_oracle = float(np.mean([r["oracle_acc"] for r in valid]))
    print(f"\n[pair] === MEAN: cmaj={mean_cmaj:.1%} verifier={mean_ver:.1%} oracle={mean_oracle:.1%} ===")
    dpp = (mean_ver - mean_cmaj) * 100
    if mean_ver >= 0.89: dec = "WIN-DECISIVE"
    elif mean_ver >= 0.87: dec = "WIN-STRONG"
    elif mean_ver >= 0.83: dec = "WIN-MINOR"
    elif mean_ver >= mean_cmaj - 0.01: dec = "INCONCLUSIVE"
    else: dec = "LOSS"
    print(f"[pair] dpp={dpp:+.2f}  >>> DECISION: {dec} <<<")
    OUT_PATH.write_text(json.dumps({
        "rich_path": str(RICH_PATH), "n_branches": len(rows),
        "n_problems": len({r["problem_id"] for r in rows}),
        "fold_results": fold_results,
        "mean_cmaj": mean_cmaj, "mean_verifier": mean_ver, "mean_oracle": mean_oracle,
        "delta_pp_vs_cmaj": dpp,
        "decision": dec,
    }, indent=2))
    print(f"[pair] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
