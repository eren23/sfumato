"""Train TF-IDF + Logistic Regression verifier on per-branch correctness labels.

5-fold CV on PROBLEM ID (not branch — branches from same problem are correlated).
Reports per-fold cmaj baseline vs verifier-rerank accuracy + 95% CIs.

Run after substrate jsonls land (or use Phase-1 only via --phase1-only flag).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from load_branches import BranchRow, load_phase1, load_substrate, extract_answer  # noqa: E402

import numpy as np  # noqa: E402

SPIKE_DIR = pathlib.Path(__file__).parent


def cp_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0: return (0.0, 1.0)
    from scipy.stats import beta
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (low, high)


def verify_rerank_accuracy(
    train_rows: list[BranchRow],
    test_rows: list[BranchRow],
    label: str = "fold",
) -> dict:
    """Train verifier on train_rows, eval verifier-rerank vs cmaj on test_rows."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X_train_text = [r.branch_text for r in train_rows]
    y_train = [int(r.correct) for r in train_rows]
    X_test_text = [r.branch_text for r in test_rows]
    y_test = [int(r.correct) for r in test_rows]

    if sum(y_train) == 0 or sum(y_train) == len(y_train):
        return {"label": label, "error": "degenerate train labels (all 0 or all 1)"}

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X_train_text, y_train)
    test_probs = clf.predict_proba(X_test_text)[:, 1]

    # Group by problem_id, apply verifier-argmax vs majority vote
    by_problem: dict[str, list[tuple[BranchRow, float]]] = defaultdict(list)
    for r, p in zip(test_rows, test_probs):
        by_problem[r.problem_id].append((r, p))

    cmaj_correct, verifier_correct, oracle_correct, n = 0, 0, 0, 0
    cmaj_minus_oracle = 0  # cases where cmaj fails BUT verifier picks wrong
    verifier_recovers = 0  # cases where cmaj fails AND verifier picks right
    for pid, branches in by_problem.items():
        n += 1
        gold = branches[0][0].gold
        votes = [r.extracted for r, _ in branches]
        cnt = Counter(votes)
        cmaj_pick = cnt.most_common(1)[0][0]
        cmaj_hit = (cmaj_pick == gold)
        if cmaj_hit:
            cmaj_correct += 1
        if any(v == gold for v in votes):
            oracle_correct += 1
        # Verifier rerank
        scored = sorted(branches, key=lambda x: x[1], reverse=True)
        verifier_pick = scored[0][0].extracted
        verifier_hit = (verifier_pick == gold)
        if verifier_hit:
            verifier_correct += 1
        if not cmaj_hit and verifier_hit:
            verifier_recovers += 1
        if not cmaj_hit and not verifier_hit and any(v == gold for v in votes):
            cmaj_minus_oracle += 1

    return {
        "label": label,
        "n_problems": n,
        "n_train_branches": len(train_rows),
        "n_test_branches": len(test_rows),
        "cmaj_acc": cmaj_correct / n,
        "cmaj_ci": cp_ci(cmaj_correct, n),
        "verifier_acc": verifier_correct / n,
        "verifier_ci": cp_ci(verifier_correct, n),
        "oracle_acc": oracle_correct / n,
        "oracle_ci": cp_ci(oracle_correct, n),
        "verifier_recovers_failed_cmaj": verifier_recovers,
        "verifier_misses_recoverable": cmaj_minus_oracle,
        "delta_pp_vs_cmaj": (verifier_correct - cmaj_correct) / n * 100,
    }


def cv_split_by_problem(rows: list[BranchRow], k: int = 5, seed: int = 0) -> list[tuple[list[BranchRow], list[BranchRow]]]:
    """5-fold CV split on problem_id."""
    rng = np.random.default_rng(seed)
    pids = sorted({r.problem_id for r in rows})
    rng.shuffle(pids)
    fold_size = len(pids) // k
    by_fold = []
    for f in range(k):
        start = f * fold_size
        end = (f + 1) * fold_size if f < k - 1 else len(pids)
        test_pids = set(pids[start:end])
        train_rows = [r for r in rows if r.problem_id not in test_pids]
        test_rows = [r for r in rows if r.problem_id in test_pids]
        by_fold.append((train_rows, test_rows))
    return by_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-only", action="store_true", help="train on Phase-1 jsonls only")
    parser.add_argument("--cv", type=int, default=5, help="k for k-fold CV")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    p1 = load_phase1()
    sub = load_substrate() if not args.phase1_only else []
    all_rows = p1 + sub
    print(f"Phase-1 branches: {len(p1)} | Substrate branches: {len(sub)} | Total: {len(all_rows)}")

    if args.phase1_only or not sub:
        print(f"\n=== Within-Phase-1 {args.cv}-fold CV (split by problem_id) ===")
        folds = cv_split_by_problem(all_rows, k=args.cv, seed=args.seed)
        fold_results = []
        for i, (train, test) in enumerate(folds):
            res = verify_rerank_accuracy(train, test, label=f"fold-{i}")
            fold_results.append(res)
            if "error" in res:
                print(f"  fold {i}: SKIP ({res['error']})")
                continue
            print(f"  fold {i}: cmaj={res['cmaj_acc']:.1%} verifier={res['verifier_acc']:.1%} "
                  f"(Δ={res['delta_pp_vs_cmaj']:+.1f}pp) oracle={res['oracle_acc']:.1%} "
                  f"recovers={res['verifier_recovers_failed_cmaj']}")
        # Aggregate
        valid = [r for r in fold_results if "error" not in r]
        if valid:
            mean_cmaj = np.mean([r["cmaj_acc"] for r in valid])
            mean_ver = np.mean([r["verifier_acc"] for r in valid])
            mean_oracle = np.mean([r["oracle_acc"] for r in valid])
            print(f"\n  MEAN: cmaj={mean_cmaj:.1%} verifier={mean_ver:.1%} oracle={mean_oracle:.1%}")
            print(f"  ΔppCMAJ={(mean_ver - mean_cmaj) * 100:+.2f} | gap-closure={(mean_ver - mean_cmaj) / max(1e-9, mean_oracle - mean_cmaj):.1%}")
    else:
        print(f"\n=== Train on Phase-1 (N={len(p1)} branches), eval on Phase-2 substrate (N={len(sub)} branches) ===")
        res = verify_rerank_accuracy(p1, sub, label="p1-train_p2-eval")
        if "error" in res:
            print(f"  SKIP: {res['error']}")
            return
        print(f"  N_problems={res['n_problems']}")
        print(f"  cmaj   = {res['cmaj_acc']:.1%}  CI={tuple(round(x,3) for x in res['cmaj_ci'])}")
        print(f"  verifier = {res['verifier_acc']:.1%}  CI={tuple(round(x,3) for x in res['verifier_ci'])}")
        print(f"  oracle = {res['oracle_acc']:.1%}  CI={tuple(round(x,3) for x in res['oracle_ci'])}")
        print(f"  Δpp_vs_cmaj = {res['delta_pp_vs_cmaj']:+.2f}pp")
        print(f"  gap-closure (ver-cmaj)/(oracle-cmaj) = {(res['verifier_acc']-res['cmaj_acc'])/max(1e-9,res['oracle_acc']-res['cmaj_acc']):.1%}")
        fold_results = [res]

    # Apply pre-registered decision rule (use MEAN across folds; best-of-folds = noise)
    out = {"folds": fold_results, "pre_reg_decision": None}
    valid = [r for r in fold_results if "error" not in r]
    if valid:
        mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
        mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
        if mean_ver >= 0.89: decision = "WIN-DECISIVE"
        elif mean_ver >= 0.87: decision = "WIN-STRONG"
        elif mean_ver >= 0.83: decision = "WIN-MINOR"
        elif mean_ver >= mean_cmaj - 0.01: decision = "INCONCLUSIVE"  # within 1pp of cmaj baseline
        else: decision = "LOSS"
        out["pre_reg_decision"] = decision
        out["mean_verifier_acc"] = mean_ver
        out["mean_cmaj_acc"] = mean_cmaj
        print(f"\n>>> PRE-REG DECISION: {decision} (mean verifier acc = {mean_ver:.1%} vs mean cmaj = {mean_cmaj:.1%})")

    out_path = SPIKE_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
