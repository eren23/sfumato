"""D1 bandit-on-replay mode router.

Trains a per-problem condition-selection policy on the 20-idx ×
12-condition full-coverage substrate from showcase/static/examples.json.

Decision rules (pre-reg):
  WIN     — LOOCV bandit acc ≥ 0.85 (≥ +5pp over always-cmajc 0.80)
  NEUTRAL — bandit acc ∈ [0.80, 0.85)
  LOSS    — bandit acc < 0.80

Usage:
  python phase2/spikes/D1-mode-router/bandit.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "phase2" / "showcase" / "static" / "examples.json"
OUT = Path(__file__).resolve().parent / "results.json"


def load_substrate():
    """Returns
        idxs: sorted list of idxs with full 12-condition coverage
        conditions: list of condition names
        correct: dict[(idx, condition)] -> 0/1
        questions: dict[idx] -> question text
        gold: dict[idx] -> gold answer string
    """
    data = json.loads(EXAMPLES.read_text())
    recs = data["records"]
    idx2conds = defaultdict(set)
    correct = {}
    questions = {}
    gold = {}
    for r in recs:
        idx = r["idx"]
        cond = r["condition"]
        idx2conds[idx].add(cond)
        # Use first record per (idx, cond) — they're seeded variants but the
        # showcase keeps them deduped.
        key = (idx, cond)
        if key not in correct:
            correct[key] = int(bool(r["correct"]))
        if idx not in questions:
            questions[idx] = r["question"]
            gold[idx] = r["gold"]
    # Find idxs covered by all 12 conditions
    full_conds = sorted({c for cs in idx2conds.values() for c in cs})
    full_idxs = sorted(idx for idx, cs in idx2conds.items() if len(cs) == 12)
    return full_idxs, full_conds, correct, questions, gold


def featurize_simple(text: str) -> dict:
    words = text.split()
    nums = re.findall(r"\d+", text)
    return {
        "len_chars": len(text),
        "len_words": len(words),
        "n_sentences": len(re.split(r"[.!?]", text)),
        "n_numbers": len(nums),
        "max_number": max((int(n) for n in nums), default=0),
        "has_then": int("then" in text.lower()),
        "has_after": int("after" in text.lower()),
        "has_more_than": int("more than" in text.lower()),
        "has_less_than": int("less than" in text.lower()),
        "has_total": int("total" in text.lower()),
        "has_each": int("each" in text.lower()),
        "starts_what": int(text.lower().startswith("what")),
        "starts_how": int(text.lower().startswith("how")),
    }


def main() -> None:
    idxs, conds, correct, questions, gold = load_substrate()
    print(f"Substrate: {len(idxs)} idxs × {len(conds)} conditions = {len(correct)} tuples")
    print(f"Conditions: {conds}\n")

    # Always-X baselines on this slice
    print("=== always-X baselines (on 20 idxs) ===")
    baselines = {}
    for c in conds:
        acc = sum(correct[(i, c)] for i in idxs) / len(idxs)
        baselines[c] = acc
        print(f"  always-{c:8s}: {acc:.3f}")
    print()

    # Oracle: any-condition-correct = upper bound
    oracle = sum(any(correct[(i, c)] for c in conds) for i in idxs) / len(idxs)
    print(f"oracle (any-condition-correct): {oracle:.3f}")
    print()

    # Best fixed-mechanism baseline
    best_fixed_cond = max(baselines, key=baselines.get)
    best_fixed_acc = baselines[best_fixed_cond]
    print(f"best-fixed-baseline: always-{best_fixed_cond} = {best_fixed_acc:.3f}")
    print()

    # Build features per idx
    simple_feats = {i: featurize_simple(questions[i]) for i in idxs}
    feat_keys = sorted(simple_feats[idxs[0]].keys())
    X_simple = np.array([
        [simple_feats[i][k] for k in feat_keys]
        for i in idxs
    ], dtype=float)

    # TF-IDF features over questions
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=80, min_df=2)
    X_tfidf = tfidf.fit_transform([questions[i] for i in idxs]).toarray()
    X = np.concatenate([X_simple, X_tfidf], axis=1)
    print(f"Feature dim: {X.shape[1]} ({X_simple.shape[1]} simple + {X_tfidf.shape[1]} tf-idf)")
    print()

    # Bandit label per idx: best-correct condition (the supervised oracle target).
    # If multiple conditions correct, pick the cheapest by FLOPs proxy
    # (lower priority index = cheaper). Tie-break order matches sweep
    # observed rough cost ranking: c1 < c2 < c2c < c2hint < c2empty <
    # crev < c3 < c3p < c4 < cmaj < cmerge < cmajc.
    cost_order = ["c1", "c2", "c2c", "c2hint", "c2empty", "crev", "c3", "c3p", "c4", "cmaj", "cmerge", "cmajc"]
    cond_to_idx = {c: i for i, c in enumerate(conds)}

    def best_action(idx: int) -> str | None:
        for c in cost_order:
            if c not in cond_to_idx:
                continue
            if correct[(idx, c)]:
                return c
        return None  # all-wrong; no good action

    y_label = np.array([cond_to_idx.get(best_action(i), -1) for i in idxs])
    valid_mask = y_label >= 0
    print(f"Idxs with at least one correct condition: {valid_mask.sum()} / {len(idxs)}")
    print()

    # LOOCV
    n = len(idxs)
    bandit_correct = []
    rf_correct = []
    chosen_actions = []
    rf_chosen_actions = []
    for held in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[held] = False
        # Skip if held-out has no positive label (oracle wrong) — bandit can't win
        # those anyway; counted as 0 in scoring.
        Xtr = X[train_mask & valid_mask]
        ytr = y_label[train_mask & valid_mask]
        if len(np.unique(ytr)) < 2:
            # Fall back to majority-class
            pred_action_idx = int(np.bincount(ytr).argmax()) if len(ytr) else cond_to_idx["cmajc"]
            pred_action_idx_rf = pred_action_idx
        else:
            try:
                clf = LogisticRegression(
                    max_iter=2000, multi_class="multinomial",
                    class_weight="balanced",
                )
                clf.fit(Xtr, ytr)
                pred_action_idx = int(clf.predict(X[held:held+1])[0])
            except Exception:
                pred_action_idx = cond_to_idx["cmajc"]
            try:
                rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
                rf.fit(Xtr, ytr)
                pred_action_idx_rf = int(rf.predict(X[held:held+1])[0])
            except Exception:
                pred_action_idx_rf = cond_to_idx["cmajc"]
        chosen_cond = conds[pred_action_idx]
        chosen_cond_rf = conds[pred_action_idx_rf]
        chosen_actions.append((idxs[held], chosen_cond))
        rf_chosen_actions.append((idxs[held], chosen_cond_rf))
        bandit_correct.append(correct[(idxs[held], chosen_cond)])
        rf_correct.append(correct[(idxs[held], chosen_cond_rf)])

    lr_acc = np.mean(bandit_correct)
    rf_acc = np.mean(rf_correct)
    print("=== LOOCV bandit results ===")
    print(f"  Logistic Regression: {lr_acc:.3f}")
    print(f"  Random Forest:       {rf_acc:.3f}")
    print()

    # Pre-reg verdict
    best_bandit = max(lr_acc, rf_acc)
    best_clf = "logistic" if lr_acc >= rf_acc else "random_forest"
    delta = best_bandit - best_fixed_acc
    print(f"Best bandit clf: {best_clf} = {best_bandit:.3f}")
    print(f"Best fixed baseline: always-{best_fixed_cond} = {best_fixed_acc:.3f}")
    print(f"Δ vs best-fixed: {delta * 100:+.1f}pp")
    print()
    if best_bandit >= 0.85:
        verdict = "WIN"
    elif best_bandit >= 0.80:
        verdict = "NEUTRAL"
    else:
        verdict = "LOSS"
    print(f"Pre-reg verdict: {verdict}")

    # Action distribution
    from collections import Counter
    print()
    print("Bandit action distribution (LR):")
    for c, k in Counter(c for _, c in chosen_actions).most_common():
        print(f"  {c}: {k}")
    print("Bandit action distribution (RF):")
    for c, k in Counter(c for _, c in rf_chosen_actions).most_common():
        print(f"  {c}: {k}")

    # Save results
    OUT.write_text(json.dumps({
        "idxs": idxs,
        "conditions": conds,
        "baselines": baselines,
        "oracle": oracle,
        "best_fixed_cond": best_fixed_cond,
        "best_fixed_acc": best_fixed_acc,
        "loocv_lr_acc": lr_acc,
        "loocv_rf_acc": rf_acc,
        "best_bandit_acc": best_bandit,
        "best_bandit_clf": best_clf,
        "delta_vs_best_fixed_pp": delta * 100,
        "verdict": verdict,
        "lr_actions": chosen_actions,
        "rf_actions": rf_chosen_actions,
    }, indent=2))
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
