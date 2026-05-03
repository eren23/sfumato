# Pre-Registration — Verifier-Based Branch Aggregation (D3.5 Spike)

**Author:** Phase-2 orchestrator | **Pre-registered at:** 2026-05-02 ~19:00 UTC
**Spike for proposal:** `phase2/proposals/verifier-based-aggregation.md`
**Budget cap:** $0.50 (target ~$0.05; CPU-only training)

## Hypothesis under test

A small text-only verifier (TF-IDF features + logistic regression) trained on
per-branch correctness labels from existing cmaj jsonls can replace majority-
vote aggregation and recover ≥ +5pp of the 12pp voting-rule gap measured at
τ=0.7 in the temperature-diversity-falsifier spike.

If true → D3.5 graduates to a Phase-2 paper section.
If false → text-only features insufficient; escalate to option 2 (Qwen-encoder
verifier) OR kill D3.5 in favor of D1/D2.

## Procedure (committed before any code runs)

### Training set (free, no GPU)

Concatenate per-branch records from existing Phase-1 jsonls:
- `e4/results/raw_cmaj_k64_seed0_b5_t0.3.jsonl` (τ=0.3, N=50, b=5 → 250 branches)
- `e4/results/raw_cmaj_k64_seed0_b5.jsonl`     (τ=0.7, N=50, b=5 → 250 branches)
- `e4/results/raw_cmaj_k64_seed0_b5_t1.0.jsonl` (τ=1.0, N=50, b=5 → 250 branches)

**Total Phase-1 training: 750 labeled (problem, branch_text, gold, correct) tuples.**

If queue Phase-2 substrate harvest completes (cmaj N=100 b=5 τ∈{0.7, 1.0}):
- adds 1000 more labeled branches
- **Total: 1750 labeled branches**

### Eval / held-out set (need to make sure this exists)

Held-out should be drawn from problems NOT in the training set. Two paths:

(a) **Within-Phase-1 split**: train on the 750 Phase-1 branches but with 5-fold
    CV split on PROBLEM ID (not branch). Report mean per-fold reranked accuracy.

(b) **Substrate-as-eval**: train on Phase-1 N=50 jsonls, eval on substrate
    N=100 jsonls. Cleaner train/test split. **Preferred IF substrate completes.**

Decision: use (b) if substrate jsonls exist by spike-fire-time, else (a).

### Architecture (option 1 — TF-IDF + LR)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Features per branch:
#   - branch text (TF-IDF on word n-grams, n=1..2, max_features=10k)
#   - extracted_answer (one-hot top-50 most common answers in train)
#   - branch length (chars), normalized
#   - simple syntactic features (count of "Answer:", "=", "$")
# 
# Label: 1 if extract_answer(branch_text) == gold else 0
# Loss: BCE (sklearn LR default)

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ("lr", LogisticRegression(C=1.0, max_iter=1000)),
])
```

### Inference rule

For each test problem with branches $\{y_i\}_{i=1}^{5}$:
1. Score each branch: $s_i = \text{clf.predict\_proba}(y_i)[1]$
2. Pick winner: $i^* = \arg\max_i s_i$
3. Predicted answer: $\hat{a} = \text{extract}(y_{i^*})$

Compare to baseline: $\hat{a}_\text{cmaj} = \text{mode}(\{\text{extract}(y_i)\})$.

## Decision rules (pre-committed)

| Outcome (verifier accuracy on held-out) | Decision |
|---|---|
| ≥ 89% (within 1pp of oracle 90%) | **WIN-DECISIVE**: full paper section, target NeurIPS/ICLR |
| ≥ 87% (oracle − 3pp) | **WIN-STRONG**: full Phase-2 follow-up experiment |
| ≥ 83% (cmaj baseline + 5pp) | **WIN-MINOR**: D3.5 graduates, paper note |
| 78%—82% (within noise of cmaj) | **INCONCLUSIVE**: try option 2 (Qwen verifier), $0.40 |
| ≤ 78% (parity with cmaj or worse) | **LOSS**: text-only insufficient, escalate to embedding-based verifier OR kill D3.5 |

Cmaj baseline at τ=0.7 from Phase-1: **78%** (from spike RESULT.md).
Oracle ceiling at τ=0.7 from Phase-1: **90%**.

## Auxiliary diagnostics (not gating)

- **Per-τ reranked accuracy**: report verifier acc separately for τ=0.3, 0.7, 1.0 to detect distribution shift.
- **Confidence calibration**: ECE on the verifier's predicted probabilities. If miscalibrated, use temperature-scaled output for inference rule.
- **Feature importance**: top-50 TF-IDF features by |coef| — sanity check (should be math-related tokens, not noise).
- **Failure mode bucket**: of cases where verifier-top-1 picks wrong, was the right answer in the b=5 branches? If yes, the verifier is failing to discriminate; if no, no aggregation method could have helped.

## Compute budget

- **Training**: CPU only, scikit-learn TF-IDF + LR on 1750 examples ≈ 30 sec.
- **Eval**: CPU only, ~2 sec to score N=100 × 5 = 500 test branches.
- **Total cost: $0** (uses local laptop CPU).

If escalation to option 2 (Qwen-0.5B encoder) needed: ~$0.40 on RTX 4090.

## Pre-reg integrity statement

I commit to publishing the result in `RESULT.md` regardless of which decision
rule fires, will not alter the win/loss thresholds after seeing data, and will
report all diagnostic findings (especially failure modes) honestly. SHA of
this file at commit time goes into RESULT.md as audit trail.
