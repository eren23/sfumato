# Pre-registration — D1 Bandit-on-Replay Mode Router

**Date:** 2026-05-06 | **Spike:** Track C of post-re-orientation plan.

## Hypothesis

A contextual bandit trained over the 20-idx × 12-condition full-coverage
substrate (showcase `examples.json`) can learn a per-problem mode-routing
policy that beats the **always-cmajc** baseline by ≥ +5pp on
leave-one-out cross-validation.

This is the offline-replay version of the original D1 proposal
(`phase2/proposals/adaptive-mode-router.md`) — sub-block-level routing
is out of scope (no real-mode trace data at scale), but **per-problem
condition selection** is testable on the existing substrate.

## Substrate

- Source: `phase2/showcase/static/examples.json` records.
- 20 idxs (problems) where ALL 12 conditions have outcomes:
  c1, c2, c2c, c2empty, c2hint, c3, c3p, c4, cmaj, cmajc, cmerge, crev.
- Per-(idx, condition) tuple: `correct ∈ {0, 1}`.
- 240 tuples total.

## Baselines

| Baseline | Meaning | Expected acc on 20 idxs |
|---|---|---|
| always-c1 | always pick Qwen-0.5B AR alone | ~0.25 (sweep cell) |
| always-cmaj | always 5-branch vote, no commit-LoRA | ~0.75 (sweep cell) |
| **always-cmajc** | always 5-branch + commit-LoRA (sfumato-v3 default) | ~0.80 (sweep cell) |
| oracle (any-condition-correct) | upper bound: pick whichever condition got it | TBD (compute from data) |

## Action space

12 discrete actions (one per condition). Constant action cost (FLOPs differ
across conditions — flagged in the result for cost-aware analysis but not
penalized in the bandit reward).

## Features

Per-problem features extracted from `question` text:
- Length (chars, words, sentence count)
- Number-token count
- First interrogative word (one-hot from {what, how, find, calculate, ...})
- TF-IDF unigram + bigram features (truncated to top 200)
- Question complexity proxy (presence of multi-step keywords like "then",
  "after", "more than")

Optional v2 features (mech-aware): for each condition outcome we have access
to the per-branch reasoning text — could mean-pool simple stats. Skip for
v1 to keep the policy general.

## Method

- **Classifier:** logistic regression with one-vs-rest, then argmax → action.
  Optionally random forest as v2.
- **Cross-validation:** leave-one-idx-out (20 folds). Train on 19, predict
  the action for the 1 held-out idx, score `correct[(held_idx, predicted_action)]`.
- **Tie-breaking:** if multiple conditions tied on training, pick lowest-FLOPs.

## Decision rules

| Outcome | Verdict |
|---|---|
| LOOCV-bandit acc ≥ 0.85 (≥ +5pp over always-cmajc 0.80) | **WIN** — paper-class result; the missing E1 contribution lands |
| LOOCV-bandit acc ∈ [0.80, 0.85) | **NEUTRAL** — non-trivial policy but not worth the complexity vs always-cmajc |
| LOOCV-bandit acc < 0.80 | **LOSS** — bandit underperforms the simple baseline; per-problem features insufficient at N=20 |

Secondary diagnostic: compare to oracle ceiling. Headroom = oracle − bandit.

## Anti-goals

- No GPU. Pure local sklearn.
- No deep features (no LLaDA hidden states, no Qwen embeddings).
- No re-running any condition — substrate is fixed.
- No held-out at N>20 — small-sample is the entire point. If signal exists
  at N=20, future work scales.

## Cost

$0. Local sklearn fit on 19-idx folds. ~30 minutes eng-time.

## Files

- `load_substrate.py` — joins examples.json into (idx, condition, correct) + features
- `bandit.py` — classifier + LOOCV
- `RESULT.md` — outcome
