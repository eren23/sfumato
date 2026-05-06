# D1 Bandit-on-Replay Mode Router — RESULT

**Pre-reg:** `phase2/spikes/D1-mode-router/PRE_REG.md` (committed before run).
**Run date:** 2026-05-06 | **Cost:** $0 (local sklearn).
**Outcome:** **LOSS** — bandit underperforms best fixed baseline at N=20.

---

## Headline numbers

| Setting | Acc on 20 idxs |
|---|---:|
| always-c1 (Qwen-0.5B AR) | 0.250 |
| always-c2 (raw LLaDA) | 0.650 |
| always-c2c (LLaDA + commit-LoRA) | 0.700 |
| **always-c2empty (best fixed)** | **0.750** |
| always-cmajc (sfumato-v3 default) | 0.750 |
| always-cmaj | 0.650 |
| **Oracle (any-condition-correct)** | **0.850** |
| **D1 LR-bandit (LOOCV)** | **0.650** |
| D1 RF-bandit (LOOCV) | 0.600 |

**Δ vs best fixed**: **−10.0 pp**. Pre-reg LOSS threshold was <0.80, hit.

(Note: on this 20-idx subset the all-conditions sweep numbers replicate
within rounding — c2empty/cmajc/cmaj/c2c match the sweep cells.)

## What this kills, what it leaves open

**Killed:** the offline-replay supervised-classifier form of D1 at this
data scale. Per-problem text features (length, number-tokens, TF-IDF
unigrams/bigrams over question text) are insufficient to discriminate
which condition will succeed on which problem when only 19 training
examples are available per fold and the action space has 12 options.

This **directly parallels the voting-rule-gap finding** in §2 of the
paper draft: per-branch supervised classification fails to close the
oracle gap. Now per-problem supervised classification *also* fails to
close the per-condition oracle gap. The same pathology — supervised
classification with surface text features, small N, large action space
— is the bottleneck in both routing problems.

**Still open (out of Phase-2 scope):**
- Sub-block-level routing (the original D1 proposal) requires
  Workstream-C real-mode trace data with `mechanism` + `entropy[]` +
  `commit_lora_active` per sub-block boundary. Not yet collected at scale.
- N=200+ replay substrate: would need to run all 12 conditions on the
  same 200 problems, ~$3-5 GPU. Possibly worth doing if a paper revision
  needs the larger-N ablation.
- Per-condition embedding features instead of TF-IDF: pass the
  question through a frozen encoder, mean-pool, attach to the bandit.
  Mirror the verifier-aggregation sweep on the bandit problem.
- Bandit actions over **outputs** rather than **conditions**: the
  problem becomes "given 5 candidate answers, pick one." That's the
  voting-rule-gap problem, which we already covered.

## Action distribution (LR-bandit on 20 LOOCV folds)

- c2: 11
- c1: 6
- c2c: 2
- c2empty: 1

The bandit collapses to a small subset (mostly c2 and c1) — interpretable
as the policy converges on "default to plain LLaDA / plain AR" because
the training-fold class distribution favors those (the easiest correct
labels). With balanced class weights LR still doesn't recover the harder
conditions reliably.

## Why this is still a publishable finding

The honest negative result here strengthens the paper's main thesis: **at
this dataset scale (N=200 problems for the cmaj substrate, N=20 for
the all-condition replay), supervised classification cannot recover the
oracle ceiling, regardless of whether you classify per-branch (verifier)
or per-problem (mode router)**. The same diagnosis applies to both
sub-problems of test-time aggregation in diffusion LMs.

The path forward is the same in both cases: more data, larger encoders,
or step-level process supervision (Lightman 2023 PRM800K). Phase-2
scope was correct to defer all three.

## Files

- `PRE_REG.md` — pre-registered hypothesis + decision rules
- `bandit.py` — substrate loader + LOOCV trainer
- `results.json` — machine-readable output (per-fold actions + accuracies)
- `RESULT.md` — this file
