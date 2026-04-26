# E2 — Protocol

## Hypothesis

E4 (inference-only) showed that LLaDA-8B-Instruct alone reaches 74% on
GSM8K-dev200, that a naive AR-plan + diffusion hybrid drops to 64%
because the plan prefix damages LLaDA's reading comprehension, and that
self-consistency over 5 LLaDA samples (`cmaj b=5 t=0.7`) lifts accuracy
to 80%. The signal is real but lossy: the hybrid pipeline is fighting
the diffusion model's prefix sensitivity, and majority voting at
inference is expensive (5× LLaDA forward passes per problem).

E2 fixes both with two small, independent LoRAs trained on top of frozen
LLaDA-8B. **Track 1** (prefix-robustness) trains a LoRA on
prefix-randomized GSM8K so that whatever an upstream planner produces —
empty, weak, strong, or oracle — LLaDA reads the question correctly and
the prefix-quality damage curve flattens. **Track 2** (commit adapter)
distils `cmaj b=5 t=0.7` into a separate small LoRA selected only when
LLaDA-greedy is mis-firing; the base path stays diverse so `cmaj`
remains available as a parallel ensemble at test time.

## Track 1 — Prefix-robustness LoRA

### Architecture

- Base: `GSAI-ML/LLaDA-8B-Instruct`, frozen (bf16).
- LoRA: r=32, α=64, dropout=0.05, target {q,k,v,o} + MLP up/down.
- Loss: standard masked-diffusion cross-entropy on `full_target` with
  `full_prompt` as the conditioning prefix (LLaDA's native objective).

### Data

`scripts/build_prefix_robust_dataset.py` → HF Hub
`eren23/sfumato-prefix-robust-gsm8k` (private). 7473 GSM8K-train
problems × 8 prefix tiers = 59,784 rows; 95/5 train/validation split
stratified on `prefix_tier`.

Tiers: `none`, `minimal` (`"Plan: "`), `hint`
(`"Let's think step by step.\n"`), `xml` (`"<plan>\n<rationale truncated to
80 chars>"`), `weak` (Qwen2.5-0.5B-Instruct greedy ≤32 tokens), `medium`
(Qwen2.5-1.5B-Instruct), `strong` (Qwen2.5-7B-Instruct), `oracle` (gold
rationale truncated to 200 chars).

### Pre-registered predictions (Track 1)

1. **Prefix damage hierarchy flattens.** Single-shot accuracy across
   `{none, hint, minimal "Plan: ", weak, medium, oracle}` spans ≤3 pp
   on GSM8K-test (vs E4's ~10 pp spread).
2. **Single-shot Track 1 LoRA accuracy ≥ 78% on GSM8K-test** (matched
   k=64, seed=0, no ensembling).
3. **Track 1 LoRA + cmaj b=5 ≥ 80%** — i.e. the LoRA must not regress
   the ensemble that E4 already established.
4. **Planner-quality threshold shifts down.** Where E4 needed `strong`
   plans for any benefit, the Track 1 LoRA makes `weak` plans no longer
   damaging and `medium` plans start helping.

## Track 2 — Commit adapter LoRA

### Architecture

- Base: `GSAI-ML/LLaDA-8B-Instruct`, frozen.
- Separate LoRA (small, r=8, α=16) — *not* merged with Track 1; a
  routing flag picks at inference whether to run base, base+T1, or
  base+T2.
- Loss: cross-entropy on the majority-vote-correct branch text from
  `cmaj b=5 t=0.7`, conditioned on the bare question.

### Data

`scripts/build_consensus_dataset.py` → HF Hub
`eren23/sfumato-consensus-gsm8k`. Run cmaj on 1,500 GSM8K-train
problems; keep rows where `majority_answer == gold AND greedy_answer
!= gold` (signal-positive cases for the commit adapter). Estimated
~30 GPU-hr on 1×4090 → script supports `--max_problems_per_run` and
`--resume_from` for preemption-safe runs. Raw jsonl
(`e2/data/consensus_raw.jsonl`) is appended on every problem and
preserved across runs.

### Pre-registered predictions (Track 2)

1. **Single-shot c2c ≥ 80% on GSM8K-test.** Base + commit adapter at
   single-shot inference reaches what E4's 5-branch cmaj reached.
2. **Base-LLaDA cmaj b=5 unchanged ±1 pp from E4's 80%.** The commit
   adapter is a *separate* LoRA; base diversity is preserved by
   construction.
3. **cmajc ≤ cmaj b=5 + 1 pp.** Stacking the commit adapter on top of
   cmaj-style ensembling produces no double-dip — confirms the gain
   came from the same source of signal, not two independent ones.

## Eval protocol

- **Benchmark:** GSM8K test set, full N=200 (the frozen
  `e4/data/gsm8k_dev_200.json` indices).
- **k:** 64 diffusion steps for all conditions (matches E4's headline
  configuration; the cmaj 80% number was at k=64).
- **Conditions:**
  - `C1` (pure AR baseline): seed=0.
  - `C2` (pure LLaDA greedy): seed=0.
  - `c2c` (pure LLaDA + commit adapter, greedy): seed=0.
  - `cmaj` (5-branch LLaDA, t=0.7, majority): seeds {0, 1, 2}.
  - `cmajc` (5-branch base + commit adapter, t=0.7, majority): seeds
    {0, 1, 2}.
- Deterministic conditions report seed=0 only; stochastic conditions
  report mean ± std across seeds {0, 1, 2}.
- All FLOPs accounted via the existing `e4/flops.py` formula. Plot
  accuracy vs `log10(flops)`, never wall-clock.

## Decision matrix

| After Track 1 | If… | Then |
|---|---|---|
| Track 1 trained | All four T1 predictions hit | Ship as the prefix-robust path; proceed to Track 2 |
| | Prediction 1 misses (spread >3 pp) | Diagnose: maybe prefix randomization not wide enough, retrain with stronger augmentation |
| | Prediction 2 misses (<78%) | Track 1 LoRA broken; fall back to base-LLaDA + cmaj only |
| | Prediction 3 misses (cmaj regresses) | LoRA collapsed branch diversity; abort merge into the inference path |

| After Track 2 | If… | Then |
|---|---|---|
| Track 2 trained | All three T2 predictions hit | Commit adapter shipped; this is the E2 deliverable |
| | Prediction 1 misses (c2c <80%) | Distillation didn't capture cmaj signal; investigate temperature / branch count |
| | Prediction 2 misses (cmaj diversity damaged) | T2 LoRA bled into base; isolate (separate LoRA file, not merged) |
| | Prediction 3 misses (cmajc > cmaj+1pp) | Surprise: independent gains. Worth a dedicated investigation, treat as positive surprise rather than failure |

## Success / kill criteria

**Success criterion (track-level):** Both Track 1 prediction 2 (≥78%
single-shot) and Track 2 prediction 1 (≥80% single-shot c2c) hit. Either
delivers a meaningful win over E4's 74% base-greedy; both delivers a
1.5–2× FLOPs reduction over E4's cmaj at matched accuracy.

**Kill criterion (track-level):** If neither Track 1 prediction 2 nor
Track 2 prediction 1 hits after one full re-run with diagnosed
hyperparameters, write up E2 as a negative result and skip directly to
E1 (gap #1 of the literature map: full mode-router) without inheriting
any LoRAs from this track. The E4 cmaj 80% number remains the published
inference-only baseline.

**Programme-level kill:** If E2 falsifies the prefix-sensitivity
diagnosis from E4 (i.e. Track 1 changes nothing about the
prefix-quality curve), the original "AR plan damages diffusion" story
is wrong and the rest of the roadmap (E1 mode router, E3 three-phase)
needs to be reconsidered before launch.
