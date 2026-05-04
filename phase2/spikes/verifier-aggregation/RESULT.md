# Spike Result — Verifier-Based Branch Aggregation (D3.5)

**Status:** PRELIMINARY (option 1, Phase-1 data only) — awaiting substrate harvest.
**Run date:** 2026-05-02 ~19:03 UTC | **Pre-reg hash:** `phase2/spikes/verifier-aggregation/PRE_REG.md` (committed 19:00 UTC)
**Outcome (preliminary):** **LOSS** on the cheapest verifier architecture (TF-IDF + LR, text-only).
**Actual cost:** $0.00 (CPU only, ~30s scikit-learn fit).

---

## Headline numbers (Phase-1 only, 5-fold CV split-by-problem)

| Fold | N (problems) | cmaj | verifier (TF-IDF+LR) | Δpp | oracle |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 100.0% | 100.0% | +0.0 | 100.0% |
| 1 | 10 | 80.0% | 50.0% | **−30.0** | 100.0% |
| 2 | 10 | 80.0% | 80.0% | +0.0 | 100.0% |
| 3 | 10 | 60.0% | 30.0% | **−30.0** | 80.0% |
| 4 | 10 | 70.0% | 60.0% | −10.0 | 80.0% |
| **Mean** | **10** | **78.0%** | **64.0%** | **−14.0** | **92.0%** |

**Verifier UNDER-performs cmaj by 14pp on average. Pre-reg decision: LOSS** for option 1.

(Source: `phase2/spikes/verifier-aggregation/results.json` after `train_verifier.py --phase1-only`.)

## What this means

The TF-IDF + Logistic Regression verifier picks up surface text features
("looks like a math solution") that do not discriminate correct from incorrect
chain-of-thought reasoning. When the cmaj majority happens to be wrong AND a
correct branch exists, the verifier rarely picks the correct one — it picks
whichever branch *reads* most fluent-mathematical, which is uncorrelated with
arithmetic correctness.

This is consistent with prior verifier literature: per-token text features
need to be augmented with *logits* / *hidden states* / *step-by-step process
reward* to catch arithmetic errors. The Cobbe et al. (2110.14168) baseline
verifier on GSM8K used GPT-3-finetuned scoring, not TF-IDF. The Lightman et al.
PRM800K result requires step-level supervision.

## What this kills, what it leaves open

**Killed (preliminary):** option 1 (TF-IDF + LR text-only) is not a viable
verifier head for this problem at N=750 training branches. The information is
not in the surface text alone.

**Still open (the proposal lives):**
- **Option 2: Qwen-encoder verifier** — fine-tune Qwen2.5-0.5B-Instruct as a
  binary classifier on (problem, branch_text). Uses the encoder's
  arithmetic-aware hidden states. Cost ~$0.40 on a 4090 spot. Worth running.
- **Option 3: Process-reward verifier** consuming Workstream-C JSONL traces
  (per-step entropy, commit-LoRA logit shifts). More speculative; needs C
  real-mode data first.
- **Larger N**: substrate harvest (in flight) adds 1000 more labeled branches;
  TF-IDF might tip from LOSS to INCONCLUSIVE at N=1750. Re-test after substrate.

## Re-evaluation plan

1. After substrate harvest lands (Phase 2 of `queue_followup.py`):
   re-run `train_verifier.py` (no flag) → uses Phase-1 train + Phase-2 eval.
   If verifier still ≤ cmaj, option 1 is dead.
2. If option 1 dead at N=1750: queue option 2 (Qwen-encoder verifier) as a new
   spike. Pre-reg threshold: same numbers (≥83% on N=200 held-out at τ=0.7).
3. If option 2 also dead: D3.5 is killed in the option-1/2 form. The proposal
   notes that a process-reward (option 3) variant remains open as a Phase-3
   candidate, but no Phase-2 spike is justified at this point.

---

## Night-1 ADDENDUM — 2026-05-03 ~01:30 UTC: option-1 confirmed LOSS at N=1750

The N=200 substrate landed (see `phase2/spikes/temperature-diversity-falsifier/RESULT.md` Night-1 ADDENDUM, raw jsonl at `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`). Re-trained option-1 (TF-IDF + LR) on the enlarged 1750-branch dataset:

### Setting (a) — train Phase-1 (N=750), eval Phase-2 substrate (N=1000, 200 problems)

| Metric | Value | 95% CP CI |
|---|---|---|
| cmaj b=5 baseline | 79.0% | [72.7%, 84.4%] |
| **TF-IDF verifier rerank** | **69.5%** | [62.6%, 75.8%] |
| Oracle ceiling | 88.0% | [82.7%, 92.2%] |
| **Δpp vs cmaj** | **−9.5 pp** | (verifier hurts) |
| Gap-closure ratio | **−105.6%** | (verifier worse than baseline) |

### Setting (b) — 5-fold CV on combined 1750 branches

| Metric | Mean across folds |
|---|---|
| cmaj b=5 baseline | 80.5% |
| TF-IDF verifier | 66.5% |
| Oracle ceiling | 89.5% |
| Δpp vs cmaj | **−14.00 pp** |

Both settings agree: **option-1 (TF-IDF + LR text-only) under-performs cmaj by ≥9pp on enlarged data.** The verifier picks up surface text features (length, math-formatting) that don't discriminate correct from incorrect arithmetic. No path forward at this architecture.

### Pre-reg decision (per `PRE_REG.md`)

**LOSS** — verifier accuracy ≤ 78% (parity with cmaj or worse). Mean verifier 66.5–69.5% across both settings. **Option-1 is dead.**

### Next step (deferred to next session)

**Option-2 (Qwen-encoder verifier)** is the remaining viable path within D3.5:
- Use Qwen2.5-0.5B-Instruct as a feature extractor; pass (problem + branch_text) → mean-pool last-layer hidden states → MLP head → P(correct)
- Train with BCE on the 1750 labeled branches (5-fold CV by problem)
- Pre-reg threshold (binding): mean verifier accuracy ≥ 83% on held-out folds
- Compute: ~$0.40 on RTX 4090 spot (~30 min training, ~5 min eval)
- Pre-reg + scaffold to be written in next session

If option-2 also fails, **D3.5 dies** in the supervised-classifier form. Process-reward (option 3 — consume Workstream-C JSONL trace schema) remains open as a Phase-3 candidate but is no longer Phase-2 graduating material.

---

## Night-1 ADDENDUM #2 — 2026-05-03 ~12:23 UTC: option-2 confirmed LOSS

Trained Qwen2.5-0.5B-Instruct as encoder, mean-pooled last-layer hidden states from `f"Problem: {q}\n\nSolution:\n{branch_text}"`, MLP head (896→256→1) with BCE loss, 5-fold CV split by problem on 1750-branch combined dataset (Phase-1 + Phase-2 substrate).

### Headline numbers (5-fold CV)

| Fold | N (problems) | cmaj | Qwen-encoder verifier | Δpp | oracle |
|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 80.0% | 72.5% | −7.5 | 95.0% |
| 1 | 40 | 80.0% | 67.5% | −12.5 | 87.5% |
| 2 | 40 | 75.0% | 62.5% | −12.5 | 85.0% |
| 3 | 40 | 82.5% | 77.5% | −5.0 | 87.5% |
| 4 | 40 | 85.0% | 80.0% | −5.0 | 92.5% |
| **Mean** | **40** | **80.5%** | **72.0%** | **−8.5** | **89.5%** |

**Pre-reg decision: LOSS** (mean verifier 72.0% < cmaj 80.5%; gap-closure −94.4% — verifier *worse* than baseline).

### Compute / cost

- Pod: spot RTX 4090 via Crucible
- Embedding extraction: 1750 branches × max_len=768 batch_size=8 → **11 sec on GPU**
- 5-fold MLP training: ~50 epochs × ~2 sec each = ~100 sec total
- Total compute on pod: ~2 min for the actual experiment
- Total spend including pod overhead: ~$0.10

### What this means

Both supervised verifier architectures (option 1 TF-IDF + LR, option 2 Qwen-encoder + MLP) significantly under-perform majority vote despite having access to the same per-branch correctness labels. The voting-rule gap (oracle 88-90% vs cmaj 79-82%) is real but **per-branch surface features — even from a 0.5B language-model encoder — cannot capture the signal needed to discriminate correct from incorrect arithmetic at this dataset size (N=200 problems, 1750 branches)**.

This is consistent with the verifier literature: Cobbe et al. (2110.14168) used a fine-tuned GPT-3 for their GSM8K verifier; Lightman et al. PRM800K used step-level human supervision. Mean-pooled embeddings from a small model + 1750 examples appears to be insufficient even for the simpler outcome-reward objective.

### Pre-reg final decision

**D3.5 is DEAD in the Phase-2 supervised-classifier form.** The two remaining paths:

- **Option 3 — Process-reward verifier** (Phase-3 candidate, not Phase-2 spike-eligible): consume Workstream-C JSONL trace schema (per-step entropy, commit-LoRA logit shifts, mechanism source). Per-step features rather than per-answer. Requires more real-mode traces (we have 1) before any spike is justified.
- **Larger encoder** (e.g., Qwen-7B or specialized math-tuned encoder): would cost >$5 per spike and is no longer "info-per-dollar best" — likely revisit only if D1 also fails.

The graduating slot reverts to **D1 — Adaptive mode router** per `phase2/proposals/RANKING.md`.

### Honest paper note

> The voting-rule gap is the most actionable Phase-2 finding (8-12pp of headroom) but our verifier architectures don't capture it. We tried (a) TF-IDF + logistic regression (text-only, free) and (b) Qwen2.5-0.5B mean-pooled encoder + MLP head; both score 8-14pp BELOW majority-vote on held-out folds. This suggests per-branch surface features are insufficient at this dataset scale; closing the gap likely requires step-level process supervision or a substantially larger encoder.

---

## Night-1 ADDENDUM #3 — 2026-05-03 ~16:16 UTC: Qwen-7B Hail Mary — encoder scale helps but D3.5 still LOSS

Same architecture as option-2, different encoder: Qwen2.5-**7B**-Instruct (14× more params than 0.5B), batch_size=2, max_len=512 to fit 24GB VRAM.

### Headline numbers (5-fold CV, same splits as 0.5B)

| Fold | N (problems) | cmaj | Qwen-**7B** verifier | Δpp |
|---:|---:|---:|---:|---:|
| 0 | 40 | 80.0% | 75.0% | −5.0 |
| 1 | 40 | 80.0% | 72.5% | −7.5 |
| 2 | 40 | 75.0% | 70.0% | −5.0 |
| 3 | 40 | 82.5% | 80.0% | −2.5 |
| 4 | 40 | 85.0% | **85.0%** | **+0.0** ← parity! |
| **Mean** | **40** | **80.5%** | **76.5%** | **−4.0** |

**Pre-reg decision: LOSS** (76.5% < 83% threshold; gap-closure −44.4%). But the trend is monotone: encoder scaling helped.

### Side-by-side encoder scale

| Encoder | Params | Mean verifier | Δ vs cmaj | Gap-closure |
|---|---:|---:|---:|---:|
| TF-IDF + LR | ~250K | 66.5% | −14.0pp | −156% |
| Qwen2.5-0.5B + MLP | 500M | 72.0% | −8.5pp | −94% |
| Qwen2.5-**7B** + MLP | 7.6B | **76.5%** | **−4.0pp** | **−44%** |

Each ~10× scale-up of encoder narrowed the gap by 5pp on average. Linear extrapolation suggests Qwen-32B or Qwen-72B might cross cmaj baseline. But that costs $$ per spike and is no longer Phase-2 budget.

### Compute / cost

- Qwen-7B fp16 + batch=2 + max_len=512: 37 sec embed + 5 × ~30 sec MLP train = ~3 min compute
- Pod runtime: ~3 min provision + 3 min manual bootstrap (skipped Crucible install_uv) + 5 min ABL_B + 5 min Qwen-7B + destroy = ~20 min
- Spend: ~$0.12 on RTX 4090 on-demand

### Honest paper-section update

The Qwen-7B result strengthens the claim that the voting-rule gap is "structural but encoder-bound": surface features are insufficient regardless of encoder size at this dataset scale, but **larger encoders monotonically narrow the gap**. Future work: 7B → 32B / 72B encoder spike (~$5-10 per attempt), OR step-level process supervision (option 3) which is cheaper but needs Workstream-C trace expansion first.

The fold-4 result (verifier MATCHED cmaj at 85% with the larger encoder, on a fold where oracle was 92.5%) is suggestive that the verifier CAN work in some problem distributions — the question is when fold-4-like gains generalize to all folds.

### D3.5 final-final verdict

D3.5 is **dead in the Phase-2 budget** but **alive as a Phase-3 candidate** with two paths:
1. Larger encoder ($5-10 per spike, see linear extrapolation above)
2. Process-reward verifier (option 3, see `phase2/proposals/option3-process-reward-verifier.md`)

Both are deferred to Phase 3. For the Phase-2 paper, the honest take is: voting-rule gap is real and reproducible; per-branch verifiers narrow but don't close it; encoder scaling shows monotone improvement (TF-IDF → 0.5B → 7B: gap closure −156% → −94% → −44%) which is itself a publishable trend.

---

## Night-1 ADDENDUM #4 — 2026-05-03 ~21:15 UTC: 3 more Qwen3 family verifiers tested, ALL LOSS

After option-2's monotone-improving trend with model scale (TF-IDF → 0.5B → 7B narrowed gap by 5pp each ~10×), tested broader architecture variants from current Qwen3 family (proper HF org-listing search instead of guessing from training memory):

### Settings (all 5-fold CV on combined N=1750 substrate)

| Encoder | Params | Mean verifier | Δ vs cmaj 80.5% | Embed time |
|---|---:|---:|---:|---:|
| Qwen3-Embedding-4B | 4B | 68.5% | **−12.0 pp** | 30s |
| Qwen2.5-Math-7B | 7B | 74.0% | −6.5 pp | 41s |
| Qwen3-Embedding-8B | 8B | 72.5% | −8.0 pp | 59s |
| Qwen3-8B (chat) | 8B | 75.0% | −5.5 pp | 55s |

Combined with prior runs:

### Full encoder-scaling table (8 architectures total)

| Encoder | Params | Mean verifier | Δ vs cmaj | Gap-closure |
|---|---:|---:|---:|---:|
| TF-IDF + LR | ~250K | 66.5% | −14.0 pp | −156% |
| Qwen3-Embedding-4B | 4B | 68.5% | −12.0 pp | −133% |
| Qwen2.5-0.5B | 500M | 72.0% | −8.5 pp | −94% |
| Qwen3-Embedding-8B | 8B | 72.5% | −8.0 pp | −89% |
| Qwen2.5-Math-7B | 7B | 74.0% | −6.5 pp | −72% |
| Qwen3-8B (chat) | 8B | 75.0% | −5.5 pp | −61% |
| **Qwen2.5-7B (chat)** | **7B** | **76.5%** | **−4.0 pp** | **−44%** ← best |

### Surprising findings

1. **Embedding-specific models are WORSE than chat models** at same param count: Qwen3-Embedding-8B (−8.0pp) loses to Qwen3-8B chat (−5.5pp) and Qwen2.5-7B chat (−4.0pp). Counter-intuitive — embedding models are PURPOSE-BUILT for representation tasks. Possible explanation: embedding models are trained for *similarity* objectives (cosine/InfoNCE) which compress features in ways that lose specific answer-correctness signal. Chat models retain richer per-token semantics in mean-pooled features.

2. **Math-tuning HURT verifier quality**: Qwen2.5-Math-7B (−6.5pp) vs plain Qwen2.5-7B (−4.0pp). Math-specific fine-tuning probably collapses features around math-vocabulary, losing discriminative signal for *correctness within math*. The math-tuned features encode "is this math?" rather than "is this math correct?".

3. **Newer Qwen3 generation didn't help vs Qwen2.5**: Qwen3-8B (−5.5pp) is essentially equivalent to Qwen2.5-7B (−4.0pp) — small architecture/training improvements don't translate to verifier improvements at this scale. The bottleneck is dataset size or feature-space, not model recency.

4. **Encoder-scaling trend within chat-LM family is monotone but flattening**: each ~10× scale narrowed gap by ~5pp early (TF-IDF → 0.5B), then ~3-4pp (0.5B → 7B). Linear extrap to crossing cmaj baseline at 32B+ may be optimistic if the curve continues to flatten.

### Final D3.5 verdict (8 architectures, all LOSS)

**No per-branch supervised classifier we tested closes the voting-rule gap at this dataset scale (200 problems, 1750 branches).** The gap is real and reproducible (8-12pp), but it appears to require either:

- (a) **Massive scale**: Qwen-32B+ encoder ($5-10+ per spike). Unclear if monotone trend continues.
- (b) **Step-level supervision**: PRM800K-style human annotation on per-step trajectory features. Expensive in human time but proven elsewhere.
- (c) **Different paradigm entirely**: not per-branch classification. E.g., generative re-ranking, debate-style critic, contrastive learning across branches of the same problem rather than independent classification.

For the Phase-2 paper, the honest punchline updates from "encoder scaling narrows the gap monotonically (3 points)" to:

> **We tried 8 verifier architectures spanning 4 orders of magnitude in encoder size. All under-perform majority vote. The encoder-scaling trend within plain chat-LMs is monotonically narrowing (−14pp → −4pp), but embedding-specific and math-specific architectures perform WORSE, suggesting the bottleneck is the supervised-classification objective itself rather than feature quality.**

This is a much stronger, more honest negative-result section than just "we tried 3 things."

### Cost

This addendum's compute: ~$0.20 (Qwen3 family) + ~$0.05 failed transformers-version-bug retries = ~$0.25.
Cumulative Phase-2: **~$3.75 / $20**.

## Deviations from pre-reg

1. **Decision rule pivot**: PRE_REG.md said "best verifier acc". This was a
   bug — best-of-folds picks up a noise-fold (10/10 trivially correct because
   all branches agreed AND all were right). Switched to MEAN across folds. The
   pre-reg thresholds (≥89/87/83/78%) still apply, just on mean. Documented
   here as audit deviation.
2. **Substrate not yet available**: pre-reg path (b) (train Phase-1, eval Phase-2)
   not runnable yet. Used path (a) (5-fold CV on Phase-1 only) as the available
   alternative. Will re-run with path (b) after substrate lands.

## Files

- `PRE_REG.md` — pre-registered hypothesis + procedure (committed before run)
- `load_branches.py` — common loader for raw_cmaj jsonls
- `train_verifier.py` — TF-IDF + LR trainer with 5-fold CV
- `results.json` — machine-readable output
- `RESULT.md` — this file (PRELIMINARY; will be superseded after substrate)
