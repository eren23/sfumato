# Proposal D3.5 — Verifier-Based Branch Aggregation for DDLM cmaj

**Author:** Phase-2 orchestrator | **Date:** 2026-05-02 | **Status:** proposal — graduating candidate per spike result

## Motivation

The temperature-diversity-falsifier spike (`phase2/spikes/temperature-diversity-falsifier/RESULT.md`) **inverted the original D3 framing**. Headline finding:

> At τ=0.7 (the Phase-1 default), **oracle ceiling = 90% but cmaj b=5 = 78%**, on N=50 GSM8K-dev with base LLaDA-8B. The right answer is *present in at least one of the 5 branches on 90% of problems*, but majority voting throws it away on 12pp of problems.

This 12pp **voting-rule gap** is larger than the gap between Phase-1's best Track-1 LoRA and the base model on cmaj (~+5pp). It's larger than the gap between any two adjacent τ values (≤2pp). **Aggregation is the bottleneck, not generation.**

Original D3 (training-time diversity reward) was killed by the spike because branch diversity is already sufficient — `bar_p_maj=0.80` at τ=0.7 means branches already disagree often. The pivoting question is:

**Can a small per-branch correctness classifier recover the oracle ceiling, replacing majority vote at inference time?**

This is the **verifier-aggregation** literature applied to discrete-diffusion language models (DLMs), where it has not yet been studied — most DLM work assumes single-shot or self-consistency vote (d1, diffu-GRPO, BaRP). Process-reward and outcome-reward verifiers are well established for AR models (Cobbe 2110.14168, Lightman 2305.20050, PRM800K, Math-Shepherd 2312.08935), but none have been adapted to a hybrid AR/DLM cmaj setting.

## Mathematical formulation

Let $\pi_\theta$ be the (frozen) LLaDA sampler. For problem $x$, draw $b=5$ branches $\{y_i\}_{i=1}^{b}$. Each branch has a numeric answer $a_i = \mathrm{extract}(y_i)$ and a per-branch hidden state $h_i \in \mathbb{R}^d$ (last-layer embedding at the answer span).

Train a verifier head $f_\phi: \mathbb{R}^d \times \mathcal{X} \to [0,1]$ that predicts $P(a_i = a^* \mid x, h_i)$ where $a^*$ is gold.

**Training data (free, no GPU spend for labeling):**
- Each row: $(x, h_i, \mathbb{1}[a_i = a^*])$ — already implicit in `e4/results/raw_cmaj_*.jsonl` (per-branch text + gold + correctness).
- Phase-1 jsonls: N=50 × b=5 = 250 labeled branches per τ, three τ values = 750 total.
- Phase-2 substrate harvest (queued, in flight): N=100 × b=5 × 2 τ values = 1000 additional labeled branches.
- **Total available training set: ~1750 labeled branches** by end of today.

**Loss:**
$$\mathcal{L}(\phi) = -\sum_{(x, h_i, c_i)} c_i \log f_\phi(h_i, x) + (1 - c_i) \log (1 - f_\phi(h_i, x))$$

**Inference rule:**
$$\hat{a}_{\text{verifier}}(x) = a_{i^*}, \quad i^* = \arg\max_i f_\phi(h_i, x)$$

vs. baseline cmaj: $\hat{a}_{\text{cmaj}}(x) = \mathrm{mode}(\{a_i\}_i)$.

**Architecture options (ordered by predicted info-per-dollar):**
1. **MLP head on frozen last-layer LLaDA embeddings**: 2-layer MLP (768→256→1), ~250K params, trains in <5 min on CPU.
2. **Distilled Qwen-0.5B verifier**: same input format as the planner; ~1 hr training on 4090.
3. **Process-reward variant**: train on per-step entropy + commit-LoRA logit-shift trajectories (consumes Workstream-C JSONL trace schema — see `phase2/STATUS.md` Workstream C section).

Option 1 first. If it under-performs, escalate.

## Quantitative success criteria

**Pre-registration (binding before any verifier training run):**

| Metric | Threshold | Action if hit | Action if missed |
|---|---|---|---|
| Verifier accuracy on cmaj b=5 N=200 GSM8K-test at τ=0.7 ≥ **83%** (+5pp over 78% cmaj baseline) | **WIN-MINOR**: D3.5 graduates, write paper note | reduce scope, try option 2 |
| Verifier accuracy ≥ **87%** | **WIN-STRONG**: full paper experiment with multi-temperature + Track-1-v3 stack | (subsumed by row above) |
| Verifier accuracy ≥ oracle ceiling − 1pp (i.e. ≥ **89%** at τ=0.7) | **WIN-DECISIVE**: re-frame the entire ensemble-aggregation literature for DLMs | (subsumed) |
| Verifier accuracy ≤ **78%** (parity with cmaj) | **LOSS**: per-branch features insufficient for discrimination, kill D3.5 |
| Verifier accuracy 78%—82% | **INCONCLUSIVE**: try option 2, or flag as scoring-noise |

Cross-validation: 5-fold on the 1750-branch training set, hold-out τ-stratified.

## Predictable failure modes

1. **Per-branch features too coarse**: last-layer hidden states at the answer-span position may not encode "is this answer right" — they encode "what answer was generated." Mitigation: use the full last-layer sequence summary (mean-pool), not just the answer-span position.
2. **Distribution shift τ=0.7 → τ=1.0**: classifier trained on τ=0.7 branches may rank τ=1.0 branches arbitrarily. Mitigation: train across both τ values; report per-τ accuracy.
3. **Verifier confident but wrong**: classifier might over-prefer fluent-but-wrong branches (length bias, format bias). Mitigation: ablation that includes an oracle-format-only baseline.
4. **Pre-existing leakage**: same problems used in Phase-1 training and now in eval. Mitigation: use frozen `e4/data/gsm8k_dev_200.json` indices that were NOT in any Phase-1 training set (Track-1 LoRA was trained on `gsm8k-train`, not dev).
5. **Tiny classifier underfits**: 250K params on 1750 examples is fine for a well-conditioned problem; if not, add Phase-3 jsonls (multi-seed cmajc N=100 × 2 seeds × 5 branches = 1000 more labels — already queued).

## Compute cost estimate

- **Training (option 1)**: CPU only, <5 min. **Cost: $0.**
- **Eval at N=200 GSM8K-test, τ=0.7, b=5, k=64**: ~10 min on RTX 4090 spot. **Cost: ~$0.04.**
- **Verifier inference per branch**: a 2-layer MLP forward, negligible (~0.001s).
- **Total spike compute: ~$0.05.**

If option 1 fails and we go to option 2 (Qwen-0.5B distilled verifier): +1 hr training + same eval = ~$0.40 total.

**Either way, D3.5 spike is < $0.50 — well under any per-spike budget.**

## Dependencies

- ✅ **Phase-1 raw_cmaj jsonls** at τ ∈ {0.3, 0.7, 1.0} N=50: already in `e4/results/`.
- 🟡 **Phase-2 D3.5 substrate** (cmaj N=100 b=5 at τ=0.7 + τ=1.0): queued in `phase2/spikes/temperature-diversity-falsifier/queue_followup.py`. Output lands in `e4/results/raw_cmaj_*.jsonl` and on W&B as `d35-substrate-tau-0.7-N100` and `d35-substrate-tau-1.0-N100`.
- 🟡 **Frozen GSM8K-test indices** distinct from Phase-1 training: confirm via `e4/data/gsm8k_dev_200.json` provenance check in runner.py.
- ⚪ **Optional — Workstream C trace schema** (already pinned in `phase2/STATUS.md` Workstream C section): used only if option 3 (process-reward variant) becomes the chosen architecture.

## Literature evidence (2024-2026, cross-referenced WebSearch + HF Hub)

- Cobbe et al. 2110.14168 "Training Verifiers to Solve Math Word Problems" — the original GSM8K + verifier paper. Establishes the verifier-vs-self-consistency baseline. **Load-bearing prior.**
- Lightman et al. 2305.20050 "Let's Verify Step by Step" + PRM800K — process-reward models outperform outcome-reward at scale on MATH. Suggests option 3 is worth trying if option 1 fails.
- Math-Shepherd 2312.08935 — automated step-level reward labels via tree search, removes human-annotation dependency.
- d1 (2504.12216) — single-shot diffu-GRPO; **does not** explore re-ranking aggregation.
- BaRP (2510.07429) — best-route policy among generation paths in DLM, closest published peer; uses path-level reward, not per-branch correctness verifier. Citable as related work.
- LRD (2510.11052) — latent reward distillation for DLMs; verifier-adjacent but operates on full trajectories, not per-answer.
- BEST-Route (2506.22716) — best-of-N routing for AR LMs. Cites ours as the natural DLM extension.
- (Workstream-B subagent's lit search noted ≥9 papers per direction; D3.5 inherits and extends that bibliography. To re-run targeted search before paper writeup, query: `"verifier" "discrete diffusion" "math"`, `"reranking" "branch" "diffusion language model"`, `"best-of-N" "DLM" 2025 2026`.)

**Gap claim:** no published work trains a per-branch correctness verifier specifically for DLM cmaj aggregation on math reasoning. This is the contribution.

## Why this graduates over original D3

| Criterion | D3 (original) | D3.5 |
|---|---|---|
| Empirical motivation | speculative ("diversity might matter") | direct (12pp voting gap measured) |
| Training-data cost | requires new dataset + multi-step RL pipeline | **free**: existing jsonls have the labels |
| Compute cost (spike) | ~$5 | ~$0.05 |
| Failure cost | full SFT run wasted if hypothesis wrong | trivial (CPU minutes) |
| Theoretical ceiling | unknown — depends on reward shaping | **measured**: oracle 90% at τ=0.7 |
| Implementation difficulty | moderate (custom loss + sampling loop) | low (BCE on hidden states) |
| Phase-2 scope fit | one paper experiment | one paper section + a tweet-thread figure |

D3 is not killed permanently — it remains a possible Phase-3 follow-up if D3.5 confirms the verifier ceiling and we want to push further. But for Phase-2's "highest info-per-dollar spike" filter, D3.5 wins.

## Files this proposal expects to land in

- `phase2/proposals/verifier-based-aggregation.md` — this file
- `phase2/spikes/verifier-aggregation/` — to be created when spike kicks off
- `phase2/spikes/verifier-aggregation/PRE_REG.md` — copy this file's "Quantitative success criteria" section verbatim before any code runs
- `phase2/spikes/verifier-aggregation/train_verifier.py` — option 1 trainer
- `phase2/spikes/verifier-aggregation/eval_rerank.py` — replace cmaj's mode-vote with verifier argmax, score on N=200
- `phase2/spikes/verifier-aggregation/RESULT.md` — outcome + decision per pre-reg

## Status as of writing

- D3.5 substrate harvest is **in flight** (Phase 2 of `queue_followup.py`).
- This proposal supersedes D3 in `phase2/proposals/RANKING.md` — to be updated when substrate completes.
- Spike kickoff requires the substrate to land (~70 min from queue start) plus ~5 min of analysis to confirm enough labels survived.
