# R4: Temporal Aggregation Across Diffusion-LM Denoising Steps

**Topic:** Phase-4 T3.C Temporal-SC × commit-LoRA aggregator validation.
**Date:** 2026-05-07. **Author:** Phase-4 research-validation subagent R4.

## TL;DR

T3.C is **partially novel with significant scooping risk**. The core
temporal self-consistency voting mechanism was published by "Time Is
a Feature: Exploiting Temporal Dynamics in Diffusion LLMs" (arXiv
2508.09138, Aug 2025) using exponential decay weights across
denoising steps. T3.C's contribution lies in **schedule-conditional
weight indexing** (1.5× when commit-LoRA active vs 1.0× otherwise)
and K-step boundary alignment, but this represents an incremental
parametrization of an established approach. The hardcoded 1.5×
amplification is not justified by prior work and risks being overfit
to commit-LoRA's specific K=3 design.

## Closest prior work table (top 5)

| Paper | ArXiv / Year | Aggregation mechanism | Domain |
|---|---|---|---|
| **Time Is a Feature** | 2508.09138 / Aug 2025 | Temporal Self-Consistency: weighted vote across T denoising steps; exponential decay f(t)=exp(α(1-t/T)), α=5 | dLLM text generation (GSM8K, MATH-500, Countdown) |
| HEX: Test-Time Scaling in Diffusion LLMs | 2510.05040 / Oct 2025 | Majority voting marginalizing over heterogeneous block schedules | dLLM reasoning (GSM8K 88.10%, MATH 40%) |
| Ranked Voting for LLM Self-Consistency | 2505.10772 / May 2025 | Borda count, IRV, MRRV weighted voting | AR / non-AR LM reasoning |
| Optimal Aggregation of LLM and PRM Signals | 2510.13918 / Oct 2025 | Learned weighting from LLM+PRM signals (calibrated logit/KDE); negative weights allowed | Reasoning w/ process rewards |
| Inverse-Entropy Voting | 2511.02309 / Nov 2025 | Weights ∝ 1/entropy across reasoning chains | Sequential AR LM reasoning |

## Direct hit: "Time Is a Feature" (arXiv 2508.09138)

**Voting formula:**
```
a* = argmax_a Σ_{t=1}^T f(t) · 𝟙(meaning(x_t^0) = a)
```

**Three weighting schemes evaluated:**
1. Fixed (f(t)=1) — equal vote, **worst performance**
2. Linear decay (f(t)=1−t/T)
3. Exponential decay f(t)=exp(α(1-t/T)), α=5 — **best performance**

**Key finding:** Accuracy improves as denoising progresses → weights
favor *later* sampling steps (closer to final answer). The TSE
(Temporal Semantic Entropy) signal correlates with correctness:
correct answers have lower TSE.

**How T3.C differs:**
- Uses **commit-LoRA state** (binary on/off) to modulate weight, not a smooth temporal function
- Applies weights at **K-step sub-block boundaries** {0,1,2,3} for k=3 schedule
- **Hardcoded 1.5× for active LoRA** vs 1.0× — no principled decay curve, no calibration

## Hardcoded 1.5× — defensibility

**Prior work consensus:** weighted aggregation either uses a
**principled decay schedule** (Time Is a Feature: exponential) or a
**learned per-position weight** (Optimal Aggregation: KDE/logit
calibration with possible negative weights).

T3.C's 1.5× is neither. It's a heuristic that:
- Ignores the exponential-decay principle from arXiv 2508.09138
- Doesn't adapt to model/task like learned approaches
- Provides no theoretical justification (why 1.5× and not 1.2× or 2.0×?)
- Treats t=1 and t=3 equally despite vastly different answer-completion rates

**Recommendation:** if defending the 1.5× weight, frame the novelty as
**schedule-conditional structure** (LoRA state × K-schedule
interaction), not the weight value itself.

## Risks specific to T3.C

### (a) Partial-answer null hypothesis at sub-blocks 0 and 1

For K=3 commit-LoRA schedule with sub-blocks {0,1,2,3}:
- **Sub-block 0:** ~0% denoised → no meaningful partial answer
- **Sub-block 1:** ~10-20% denoised → mostly noise, few legible words
- **Sub-block 2:** ~50-70% denoised → partial answer emerges
- **Sub-block 3:** ~99% denoised → final answer (this is what cmajc-vote already uses)

If t∈{0,1} partial answers are predominantly null:
- Weight[t=1]=1.5× amplifies noise, not signal
- The "active LoRA" condition is a fixed schedule side-effect, not learned signal
- Pre-reg +6pp claim risks confounding with **late-stage dominance** (sub-block 3 already does the heavy lifting)

**Mitigation:** quantify % valid `partial_answer_strict` per sub-block before claiming WIN; ablate against uniform-weight temporal voting baseline.

### (b) K-schedule prior double-counting

Commit-LoRA was trained with K-step-aware denoising (sub-blocks 2-4
ON). Using the SAME K-schedule to weight aggregation = re-applying
the same prior twice:
- The model learned K-step-aware denoising
- The aggregator re-weights by K-step state
- The +6pp gain may be **already encoded in the LoRA training
  objective** rather than added by aggregation

**Mitigation:** ablate against schedule-agnostic baseline (uniform
weights across all sub-blocks); if randomly-initialized LoRA + T3.C
voting still gains ≥2pp, the aggregation contribution is independent.

## Recommended §3.5 framing if T3.C WINs

> *"We combine temporal self-consistency voting (Zhang et al. 2025,
> arXiv 2508.09138) with schedule-aware weighting parameterized by
> commit-LoRA activation state, achieving a +X pp improvement over
> majority vote on N=100 GSM8K. This establishes that
> denoising-step-conditional aggregation — when conditioned on a
> learned schedule prior — is a viable test-time scaling lever for
> diffusion language models. We acknowledge concurrent work on
> temporal SC for dLLMs and frame our contribution as the K-step
> schedule × LoRA state interaction."*

**Key wording:** acknowledge "Time Is a Feature" as direct prior;
frame T3.C novelty as the K-schedule × LoRA-state interaction; avoid
overclaiming on the 1.5× weight value.

## Realistic pre-reg target

With proper ablation discipline (uniform-weight + frozen-LoRA
controls), expect **+3-5pp realistic gain** over cmajc-vote, not the
locked +6pp WIN threshold. The +6pp threshold is ambitious; +3pp
would still land PARTIAL per the locked rules and remain publishable.

## References

- Time Is a Feature: arXiv 2508.09138 (direct scoop risk)
- HEX: arXiv 2510.05040
- Ranked Voting: arXiv 2505.10772
- Optimal Aggregation: arXiv 2510.13918
- Inverse-Entropy Voting: arXiv 2511.02309
- Self-Consistency original: Wang et al. arXiv 2203.11171

## 100-word verdict

T3.C is **incremental, not novel.** "Time Is a Feature" (arXiv 2508.09138, Aug 2025) directly publishes temporal SC voting on diffusion LMs with principled exponential decay weights. T3.C's contribution is a hardcoded 1.5× schedule-conditional weight on top of that mechanism — clean methodologically but risks scoop. Two specific risks identified: (a) partial answers at sub-blocks 0-1 are mostly null, making the weight asymmetry forced; (b) K-schedule used for LoRA training and aggregation creates double-counting. Realistic gain estimate: +3-5pp (PARTIAL territory), not the +6pp WIN threshold. **Recommended if executed: ablate against uniform-weight + random-LoRA controls, frame as schedule × state interaction, cite arXiv 2508.09138 prominently.**
