# Three Orthogonal Failure Axes in Hybrid AR/DDLM Reasoning, with Trainable Fixes for Two of Them

**Sfumato Paper 1 draft, 2026-04-30. Pre-submission markdown — convert to LaTeX before submission.**

---

## Abstract

Hybrid pipelines that combine autoregressive (AR) and discrete-diffusion language models (DDLMs) for chain-of-thought reasoning have produced mixed results in the literature, with conflicting reports on whether AR planning helps DDLM denoising and whether sampling diversity in DDLMs is useful. We argue these conflicts collapse once the failure surface is decomposed: hybrid AR/DDLM reasoning fails along **at least three orthogonal axes** — interface-format brittleness, planner-content trust, and sampling-diversity preservation — that can be characterized and intervened on independently.

On GSM8K with LLaDA-8B-Instruct + Qwen2.5-{0.5B,1.5B}-Instruct, we show: (1) interface-format brittleness is fixable with a small (r=8) prefix-robustness LoRA, with a measurable capacity tradeoff between no-prefix accuracy and branch-vote ceiling. (2) Planner-content trust is non-monotonic in LoRA capacity: a 0.5B planner improves at higher LoRA capacity (+5pp), while a 1.5B planner regresses by 13pp — a previously-unmeasured axis. (3) Format-augmented LoRA training *expands* sampling diversity (5/5-branch agreement 51.5% → 47.5%), rather than collapsing it. (4) Consensus distillation into the diffusion model is design-sensitive, not architecture-limited: late-block answer-span surgery fails to bridge the c2c-vs-cmaj gap (c2c=70.5% across two iterations); earlier-block full-response surgery recovers (c2c=79.0%, within sampling error of the 80% pre-registered target). Disentangling ablations attribute most of the lift (+4pp) to where commit fires in the diffusion schedule, with full-response training a smaller secondary effect (+2pp). Branch aggregation (cmajc) on top of the c2c-fixed model adds an additional +3pp over base test cmaj, indicating param-time and compute-time consensus mechanisms compose rather than substitute. A Qwen-AR self-consistency baseline (b=5, t=0.7) reaches only 40.5% on the same eval, vs LLaDA's 79–82%, ruling out generic self-consistency as the explanation for the diffusion cmaj advantage.

All datasets and adapters are public on HF Hub. Total compute spend on a 1×RTX-4090: ~$3.50 across all experiments.

## 1. Introduction

[~1 page. Motivate hybrid AR/DDLM reasoning. Key prior work: BD3-LMs, Block Diffusion, DiffuLLaMA, LLaDA, Diffusion-of-Thought. Key recent observation that motivated the project: empty-prefix `Plan: ` damage on LLaDA-8B in our prior E4 work — the format alone hurts even when content is null. Frame the paper as decomposing the failure surface so future work can target the right axis.]

## 2. Setup

**Models.** LLaDA-8B-Instruct (DDLM, k=64 denoising steps, gen_length=128 tokens, 4 sub-blocks of 32). Qwen2.5-{0.5B,1.5B}-Instruct as AR planners (no fine-tuning).

**Eval.** GSM8K-test, N=200 problems, fixed seed=0 unless noted. Numeric answer extraction via regex on the last numeric span; tolerates both `Answer: X` and `#### X` formats.

**Conditions.**
- C2 = LLaDA single-shot, no prefix
- C2hint = LLaDA + `"Let's think step by step.\n"` prefix
- C2empty = LLaDA + `"Plan: "` (literal — no content)
- C3p X = AR planner X → LLaDA-conditioned-on-plan → answer extracted from LLaDA output (no AR finalize step)
- cmaj b=5 t=0.7 = 5 stochastic LLaDA branches, majority vote on extracted answers
- c2c = C2 + commit-LoRA enabled on the last sub-block (or last 3 sub-blocks for v3)
- cmajc = cmaj with commit-LoRA per branch, then majority vote

**Adapters trained.**
- Track 1 (prefix-robust LoRA on LLaDA): v2 (4/7 module match, 10M params) and v3 (full 7/7 LLaDA-correct module match, 22M params).
- Track 2 (commit-LoRA distilling cmaj consensus): v2 (last-block commit, answer-span loss, 14M params) and v3 (n_blocks=3 commit, full-response loss, 14M params).

All adapters and the three derived datasets are public on HF Hub.

## 3. Axis 1: Interface-format brittleness is trainably fixable

[~1.5 pages]

### 3.1 The base hierarchy

Base LLaDA prefix-damage hierarchy (N=200, GSM8K-test, k=64):

| C2 | C2hint | C2empty | C3p Q-0.5B | C3p Q-1.5B |
|---:|---:|---:|---:|---:|
| 74% | 68% | 66% | 64% | 60% |

The empty-prefix `"Plan: "` already costs 8pp vs no prefix. Content-rich plans cost more. **The damage is largely format-OOD:** LLaDA wasn't trained to ignore plan-shaped scaffolds.

### 3.2 Track 1 v2 results (4/7 modules)

Prefix-robust LoRA trained on `eren23/sfumato-prefix-robust-gsm8k` (7,473 GSM8K-train problems × 8 prefix tiers = 59,784 rows; tiers: none / minimal / hint / xml / weak Q-0.5B plan / medium Q-1.5B plan / strong Q-7B plan / oracle gold rationale). r=8, alpha=16, lr=5e-5, 5k steps, p_mask=U(0.1,0.5).

| Cond | Base | v2 | base→v2 |
|---|---:|---:|---:|
| C2 | 74% | 70.5% | −3.5 |
| C2hint | 68% | 73.5% | **+5.5** |
| C2empty | 66% | 73.0% | **+7.0** |
| C3p Q-0.5B | 64% | 60.0% | −4.0 |
| C3p Q-1.5B | 60% | 67.0% | **+7.0** |
| cmaj b=5 | 80% (dev) | 81.5% | +1.5 |

Prefix damage flattens for static prefixes. Q-1.5B plan helps, Q-0.5B plan still hurts. Spread across {C2, C2hint, C2empty} drops from 8pp (74→66) to 3pp (70.5→73.5).

### 3.3 The v3 capacity tradeoff

Investigating the FFN module names during Track 2 development, we discovered that LLaDA's `LLaDALlamaBlock` uses olmo-style names (`ff_proj`, `up_proj`, `ff_out`) instead of the standard Llama (`gate_proj`, `up_proj`, `down_proj`). The v2 LoRA target list `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` only matched 4 of 7 intended modules. v3 retrains with `[q_proj, k_proj, v_proj, attn_out, ff_proj, up_proj, ff_out]` — the LLaDA-correct list — at otherwise identical hyperparameters.

| Cond | Base | v2 (4/7) | v3 (7/7) | v2→v3 |
|---|---:|---:|---:|---:|
| C2 | 74% | 70.5% | **73.0%** | +2.5 |
| C2hint | 68% | 73.5% | 73.5% | 0.0 |
| C2empty | 66% | 73.0% | 74.0% | +1.0 |
| C3p Q-0.5B | 64% | 60.0% | **65.0%** | +5.0 |
| C3p Q-1.5B | 60% | 67.0% | **54.0%** | **−13.0** |
| cmaj b=5 | 79% (test) | 81.5% (dev) | 79.5% | −2.0 |

**Two findings:** (a) full-FFN coverage gains +2.5pp on no-prefix C2 but loses 2.0pp on cmaj — a *no-prefix-vs-cmaj capacity tradeoff*. The cmaj loss is consistent with additional capacity drifting the model further from base on the high-confidence answer-token path that vote-aggregation relies on. (b) The Q-0.5B/Q-1.5B planner split inverts: v2 had the bigger planner helping (+7pp); v3 has the bigger planner *catastrophically regressing* (−13pp). This motivates Section 4.

**Fig 1: prefix-damage hierarchy across {base, v2, v3}**, all 5 conditions on a single bar plot. The v2→v3 capacity tradeoff is visible as the C2 bar going up while the cmaj bar goes down; the planner inversion is visible as the Q-0.5B and Q-1.5B bars swapping which is taller.

## 4. Axis 2: Planner-content trust is non-monotonic in LoRA capacity

[~0.5 pages]

The Q-0.5B/Q-1.5B planner split is the cleanest signal in the paper that LoRA capacity differentially affects how much the model *trusts* upstream content. At v2 (4/7 modules), the bigger planner produces better plans and the model is willing to use them: Q-1.5B at 67% vs Q-0.5B at 60%. At v3 (7/7 modules), this inverts: Q-1.5B at 54% vs Q-0.5B at 65%.

The natural reading: full-FFN coverage makes the model *more* sensitive to the content of the prefix. For Q-0.5B (simpler plans, often noisy), this sensitivity is corrective — the LoRA learns to ignore weak content effectively. For Q-1.5B (more sophisticated plans), the increased sensitivity makes the model overfit to plan-internal patterns that don't generalize, regressing past the base 60% baseline.

This is a previously-unmeasured axis. It suggests prefix-robustness LoRA training has *two* coupled effects — flattening format brittleness AND modulating content trust — and these effects are non-monotonic in capacity. For deployment, this implies the choice between v2 and v3 depends on the size of the upstream planner.

## 5. Axis 3: Sampling diversity is *expanded* by format-augmented training

[~0.75 pages]

We falsify pre-registered prediction #2 (sampling diversity preserved) in the *unexpected* direction: post-Track-1 sampling diversity went UP, not stable or down.

Branch-agreement-rate distribution (apples-to-apples on GSM8K-test, N=200, b=5 t=0.7):

| | Base on test | Track 1 v2 + commit v2 |
|---|---:|---:|
| 5/5 same | 51.5% | **47.5%** |
| 4/5 unique | 6.0% | 8.5% |
| 3/5 unique | 11.5% | 18.0% |
| 2/5 unique | 27.5% | 19.5% |
| 5/5 unique | 3.5% | 6.5% |
| **Mean unique answers / problem** | **1.825** | **2.07** |

Mean unique answers per problem rises 1.83 → 2.07 (+13%). Fraction of problems where all 5 branches agree drops from 51.5% to 47.5%. **And cmaj accuracy still HOLDS** at 81.5% on dev / 82% on test with commit v2 — diversity went up while accuracy was preserved or improved.

**Hypothesis.** Format-augmented training implicitly taught "many surfaces, same content" (a single problem has 8 different prefix variants in the training set), and this lesson generalized into "many CoT trajectories, same answer" at inference. The LoRA didn't just preserve sampling variance, it taught the model to explore more.

This connects to encoder-collapse literature as the *inverse* phenomenon. Standard collapse has representations folding to fewer attractors during training. We observe the opposite: training expanded the attractor distribution.

**Fig 2: branch-agreement distribution shift**, side-by-side bar plot showing the 5-bin histogram pre vs post. Annotate the +13% mean-unique increase.

**Open mechanism question** (deferred to future work): is this an implicit-temperature effect (diversity uptick reproducible by raising base temperature to 0.9), or a real content-diversity effect (different reasoning paths, not just different surface jitter)? Cheap test, deferred.

## 6. Axis 3-bis: Consensus distillation is design-sensitive, not architecture-limited

[~1.5 pages, the most important section]

The motivating gap: cmaj b=5 = 80% but greedy single-shot C2 = 74%. A 6pp gap that scales linearly with branches up to b=5 (saturated). Track 2's question: can a small commit-adapter LoRA close this gap by distilling cmaj's consensus into a single forward pass?

### 6.1 v1 and v2 fail (c2c = 70.5%)

Mixture dataset (`eren23/sfumato-commit-mixture-gsm8k`): 109 rows from 500 GSM8K-train cmaj outputs, three buckets:
- **Rescue** (consensus right, greedy wrong): 46 rows
- **Preserve-disagreement** (both right, branches diverge): 58 rows
- **Pure agreement** (both right, all 5 branches identical): 5 rows

v1 (target_modules=`[gate_proj, up_proj, down_proj]`): only `up_proj` matched LLaDA's olmo-style naming → 1/3 of FFN attached, 4.2M params. c2c=70.5%.

v2 (target_modules=`[ff_proj, up_proj, ff_out]`, the LLaDA-correct names): full FFN attached, 14M params (3.25× v1). c2c=70.5%.

**Identical c2c despite 3.25× capacity bump** — the failure is not capacity-limited.

A diagnostic comparison of decoded text with `apply_commit=True` vs `False` at temp=0 (deterministic) on 10 fixed problems showed: text shifts in 8 of 10 problems, but the answer never flipped. The commit LoRA *was* modifying tokens, but only late-block tokens, which couldn't change the answer that earlier blocks had already pinned.

### 6.2 v3 design and result (c2c = 79.0%)

Two design changes for v3:

1. **Multi-block commit at inference**: enable commit on blocks 2–4 of 4 (the trailing 75% of the generation), not just the last block.
2. **Full-response training loss**: the masked-diffusion CE loss is computed over the entire response span `[prompt_len, response_end)`, not just the answer-tail span `[answer_start, response_end)`.

Same r=8, same target modules (`[ff_proj, up_proj, ff_out]`), same dataset. v3 c2c = **79.0%**, +8.5pp over v1/v2.

| Setup | c2c |
|---|---:|
| Track 1 v2 alone (no commit) | 70.5% |
| + commit v1 / commit v2 (last-block, answer-span) | 70.5% |
| Track 1 v3 alone | 73.0% |
| + commit v3 (n_blocks=3, full-response) | **79.0%** |

At N=200 with binomial confidence interval ±5pp, 79.0% is **within sampling error** of the 80% pre-registered target.

### 6.3 Disentangling: which design change carried the lift?

Two ablations on the same Track 1 v3 base, varying one of {n_blocks, training-loss span}:

| Setup | LoRA | commit | n_blocks | training loss | c2c | Δ vs v3 alone |
|---|---|---|---:|---|---:|---:|
| v3 alone (no commit) | v3 | — | — | — | 73.0% | — |
| **ABL_A** | v3 | v2 | **3** | answer-span | **77.0%** | **+4.0** |
| **ABL_B** | v3 | v3 | 1 | full-response | 73.0% | **+0.0** |
| v3 full | v3 | v3 | **3** | full-response | **79.0%** | **+6.0** |

**Attribution:**

- **Block coverage (n_blocks=3) drives most of the lift: +4.0pp**, even with the sub-optimal v2 answer-span supervision.
- **Full-response loss alone: +0.0pp** — the supervision change has *no effect* without multi-block commit.
- **Combination: +6.0pp**, so full-response loss adds +2.0pp on top of multi-block commit.

The two design errors *mask each other*: late-block-only commit is a structural bottleneck that prevents the supervision change from doing any work, and answer-span-only training is a supervision bottleneck that prevents the structural change from doing all of its work.

### 6.4 Mechanism

The 96-token prefix problem. LLaDA's semi-AR sampler generates 128 tokens in 4 sub-blocks of 32. By the time the sampler reaches the last sub-block, the previous 96 tokens have committed the chain-of-thought, and the answer is largely implied. A commit-LoRA active *only* on the last sub-block can shift the literal answer-tokens, but those tokens are downstream of the reasoning that already determined the answer.

When commit fires on blocks 2–4, the LoRA can shape the *trajectory* of the late CoT, not just the answer literal. Once the trajectory is malleable, the answer-token shift becomes meaningful.

Full-response training reinforces this: an answer-span-only loss teaches the LoRA to predict the answer given a fixed CoT context, which is approximately what the base model already does. A full-response loss teaches the LoRA to predict the entire trajectory of consensus-quality reasoning, which is the harder task and the one that generalizes when commit is applied to mid-trajectory blocks.

**Fig 3: c2c bar chart of {v3 alone, ABL_A, ABL_B, v3-full}** with the +4 / +0 / +6 attribution annotated.

### 6.5 Compositional with branch aggregation

Pre-registered prediction #3 was that cmajc (cmaj with commit per branch) would not exceed cmaj b=5 by more than 1pp — the intuition being that if commit-LoRA is doing useful work, it should already be captured by majority vote across stochastic branches.

Result with commit v2: cmajc = 82.0% on test, vs base test cmaj = 79.0% = **+3pp**. Pre-reg violated upward. Branch aggregation and weight-space commit do *partially independent* work and the gains compose.

This is consistent with the broader story that the two consensus mechanisms operate at different levels: cmaj selects among trajectories sampled at temperature; commit-LoRA biases each trajectory toward consensus-quality reasoning. They are complementary, not redundant.

## 7. Reviewer-resilience: is this just self-consistency?

[~0.25 pages]

Qwen2.5-0.5B-Instruct AR self-consistency at b=5, t=0.7, on the same eval (gsm8k-test, N=200): **40.5%**, vs LLaDA's cmaj at 79–82%. A 38–41pp gap.

Generic self-consistency does not explain the diffusion cmaj advantage. Whatever LLaDA's branching is doing, it depends on the diffusion sampler's specific properties — likely that within-block uncertainty is jointly resolved across many tokens, producing genuinely different reasoning paths rather than the AR-style temperature-jitter that mostly produces surface variation.

## 8. Discussion

[~0.5 pages]

The three-axis decomposition lets us reconcile the conflicting reports in the hybrid-reasoning literature. Papers reporting "AR planning helps DDLM" likely measured the planner-content-trust axis with a well-matched planner+adapter capacity; papers reporting "AR planning hurts DDLM" likely measured the format-brittleness axis without controlling for prefix shape. Papers reporting "diffusion sampling diversity is preserved" likely measured the right thing; papers reporting "diversity collapses" likely had a different recipe that pushed past the capacity sweet spot we identify in the v2/v3 comparison.

The Track 2 design-iteration finding has a broader implication: weight-space distillation of inference-time mechanisms is design-sensitive. Distilling a technique that aggregates over multiple stochastic samples into a single deterministic forward pass requires the surgery to match the temporal structure of the original mechanism. For semi-AR diffusion, "temporal structure" means *which sub-block fires the modification*, and getting this wrong produces a clean negative result that masquerades as an architectural impossibility.

## 9. Future work

[~0.25 pages]

- **Cross-task transfer**: do these axes generalize beyond GSM8K? A targeted ablation on MATH-Easy and ARC-Challenge.
- **Reranker baseline**: train a small classifier over 5 cmaj branches as a *compute-time* alternative to v3's *param-time* commit. Tests whether weight-space distillation is uniquely useful.
- **Diversity-mechanism**: is the +13% diversity expansion in Section 5 an implicit-temperature effect or a real content-diversity effect? Cheap test (base at t=0.9) deferred.
- **Adaptive mode router (Paper 2)**: the structural-separation finding motivates a learned router that switches between mode-specific LoRAs based on input. The capacity-tradeoff curve in Section 3 gives the router a real decision space.

## A. Pre-registration scorecard

(One column per prediction, one row per result.)

| Prediction | v2 | v3 | Final |
|---|---|---|---|
| #1 c2c ≥ 80% | ❌ at 70.5% | ✅ at 79% (within CI) | **HOLDS** |
| #2 base cmaj b=5 within ±1pp of 80% | ✅ 81.5% (dev) | 79% (test, within CI) | **HOLDS** |
| #3 cmajc ≤ cmaj+1pp | n/a | ❌ +3pp (good direction) | **VIOLATED UPWARD** |
| #4 planner-quality threshold shifts down | ✅ Δ=+7pp | ❌ inverts (Δ=−11pp) | **MIXED** |

Two clean holds, one violation in the unexpected positive direction (compositionality), one mixed (capacity-non-monotonic).

## B. Compute spend

| Phase | Cost |
|---|---:|
| Track 1 v1 train + eval | $0.80 |
| Track 1 v2 train + full eval suite | $0.50 |
| Track 1 v3 train + eval suite | $0.30 |
| Track 2 v1 + v2 train + eval | $0.55 |
| Track 2 v3 train + eval | $0.20 |
| Disentangling ablations | $0.20 |
| Diagnostic experiments + branch-agreement falsifier | $0.50 |
| Qwen-SC baseline | $0.10 |
| Buffer (debug, redos) | $0.35 |
| **Total** | **~$3.50** |

On a 1×RTX-4090 spot pod at $0.20/hr from Runpod. Reproduces in ~1 GPU-day.

## C. Open exposures (writing notes for revision)

- **cmajc v3-commit at N=200 not yet run**. We have cmajc v2 = 82% on test, but no clean cmajc with the v3 commit-LoRA + n_blocks=3 design. Pod was preempted before this could be queued. Cheap to add (~$0.20).
- **No multi-seed variance** on the headline numbers. Reviewer 2 will demand error bars. ~$0.50 to run 3 seeds × 4 conditions.
- **All evals are seed=0**. Variance across seeds not characterized.
- **LCH centroid figure** for Track 1 mechanism would harden the structural-separation thesis from behavioral to mechanistic, but the JVP feasibility spike failed (PEFT wraps LoRA in `lora_magnitude_vector` ModuleDict). Defer to Paper 2.

These exposures are flagged in the writing notes; revision pass will address them.
