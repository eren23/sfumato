# Sfumato — The Idea, the Novelty, the Open Directions

**Audience:** kernel-survey agents (questions 4 & 5) and the human PI.
**Author:** research subagent | **Date:** 2026-05-04 | **Cwd:** `/Users/eren/Documents/AI/sfumato`
**Sources read:** `README.md`, `PLAN.md`, `e2/PROTOCOL.md`, `phase2/STATUS.md`,
`phase2/proposals/RANKING.md`, `phase2/proposals/{adaptive-mode-router,
diversity-as-objective,latent-coupled-joint-lora,option3-process-reward-verifier}.md`,
`phase2/PHASE2_FINAL_SUMMARY.md`, `e4/diff_llada.py`, `e4/runner.py`.

---

## 1. The idea in 5 sentences

Sfumato is a **plan-then-fill hybrid CoT pipeline** for GSM8K reasoning:
a small AR planner (Qwen2.5-0.5B/1.5B-Instruct) emits a short prefix, a
mask-diffusion LM (LLaDA-8B-Instruct, `2502.09992`) iteratively denoises a
128-token chain-of-thought conditioned on that prefix under a **prefix-robust
LoRA v3** that flattens the AR-prefix-quality damage curve, and a **commit-LoRA v3**
biases logits at sub-block transitions 2-4 of 4 toward stable answer-format
patterns distilled from `cmaj b=5` consensus. We then run **5 stochastic
LLaDA branches at τ=0.7** in parallel from the same prefix, apply the
commit-LoRA per branch, and **majority-vote on the regex-extracted numeric
answers** — this is the `cmajc` condition. The commercial unit of the
contribution is `cmajc-v3 = 82.5% on GSM8K-test` (`e2/PROTOCOL.md` Pred 3,
multi-seed mean = 82.2%, σ ≈ 0.3pp across seeds {0,1,2}, N=200; STATUS line 58-62).
Phase-1 also published an "orthogonal-failure-axes" decomposition (prefix
damage / planner trust / sampling diversity) that says the +3.5pp
**violation upward** of the no-double-dip prediction is the headline puzzle —
the LoRAs and the ensemble closed two different gaps despite our prediction
that they were correlated.

---

## 2. What is novel

PLAN.md mapped 226 papers across 11 threads. Triangulated against that map
plus targeted re-checks (HF Hub, GitHub, arXiv 2024-2026):

### 2.1 Where sfumato sits in the gap-list

| PLAN.md gap | Sfumato status | Honest verdict |
|---|---|---|
| **G1** model-decided AR↔diffuse alternation | Not in shipped Phase-1 (cmajc is fixed-schedule); D1 mode-router proposes it | gestures at, does not close |
| **G2** backtracking as first-class action | No | does not touch |
| **G3** mixed-granularity refinement | Partial — commit-LoRA on sub-blocks 2-4 of 4 is a *coarse* span-typed refinement, but the spans are positional not semantic | partially gestures at |
| **G4** uncertainty-triggered diffusion budget | No (k=64 is fixed) | does not touch |
| **G5** semantically-named diffusion timesteps | No | does not touch |
| **G6** compute-vs-iteration disambiguation | Acknowledged not closed; multi-seed σ closes one slice | open exposure |
| **G7** three-phase end-to-end staged inference | Phase-1 does plan→diffuse→extract but with frozen modules and no halting policy | partially gestures at |

Net: **sfumato closes none of the seven gaps in full**. What it *does*
contribute is three engineering primitives the gap-list never names:

1. **Prefix-robust-LoRA on a mask-diffusion LM** trained over an 8-tier ×
   7,473-problem prefix-augmentation grid (`eren23/sfumato-prefix-robust-gsm8k`,
   59,784 rows; 8 tiers = `none|minimal|hint|xml|weak|medium|strong|oracle`).
2. **Commit-LoRA at sub-block boundaries 2-4 of 4**, not just the final
   block — a multi-block commit schedule with `COMMIT_N_BLOCKS=3`.
3. **Majority-vote-with-per-branch-commit-LoRA (`cmajc`)**, where the
   commit adapter is hot-swapped on every branch independently before the
   regex vote.

### 2.2 Precedent search per primitive

#### A. Prefix-robust LoRA (8 tiers × 7,473 problems)

| Source checked | Result |
|---|---|
| arXiv title/abs search "prefix-robust" + "diffusion language model" | no hits |
| HF Hub model search `prefix-robust` | only `eren23/sfumato-llada-prefix-robust-{v2,v3}` (own artifact) |
| LLaDA paper (`2502.09992`) §SFT recipe | 4.5M-pair SFT, **no** prefix-tier randomization |
| d1 / diffu-GRPO (`2504.12216`) | RL on correctness; no prefix-augmentation curriculum |
| Planner-Executor DDLM (Berrayana, `2510.15244`) | latent-space planner; AR-prefix-text damage curve never measured |
| GIFT (`2509.20863`) | per-token entropy weighting at single trajectory; not prefix-tier |
| LRD (`2510.11052`) | belief-state refinement on DDLM internals; no prefix curriculum |
| Robust-prefix-tuning text (`2106.05665`, AR setting) | conceptually related ("prefix-robust" trained for AR), but pre-LLM; no diffusion variant |

**Verdict: no precedent found.** The 8-tier prefix damage curve as a
training signal for a DDLM is an unpublished primitive.

#### B. Commit-LoRA at sub-block boundaries (not just final block)

| Source checked | Result |
|---|---|
| BD3-LMs (`2503.09573`) | commits *whole* blocks AR-style; no per-sub-block adapter swap |
| SDAR (`2510.06303`) | semi-AR with fixed commit schedule, no LoRA at boundaries |
| Self-Correcting Masked Diffusion (`2602.11590`) | re-masks tokens; **the inverse** of committing them |
| Self-Speculative Decoding for DLMs (`2510.04147`) | confidence-thresholded commitment; no separate adapter |
| Diffusion-in-Diffusion (`2601.13599`) | block-AR draft + global polish; one schedule, no per-block adapter |
| FailFast (`2512.20573`) | confidence-conditioned draft length, AR setting |
| LLaDA paper App. B.4 | semi-AR low-confidence remasking only |
| GitHub `peft` MultiPLEx / hot-swap recipes | adapters can be enabled/disabled per call, but no published recipe **schedules adapter activation across mid-generation sub-block boundaries inside a DDLM** |

**Verdict: no precedent found** for a DDLM-side LoRA whose activation
window is restricted to sub-blocks 2..N-1 of an in-flight semi-AR
generation. Commit-LoRA-on-final-block has cousins (SUNDAE-style finalization
heads `2112.06749`); commit-LoRA-multi-block does not.

#### C. cmaj-with-commit-LoRA-per-branch (`cmajc`)

| Source checked | Result |
|---|---|
| Self-Consistency for AR (Wang et al. `2203.11171`) | majority-vote, no per-sample adapter swap |
| diffu-GRPO / d1 (`2504.12216`) | self-consistency reward in RL; no per-branch adapter |
| AGRPO/GDPO (`2510.08554`) | cmaj-style at training, no per-branch commit |
| Self-Speculative DLM (`2510.04147`) | per-branch *verification*, not per-branch *generation-side adapter* |
| Tree-of-Thoughts (`2305.10601`) | search trees, AR only, no DDLM adapter |

**Verdict: no precedent found** for the specific composition. Closest
cousins are AGRPO (cmaj-as-training-objective) and self-speculative DLM
(per-branch verifier head), neither of which combines per-branch on-policy
adapter activation with regex-vote at inference.

#### D. Was the +3.5pp "no double-dip" violation predicted?

`e2/PROTOCOL.md` Track-2 Prediction 3 says `cmajc ≤ cmaj+1pp`. Outcome:
**violated upward by +3.5pp for v3**. This is the most interesting Phase-1
finding and almost certainly the most novel observation in the paper.

| Closest published intuition | Notes |
|---|---|
| Self-Refine (`2303.17651`) | generator + critic interaction can be **multiplicative**, not additive — qualitatively related but not quantified for DDLM |
| ReAct / chain-of-verification | composition of reasoning + verification gives super-additive lifts in AR — anecdotal |
| ICL-aware fine-tuning (Chen et al. `2310.10616`) | LoRA + few-shot prompting compose super-additively at small scales |
| Rolnick et al. ensemble theory (general ML) | additive bound assumes uncorrelated errors; LoRA changes the error correlation structure — the mechanism by which sfumato could have predicted the violation |

**Verdict: not predicted by anyone we found.** The empirical observation
itself is the first published instance for DDLMs that we are aware of.

### 2.3 Honest one-line novelty claim

> Sfumato is the first published recipe to (a) measure and flatten the
> AR-prefix-quality damage curve on a mask-diffusion LM via tier-augmented
> LoRA SFT, (b) hot-swap a `cmaj`-distilled commit adapter at multi-block
> sub-boundaries during a single semi-AR diffusion pass, and (c) document
> a ≥3pp super-additive violation when those two ablations are stacked
> under majority vote.

Everything else (plan-then-fill, semi-AR diffusion, self-consistency-on-DDLM,
LoRA-on-LLaDA) has prior art.

---

## 3. What other things to try (non-kernel directions)

### 3.1 Status of the existing Phase-2 slate (RANKING.md final)

| ID | Direction | Status | Why dead/alive |
|---|---|---|---|
| **D1** | Adaptive mode router (E1) | **alive — graduating** | proposal-only; substrate exists (7 STATUS-schema traces in `phase2/inference_viz/traces/`); needs 50-100 more before a useful bandit can be fit |
| **D2** | Latent-coupled joint LoRA (Berrayana 2510.15244 ext.) | **alive — Phase-3 only** | $5.50+ spike floor, beyond Phase-2 cap; bigger paper bet |
| ~~D3~~ | Diversity-as-objective | **DEAD** | spike showed `a_b` monotone-decreasing in τ — diversity is not a free lever |
| ~~D3.5~~ | Verifier-based aggregation | **DEAD for peer-class verifiers, CONFIRMED for frontier judges** | Claude Sonnet 4.5+CoT closes 86% of gap (`PHASE2_FINAL_SUMMARY.md` headline); **17 peer-class verifiers all lost** |

### 3.2 Phase-1 open exposures still in play

- **ABL_B** sanity probe: closed 2026-05-03 (`phase2/spikes/abl_b_RESULT.md`).
- **Multi-seed v3 cmajc**: closed 2026-05-03, σ ≈ 0.3pp.
- **LCH JVP shim** (`scripts/lch_feasibility.py`, PEFT `lora_magnitude_vector`
  ModuleDict bug): **open**.

### 3.3 Five new directions surfaced from re-survey

#### NEW-1. Frontier-judge-as-verifier, productized (extension of D3.5 WIN)

**Spec.** Use Claude Sonnet 4.5 / GPT-5 / Gemini 2.5 Pro as a per-branch
CoT-prompted judge, with **cost-optimized hybrid routing**: only call the
frontier judge when cmaj has no ≥50% majority (~15% of GSM8K problems).
Cross-family confirmation needed (`PHASE2_FINAL_SUMMARY.md` followup A).
**Info-per-dollar:** very high — Phase-2 already showed +6.16pp at 86% gap
closure on GSM8K-N=500. Phase-3 would (i) confirm Claude-specific vs general
frontier-judge effect, (ii) test on MATH-500 / AIME where cmaj is weaker.
**Kill criterion:** if GPT-5 judge closes <50% of gap, Claude-CoT WIN may
not generalize → publish as Claude-specific finding only.
**$ budget:** ~$8 ($2 each cross-family judge × 3 + $2 MATH-500 substrate).

#### NEW-2. cmajc → single-pass distillation

**Spec.** Distill the b=5 cmajc trace ensemble into a **single-pass LoRA on
LLaDA** (no branching at inference). Training data: (question →
cmajc-majority-correct trace) pairs filtered for `majority == gold`.
Closest precedent: distilling self-consistency (Lin et al. `2310.06825`,
AR setting). **Info-per-dollar:** medium — directly tests "does the
ensemble carry information that a single pass cannot, or are we just paying
for compute?" Closes PLAN.md gap 6 partially.
**Kill criterion:** distilled single-pass < 80% on N=200 → ensemble was
genuinely lossy, refute distillability claim.
**$ budget:** ~$3 ($1 distill train + $0.5 eval × 4 hyperparam).

#### NEW-3. RL with branch-agreement reward on the LLaDA side

**Spec.** GRPO/diffu-GRPO-style RL on LLaDA-8B with reward = `mode_correct ×
(1 + λ · disagreement_among_correct_branches)` (cf. AGRPO `2510.08554`,
diffu-GRPO `2504.12216`). This is D3 reborn but conditioned on the D3.5
finding that **diversity is not a free lever**: the new reward signal
exists only on problems where cmaj already lands the right mode.
**Info-per-dollar:** low-medium — D3 spike already killed the naive
formulation; needs a concrete reason to expect the conditioning helps.
**Kill criterion:** if at convergence cmaj N=200 ≤ baseline+1pp, kill.
**$ budget:** ~$8 (3 GPU-h train × 4 λ + eval).

#### NEW-4. Test-time-compute trade curves (o1-style)

**Spec.** Plot accuracy vs `log10(FLOPs)` for sfumato sweeping
`(k_steps, b_branches, COMMIT_N_BLOCKS)` jointly, against o1-preview /
o3-mini at matched API-token-budget. Crucially, this is the experiment that
**closes PLAN.md gap 6** (compute-vs-iteration confusion) at the project
level. Closest precedents: o1 system card scaling plots (OpenAI, 2024-09)
and `2407.21787` ("Scaling LLM Test-Time Compute Optimally").
**Info-per-dollar:** high — converts an open exposure into a paper
section. Clean comparison curve is a strong reviewer-defense artifact.
**Kill criterion:** none, this is descriptive.
**$ budget:** ~$5 (1 GPU-day matrix × 1 sweep × 1 eval round).

#### NEW-5. Larger-encoder / process-reward verifier (option-3 + Hail Mary)

**Spec.** Train a process-feature verifier on the
`rich_substrate_n500.jsonl` dataset (1051 branches × per-step features +
correctness) using a Qwen-7B / Qwen-14B encoder, plus a step-level PRM
head (à la PRM800K, Lightman et al. `2305.20050`). Phase-2 showed
**monotone encoder scaling**: TF-IDF (-14pp) → Qwen-0.5B (-8.5pp) →
Qwen-7B (-4pp). Linear extrapolation crosses zero around Qwen-32B, then
a frontier judge wins. So the question is: **at what encoder size does
self-trained per-branch verification beat cmaj?** This is option-3 of the
D3.5 family + the Hail Mary.
**Info-per-dollar:** medium — answers a specific scaling question, but
already-published frontier-judge result probably dominates economically.
**Kill criterion:** Qwen-14B verifier still <80% → process-reward at this
scale doesn't help; conclude "must use frontier judge in production".
**$ budget:** ~$5-7 ($3-4 train Qwen-14B + $1 eval; substrate exists).

### 3.4 Scoring summary

| New direction | Headline-paper potential | Info-per-$ | $ |
|---|---|---|---|
| NEW-1 frontier-judge productized | high | very high | $8 |
| NEW-2 cmajc distillation | medium | medium | $3 |
| NEW-3 RL with branch-agreement | low | low-medium | $8 |
| NEW-4 TTC trade curves | medium-high | high | $5 |
| NEW-5 larger-encoder PRM | medium | medium | $5-7 |

---

## 4. Open questions for the kernel agents

PLAN.md gap 6 is the kernel-relevant exposure: **was the +3.5pp from the
LoRAs, or from extra inference compute?** The Phase-1 paper does not
disambiguate — `cmajc-v3` runs at `(k=64, b=5, COMMIT_N_BLOCKS=3)` while
the cmaj-base baseline runs at `(k=64, b=5, COMMIT_N_BLOCKS=0)`. The
extra commit-LoRA forward passes at sub-blocks 2-4 also add FLOPs. Three
kernels-friendly experiments would close this:

### K1. Match-FLOPs ablation: `b=5 cmaj at k=32` vs `b=1 cmaj at k=160`

Test whether 5-branch ensembling at lower k recovers the same gain as
single-branch at higher k. Both spend ~5× the single-branch-k=32 budget.
**If single-branch-deep ≥ ensemble-shallow ± 1pp**, the headline lift
came from compute, not iteration topology. **If ensemble-shallow wins by
≥3pp at matched FLOPs**, the topology matters.

- Conditions: `cmaj b=5 k=32`, `c2 k=160`, `c2c k=160`.
- N=200, seeds {0,1,2}.
- FLOPs accounted via `e4/flops.py`.
- Kernel ask: **prefix-KV reuse** across the 5 branches in `cmaj b=5 k=32`
  is the only way this matches FLOPs accurately. A naïve implementation
  re-runs the prefix forward 5×.

### K2. Commit-LoRA activation-window ablation

Sweep `COMMIT_N_BLOCKS ∈ {0, 1, 2, 3, 4}` with all other hyperparams
fixed. Phase-1 v3 picked 3 by grid search but didn't publish the per-N
table for `cmajc`. **If accuracy is monotone in N**, commit-LoRA is doing
"more compute = more gain" and we can't separate from K1. **If there's a
sweet spot at N=3 with N=4 *worse***, the sub-block-localized adapter
pattern carries genuine signal.

- Kernel ask: efficient PEFT adapter hot-swap mid-generation. Currently
  done by `enable_adapter()` / `disable_adapter()` calls per sub-block;
  cost-per-swap is unmeasured. Profile this. If swap is non-trivial,
  fused-LoRA-skip (skip the LoRA contribution at certain blocks instead
  of unloading) is a kernel-side win.

### K3. Branch-prefix-cache reuse across cmaj branches

The 5 cmaj branches share the same prompt prefix and the same first
sub-block (until temperature noise diverges them). KV-cache for the prefix
+ `block_0` could be shared. **Question for kernels:** does prefix-KV
sharing across branches actually save measurable wall-clock on LLaDA-8B
at our shapes (`prompt_len ≈ 100, gen_length=128, sub_block_length=32`)?
If yes, this is a free 2-3× cmaj speedup.

- Reference: `2402.10379` ("Hydragen") for AR; no published DDLM analog.
- Sub-question: does LLaDA's bidirectional attention break per-branch KV
  reuse at the block boundary, or only at the temperature-divergence
  boundary?

### K4. (bonus) Compute-curve flattening from the prefix-robust LoRA

The Phase-1 prefix-robust LoRA is trained over 8 tiers. The conjecture is
that at inference the LoRA compresses the prefix-quality-induced
*per-token entropy* curve, which would let us run **fewer denoising
steps** for the same accuracy on a `weak`-prefixed input than base LLaDA
needs. Test: re-run the `c3` condition with prefix-robust-v3 at
`k ∈ {16, 32, 48, 64}` and base LLaDA at the same. **If the LoRA flattens
the k-vs-accuracy curve**, the LoRA is doing entropy-compression — a
genuine computational benefit, not just static accuracy.

### K5. (bonus) AR-extend-mid-DDLM efficiency

The D1 mode-router proposes `switch_to_ar(n)` directives mid-generation
(`e4/diff_llada.py:84`). Profile: cost of cross-model handoff (Qwen ↔
LLaDA) per directive at our shapes. If overhead dominates the AR-extend
cost itself, kernel work on shared-tokenizer / shared-KV cross-model
handoff is the bottleneck-blocker for the whole D1 line.

---

## Applicability to sfumato

Mapping each direction to which conditions it touches, expected delta on
GSM8K-test cmajc-v3=82.5% headline, and engineering days.

| Direction | Conditions touched | Expected Δ vs 82.5% | Eng-days |
|---|---|---|---|
| **D1** mode router | `cmaj`, `cmajc`, new `crouted` | +1.5 to +3pp (proposal §"Quantitative success criteria") | 8-12d (substrate harvest + bandit + eval) |
| **D2** latent-coupled joint LoRA | new condition (Berrayana-style); replaces `c3` | Berrayana floor +27pp on DART-5; on GSM8K likely +1-2pp on top of cmajc | 15-25d (port Berrayana, train, eval) |
| **NEW-1** frontier-judge productized | post-hoc on `cmaj` outputs | +6.16pp confirmed for Claude+CoT (Phase-2 result) | 3-5d (cross-family + MATH-500) |
| **NEW-2** cmajc → single-pass distillation | new `c2cd` (distilled c2c), no branching | target: match cmajc within −2pp at 1/5 the FLOPs | 5-8d |
| **NEW-3** RL with branch-agreement | retrain prefix-robust-v3 → v4 | if works: +1pp; risk of regression | 7-10d |
| **NEW-4** TTC trade curves | all of `c2`, `c2c`, `cmaj`, `cmajc` | descriptive; closes G6 | 3-4d |
| **NEW-5** larger-encoder PRM | post-hoc on `cmaj` outputs | extrapolation says cross zero ~Qwen-32B | 5-7d |
| **K1** match-FLOPs ablation | `cmaj`, `c2`, `c2c` at varied (k, b) | descriptive; closes G6 | 2-3d |
| **K2** commit-LoRA window sweep | `c2c`, `cmajc` × `COMMIT_N_BLOCKS={0..4}` | descriptive; +0 to +1pp at sweet spot | 1-2d |
| **K3** prefix-KV reuse across cmaj branches | `cmaj`, `cmajc` (kernel only) | wall-clock 2-3× speedup, accuracy unchanged | 5-8d kernel work |
| **K4** prefix-robust k-curve | `c3`, `c2` (with prefix-robust-v3 on/off) | k=32 with LoRA matches k=64 base → 2× FLOPs win | 2d |
| **K5** AR-extend cross-model handoff profiling | enables D1 | unblocker for D1 (no Δ alone) | 3-5d kernel work |

### Headline takeaways for the kernel agents

1. **K1, K2, K3 directly settle the LoRA-vs-compute question** (PLAN.md gap 6).
   These are the kernel-survey questions with paper-defense value.
2. **K3 prefix-KV reuse is the only kernel-side change with a clear free-lunch
   profile**: 5-branch cmaj currently rebuilds the prefix KV 5×. If LLaDA's
   bidirectional attention permits prefix-cache reuse to the first
   sub-block boundary, this is a free 2-3× cmaj wall-clock speedup at zero
   accuracy cost.
3. **D1 mode-router is unblocked by K5** (cross-model handoff cost). If
   handoff is cheap, D1 is the natural Phase-2 graduating direction; if
   handoff is expensive, D1 needs a kernel patch first.
4. **NEW-1 (frontier-judge productized) is the closest-to-shipping new
   direction**, requiring zero kernel work — its cost is API spend.

---

## Citations

arXiv IDs and URLs cited above (≥10):

1. LLaDA — `2502.09992` — https://huggingface.co/papers/2502.09992
2. BD3-LMs — `2503.09573`
3. SDAR — `2510.06303`
4. Planner-Executor DDLM (Berrayana) — `2510.15244` — https://huggingface.co/papers/2510.15244
5. SongBloom — `2506.07634`
6. Self-Refine — `2303.17651`
7. Self-Consistency (Wang et al.) — `2203.11171`
8. d1 / diffu-GRPO — `2504.12216` — GitHub: https://github.com/dllm-reasoning/d1
9. AGRPO / GDPO — `2510.08554`
10. GIFT — `2509.20863`
11. LRD — `2510.11052`
12. Self-Speculative DLM — `2510.04147`
13. Self-Correcting Masked Diffusion — `2602.11590`
14. Diffusion-in-Diffusion — `2601.13599`
15. FailFast — `2512.20573`
16. SUNDAE — `2112.06749`
17. Coconut — `2412.06769`
18. CCDD — `2510.03206`
19. ACDiT — `2412.07720`
20. Hydragen (KV reuse, AR) — `2402.10379`
21. PRM800K (Lightman et al.) — `2305.20050`
22. Scaling LLM Test-Time Compute — `2407.21787`
23. BaRP — `2510.07429` — https://huggingface.co/papers/2510.07429
24. Expert-Choice DLM — `2604.01622`
25. Tree-of-Thoughts — `2305.10601`
26. Robust prefix-tuning (AR) — `2106.05665`
27. Dual-Objective LMs — `2512.14549`
28. ICL-aware fine-tuning (Chen et al.) — `2310.10616`
29. Self-Consistency distillation (Lin et al.) — `2310.06825`

Sfumato HF artifacts:

- `eren23/sfumato-llada-prefix-robust-v3` (headline LoRA, 22M params, 7/7 modules)
- `eren23/sfumato-llada-commit-v3` (headline commit LoRA, 13.6M params)
- `eren23/sfumato-prefix-robust-gsm8k` (8 prefix tiers × 7,473 problems = 59,784 rows)
- `eren23/sfumato-consensus-gsm8k` (cmaj b=5 outputs, 500 train)
- `eren23/sfumato-commit-mixture-gsm8k` (109-row training mixture)

Sfumato repos:

- Source: https://github.com/eren23/sfumato
- Paper: https://github.com/eren23/sfumato_paper

Internal anchors:

- `e4/diff_llada.py` (StepCallback / StepDirective contract — lines 25-101)
- `e4/runner.py` (env-driven condition dispatch — lines 95-160)
- `e2/PROTOCOL.md` (pre-registered predictions; Track 1 Pred 2 ≥78%, Track 2 Pred 3 ≤cmaj+1pp; **Pred 3 violated upward by +3.5pp** — the headline)
- `phase2/STATUS.md` lines 51-63 (STATUS-schema JSONL trace format)
- `phase2/proposals/RANKING.md` (final Phase-2 ranking — D1 graduating, D2 Phase-3, D3/D3.5 dead for self-trained verifiers)
- `phase2/PHASE2_FINAL_SUMMARY.md` (Claude-Sonnet-4.5+CoT WIN-MINOR, 86% gap closure)
