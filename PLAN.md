# Hybrid AR+Diffusion "Iterative Thinking" — Research Map & Experiment Plan

## Context

User's intuition: human cognition is neither pure left-to-right autoregressive nor pure denoise-from-noise diffusion. We sketch a concept, parts stay vague and iterate, we revise, change direction, then continue serially until we have a full picture. The question: what's already been published in this space, and what's genuinely novel territory worth running experiments on?

Goal of this document: map the existing literature (4 parallel research agents, 226 papers examined — well above the 100+ threshold the user requested), identify the closest precedents, isolate the genuinely open gaps, and propose a small ranked set of experiments.

## Coverage

Four parallel research agents covered overlapping but distinct angles:

| Angle | Papers examined |
|---|---|
| Hybrid AR+diffusion architectures (text/code/multimodal) | 84 |
| Diffusion-as-reasoning, latent thought, recurrent depth, test-time compute | 47 |
| Iterative refinement, edit-based decoders, self-refine, search-as-reasoning | 40 |
| Dual-process AI, unified AR+diffusion multimodal, world-model reasoning, JEPA | 55 |
| **Total (with overlap across agents)** | **~226 distinct paper hits, ~180 unique arXiv IDs** |

## Convergent findings (what the literature already covers)

The user's idea decomposes into several threads. Most threads are partially mature:

1. **Block / semi-autoregressive diffusion** — AR across blocks, diffusion within. Mature recipe.
   Anchors: Block Diffusion / BD3-LMs (`2503.09573`), SDAR (`2510.06303`), ACDiT (`2412.07720`), MADFormer (`2506.07999`), Fast-dVLM (`2604.06832`), SSD-LM (`2210.17432`), SUNDAE (`2112.06749`).

2. **AR-with-diffusion-head** — AR backbone, per-token diffusion head produces continuous output (image/audio).
   Anchors: MAR (`2406.11838`), MarDini (`2410.20280`), DART (`2410.08159`), HMAR (`2506.04421`), VideoMAR (`2506.14168`).

3. **Plan-then-fill switching** — AR emits a plan/outline; diffusion fills in parallel.
   Anchors: Planned Diffusion (`2510.18087`), Planner-Executor DDLM↔ARM (`2510.15244`), Think First Diffuse Fast (`2603.13243`), LaDiR (`2510.04573`).

4. **Diffusion-as-reasoning** — denoising steps act as reasoning iterations.
   Anchors: Diffusion-of-Thought (`2402.07754`), d1/d2 (`2504.12216`, `2509.21474`), DCoLT (`2505.10446`), CCDD (`2510.03206`), IRED (`2406.11179`).

5. **Recurrent / looped depth** — iterate the same block r times in latent space; r is test-time configurable.
   Anchors: Geiping "Huginn" (`2502.05171`), Saunshi looped reasoning (`2502.17416`), Universal Transformer (`1807.03819`), MoEUT (`2405.16039`). Geiping's `2510.14961` *proves* recurrent-depth is samplable as a diffusion LM — direct unification.

6. **Latent / continuous CoT** — feed last hidden state back as next input embedding; reasoning happens in latent space.
   Anchors: Coconut (`2412.06769`), Quiet-STaR (`2403.09629`), Pause Tokens (`2310.02226`), SoftCoT++ (`2505.11484`), Token Assorted (`2502.03275`), KaVa (`2510.02312`), PLaT (`2601.21358`).

7. **Edit-based decoders** — atomic insert/delete/replace operations over partial text.
   Anchors: Levenshtein Transformer (`1905.11006`), Insertion Transformer (`1902.03249`), DiffusER (`2210.16886`), Mask-Predict / CMLM (`1904.09324`), EdiT5 (`2205.12209`).

8. **Unified multimodal AR+diffusion (single transformer, dual loss)**
   Anchors: Transfusion (`2408.11039`), Show-o (`2408.12528`), Janus / Janus-Pro (`2410.13848`, `2501.17811`), JanusFlow (`2411.07975`), Chameleon (`2405.09818`), BAGEL (`2505.14683`), MANZANO (`2509.16197`).

9. **Self-refine / post-hoc revision** — generator + critic + rewriter.
   Anchors: Self-Refine (`2303.17651`), Reflexion (`2303.11366`), CYCLE (`2403.18746`).

10. **Dual-process VLA** — System-2 plans, System-1 executes; mostly engineering, mostly Kahneman-as-decoration.
    Anchors: Dualformer (`2410.09918`, only honest-to-name in the family), Hume (`2505.21432`), OneTwoVLA (`2505.11917`), AlphaOne (`2505.24863`), Distilling System-2 into System-1 (`2407.06023`).

11. **Cross-modal interleaved sketch-then-refine** — closest existing instance of the user's loop.
    Anchor: SongBloom (`2506.07634`) literally titled "Interleaved Autoregressive Sketching and Diffusion Refinement" — but for music, not language.

## Closest existing matches to the user's exact intuition

Ranked from most-direct to least-direct match for **"AR-extend → diffuse-refine some chunk → AR-continue → diffuse again"**:

1. **SongBloom (`2506.07634`)** — exactly the iterated AR-sketch + diffusion-refine pattern, but cross-modal (audio).
2. **Block Diffusion / BD3-LMs (`2503.09573`)** — exactly the pattern in language at the *block* token level, with fixed block size and one round per block.
3. **Planner-Executor DDLM→ARM (`2510.15244`)** — diffusion model produces a latent plan, AR articulates from it. One round only. Closest empirical demo of "diffuse the abstract scaffold, then AR-continue."
4. **Geiping recurrent-depth + diffusion-forcing sampler (`2502.05171` + `2510.14961`)** — the model's forward pass *is* iterative latent refinement; a diffusion sampler can refine future positions while AR commits earlier ones.
5. **Diffusion-in-Diffusion (`2601.13599`)** — block-AR draft → global bidirectional diffusion polish with confidence-based remasking. One round only.
6. **ACDiT (`2412.07720`)** — block-wise continuous AR with diffusion denoising inside each block. Pixel domain.
7. **Coconut (`2412.06769`)** — latent thoughts encode a superposition of reasoning paths before collapsing to AR tokens. The "iterate-on-private-sketch-before-externalising" leg of the loop.
8. **CCDD (`2510.03206`)** — coevolves a continuous diffusion latent with discrete diffusion tokens; explicitly framed as making a diffusion LM into a latent reasoner.

## Genuinely open territory

Across all 4 agents, the same gaps appeared independently:

**Gap 1 — Multi-round, model-decided alternation in language.**
Almost all existing hybrids do AR→diffuse *once*, or one fixed block diffusion per chunk. SongBloom does multi-round but in audio. **There is no published text LM where the model itself decides at each step "should I serially extend, or revisit and re-diffuse the last K tokens, or change direction?"** — i.e. a learned policy over operation type.

**Gap 2 — Backtracking / direction-change as a first-class action.**
Block diffusion is irreversible by design. Self-correcting masked diffusion (`2602.11590`) remasks tokens but not strategic chunks. A dedicated "rewrite the last paragraph because the next sentence I drafted contradicts it" action is unbuilt.

**Gap 3 — Mixed-granularity refinement.**
The user's intuition contains "*some parts* diffuse, others AR" within the same generation. No published work studies *masking specific spans* (a sub-question's answer, a numeric value, a planning concept) and iteratively refining only those while surrounding text is AR.

**Gap 4 — Adaptive, uncertainty-triggered diffusion budget.**
Adaptive Latent Reasoning via RL (`2511.21581`) learns when to *stop* latent reasoning. Nobody learns when to *re-enter* it after AR continuation. Geiping Huginn supports per-token adaptive recurrence but the policy is local convergence, not concept-level uncertainty.

**Gap 5 — Diffusion timesteps as semantically-named reasoning hops.**
DoT/CCDD treat denoising as a continuous schedule with no interpreted intermediate states. No work supervises diffusion timesteps to correspond to specific reasoning phases (abstraction → decomposition → grounding).

**Gap 6 — Mechanistic faithfulness of diffusion-as-thought.**
For latent CoT we now have `2512.21711` (Coconut tokens are largely uninterpretable shortcuts) and `2504.09762` (warning against anthropomorphizing intermediate tokens). The equivalent careful study for *diffusion* reasoning steps is missing.

**Gap 7 — Three-phase staged inference end-to-end.**
The full loop *AR plan tokens → diffusion-refine a latent sketch → AR articulate final tokens conditioned on refined sketch*, with a learned halting policy on the refinement stage, is **not formalized as a single end-to-end trained system anywhere**. ACDiT, JanusFlow, RIG each cover two of the three legs; CCDD is the closest unified formalism but is text-only and not staged as plan→sketch→articulate.

## Bottom line

- "AR + diffusion in one model" — crowded.
- "AR-then-diffuse-once" — crowded.
- "AR ↔ diffuse ↔ AR with a learned policy over which mode to take next, in language, with explicit backtracking" — **open territory.**

The user's idea is a natural composition of three established threads (block diffusion, latent reasoning, dual-process control), but the composition itself is unpublished.

## Compute envelope

User constraints (clarified): training on 2-4×RTX 4090 on Runpod, often spot pricing, modest budget, **goal is plausibility proof, not SOTA**. Practical implications:

- 2×4090 = 48 GB VRAM. 4×4090 = 96 GB. Spot price roughly $0.30-0.40/GPU/hr.
- Base models: ≤1.5B for from-scratch training, ≤7B for QLoRA fine-tunes, ≤3B for full fine-tunes.
- Every job must be **checkpointable** (spot can preempt). Use short shards (≤6 hr), resume from latest checkpoint.
- Each experiment needs a **kill criterion**: an early signal that says "stop, this isn't going to work" before burning weeks of compute.
- Datasets: small enough to iterate fast. GSM8K-Aug subsets, custom synthetic reasoning, ARC-Easy. No GSM8K-full pretrain runs.

## Roadmap — 4 tracks, sequenced

Total estimated wall-clock if all four run in sequence: ~10-14 weeks, ~$800-1500 in spot compute. Branch points let earlier kills save the rest.

```
Week 0         Week 1-2       Week 3-5        Week 6-9            Week 10-14
  │              │               │                │                    │
  Setup ─→ E4 (no train) ─→ E2 (LoRA) ─→ E1 (main contrib) ─→ E3 (stretch)
                  │               │                │                    │
              kill if          kill if          kill if            paper-grade
              hybrid no        span-only        router not         result, or
              better than      not equal        learning           just a strong
              pure DoT         to whole-CoT     hard/easy          ablation
                               diffusion        distinction
```

- **If E4 fails** (no signal that hybrid AR↔diffuse beats pure paradigms at matched compute): **stop the whole programme**; the user's intuition is not falsifiable cheaply at this scale and needs different tooling.
- **If E2 fails** (span-marked refinement doesn't match whole-CoT diffusion): jump straight to E1; mixed-granularity is not the right axis.
- **If E1 succeeds** (router learns nontrivial mode policy + improves over BD3-LM-no-router): consider E3 as a bigger writeup.
- **If E1 fails**: write up the negative result with E4+E2 ablations as the contribution.

## Per-track plans

### E4 — Inference-Time Hybrid CoT Pipeline (Track 0: plausibility check)

**Why first.** No training. Cheapest possible test of the core hypothesis: does interleaving AR and diffusion buy *anything* over either paradigm alone at matched compute? Calibrates whether the rest of the roadmap is worth running.

**Architecture (no new params):**
- AR backbone: Qwen2.5-1.5B-Instruct (fits comfortably on 1×4090).
- Diffusion module: LLaDA-1.5B (`2502.09992`) or DiffuLLaMA-1B (`2410.17891`) — pick whichever has cleaner inference code.
- Pipeline: `[user prompt] → AR plan tokens (≤32) → switch → diffusion fills <thinking>...</thinking> for k denoising steps → switch → AR final answer`.

**Data:** GSM8K test set (1319 problems). For dev: 200-problem subset.

**Conditions to compare (matched FLOPs):**
1. Pure AR CoT (Qwen2.5-1.5B baseline).
2. Pure diffusion CoT (LLaDA, k denoising steps for full trace).
3. AR-plan → diffusion-CoT → AR-answer (the hybrid).
4. AR-plan → multi-round (diffuse k steps, AR-extend, diffuse k steps again) — closest to the user's loop, harder to wire up.

**Kill criterion:** if condition 3 doesn't beat the better of (1, 2) by ≥1.5 pp on GSM8K at matched FLOPs across at least three k values, kill.

**Budget:** ~3-5 days, ~$50-80 spot. No training, only inference.

**Deliverable:** numbers + a write-up plot (accuracy vs compute, three curves). This alone is a workshop-paper-worthy negative-or-positive ablation if no one has published it cleanly (and per the survey, the disentanglement of compute-vs-iteration in DoT-style results is gap #6 — open territory).

### E2 — Span-Level `<sketch>` Refinement (Track 1: mixed-granularity)

**Why second.** Tests whether refining *some parts* (not all) is the right granularity. Cheaper than E1 because most parameters are frozen.

**Architecture:**
- Frozen Qwen2.5-1.5B (AR backbone).
- Small diffusion adapter (~100-150M params, DiT-style or LLaDA mini head) attached to the frozen AR at a layer-15 hook.
- LoRA adapters on AR for span-marker generation only.
- Format: `<sketch_open> ... <sketch_close>` markers; AR emits openers and closers, diffusion adapter denoises the content for k steps before AR continues past the closer.

**Training:**
- Synthetic CoT dataset (~50k problems): take GSM8K-Aug problems, mark sub-question answers as sketch spans. Augment with a few thousand "wrong → correct" sketch refinement traces so diffusion learns to repair.
- Two-stage training: (a) freeze AR, train diffusion adapter on span content denoising; (b) joint LoRA + adapter fine-tune for marker-emission policy.
- Hardware: 2×4090, mixed precision, gradient checkpointing. ~3-5 days for stage (a), ~1 week for stage (b).

**Eval:** GSM8K, MATH-Easy subset, custom synthetic with controlled span difficulty.

**Kill criterion:** if span-only refinement doesn't match whole-CoT diffusion (E4 condition 2) at half the FLOPs, mixed-granularity is not the right axis. Skip to E1.

**Budget:** ~2-3 weeks, ~$200-300 spot.

### E1 — Adaptive Diffuse-Continue Loop with Mode Router (Track 2: main contribution)

**Why this is the headline.** Matches the user's intuition most directly: a learned policy at every block boundary chooses among {extend-AR, diffuse-current-block, re-diffuse-last-K-blocks, commit}. This addresses gaps 1, 2, and 4 simultaneously — none of which are published.

**Architecture:**
- Backbone: BD3-LM-style hybrid transformer, ~170-400M params (the BD3-LM paper has small variants that fit single-4090 from scratch). Alternative: adapt DiffuLLaMA-1B (cheaper warm-start).
- **Mode-router head:** small MLP on the final hidden state of the last block boundary, outputs softmax over 4 actions. Trained jointly with the LM loss + a router-regularization loss (entropy bonus + cost-of-iteration penalty).
- KV cache tracks "committed" vs "tentative" tokens; re-diffuse-last-K invalidates the last K blocks of cache and re-runs diffusion conditioned on prior committed context.

**Training:**
- Stage 1: standard BD3-LM training on a small text mix (subset of FineWeb-Edu, ~5B tokens) — establishes baseline.
- Stage 2: synthetic reasoning dataset with controlled difficulty per span (so a working router *should* spend more iterations on hard spans). RL or supervised: start supervised (oracle action labels from cost-vs-accuracy tradeoff), only RL if needed.
- Hardware: 4×4090 ideal, 2×4090 viable for the smaller variant. ~2-3 weeks.

**Eval (and falsifiability):**
- GSM8K + a custom **dependency-injected reasoning** test set where late steps imply earlier ones are wrong. A router that doesn't learn to backtrack will fail here.
- Forced-mode ablations: freeze router to each single action, recover the corresponding baseline (BD3-LM, full re-diffusion, etc.).
- Per-problem compute distribution: hard problems should use more iterations. Plot this — if the distribution is flat, the router didn't learn anything.

**Kill criterion:** if after stage 2 the router's compute distribution is flat across difficulty bins, or forced-mode ablation matches the routed model, the router isn't doing anything. Write up as negative result.

**Budget:** ~4-6 weeks, ~$500-800 spot.

### E3 — Three-Phase Plan→Sketch→Articulate End-to-End (Track 3: stretch goal)

**Why last.** Highest risk. Only justified if E1 produces a working learned router — that proves dynamic mode-switching is trainable at this scale. E3 then asks the bigger question: can we train the *full* loop (AR plan → latent diffuse → AR articulate) end-to-end with a learned halting policy on the middle stage?

**Architecture:**
- Single transformer (~500M-1B). Three phases share parameters but use different attention masks and loss heads.
- Phase 1: AR over plan tokens.
- Phase 2: introduce N latent slots (16-32), iteratively diffuse them in *continuous* latent space (CCDD-style, `2510.03206`) for T steps; T is dynamic, controlled by a learned halting head reading slot-state convergence.
- Phase 3: AR articulates final answer with cross-attention to refined slots.

**Training:**
- End-to-end via a CCDD-style joint loss + halting penalty.
- Datasets: planning benchmarks where Coconut already shows superposition-of-plans helps (Game-of-24, ProsQA, Blocksworld), to maximise expected signal.
- Hardware: 4×4090, full fine-tune from a Coconut-initialised checkpoint if available, else from a small Qwen base.

**Eval:** Game-of-24, ProsQA, Blocksworld, GSM8K. Compare against plain Coconut, against E1 router model, against pure AR CoT — all at matched FLOPs.

**Budget:** ~4-6 weeks, ~$800-1500 spot. Only run if E1 succeeded.

**Deliverable:** if it works, this is the paper. If E1 already worked and E3 doesn't add over E1, that's also a clean finding (the dynamic mode-router is sufficient; full three-phase is not necessary).

## Branch decision tree (compact)

| After... | If... | Then... |
|---|---|---|
| E4 | Hybrid > pure paradigms at matched FLOPs | Continue to E2 |
| E4 | Hybrid no signal | Stop programme; rethink |
| E2 | Span refinement ≥ whole-CoT diffusion at lower FLOPs | Continue to E1 with span granularity baked in |
| E2 | Span refinement worse | Skip span granularity in E1; use block granularity only |
| E1 | Router learns nontrivial difficulty-conditioned compute | Continue to E3 |
| E1 | Router degenerate (flat policy) | Write up E1 negative result + E4/E2 positives; stop |
| E3 | Beats E1 at matched FLOPs | Paper |
| E3 | Doesn't beat E1 | E1 is the paper, E3 is an appendix ablation |

## Critical files / references for execution

- BD3-LM repo (Block Diffusion): paper `2503.09573` — anchor codebase for E1.
- DiffuLLaMA (`2410.17891`) — established AR→diffusion adaptation recipe; reuse for E2 and E4.
- Coconut implementation (Meta FAIR, paper `2412.06769`) — reference for latent slot training in E3.
- CCDD (`2510.03206`) — reference for joint continuous-discrete diffusion in E3.
- Geiping Huginn (`2502.05171`) + diffusion-forcing-sampler bridge (`2510.14961`) — reference for adaptive depth, comparable baseline.
- Survey on Parallel Text Generation (`2508.08712`) — bibliography shortcut.

## Verification plan (applies to every track)

1. Reproduce a single-paradigm baseline first (BD3-LM-no-router for E1, plain LLaDA for E4, plain Qwen2.5 for E2) before measuring novelty. Cite the baseline number; never claim improvement without it.
2. **Match FLOPs across conditions.** Iteration = compute. Gap 6 in the literature is precisely the confusion between "extra refinement helps because it's iterative refinement" and "extra refinement helps because it's more compute." Always plot accuracy vs. FLOPs, not vs. wall-clock.
3. **Forced-mode ablations.** Any model with a router/halting policy must reduce to known baselines under forced single-mode. If forced-AR-only doesn't match Qwen2.5 baseline, training is broken.
4. **Bench on tasks where the structure should matter and tasks where it shouldn't.** Reasoning matter: GSM8K, Blocksworld, dependency-injected synthetic. Shouldn't matter: TriviaQA, simple completion. If the hybrid gain transfers to non-reasoning tasks identically, we are not measuring iterative thinking, we are measuring more compute.
5. **Spot-checkpoint hygiene.** Every job saves every 30-60 min. On preempt, resume. Lost wall-clock budget per preemption ≤ 1 hr.
6. **Cost tracking.** Log $ spent per experiment; if a track exceeds 2× the planned budget without hitting kill or success criterion, stop and reassess.

## Total budget summary

| Track | Time | $ (spot) | Hardware |
|---|---|---|---|
| E4 | 3-5 days | $50-80 | 1-2×4090 |
| E2 | 2-3 weeks | $200-300 | 2×4090 |
| E1 | 4-6 weeks | $500-800 | 2-4×4090 |
| E3 (stretch) | 4-6 weeks | $800-1500 | 4×4090 |
| **Sum if all run** | **~13-20 weeks** | **~$1.5k-2.7k** | |
| **If kill at E1** | **~10 weeks** | **~$0.7-1.1k** | |
| **If kill at E4** | **~1 week** | **<$100** | |

The kill criteria are designed so the worst-case spend is small.

## Implementation entry points

For when execution begins:

- **E4 first:** fork [LLaDA inference repo] + [Qwen2.5 inference]; write a 200-line pipeline that pipes between them. No training infra needed.
- **E2 second:** fork DiffuLLaMA repo; LoRA on Qwen via PEFT; synthetic data generation script.
- **E1 third:** fork BD3-LM official repo (`https://github.com/kuleshov-group/bd3lms` likely); implement mode-router head as a small head on top; modify training loop for stage 2 RL/supervised.
- **E3 last:** fork Coconut implementation as the latent-slot reference; CCDD as the joint diffusion reference; build on top of whichever base model E1 used.

## Outstanding question (carried into execution)

Whether to start E4 with LLaDA (`2502.09992`) or DiffuLLaMA (`2410.17891`) — depends on which has cleaner inference code and stronger GSM8K baselines as of execution time. Decide at week 0 setup, not now.
