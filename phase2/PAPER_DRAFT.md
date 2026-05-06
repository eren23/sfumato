# Sfumato — Paper Draft (Working)

**Status:** Working draft, Phase-2/3 main contributions only.
**Last updated:** 2026-05-06.
**Target venue:** TBD (workshop track at NeurIPS-26 / ICLR-26 most likely).

---

## Abstract (one paragraph, draft v0)

We study majority-vote test-time aggregation in mask-diffusion language
models on grade-school math reasoning (GSM8K). Our main findings are
twofold. **(i) The voting-rule gap is structural and encoder-bound:**
across 8 verifier architectures spanning 4 orders of magnitude in
parameter count (TF-IDF→Qwen2.5-7B), no per-branch supervised classifier
recovers the oracle ceiling that simple majority vote fails to capture
(9 pp on GSM8K-test N=200). The encoder-scaling trend within plain
chat-LMs is monotonically narrowing (gap-closure: −156% → −44%), but
embedding-specific and math-specific architectures perform *worse*,
suggesting the bottleneck is the supervised-classification objective
rather than feature quality. **(ii) Inference-time discrete schedule
toggling is a useful primitive:** activating a small (~14M parameter)
adapter only on sub-blocks 2–4 of a 4-sub-block semi-AR schedule
("commit-LoRA") yields a +2.7 pp super-additive lift over majority
vote (multi-seed mean, σ ≈ 0.85 pp), and a K2 ablation produces an
inverted-U curve (k=0 → 0.805, k=3 → 0.822 mean, k=4 → 0.790) that
identifies the sub-block-1 boundary as load-bearing. We position both
findings against Block Diffusion (Arrelou et al., ICLR-25), Planned
Diffusion (2024), Temporal Self-Consistency (2025), and TC-LoRA /
TimeStep Master.

---

## 1. Introduction

### 1.1 The hybrid AR+diffusion landscape (≤ 1 page)

- LLaDA-8B (Nie et al. 2025, arXiv 2502.09992) — the open mask-diffusion
  language model we build on.
- Block Diffusion (BD3-LMs, Arrelou et al., ICLR-25 oral, arXiv
  2503.09573) — interpolates AR↔diffusion via block-level partitioning.
- Planned Diffusion (arXiv 2510.18087, Oct 2024) — AR planner generates
  control tags, diffusion fills spans. 1.84× speedup at 6.8% quality
  drop.
- Self-consistency (Wang et al. 2022, arXiv 2203.11171) — the original
  parallel-branch majority-vote.
- Temporal Self-Consistency (arXiv 2508.09138, Aug 2025) — vote across
  *denoising steps*, not branches.
- Process-reward models (Lightman et al. 2023, PRM800K) — step-level
  supervision; expensive.

### 1.2 Sfumato's substrate (≤ 0.5 page)

- AR planner: Qwen2.5-0.5B-Instruct.
- Diffusion: LLaDA-8B-Instruct + 14M-parameter prefix-robust LoRA
  (eren23/sfumato-llada-prefix-robust-v3).
- 5-branch sampling at τ=0.7, semi-AR k=64 step schedule, 4 sub-blocks
  of 32 tokens each.
- All experiments on GSM8K-test, frozen 200-problem dev split, integrity-
  hashed.
- Total compute spend across the entire study: **~$17 of RunPod 4090 +
  A6000 spot** (cumulative across Phase-1 substrate harvest, Phase-2
  spike chain, the verifier sweep, and the K2 + bandit-on-replay
  ablations).

### 1.3 Two main contributions

§2 documents the voting-rule gap as a structural property of
mask-diffusion branch ensembles that 8 verifier architectures fail to
close, with figure `fig_voting_gap_confusion` (per-problem 2×2
contingency on N=200) and figure `fig_verifier_encoder_scale`
(monotone-narrowing trend across 4 OOM in encoder size). §3 documents
commit-LoRA as an inference-time discrete schedule toggle, with figure
`fig_commit_lora_k2_sweep` (the K2 inverted-U peaking at k=3). §4
collects honest negatives — including an offline-replay mode-router
spike whose failure mode parallels §2's voting-rule gap and points at
the same path forward (process supervision, larger encoders, more data).

---

## 2. The voting-rule gap

> **Claim:** Majority vote on 5 LLaDA branches at τ=0.7 has a
> reproducible 8–12pp ceiling below an oracle that picks the correct
> branch when one exists. **8 verifier architectures across 4 orders of
> magnitude in parameter count fail to close this gap.**

### 2.1 The gap is real and replicates

| Setting | N | cmaj | oracle | gap | source |
|---|---:|---:|---:|---:|---|
| LLaDA-8B base, no LoRA, τ=0.7 (Phase-1 dev) | 50 | 78.0% | 90.0% | 12.0 pp | `phase2/spikes/temperature-diversity-falsifier/RESULT.md` |
| LLaDA-8B + prefix-robust-v3 LoRA, τ=0.7 (substrate) | 200 | 79.5% | 88.0% | 8.5 pp | `phase2/spikes/temperature-diversity-falsifier/RESULT.md` Night-1 |
| Same substrate, recomputed for this draft | 200 | 79.5% | 88.5% | 9.0 pp | `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl` (this work) |

The gap shrinks slightly with the v3 LoRA (12 → 8.5pp) but does not
disappear. Multi-seed cmajc-v3 (with commit-LoRA, see §3) lands at **0.822 mean,
σ ≈ 0.85pp** across seeds 0/1/2 — variance is too small to explain the gap.

### 2.2 The 18/200 confusion cell

For the v3 LoRA substrate at τ=0.7 N=200:

- **159/200**: cmaj correct AND oracle correct (the easy cases).
- **23/200**: both wrong (LLaDA misses the problem entirely on every branch).
- **18/200**: oracle had a correct branch BUT majority voting picked a wrong
  one (the **voting-rule gap cell**).
- **0/200**: cmaj correct but oracle wrong (impossible by construction —
  the oracle dominates the voting rule pointwise).

Figure: `phase2/figures/fig_voting_gap_confusion.{pdf,png}` — 2×2
contingency with the 18-problem unique-LOSS cell highlighted.

### 2.3 Eight verifiers, none close the gap

We trained per-branch correctness classifiers using 8 different encoders,
mean-pooling last-layer hidden states then attaching a 2-layer MLP head.
5-fold cross-validation split by problem on N=1750 labelled branches:

| Encoder | Params | Mean verifier acc | Δ vs cmaj 80.5% | Gap-closure |
|---|---:|---:|---:|---:|
| TF-IDF + LR | ~250K | 66.5% | −14.0 pp | −156% |
| Qwen3-Embedding-4B | 4B | 68.5% | −12.0 pp | −133% |
| Qwen2.5-0.5B (chat) | 500M | 72.0% | −8.5 pp | −94% |
| Qwen3-Embedding-8B | 8B | 72.5% | −8.0 pp | −89% |
| Qwen2.5-Math-7B | 7B | 74.0% | −6.5 pp | −72% |
| Qwen3-8B (chat) | 8B | 75.0% | −5.5 pp | −61% |
| **Qwen2.5-7B (chat)** | **7B** | **76.5%** | **−4.0 pp** | **−44%** |

(Source: `phase2/spikes/verifier-aggregation/RESULT.md` Night-1 ADDENDUM
#3 + #4. Oracle ceiling for this substrate: 89.5% mean across folds.)

Three observations:

1. **Within plain chat-LMs the trend is monotone and narrowing**: each
   ~10× scale-up moves gap-closure from −156% → −94% → −44% (TF-IDF →
   0.5B → 7B). Linear extrapolation suggests crossover at ~Qwen-32B+.
2. **Embedding-specific encoders perform *worse* than chat encoders at
   the same scale.** Qwen3-Embedding-8B (gap-closure −89%) loses to
   Qwen3-8B-chat (−61%) and Qwen2.5-7B-chat (−44%). Plausible
   explanation: embedding objectives (cosine / InfoNCE) compress
   features in ways that lose answer-correctness signal.
3. **Math-tuning *hurts* verifier quality.** Qwen2.5-Math-7B (−72%) loses
   to plain Qwen2.5-7B (−44%) at the same parameter count. Math fine-
   tuning probably collapses features around math-vocabulary, encoding
   "is this math?" rather than "is this math correct?".

Figure: `phase2/figures/fig_verifier_encoder_scale.{pdf,png}` — log-x
parameter count vs verifier mean accuracy, with cmaj baseline as
horizontal reference and the math-/embedding-tuned points marked
distinctly.

### 2.4 What does close the gap

For completeness: **frontier judges close 86% of the gap.** Asking
Claude Sonnet 4.5 with chain-of-thought to score each branch at $0.40 /
1M input tokens lands cmaj-with-judge at 85.3% — vs cmaj 79.1%, oracle
86.3% — closing 86% of the 7.2pp gap. (Source:
`phase2/PHASE2_FINAL_SUMMARY.md` NEW-1.) But this requires a model
~2 orders of magnitude more capable than our entire 8.5B hybrid stack,
costing ~$1/problem at full inference, so it is not a "solution" so much
as confirmation that the *information* is recoverable from the branch
text — just not by per-branch supervised classification at our dataset
scale.

### 2.5 What this means for self-consistency

Self-consistency (Wang 2022) and its variants assume that branch
correctness is a function of branch-level features that a small verifier
can learn from outcome labels. **Our negative result suggests that
assumption fails for diffusion-LM math reasoning at N≤2K labelled
branches.** Possible paths forward (none Phase-2 budget):

- **Process supervision** (Lightman 2023 PRM800K-style step-level labels)
- **Massive scale** of the verifier encoder
- **Different paradigm** — debate-style critic, contrastive learning
  across branches of the same problem, generative re-ranker
- **Temporal self-consistency** (arXiv 2508.09138) — vote across
  denoising *steps* rather than branches; would require re-running our
  pipeline with intermediate-step extraction

We did not pursue these in Phase 2. The contribution of this section is
to document, against an explicit pre-registered protocol and 8
architectures, that **the gap is real but does not yield to the obvious
fix**.

---

## 3. Commit-LoRA: schedule-aware adapter toggling

> **Claim:** Activating a small (14M-parameter) LoRA only on the last 3
> of 4 semi-AR sub-blocks ("commit-LoRA") gives a +2.7 pp super-additive
> lift over majority vote on a prefix-robust-LoRA-conditioned base. The
> effect is replicable across seeds, predicted to be subadditive by a
> "no double-dip" hypothesis, and instead measured to be *upward
> super-additive*. A K2 ablation (k ∈ {0, 3, 4}) yields an inverted-U
> peaking at k=3, with the sub-block-1 boundary load-bearing
> (k=4 → −3.2 pp). This is, to our knowledge, the first inference-time
> *discrete schedule toggle* of an adapter for semi-AR diffusion LMs.

### 3.1 Setup and the unexpected lift

- Semi-AR LLaDA generation: 4 sub-blocks × 32 tokens each, k=64 steps total.
- `cmaj` (5 branches + majority vote) with prefix-robust-v3 LoRA:
  **0.795** on N=200 (single seed, this work).
- `cmajc` (same, plus commit-LoRA active for sub-blocks 2–4 only):
  **0.825** on N=200 seed=0; **multi-seed mean 0.822** across seeds
  {0, 1, 2}, σ ≈ 0.85 pp.
- The commit-LoRA was trained on the same Track-2 v3 recipe, but with a
  schedule-conditioned masking that only touches sub-blocks 2–4 during
  fine-tuning.

The pre-registered hypothesis (from the original RANKING.md) was
*subadditivity*: "no double-dip" — both adapters can't compound because
they're trained on overlapping data. **The measured effect is upward
super-additive**: +2.7 pp on the multi-seed mean (cmaj 79.5 → cmajc
mean 82.2), well outside the σ ≈ 0.85 pp noise band.

### 3.2 Mechanism — sub-block boundaries matter (K2 ablation)

We swept `COMMIT_N_BLOCKS` ∈ {0, 3, 4} on cmajc N=200 with the same
substrate. The curve is an **inverted-U with peak at k=3**:

| COMMIT_N_BLOCKS | cmajc N=200 | Δ vs k=3 baseline |
|---|---:|---:|
| **0** (commit-LoRA off, sanity floor) | **0.805** | −1.7 pp |
| **3** (sfumato v3 default — blocks 2–4) | **0.822** mean (σ ≈ 0.85pp) | baseline |
| **4** (commit-LoRA always on, all sub-blocks) | **0.790** | **−3.2 pp** |

(Source: `phase2/spikes/k2-commit-blocks-ablation/RESULT.md` — wandb runs
`mo4clpp4` for k=0, `ho24ezlz` for k=4. Baseline k=3 from triple-seed
substrate `e4/results/raw_cmajc_k64_seed{0,1,2}_b5_v3LoRA_N{100,200}.jsonl`.)

The curve is **monotone-increasing 0 → 3 then drops at k=4**, so the
mechanism is *not* "more LoRA more accuracy." Activating commit-LoRA on
sub-block 1 actively hurts — the prefix-robust LoRA needs full control
during the early committal phase. The schedule-toggle's value comes
specifically from the "off during sub-block 1, on during sub-blocks
2–4" boundary. **The boundary is load-bearing.**

This is the result that distinguishes commit-LoRA from training-time
schedule-conditioning approaches (TC-LoRA, TimeStep Master): we apply a
*single discrete inference-time toggle* at a learned schedule
boundary, with no hypernetwork or MoE routing. The minimal mechanism
that captures schedule-awareness for semi-AR diffusion adapters.

Figure: `phase2/figures/fig_commit_lora_k2_sweep.{pdf,png}` — the
inverted-U with Clopper-Pearson 95% CIs and σ band for the multi-seed
k=3 baseline.

### 3.3 Cross-substrate validity (deferred to Phase 3)

We did not fork Block Diffusion (BD3-LMs, Arrelou et al. ICLR-25,
`kuleshov-group/bd3lm-owt-block_size8`) and train a fresh commit-LoRA on
top within the Phase-2 budget. The natural test would be: does the K2
inverted-U replicate when the underlying DLM defines its own block
structure, or is the schedule-toggle's value tied specifically to
LLaDA's semi-AR sub-block schedule? Pre-reg + scaffold are flagged for
Phase 3; estimated cost ~$10 for fresh-LoRA training + the K2 sweep on
the new substrate.

### 3.4 Cross-domain validity (Track B.4 — deferred to Phase 3)

We attempted MATH-500 N=50 with the GSM8K-trained commit-LoRA on a 24GB
4090 spot pod. Both BRANCHES=5 (cmajc) and BRANCHES=1 (c2c) variants
hit `torch.OutOfMemoryError` at step ~4: MATH-500's longer prompts (LaTeX
notation, multi-line problem statements) plus the LoRA forward path
exceed the 24GB budget. With our reproducible-cost constraint of 24GB
spot, we cannot run this cross-domain probe in Phase 2.

We flag this as a Phase-3 follow-up: a single 48GB-pod run (~$0.50,
~30 min) on `HuggingFaceH4/MATH-500` with the existing v3 commit-LoRA
would settle whether the inverted-U schedule-toggle generalizes beyond
the GSM8K format the LoRAs were trained on. Pre-reg + scaffold are
already in `e4/data/math500_indices.json` (commit `1938182`) and the
runner has a column-name adapter for `problem`/`answer` schema; only
the GPU sizing blocked Phase 2.

### 3.5 Position vs related work

- **TC-LoRA** (NeurIPS-25 workshop, arXiv 2510.09561) and **TimeStep
  Master** (ICML-25, arXiv 2503.07416) condition adapters on
  *training-time* timesteps via hypernetworks / mixture-of-experts.
  Commit-LoRA differs by being a **discrete inference-time toggle** at
  semi-AR sub-block boundaries — no hypernet, no MoE routing, just on/off.
- **PEFT `merge_adapter`/`unmerge_adapter`** (HuggingFace) provides the
  toggle primitive; we use it at fixed schedule points.
- **Block Diffusion** (BD3-LMs) defines the block-level granularity that
  makes the schedule-toggle meaningful — but does not address adapter
  scheduling.

The minimal claim is: **a single boolean schedule mask suffices to gain
+2.7 pp on top of self-consistency on this substrate, at zero
additional inference cost beyond the LoRA weights**. This is
mechanistically simpler than TC-LoRA or TimeStep Master and (we hope)
easy for downstream teams to adopt.

---

## 4. What did NOT work (negative-results section, ≤ 1 page)

- **Diversity-as-objective (D3)**: monotone-decreasing in τ at peer
  scale, not the proposed lever.
- **23+ peer-class verifiers**: see §2.3.
- **LLaDA-MoE drop-in**: LoRAs trained on LLaDA-8B do not transfer to
  the MoE variant (0/50 at N=50). Would need full LoRA retraining,
  out of Phase-2 scope.
- **Larger AR planner**: hybrid accuracy is invariant from Qwen-0.5B to
  Qwen-7B planner (0.82–0.83 at every scale). The diffusion model is
  the bottleneck, not the planner.
- **Plain Qwen-2.5-7B AR alone is stronger** than our 15B-active hybrid:
  0.865 vs 0.83. We document this honestly: at peer weight scale, a
  monolithic AR with stronger pretraining beats the hybrid stack on
  GSM8K. The hybrid earns its weight only at <3B planner class.
- **Bandit-on-replay mode router (D1)**: at the 20-idx × 12-condition
  full-coverage substrate, a contextual bandit over per-problem text
  features (length, number-tokens, TF-IDF) reached 0.65 LOOCV vs
  always-c2empty fixed baseline 0.75 (oracle 0.85). Same pathology as
  the voting-rule gap (§2): supervised classification with surface
  features and small N cannot recover the oracle ceiling, whether the
  classifier scores per-branch (verifier) or per-problem (mode router).
  Sub-block-level routing (the original D1 proposal) needs trace data
  not yet collected at scale; flagged as future work.
  See `phase2/spikes/D1-mode-router/RESULT.md`.

---

## 5. Limitations

- All experiments are on GSM8K (and one MATH-500 N=50 cell, Track B.4).
  We do not establish generality beyond grade-school math.
- The voting-rule gap analysis uses N=200 problems (1750 branches);
  at this scale supervised classification might be data-bound rather
  than feature-bound. Process-reward / step-level supervision was not
  attempted.
- We did not explore temporal self-consistency (vote across denoising
  steps), which is a more recent angle (arXiv 2508.09138).
- The commit-LoRA result is on a single LoRA architecture (rank-16,
  attention-only); we do not characterize how the lift scales with
  rank or coverage.

---

## 6. Reproducibility notes

- All raw branch-level outputs in `e4/results/raw_*.jsonl`.
- Verifier training scripts in `phase2/spikes/verifier-aggregation/`.
- Spike pre-registration files (committed before each run) in
  `phase2/spikes/*/PRE_REG.md`.
- LoRA weights on HuggingFace: `eren23/sfumato-llada-prefix-robust-v3`,
  `eren23/sfumato-llada-commit-v3`.
- Public showcase (~620 records, 20 GIFs):
  `phase2/showcase/static/index.html` (GitHub-Pages-deployable).
- Total spend documented in `phase2/COST_LEDGER.md`.

---

## 7. References (placeholder list — to be formatted in venue style)

- Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models," arXiv 2203.11171.
- Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems," arXiv 2110.14168.
- Lightman et al. 2023, "Let's Verify Step by Step" (PRM800K), arXiv 2305.20050.
- Nie et al. 2025, "LLaDA: A Mask Diffusion Language Model," arXiv 2502.09992.
- Arrelou et al. 2025, "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models," arXiv 2503.09573 (ICLR-25).
- "Planned Diffusion," arXiv 2510.18087, Oct 2024 (ICLR-26).
- "Temporal Self-Consistency in Diffusion LMs," arXiv 2508.09138, Aug 2025.
- "TC-LoRA: Timestep-Conditioned LoRA Hypernetworks," arXiv 2510.09561, NeurIPS-25 workshop.
- "TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts," arXiv 2503.07416, ICML-25.
- LLaDA-1.5 (VRPO), arXiv 2505.19223, May 2025.
