# Kernel & Free-Lunch Survey — SUMMARY

**Date:** 2026-05-04 | **Authoring run:** 5 parallel research subagents over `phase2/proposals/kernel-survey/`.
**Cost:** $0.00 (research-only). No GPU spend, no fleet calls, no code edits outside this directory.

---

## 0. Per-agent one-paragraph summaries

### `00_OUR_IDEA.md` — sfumato's idea, novelty audit, non-kernel directions
Sfumato is a **plan-then-fill hybrid**: Qwen2.5 plans, LLaDA-8B diffuses
the CoT under prefix-robust LoRA v3, commit-LoRA v3 fires on sub-blocks
2–4 of 4, b=5 stochastic branches majority-vote. Headline:
**cmajc-v3 = 82.5%** on GSM8K-test (multi-seed σ ≈ 0.3pp). Triangulated
against PLAN.md's 7-gap list: sfumato closes **none in full** but
contributes three engineering primitives with **no precedent found** in
arXiv / HF / GitHub: (A) prefix-robust LoRA on a DDLM via 8-tier prefix
augmentation, (B) commit-LoRA at sub-block boundaries 2..N-1 (not just
final), (C) `cmajc` per-branch commit-LoRA + regex vote. The
super-additive **+3.5pp** "no double-dip" violation (e2 PROTOCOL Pred 3)
also appears unpredicted. Five new non-kernel directions surfaced:
NEW-1 frontier-judge productized (already +6.16pp validated), NEW-2
cmajc → single-pass distillation, NEW-3 RL with branch-agreement reward,
NEW-4 TTC trade curves, NEW-5 larger-encoder PRM. Five kernel-friendly
gap-6 disambiguation experiments (K1–K5) framed for the kernel agents.

### `01_diffusion_lm_kernels.md` — discrete-diffusion-LM kernel catalog
**30 kernels** catalogued (≥15 floor cleared 2×). Hot-path map:
`e4/diff_llada.py:444` (the per-step `self._model(x).logits` call) is
**>95% of wall-clock** at sfumato's 320 forwards/cmajc-v3-problem.
Strongest drop-in: **Fast-dLLM v1** (NVlabs, `2505.22618`, Apache-2.0,
LLaDA-8B-targeted, **5–12× on c2c** GSM8K). Runner-up: **dKV-Cache**
(`2505.15781`, NeurIPS'25, simpler, 2–4×). Schedule-side wins:
**SlowFast Sampling** (`2506.10848`, 3–8× alone, 34× combined with
caching), **dParallel** LoRA (`2509.26488`, 8.5×), **Prophet**
early-commit (`2508.19982`, 3.4×). Sampling-path: **FlashInfer** top-k
kernels (5–15%). Largest *missed* opportunity is non-kernel:
`runner.py:203-261`'s sequential `for b in range(5)` — branch batching is
**~4×** with no kernel work. Combined sequenced rollout (Fast-dLLM v1 +
SlowFast + branch-batching) collapses N=200 cmajc-v3 from ~3.5h ($0.70)
to **~10–15min ($0.05)**.

### `02_hybrid_kernels.md` — AR+diffusion hybrid kernels
**11 hybrid-mask kernels + 6 KV-bridge entries + paged-attention state-of-art**.
Headline finding: sfumato's hybrid mask (causal Qwen prefix → bidi LLaDA
gen → causal Qwen finalize) is **FlexAttention-shaped**, and BD3-LMs +
dFactory have already published the exact recipe — only work is wiring
into LLaDA's `trust_remote_code` modeling. **No fused kernel exists** for
a Qwen→LLaDA cross-architecture KV bridge — sfumato-specific gap. SGLang
shipped day-0 LLaDA-2.0 support Dec 2025 with RadixAttention but does
**not** wire sfumato's specific AR-plan→diffuse→AR-finalize pipeline; vLLM
still experimental. Eight concrete missing kernels listed (cross-arch KV
bridge, branch-batched diffusion, phase-switching paged attention, etc.).
If sfumato lifts only **Fast-dLLM + RadixAttention** (~5–7 eng-days, two
OSS repos), `c2/c2c/cmaj/cmajc/cmerge` get **3–8× wall-clock** with zero
algorithmic change.

### `03_branching_free_lunch.md` — cmaj/cmajc free lunch
**17 techniques** across (A) prefix-sharing, (B) paged-KV, (C) speculative,
(D) pruning, (E) continuous batching, (F) sampler kernels. Quantified the
redundancy: at L_pref=150, L_gen=128, b=5, **prefix is 54%** of every
forward, computed 5× redundantly. Hydragen analytic ceiling: **1.76×**;
realistic **1.4–1.6×** after overhead. Speculative decoding (C1, AR-style)
**does not port** to mask diffusion — architectural mismatch. SpecDiff-LM
(`2502.06768`) and AccSpec-dLLM (`2508.02193`) are the dLLM-specific
analogues. Spike candidate: **A1 (b-batching) + D1 (early-exit
self-consistency at quorum=3)** — ~$3–5 GPU + ~2 eng-days, expected
**1.3–1.7×** wall-clock with ≤1pp accuracy delta. Pre-reg sketch
included with WIN/LOSS/INCONCLUSIVE thresholds. Mid-denoising D3 pruning
(novel; Phase-1 ABL_B closure shows commit-LoRA reinforces rather than
flips, supporting low false-prune-rate prior) adds another ~1.25× on top.

### `04_commit_lora_free_lunch.md` — commit-LoRA + denoise-loop kernels
**17 techniques** across PEFT-fusion, logit-shift caching, top-k Triton,
mask-only attention, and quality plays. Cheapest spike: **E1
(merge-on-toggle) + B3 (populate `logit_shift_norm`)** — ≤20-line patch,
PEFT API already exists, exact-equivalence numerical guarantee, ~5%
per-step on commit-active sub-blocks (~2–3% end-to-end on c2c, **5×
absolute on cmajc**), ~$1 GPU + 1.5–2 eng-days. Sparse-dLLM
(`2509.24095`) port plan: 6–8 eng-days, expected **8–15%** wall-clock cut
at k=32–128, stacks with E1, applies to *every* LLaDA-using condition.
Negative finding: **FlashLoRA** has no canonical OSS repo as of
2026-05-04. `merge_and_unload` patch sketch (≤20 lines) included.

---

## 1. Cross-cutting themes (which optimizations multiple agents converged on)

| Theme | Where it appears | Status |
|---|---|---|
| **KV-cache for LLaDA** (Fast-dLLM v1, dKV-Cache) | 01 (drop-in #1), 02 (KV-cache survey), 03 (B3 entry) | Three independent agents nominate **Fast-dLLM v1** as the strongest single drop-in. |
| **Branch batching** (`cmaj`/`cmajc` `for b in range(5)` → batch dim) | 01 (#3.5), 02 (gap #2), 03 (A1) | Three agents identify this as the **largest missed engineering opportunity** — no kernel work, ~4× wall-clock, exact-equivalence in expectation if seed mapping is preserved. |
| **FlexAttention / mask-aware attention** | 01 (#20), 02 (kernel #1, #11) | sfumato's hybrid mask is FlexAttention-shaped; BD3-LMs/dFactory recipes already published. |
| **Confidence-aware parallel decoding / SlowFast scheduler** | 01 (#2, #6, #11, #13), 02 (gap #7), 04 (mentioned for B-row stacking) | Schedule-side wins compound multiplicatively with cache wins. |
| **Early-exit self-consistency / quorum-based pruning** | 00 (K1 framing), 03 (D1, D2, D3) | Trigger-rate substrate (1750 labeled branches) already exists. |
| **Sparse-dLLM mask-only attention** | 01 (cited in passing), 04 (D1 — concrete port plan) | Needs LLaDA `trust_remote_code` patch; 6–8 eng-days. |
| **Frontier-judge / verifier post-hoc** | 00 (NEW-1, NEW-5) | Non-kernel; +6.16pp Claude-Sonnet-CoT WIN already validated in Phase-2 finals. |
| **No published Qwen→LLaDA KV bridge** | 02 (null-result row) | Sfumato-specific *open* problem — not free lunch, but flagged for Phase-3 paper bet. |

---

## 2. Ranked spike-candidate table (≤$5 each, sorted by info_per_dollar)

`info_per_dollar = expected (speedup × confidence × paper-defense-value) / (GPU-$ × eng-days)`.
All entries are read-only research today; the table is the *menu* for a follow-up "Workstream D — kernel free-lunch spike" pre-reg session.

| Rank | ID | Spike | Source agent | Expected payoff | Eng-days | GPU-$ | Risk | Stacks with |
|---|---|---|---|---|---|---|---|---|
| **1** | **S0** | **Branch batching** for `cmaj`/`cmajc`/`cmerge` (`runner.py:203-288` rewrite, no kernel work) | 01 §3.5, 03 A1 | **~4× wall-clock on cmaj/cmajc**, exact-equivalence under controlled seed remap | 2–3 | ~$3 (4090, N=200 validation) | M (memory ceiling at b=5 on 4090; falls back to b=2) | every other spike below |
| **2** | **S1** | **Early-exit self-consistency (ESC) at quorum=3** on top of S0 | 03 D1 | **+1.3–1.5× on top of S0**; bounded above 1.7× combined | +0.5 | ~$1 marginal | L | S0 mandatory |
| **3** | **S2** | **LoRA merge-on-toggle** (PEFT `merge_adapter`/`unmerge_adapter` at `_enable_commit`/`_disable_commit`) | 04 E1 | **~5% per-step on commit-active sub-blocks**, exact numerical equiv (within bf16); 5× absolute on cmajc; **also enables clean B3 measurement** | 0.5–1 | ~$1 | L | S0, S1, S3 |
| **4** | **S3** | **`logit_shift_norm` populated** (one extra base-only forward at last step of each commit-active sub-block) | 04 B3 | Diagnostic only — quantifies whether v3's COMMIT_N_BLOCKS=3 is uniformly utilised across sub-blocks 1/2/3; gates B1/B2 future spikes | 0.5 | $0 (trace mode) | L | S2 |
| **5** | **S4** | **Fast-dLLM v1 drop-in** under `c2c` only (smallest condition first) | 01 §3.1, 02 KV row #2 | **5–12× wall-clock on c2c**; lowest-risk kernel-side win; multiplicative with S0 on cmaj/cmajc later | 3–5 | ~$5 (1d on 4090: 5-problem mock + N=20 real + N=200 CI confirm) | L–M (paper claims "negligible accuracy loss"; we verify on dev-200) | S0, then `c2c` first then promote |
| **6** | **S5** | **TTC trade curves** sweeping `(k_steps × b_branches × COMMIT_N_BLOCKS)` against API-token-matched o1/o3-mini | 00 NEW-4, K1 | Closes PLAN.md gap-6 (compute-vs-iteration disambiguation); **paper-section artifact** | 3–4 | ~$5 | L | independent |
| 7 | S6 | **dKV-Cache** drop-in (alternative to Fast-dLLM v1, simpler) | 01 §3.2 | 2–4× on c2c; safer baseline before S4 | 2–4 | ~$3 | L | mutually exclusive with S4 |
| 8 | S7 | **SlowFast Sampling** schedule replacement | 01 §3.4 | 3–8× alone, 34× combined with caching | 1–2 | ~$2 | M (commit-schedule change; verify cmaj voting) | S4/S6 |
| 9 | S8 | **FlashInfer top-k sampling kernel** for the categorical step | 01 #21, 04 C2 | 5–15% end-to-end (sampling not the bottleneck) | 1 | ~$1 | L | always |
| 10 | S9 | **Mid-denoising D3 branch pruning** (sfumato-novel, leverages 1750-branch substrate + Workstream-C `step_callback`) | 03 D3 | +1.25× on top of S0+S1 | 2–3 | ~$0.20 substrate augmentation | M (false-prune risk; ABL_B suggests low) | S0, S1 |
| 11 | S10 | **K2 commit-LoRA window sweep** `COMMIT_N_BLOCKS ∈ {0..4}` | 00 K2 | Descriptive; ablates v3's choice of 3; cheap | 1–2 | ~$3 | L | S2 |
| 12 | S11 | **Sparse-dLLM port (mask-only attention)** | 04 D1 | 8–15% wall-clock at k=32–128; applies to every LLaDA-using condition | 6–8 | ~$5 | M (LLaDA `trust_remote_code` monkey-patch + accuracy regression risk) | S2, S4 |
| 13 | S12 | **K3 prefix-KV reuse across cmaj branches** (kernel-side variant of S0+Hydragen) | 00 K3 | +1.4–1.6× on top of S0 (Hydragen analytic ceiling 1.76×) | 5–8 | ~$5 | M–H | S0 |

> **Excluded (over budget or low signal):** Hydragen full port (~$10), SGLang LLaDA backend (~$15), vLLM LLaDA backend (~$15), Cross-arch Qwen→LLaDA KV bridge (Phase-3-only, ~$30+, novel research not engineering). Frontier-judge productized (NEW-1) is a separate **quality** spike at ~$8 already validated upstream — see PHASE2_FINAL_SUMMARY.md.

---

## 3. Top-3 picks with pre-registration sketches

The following are ordered by recommended execution sequence — **S0 must
land first** because S1, S2, S9, S11 either depend on it or compose
multiplicatively. Each pre-reg follows the discipline of `e2/PROTOCOL.md`
and `phase2/spikes/*/PRE_REG.md`.

### Top pick — **S0 + S1: branch batching + ESC quorum exit**
Pre-reg lifts directly from `03_branching_free_lunch.md` §4. Reproduced concisely:

- **Hypothesis.** A1 (`cmaj` branches batched as `(B=5, L+128)` instead
  of 5 sequential calls) + D1 (`break` when `Counter(extracted_answers)
  .most_common(1)[0][1] >= 3`) yields **≥1.3× wall-clock speedup** on
  `cmaj b=5` with **≤1pp accuracy delta** vs the published baseline
  `cmaj=79.5% [73.3, 84.9]` at N=200, seed=0, k=64, v3-LoRA.
- **WIN.** Median per-problem wall-clock at N=20 ≥1.3× vs paired
  baseline AND N=200 accuracy ∈ `[78.0%, 81.0%]`.
- **LOSS.** Speedup <1.15× OR accuracy outside `[77.5%, 81.5%]`.
- **INCONCLUSIVE.** Speedup ∈ `[1.15×, 1.3×)` → graduate to S12 (Hydragen).
- **Files touched.** `e4/diff_llada.py:_generate` to accept `(B, L+gen)`
  inputs; `e4/runner.py:216-223` (`cmaj`) and `:243-251` (`cmajc`) to
  dispatch the batched call + ESC quorum check.
- **Measurement.** Paired Wilcoxon on per-problem wall-clock (logged via
  new `wallclock_ms` column in `runner.py:415-422`); accuracy via
  `scripts/binom_ci.py`.
- **Budget.** ≤$5 GPU (1 spot 4090-day at $0.20–0.50/hr) + 2.5 eng-days
  (S0=2d + S1=0.5d).
- **Artifacts land in.** `phase2/spikes/branching-free-lunch/`.

### Second pick — **S2 + S3: LoRA merge-on-toggle + logit_shift_norm**
Pre-reg lifts from `04_commit_lora_free_lunch.md` §4. Reproduced concisely:

- **Hypothesis (S2).** `merge_adapter` / `unmerge_adapter` around
  commit-LoRA toggle reduces per-step wall-clock by **≥3%** on LLaDA-8B
  bf16 on a single RTX 4090 with **|Δaccuracy| ≤0.5pp** on GSM8K dev-200
  at k_steps ∈ {32, 64}, seed ∈ {0, 1, 2}.
- **Hypothesis (S3).** Mean `logit_shift_norm` at sub-block 1 is **≥2×**
  sub-block 3 — i.e. commit adapter shifts logits more on early commit
  blocks than the answer-formatting block.
- **WIN.** Both hypotheses confirmed at the stated thresholds.
- **LOSS.** Accuracy regresses >0.5pp on any (k, seed) cell.
- **Files touched.** `e4/diff_llada.py:554-572` (`_enable_commit` /
  `_disable_commit`) — add `merge_adapter` / `unmerge_adapter` calls;
  `:506` populate `state.logit_shift_norm` with shadow base-only forward.
- **Measurement.** `runner.py:415-422` adds `wallclock_ms`;
  `logit_shift_norm` flushed to JSONL trace.
- **Budget.** ≤$1 GPU (~25 min on 4090 spot) + 1.5–2 eng-days.
- **Artifacts land in.** `phase2/spikes/lora-merge-toggle/`.

### Third pick — **S4: Fast-dLLM v1 drop-in on `c2c`**
Pre-reg follows `01_diffusion_lm_kernels.md` §3.1:

- **Hypothesis.** Wrapping LLaDA-8B with NVlabs Fast-dLLM v1
  (block-wise approximate KV cache + confidence-aware parallel decoding)
  yields **≥3× wall-clock speedup** on `c2c` N=200 with `cmajc-v3`
  accuracy on the same N=200 inside the **[80.0%, 85.0%]** band of the
  current 82.5% headline.
- **WIN.** Speedup ≥3× AND accuracy ∈ [80.0%, 85.0%].
- **LOSS.** Accuracy <80.0% OR speedup <2×.
- **INCONCLUSIVE.** Speedup ∈ [2×, 3×) → keep as `c2c`-only optimisation
  but do not promote to `cmajc` until S0+S1 land first.
- **Files touched.** `e4/diff_llada.py:_ensure_loaded` to wrap LLaDA with
  Fast-dLLM's `LLaDAModelWithKVCache`; `:442-460` denoising body to call
  `parallel_decode_with_kv_cache(x, threshold=τ)`.
- **Measurement.** Per-problem wallclock JSONL diff; binom-CI on N=200
  accuracy.
- **Budget.** ~$5 GPU (1 day 4090 spot) + 3–5 eng-days.
- **Order.** *After* S0+S1+S2 — `c2c` first, then promote to `cmajc` once
  branch batching is stable so the speedups multiply cleanly.
- **Artifacts land in.** `phase2/spikes/fast-dllm-c2c/`.

---

## 4. Open exposures parked for future phases

These were surveyed but didn't reach the spike-candidate ranking:

- **Cross-arch Qwen→LLaDA KV bridge** (no published kernel; needs
  learned-projection adapter + dKV-Cache combo). Phase-3 paper bet, not
  Phase-2 spike. Surfaced in `02_hybrid_kernels.md` §2 row 4.
- **SGLang LLaDA backend** for sfumato's specific c3/cmaj/cmajc pipeline.
  Day-0 LLaDA-2.0 support landed Dec 2025 but does not wire AR-plan →
  diffuse → AR-finalize as a single program. ~5 eng-days. `02` §3.
- **vLLM LLaDA registration** with custom block-causal mask. Tracking
  issue `vllm-project/vllm#18532` open since 2024. ~7+ eng-days. `02` §3.
- **Hydragen full port** (`arXiv:2402.05099`). Highest analytic ceiling
  (1.76×) but LLaDA's bidi attention isn't on Hydragen's supported list →
  needs a fork. `03` A2.
- **D2F / WeDLM / CDLM / dParallel** as **alternative base models**.
  All require retraining; out of scope under Phase-2 budget but worth
  cataloguing if a Phase-3 retrain ever happens. `01` rows #7–10.
- **NEW-1 frontier-judge productized** (Claude-Sonnet-4.5+CoT closes 86%
  of voting-rule gap, +6.16pp validated). Not a kernel spike — a
  product/cost-routing decision. `00` §3.3, `phase2/PHASE2_FINAL_SUMMARY.md`.
- **K5 AR-extend cross-model handoff profiling** — unblocker for D1
  mode-router (the surviving Phase-2 graduating proposal). 3–5 eng-days,
  necessary before any D1 substrate run on real models. `00` §4.

---

## 5. Verification

End-to-end test that the dispatch-and-synthesis pipeline worked:

1. **Five files exist + non-empty:** `ls phase2/proposals/kernel-survey/`
   shows `00..04*.md` + this `SUMMARY.md`. ✅ (verified at write time;
   line counts: 00=409, 01=402, 02=152, 03=346, 04=290.)
2. **≥10 citations per file:**
   - `00_OUR_IDEA.md`: 29 numbered citations ✅
   - `01_diffusion_lm_kernels.md`: 30 papers + 10 repos/blogs = 40 ✅
   - `02_hybrid_kernels.md`: 30 numbered citations ✅
   - `03_branching_free_lunch.md`: 19 footnote citations ✅
   - `04_commit_lora_free_lunch.md`: 17 references ✅
3. **No code outside this directory touched:** `git status` should show
   only new files in `phase2/proposals/kernel-survey/`.
4. **Spike candidates with eng-day estimates:** §2 ranks 12+ candidates;
   §3 pre-registers 3 with WIN/LOSS thresholds, budgets, file ranges.
5. **COST_LEDGER row to add when Phase-2 wraps this workstream:**
   `2026-05-04 | Workstream-D kernel-survey research | $0.00 | research-only, 5 parallel agents, 5 markdown deliverables`.

---

## 6. Recommended next action

**Pick S0 + S1 as the Phase-2-or-3 lead spike** — branch batching is
the single highest-leverage change in the entire project (3 of 5 agents
converged independently), it is pure Python (no kernel work), the
expected speedup is **~4× on cmaj/cmajc/cmerge**, it is exact-equivalence
under controlled seed mapping, and the cumulative wall-clock saving
**multiplies with every subsequent spike** (S2, S4, S9, S11). Pre-reg
sketch ready in §3. Budget: ~$5 GPU + 2.5 eng-days.

Run S2+S3 next on the same pod (≤$1 marginal, 1.5–2 eng-days, exact
numerical equivalence).

Defer S4 (Fast-dLLM v1) to a third spike *after* S0 lands so the 5–12×
on c2c multiplies with the 4× from branch batching on cmajc instead of
being measured separately.
