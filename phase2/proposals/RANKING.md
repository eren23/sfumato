# Workstream B Ranking — info-per-dollar selection

**Author:** Workstream B subagent | **Date:** 2026-05-02

## Scoring

| Proposal | Spike cost | Spike yields decisive signal? | Mech depth | Phase 1 lift | Score | Rank |
|---|---:|---:|---:|---:|---:|---:|
| **D3 — Diversity-as-objective** | **$2.00** | YES (temperature falsifier kills the whole proposal cheaply if branch-diversity isn't a free lever) | high — directly tests the Phase 1 falsifier finding | direct continuation of `e2/RESULTS_TRACK1.md` "Diversity-expansion finding" | **9.5** | **#1** |
| D1 — Mode router | $1.50 | partial — needs WS-C trace data to land first; spike-able only as feature-importance ablation today | high once data exists; routing policy is Phase 2's natural Phase 1 extension | medium — uses commit-LoRA + cmaj infra, doesn't directly test Phase 1 falsifier | 7.5 | #2 |
| D2 — Latent-coupled joint LoRA | $5.50+ (over cap) | weak — joint training needs ≥4-cell ablation to falsify; no $≤$5 single-cell test | very high if it works (paper-headline number) | low — orthogonal to Phase 1 axes; depends on Berrayana code | 6.0 | #3 |

**Selection: D3 graduates to spike (cost $2.00).**

## Rationale

D3 wins on three dimensions:

1. **Cheapest decisive falsifier.** A 4-temperature sweep on N=20 problems
   tells us whether branch-diversity is even a causal lever on cmaj
   accuracy. If it's not, the entire proposal dies for $2 and we save the
   ~$5.50 graduating-experiment cost. If it is, we have direct empirical
   support for the diversity-reward training run.

2. **Direct continuation of Phase 1's most interesting falsifier.** Phase 1
   turned up the diversity-expansion finding (52.4% → 47.5% 5/5 agreement
   rate) but didn't have time to test whether diversity *causes* cmaj
   accuracy or merely co-varies with it. D3's spike is exactly that test.

3. **Reuses existing infrastructure.** No new code paths needed:
   `e4/runner.py` with `CONDITION=cmaj`, the v3 LoRA, `scripts/branch_agreement.py`
   for the diversity score, `scripts/binom_ci.py` for CIs. Spike is one
   `crucible run_project` invocation with a TEMP env override.

D1 ranks #2 because the proposal stands alone but the spike is gated on
Workstream C's trace JSONL data, which won't exist until C finishes the
Gradio app. Running D1's bandit-on-replay before C ships traces would only
test on synthetic data, which doesn't validate the real signal.

D2 ranks #3 not because it's a bad idea — Berrayana's +27pp is the
paper-headline-grade lift, and joint LoRA is genuinely novel — but because
a spike under $5 cannot give a decisive signal. Even one ablation cell
($2.40) leaves us with no comparison point. D2 belongs in a Phase 3 run
where ~$15 of compute is allocated for a 4-cell ablation.

## Graduating direction

**Pre-spike recommendation:** D3 — Diversity-as-objective fine-tuning for DDLMs.
Spike: `temperature-diversity-falsifier` (see `phase2/spikes/temperature-diversity-falsifier/PRE_REG.md`).

## Post-spike addendum (2026-05-02 ~15:10 UTC)

The temperature-diversity-falsifier spike returned **INCONCLUSIVE-tilting-LOSS**
on D3's original hypothesis (cmaj a_b is monotone-decreasing in τ over
{0.3, 0.7, 1.0}; CIs overlap at N=50). However it surfaced a *stronger,
unexpected* finding: at the default τ=0.7, the oracle ceiling (90%) exceeds
cmaj (78%) by **12pp** — the right answer is in some branch most of the time,
but majority voting throws it away. This re-frames the Phase 2 question as
**verifier / re-ranker** training, not diversity training.

**Updated graduating recommendation:** open a new "D3.5 — Verifier-based
branch aggregation" proposal in a follow-up Phase 2 cycle. D2 (latent-coupled
joint LoRA) and D1 (mode router) remain unchanged in their original ranking;
D1's value goes UP if a verifier head becomes a routable mode. See
`phase2/spikes/temperature-diversity-falsifier/RESULT.md` § "What this kills,
what it opens" for the proposal seed.

## Post-D3.5-write addendum (2026-05-02 ~19:00 UTC)

D3.5 proposal landed at `phase2/proposals/verifier-based-aggregation.md`.
Substrate harvest (cmaj N=100 b=5 at τ=0.7 + τ=1.0) is in flight via
`phase2/spikes/temperature-diversity-falsifier/queue_followup.py` Phase 2.

Updated final ranking with D3.5 added:

| Rank | Proposal | Spike cost | Status |
|---|---|---:|---|
| **#1 (graduating)** | **D3.5 — Verifier-based aggregation** | **~$0.05** | Substrate harvest queued; spike eligible after substrate completes |
| #2 | D1 — Mode router | $1.50 | Unchanged; trace data still gated on Workstream C real-mode pass (queued Phase 4) |
| #3 | D2 — Latent-coupled joint LoRA | $5.50+ | Unchanged; Phase 3 candidate |
| ~~#4~~ | ~~D3 — Diversity-as-objective~~ | — | **Killed by spike** (a_b monotone-decreasing in τ) |

D3.5 wins on every axis: empirical motivation (12pp measured gap), zero
training-data acquisition cost (existing jsonls + queued substrate), and
~$0.05 spike — 40× cheaper than original D3 with stronger prior.

## Night-1 update — 2026-05-03 ~01:30 UTC

| Item | Status |
|---|---|
| **D3.5 substrate** (cmaj N=200 b=5 τ=0.7 v3-LoRA) | ✅ **DONE**. a_b=79.5%, oracle=88%, gap=8.5pp. 1000 labeled branches at `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`. Replicates offline N=50 baseline. |
| **D3.5 option-1 spike** (TF-IDF + LR text-only verifier) | ❌ **LOSS @ N=1750**. Verifier 66.5–69.5% vs cmaj 79.0–80.5% (Δpp −9.5 to −14.0). Architecturally insufficient. |
| **D3.5 option-2 spike** (Qwen-encoder verifier) | 🟡 **PENDING** next session. Pre-reg threshold ≥83%. ~$0.40 GPU. |
| **D3** (diversity-as-objective) | ❌ Killed in original spike. |
| **D1** (mode router) | 🟡 Unchanged; gated on Workstream C real-mode trace data which we now have a sample of. |
| **D2** (latent-coupled joint LoRA) | 🟡 Unchanged; Phase-3 candidate. |

**Updated final ranking:**

| Rank | Proposal | Spike status | Next action |
|---|---|---|---|
| **#1** | **D3.5 — Verifier-based aggregation** | option-1 LOSS, option-2 pending | Train Qwen-encoder verifier (~$0.40) — if WIN, full Phase-2 paper section; if LOSS, D3.5 dies and graduating slot reverts to D1 |
| #2 | D1 — Mode router | gated on traces; Phase-2-ready when 5-10 more real-mode traces exist | Sketch Workstream-C-trace-consumer in pseudocode for next session |
| #3 | D2 — Latent-coupled joint LoRA | Phase-3 only | Park |

Honest note: D3.5 may die in option-2 too. The 12pp voting-rule gap is real but **may simply not be addressable by any per-branch verifier on this size of math problems** — the right answer may require process-level features (per-step entropy, commit-LoRA logit shifts, AR/DDLM choice — option-3) rather than per-answer features. If option-2 fails, the honest paper note becomes: "Voting-rule gap is real but our text-only verifier doesn't close it; process-reward variant remains future work."

## Night-1 final — 2026-05-03 ~12:25 UTC: D3.5 confirmed DEAD, RANKING reverts to D1

Option-2 (Qwen-encoder verifier) result: **LOSS** at 72.0% mean vs cmaj 80.5% (5-fold CV on 1750 branches). Both supervised verifier paths in Phase-2 budget have now been tested and killed.

**Final Phase-2 ranking:**

| Rank | Proposal | Status | Next |
|---|---|---|---|
| **#1 (graduating)** | **D1 — Adaptive mode router** | proposal-only; substrate (Workstream-C traces) needed | Drive `phase2/inference_viz/server.py` through Gradio with manual interventions on 5-10 GSM8K problems → 5-10 STATUS-schema JSONL traces → bandit-on-replay sketch |
| #2 | D2 — Latent-coupled joint LoRA | Phase-3 candidate | Park (>$5 spike cost) |
| ~~#3~~ | ~~D3.5 — Verifier-based aggregation~~ | **DEAD (both options LOSS)** | Defer to Phase-3 if process-reward (option 3) becomes viable with C-trace expansion |
| ~~#4~~ | ~~D3 — Diversity-as-objective~~ | killed earlier in night-1 | — |

**Honest paper-section take**: the voting-rule gap (oracle 88-90% vs cmaj 79-82%, 8-12pp) is the most actionable Phase-2 finding, BUT we showed it's not addressable by per-branch supervised classifiers at this dataset size with text-only or 0.5B-encoder features. Future work: (a) larger encoder ($5+ spike), (b) process-reward verifier consuming per-step trajectory features (Workstream-C trace schema). For Phase-2 paper, document the gap as the headline observation + the 2 negative results as falsifiers + flag option-3 / larger-encoder as Phase-3 work.
