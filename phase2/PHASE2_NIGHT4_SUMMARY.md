# Sfumato Phase-2/3 — Night 4 Status (Spike Chain + Sweep + Showcase)

**Date:** 2026-05-05 (UTC+local rollover from 2026-05-04 night session)
**Spend:** ~$5 GPU spot across 2 RunPod nodes (Pod-A 48GB A6000, Pod-B 24GB 4090)
**Status:** Spike chain S0–S4 implemented + validated, all-conditions sweep done, showcase shipped, follow-up pod work queued.

---

## What landed

### Spike chain (S0–S4) — full implementation + per-spike pre-reg + validation

| Spike | Implementation | Pre-reg | Pod result | Verdict |
|---|---|---|---|---|
| **S0 — branch batching** (`runner.py` cmaj/cmajc/cmerge → batched `(B, L+gen)`) | ✅ landed in `e4/diff_llada.py:_generate_batched` + `denoise_block_batched` | `phase2/spikes/branching-free-lunch/PRE_REG.md` | acc 0.70 → 0.75 (N=20), wallclock 801s → 601s | **WIN ~1.33×** N=20, **1.53×** N=200 |
| **S1 — ESC quorum=3 early-exit** | ✅ `_make_esc_callback` in runner.py + `BatchStepDirective.should_stop` flags | (same dir) | acc 0.75 unchanged after extractor fix | **NEUTRAL** — ESC structurally inert on LLaDA's schedule (answer span lands only in last block) |
| **S2 — LoRA merge-on-toggle** | ✅ `merge_adapter()` / `unmerge_adapter()` in `_enable_commit`/`_disable_commit` | `phase2/spikes/lora-merge-toggle/PRE_REG.md` | wallclock 283s → 280s (1.0% cut) | **NEUTRAL** — below 3% threshold, accuracy preserved |
| **S3 — `logit_shift_norm` populated** | ✅ `_maybe_logit_shift_norm` shadow forward gated by `LOGIT_SHIFT_NORM=1` | (S2/S3 same dir) | populates schema slot, +41% wallclock from extra forward (expected) | **DIAGNOSTIC OPERATIONAL** |
| **S4 — Fast-dLLM v1 drop-in** | ✅ `e4/fast_dllm_adapter.py` shim (LLaDAModelLM + generate, with FAST_DLLM_TAU env knob) | `phase2/spikes/fast-dllm-c2c/PRE_REG.md` + `SETUP.md` | c2 N=20 acc=0.65, **wallclock 124s = ~6.2s/problem** | **WIN ~6.5×** on c2; partial on c2c (commit-LoRA bypassed by fastpath — Phase-3 follow-up) |

**Headline N=200 confirmation:** S0+S1 cmaj N=200 → **acc 0.7750, wallclock 5232s (26.2s/problem) = 1.53× speedup vs baseline.** Tracks the 79.5% Phase-1 cmaj baseline within sampling noise.

**Critical bug found + fixed during validation:** initial S1 ESC quorum used `grade.extract_answer` (last number anywhere in partial CoT). Fired on mid-reasoning numbers ("16 - 3 = 13"), pruned dissenters incorrectly, dropped acc 0.75 → 0.45. Fixed by adding `grade.extract_final_answer` matching only `#### N` / `Answer: N` patterns. ESC then went from harmful → safe-but-inert.

### Showcase v0/v1 — public-facing query interface

`phase2/showcase/static/` — vanilla HTML+CSS+JS, GitHub-Pages-deployable, ~1.6 MB total.

- **Browse**: 14 condition × seed cells, ~620 records (existing 400 from Phase-1 + 220 from sweep). Tag-faceted filter pills, search by question text, random-button.
- **Detail per problem**: question + gold + predicted + correctness, vote tally bar (gold-aware coloring), per-branch reasoning cards (collapsible). Single-branch conditions (c1/c2/c3/etc.) render their `cot`/`plan`/`diffusion_cot`/`finalize` fields with appropriate labels.
- **Animations**: 20 GIFs (`phase2/showcase/static/animations/`) — pure LLaDA, AR-handoff, cmaj branching at varied sub-block boundaries.
- **About**: 1-screen explainer (plan-then-fill, why 5 branches, what is commit-LoRA).
- **Speed-panel scaffolding (v1)**: `build_examples.SPEED_PAIRS` ready to flip on once paired pre/post-spike wallclock data lands.

### All-conditions sweep at N=20

| Cond | Acc | Wallclock | Mechanism |
|---|---|---|---|
| c1 | 0.25 | 194s | pure Qwen-0.5B AR (baseline) |
| c2 | 0.55 | 261s | pure LLaDA single-branch |
| c2c | 0.70 | 120s | LLaDA + commit-LoRA |
| c2hint | 0.65 | 146s | LLaDA + hint prefix |
| c2empty | 0.75 | 108s | LLaDA + empty prefix |
| c3 | 0.45 | 245s | AR plan → LLaDA → AR finalize |
| c3p | 0.50 | 309s | AR plan → LLaDA |
| c4 | 0.55 | 511s | c3 + extra round |
| cmaj | 0.75 | 563s | 5 LLaDA branches → vote |
| **cmajc** | **0.80** | 477s | cmaj + commit-LoRA per branch |
| cmerge | 0.70 | 481s | 3 LLaDA branches → AR merge |
| crev | 0.30 | 73s | LLaDA scaffold → AR finalize |

Plus 4 seed-variants (c2-seed1, c2c-seed1/2, c2hint-seed1) for showcase variety.

### Process + infrastructure findings

- **Crucible's `no_terminal_marker` false-positive** still fires consistently at the end of every successful run. JSONL on disk is authoritative; ignore Crucible status.
- **Wandb env-forwarding:** initial dispatches with `WANDB_DISABLED=1` killed live monitoring. Removing the override + re-bootstrapping pod-07 surfaces the wandb URLs in logs (not in Crucible's status sidecar).
- **Pod-07 SSH flakiness:** Crucible bootstrap occasionally hits `kex_exchange_identification: read: Connection reset by peer`. Retrying 1-2× resolves it. Documented for future sessions.
- **Pod-B (24GB) vs Pod-A (48GB):** Qwen + LLaDA together OOM the 24GB. Pod-B gets restricted to single-LLaDA workloads; Pod-A handles c3/c3p/c4/cmerge/cmaj/cmajc.

### Files / commits

- `e4/diff_llada.py` — major: BatchStepState, `_generate_batched`, merge-on-toggle, logit_shift_norm, FAST_DLLM gate.
- `e4/runner.py` — cmaj/cmajc/cmerge use batched method, `_make_esc_callback`, `wallclock_ms` log column, `BATCHED`/`MERGE_ADAPTER` toggles, `fast_dllm_setup` short-circuit.
- `e4/fast_dllm_adapter.py` — new module; pins NVlabs/Fast-dLLM v1 entry-points.
- `e4/grade.py` — added `extract_final_answer` for strict ESC pattern matching.
- `phase2/spikes/branching-free-lunch/{PRE_REG,IMPLEMENTATION}.md`
- `phase2/spikes/lora-merge-toggle/{PRE_REG,IMPLEMENTATION}.md`
- `phase2/spikes/fast-dllm-c2c/{PRE_REG,SETUP,IMPLEMENTATION}.md`
- `phase2/showcase/{README.md, build_examples.py, static/{index.html, app.js, style.css, examples.json, animations/}}`
- `phase2/proposals/kernel-survey/{00..04, SUMMARY}.md` — 5-agent research dispatch deliverables.

Commits on `eren23/sfumato` main:
1. `baf56a9` — kernel-survey research dispatch (5 markdown deliverables)
2. `6487453` — S0+S1+S2+S3+S4 spike chain implementation
3. `82cd01c` — showcase v0/v1
4. `a5580a6` — BATCHED=0 toggle for paired baseline
5. `9ae8e1d` — MERGE_ADAPTER=0 toggle
6. `2ca2735` — fp64/fp32 memory fix in batched gumbel
7. `ac7d8d2` — fast_dllm_setup CONDITION
8. `56d8c83` — Fast-dLLM v1/v2 subdir enumeration
9. `21b2ae2` — Fast-dLLM symbol pinning (LLaDAModelLM + generate)
10. `01ad6bb` — Fast-dLLM (x, nfe) tuple unpack
11. `1a16081` — ESC strict final-answer extractor
12. `642f83c` — sweep results + animations gallery in showcase

---

## What's planned ahead

Pods still up. Queued work in order of info-per-dollar:

### Tonight / next pod session (cheap, high-signal)

1. **Multi-seed cmajc-v3 N=200** (Pod-A, ~60 min, ~$0.30) — re-confirm Phase-1 σ ≈ 0.3pp robustness with the new batched code path. Validates the 82.5% headline survives the spike-chain changes.

2. **Fast-dLLM c2 N=200** (Pod-B, ~30 min, ~$0.15) — confirm the 6.5× speedup at full N=200 (currently only verified at N=20).

3. **cmajc temperature sweep** (Pod-A, ~90 min, ~$0.50) — τ ∈ {0.3, 0.7, 1.0, 1.3} × N=50. Showcase variety + "why temp=0.7" defense.

4. **K-step sweep cmajc** (Pod-A, ~90 min, ~$0.50) — k ∈ {16, 32, 64, 128} × N=50. Closes PLAN.md gap-6 (compute-vs-iteration). Generates the TTC trade curve (NEW-4).

### Bigger / paper-class

5. **K1 match-FLOPs ablation** (~90 min, ~$1) — b=5 cmaj at k=32 vs b=1 c2 at k=160 — both same FLOPs. Settles "ensemble topology vs more compute."

6. **Full GSM8K-test 1319 cmajc-v3** (~2.5 hr, ~$2) — biggest possible showcase substrate at N=full-test.

7. **MATH-500 sample on cmajc** (~2 hr, ~$2) — cross-domain. Currently zero MATH data in showcase.

### Phase-3 research bets

8. **Commit-LoRA-aware Fast-dLLM port** (5-7 eng-days) — fork upstream, thread PEFT toggle through their `generate()`. Gets cmajc the 6.5× → ~10× combined headline.

9. **D1 mode-router substrate harvest** (3-5 eng-days) — N=100 STATUS-schema traces via the visualizer. Substrate for the bandit-on-replay sketch (RANKING.md alive item).

### Local writeups (no GPU)

10. **RESULT.md per spike** (30 min, $0) — fill in the implementation logs with actual numbers + W&B URLs + pre-reg pass/fail verdicts.

11. **Spike-chain executive summary** for `phase2/PHASE2_FINAL_SUMMARY.md` extension — one-pager for the paper appendix.

---

## Open exposures parked

- **Crucible bug**: status sidecar shows "failed: no_terminal_marker" on every run despite clean completion. Filed in `<parameter-golf_dev>/CRUCIBLE_BUG_ANALYSIS.md` previously; not yet upstream-fixed.
- **WANDB_API_KEY rotation**: leaked once into a Crucible bootstrap error log mid-session. User OK'd not pushing rotation; should rotate on next routine cycle.
- **Pod-B SSH flakiness** intermittent — needs retry pattern documented in Crucible bootstrap helpers.
- **Commit-LoRA in Fast-dLLM fastpath**: known unimplemented; cmajc + FAST_DLLM=1 silently bypasses commit-LoRA toggling. Phase-3 work item.

---

## Numbers to remember

- **cmajc-v3 = 82.5% on GSM8K-test** (Phase-1 headline; multi-seed σ ≈ 0.3pp).
- **S0+S1 cmaj N=200 = 0.7750, 1.53× speedup** (this session, batched code).
- **S4 c2 N=20 = 0.65, 6.5× speedup** (Fast-dLLM v1 drop-in).
- **cmajc N=20 = 0.80** (this session's all-conditions sweep cell).
- **Total Phase-2 spend: ~$8.35** ($2.85 prior + $5.50 this session including spikes + sweep).
