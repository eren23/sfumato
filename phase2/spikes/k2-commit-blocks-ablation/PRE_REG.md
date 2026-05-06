# Pre-registration — K2 Commit-LoRA Sub-Block Ablation

**Date:** 2026-05-06 | **Spike:** Track B.1 of post-re-orientation plan
(`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`).

## Hypothesis

Commit-LoRA's value comes from its **schedule-toggle** at the
sub-block-1 boundary, not from "more adapter parameters everywhere."
Sweeping `COMMIT_N_BLOCKS ∈ {0, 1, 2, 3, 4}` on cmajc N=200 with a
prefix-robust-v3 LoRA-conditioned base should produce a **monotone
increase from k=0 (off) to k=3 (sfumato-v3 default, blocks 2–4)**
followed by a **drop at k=4 (always on)** — an inverted-U with peak
at k=3. The drop at k=4 would be direct evidence that activating
commit-LoRA on sub-block 1 (where the prefix-robust LoRA needs to
drive the early committal phase) actively hurts.

## Substrate

- Same as Track A's voting-gap §2.1: GSM8K-test idx 0..199, frozen
  via `e4/data/gsm8k_dev_200.json`.
- LLaDA-8B-Instruct base + `eren23/sfumato-llada-prefix-robust-v3`
  prefix LoRA + `eren23/sfumato-llada-commit-v3` commit LoRA.
- Condition: `cmajc` (5-branch majority vote with commit-LoRA toggle).
- N_PROBLEMS=200, BRANCHES=5, K_STEPS=64, TEMP=0.7, SEED=0.
- Branch path: `BATCHED=0` (sequential — required to fit 24GB pods).

## Pre-registered cells

| Cell | COMMIT_N_BLOCKS | Description |
|---|---|---|
| C1 | 0 | commit-LoRA never activated (sanity floor) |
| C2 | 1 | activated only on the LAST sub-block (Phase-1 default) |
| C3 | 2 | activated on sub-blocks 3–4 |
| C4 | 3 | activated on sub-blocks 2–4 (sfumato-v3 default) |
| C5 | 4 | activated on ALL sub-blocks (always on) |

Cells C1, C4, C5 are the diagnostic minimum for the inverted-U
hypothesis. C2 and C3 are nice-to-have but skipped within Phase-2
budget — see RESULT.md "deviations" section.

## Decision rules

| Outcome | Verdict |
|---|---|
| C1 < C4 AND C5 < C4, with both deltas > σ ≈ 0.85pp (multi-seed noise) | **WIN** — schedule-toggle hypothesis confirmed; sub-block-1 boundary load-bearing |
| Curve monotone-increasing through C5 (k=4 ≥ k=3) | **LOSS** — value is "more LoRA" not scheduling |
| Curve flat (all cells within σ band) | **NEUTRAL** — commit-LoRA has no measurable effect |

## Expected secondary diagnostic

The k=3 baseline at multi-seed mean 0.822 (σ ≈ 0.85pp) is locked from
prior triple-seed runs (seed 0/1/2 = 0.825/0.81/0.83). Pre-reg expects
both k=0 and k=4 to land at least 1pp below k=3 (≥1× σ), with k=4
hurting more than k=0 hurts (since k=4 actively miscommit-LoRAs the
prefix-driven sub-block 1, vs k=0 which just doesn't help).

## Anti-goals

- No multi-seed for k=0 / k=4 in Phase 2 — too expensive at sequential
  branch path. Single-seed inverted-U with σ-band reference from the
  triple-seed k=3 mean is sufficient.
- No ablation of `MERGE_ADAPTER` for k=0 — set to 0 to skip the
  no-op merge path (commit-LoRA never activated → nothing to merge).

## Cost

~$1.30 estimated (2 cells × ~75min × $0.53/hr A6000) — landed at
~$1.30 actual.

## Files

- `PRE_REG.md` — this file
- `RESULT.md` — outcome
- `e4/results/raw_cmajc_*_K2-cmajc-k{0,4}*.jsonl` — raw outputs
  (synced via wandb runs `mo4clpp4` for k=0, `ho24ezlz` for k=4)
