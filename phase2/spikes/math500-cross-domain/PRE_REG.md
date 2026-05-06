# Pre-registration — MATH-500 cross-domain commit-LoRA K2 inverted-U

**Date:** 2026-05-06 | **Spike:** Phase-3 T2.A of
`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`.

## Hypothesis

The K2 inverted-U finding (Phase-2 §3 / spike `k2-commit-blocks-ablation`):

> On GSM8K cmajc-v3 N=200 seed=2: k=0 → 0.805, k=2 → 0.815, k=3 → 0.822
> (peak), k=4 → 0.790. Sub-block-1 boundary load-bearing.

Tests whether this inverted-U replicates on **MATH-500** with the
**same** GSM8K-trained commit-LoRA (`eren23/sfumato-llada-commit-v3`),
and whether the optimal block boundary shifts.

## Substrate

- **Dataset:** MATH-500 numeric-answer subset (316 of 500 problems
  with purely numeric gold answers — `e4/data/math500_numeric_indices.json`).
  Restricted from full 500 because `e4/grade.is_correct` only handles
  numeric equality; LaTeX-formatted answers (`\frac{2}{3}`, `90^\circ`)
  would be auto-LOSS regardless of model output.
- **Sample size:** N=50 problems (first 50 indices of the numeric
  subset: 3, 5, 6, 9, 10, 12, 13, 16, 18, 19, ...).
- **Model:** LLaDA-8B-Instruct + Qwen-2.5-0.5B-Instruct + prefix-robust-v3
  + commit-v3 (frozen GSM8K-trained adapters; this probe tests
  generalization without any MATH-500 fine-tuning).
- **K-sweep:** 4 conditions, each N=50 BATCHED=1 BRANCHES=5 K_STEPS=64
  TEMP=0.7 SEED=0:
  - **c2c**       — pure diffusion + commit-LoRA on last sub-block only
                    (COMMIT_N_BLOCKS=1 ≡ k=0 baseline; matches Phase-2
                    K2 row "k=0 = 0.805")
  - **cmajc k=2** — COMMIT_N_BLOCKS=2 (commits on sub-blocks 3-4)
  - **cmajc k=3** — COMMIT_N_BLOCKS=3 (commits on sub-blocks 2-4; the
                    Phase-2 GSM8K peak)
  - **cmajc k=4** — COMMIT_N_BLOCKS=4 (commits on all sub-blocks 1-4)
- **Pod:** 48GB A40 spot (24GB OOMs on MATH-500 prompt lengths +
  BATCHED=1 BRANCHES=5 — Phase-2 deferral confirmed).

## Prediction

| Condition | Predicted MATH-500 acc (numeric subset, N=50) | Phase-2 GSM8K analogue |
|---|---:|---:|
| c2c (k=0) | 0.30 ± 0.07 | 0.805 |
| cmajc k=2 | 0.32 ± 0.07 | 0.815 |
| cmajc k=3 | 0.34 ± 0.07 | 0.822 (peak) |
| cmajc k=4 | 0.30 ± 0.07 | 0.790 |

Cross-domain accuracy will be lower in absolute terms (LLaDA + adapters
are GSM8K-tuned), but the *shape* of the K-sweep is the diagnostic. The
inverted-U structure is what makes commit-LoRA mechanistically novel
vs schedule-agnostic prior work.

## Decision rules

| Outcome | Verdict |
|---|---|
| cmajc-best ≥ 0.30 AND k-sweep produces inverted-U with peak in [k=2, k=3] | **WIN** — primitive generalizes; §3.5 grows a "MATH-500 generalization replication" subsection |
| cmajc-best ≥ 0.30 AND positive lift over c2c but no inverted shape (e.g., monotone or k=4 peak) | **PARTIAL** — commit-LoRA helps cross-domain but the schedule-boundary claim is GSM8K-specific; §3.5 grows a "boundary is GSM8K-specific" subsection |
| Any cmajc < c2c (commit-LoRA hurts cross-domain) | **LOSS** — commit-LoRA is GSM8K-overfit; §3 caveat must say "validated on GSM8K only, MATH-500 disconfirms generalization" |

## Method

1. Provision 48GB A40 spot via `provision_project sfumato_e4` after
   commenting `RTX 4090` out of the project spec's `pod.gpu_type`
   allowlist (so the provisioner can't fall back to 24GB).
2. Bootstrap project, sync code.
3. Dispatch 4 sequential `run_project` calls with overrides:
   `DEV_INDICES_FILE=math500_numeric_indices.json`, `CONDITION` and
   `COMMIT_N_BLOCKS` per row.
4. Each run: ~10 min wall on 48GB BATCHED=1 (Phase-2 GSM8K cmajc N=200
   ran ~30 min on 48GB BATCHED=1; N=50 should be ~7-8 min plus model load).
5. After all 4 finish, rsync `e4/results/raw_*_k64_seed0.jsonl` ×4
   → local; compute K-sweep accuracy table; verdict against decision
   rules.

## Anti-goals

- No multi-seed at this scale. Single seed=0 is sufficient to verify
  shape; multi-seed lift only if cmajc k=3 lands ≥ 0.30 AND the
  inverted-U is visible.
- No N>50. The K2 ablation cost on Phase-2 was ~$2-3 across multiple
  retries; this probe is a generalization check at half the
  resolution. Scale to N=100 only if WIN at N=50.
- No grader change. The numeric-answer subset is the substrate;
  LaTeX-equivalence checking is a separate spike with its own
  grader-correctness cost.
- No MATH-500 fine-tuning. The point is generalization of the
  GSM8K-trained adapter. If cmajc loses to c2c, the commit-LoRA is
  overfit.
- No new env knobs. Reuses existing CONDITION / COMMIT_N_BLOCKS /
  DEV_INDICES_FILE / BATCHED machinery. Only the indices file is new.

## Cost

| Item | $ |
|---|---:|
| Pod boot + bootstrap (48GB A40 spot @ $0.24/hr) | ~$0.05 |
| 4× cmajc/c2c N=50 BATCHED=1 BRANCHES=5 (~10 min each, 40 min total) | ~$0.16 |
| Buffer (one re-run if eviction) | ~$0.10 |
| **Total spike cost cap** | **~$0.31** |

Of the <$1 Tier-1 GPU budget remaining ~$0.67 (after T1.B + T1.C ate $0.33),
this stays well under cap.

## Files

- `PRE_REG.md` — this file
- `e4/data/math500_numeric_indices.json` — substrate scaffold
  (committed prior to harvest)
- `RESULT.md` — to be filled after 4× harvest + analysis

## Implementation refs

- `e4/runner.py:load_problems` — already parameterizes `question_col`,
  `answer_col`, `answer_extract_split` per spec (commit `1938182`).
- `.crucible/projects/sfumato_e4.yaml` — `pod.gpu_type` allowlist will
  be temporarily edited to remove 4090 before provision.
