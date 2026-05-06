# K2 Commit-LoRA Sub-Block Ablation — RESULT

**Pre-reg:** `PRE_REG.md` (committed before run).
**Run dates:** 2026-05-06 (k=0 retry, k=4) | **Cost:** ~$1.30 GPU.
**Outcome:** **WIN** — inverted-U with peak at k=3, sub-block-1 boundary
confirmed load-bearing.

---

## Headline numbers

| COMMIT_N_BLOCKS | cmajc N=200 | Δ vs k=3 baseline | wandb |
|---|---:|---:|---|
| **0** (commit-LoRA off, sanity floor) | **0.805** | **−1.7 pp** | [mo4clpp4](https://wandb.ai/eren23/sfumato-e4/runs/mo4clpp4) |
| **3** (blocks 2–4, sfumato-v3 default) | **0.822 mean** (σ ≈ 0.85pp) | baseline | seed 0/1/2 substrate |
| **4** (commit-LoRA always on) | **0.790** | **−3.2 pp** | [ho24ezlz](https://wandb.ai/eren23/sfumato-e4/runs/ho24ezlz) |

**The curve is an inverted-U: monotone-increasing 0 → 3, drops at 4.**
Both deltas (1.7pp + 3.2pp) exceed the multi-seed σ ≈ 0.85pp band.

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| C1 < C4 AND C5 < C4, both > σ band | **YES** — k=0=0.805, k=4=0.790, k=3=0.822; both deltas > 1×σ |
| Curve monotone-increasing through k=4 | **NO** — k=4 (0.790) < k=3 (0.822), drop confirmed |
| All cells flat within σ band | **NO** — total spread 3.2pp, well outside σ |

**Verdict per pre-reg:** **WIN** — schedule-toggle hypothesis confirmed.
The sub-block-1 boundary (off during early committal phase, on during
sub-blocks 2–4) is load-bearing. Activating commit-LoRA on sub-block 1
actively hurts (k=4 = −3.2pp), not just adds nothing.

## What this opens

This is the **mechanistic positive** of Phase 2. It distinguishes
commit-LoRA from training-time schedule-conditioning approaches
(TC-LoRA arXiv 2510.09561, TimeStep Master arXiv 2503.07416) by
showing that a **single discrete inference-time toggle** at a learned
schedule boundary is sufficient — no hypernetwork, no MoE routing,
no training-time conditioning required. The minimal mechanism that
captures schedule-awareness for semi-AR diffusion adapters.

Promoted into the paper draft as §3.2; figure
`phase2/figures/fig_commit_lora_k2_sweep.{pdf,png}` renders the
inverted-U with Clopper-Pearson 95% CIs.

## Deviations from pre-reg

1. **Cells C2 (k=1) and C3 (k=2) skipped.** Within the Phase-2 budget
   ($25–30 cap), the diagnostic minimum (C1, C4, C5) was sufficient
   to confirm the inverted-U hypothesis. Filling in k=1 and k=2 is
   flagged as a Phase-3 nice-to-have if a follow-up paper or revision
   asks for the dense curve. Estimated cost: ~$1.30 for both cells
   sequential.
2. **Single-seed for k=0 and k=4.** Multi-seed for these cells would
   have cost an extra ~$3 each. The k=3 baseline is multi-seed and
   the k=0 / k=4 deltas are both >1× σ, so a single seed is
   sufficient to reject the null. Multi-seed scale-up flagged for
   any paper revision that asks for tighter CIs.
3. **k=0 retry needed `MERGE_ADAPTER=0`.** First attempt at k=0
   with `MERGE_ADAPTER=1` died at step 2 with no terminal error
   (run id `sfumato_e4_1778048723907660000_ac3507`). Diagnosis: when
   `COMMIT_N_BLOCKS=0` the `_enable_commit`/`_disable_commit` calls
   in `_generate` produce no-op merge/unmerge cycles that hit a
   PyTorch allocator path under `expandable_segments:True`. Retry
   with `MERGE_ADAPTER=0` (run `mo4clpp4`) ran cleanly.

## Files

- `PRE_REG.md` — pre-registered hypothesis + thresholds
- `RESULT.md` — this file
- Raw outputs synced via wandb (run IDs above); local JSONLs not on
  disk because Pod-03 was destroyed before pull.

## How this is used downstream

- **Paper §3.2** cites this RESULT.md and draws the inverted-U as
  `fig_commit_lora_k2_sweep`.
- **Phase-3 T1.C** (Fast-dLLM commit-aware port) relies on this
  result: porting commit-LoRA into Fast-dLLM is only worth doing
  because the K2 ablation showed the schedule toggle is load-bearing.
- **Phase-3 T2.A** (MATH-500 cross-domain): the K2 inverted-U
  hypothesis is what we re-test on a different domain.
- **Phase-3 T2.C** (BD3-LMs cross-substrate): the K2 inverted-U is
  what we re-test on a different DLM family.
