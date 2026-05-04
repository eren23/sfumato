# Pre-registration — LoRA merge-on-toggle + logit_shift_norm (S2 + S3)

**Phase:** Phase 2/3 spike chain
**Spike ID:** S2 (PEFT merge/unmerge at commit-LoRA boundary) + S3
(populate `logit_shift_norm` schema slot via shadow base-only forward).
**Author:** main session
**Date:** 2026-05-04
**Plan source:** `phase2/proposals/kernel-survey/SUMMARY.md` §3 second pick;
detailed cost model in `phase2/proposals/kernel-survey/04_commit_lora_free_lunch.md` §2.

## Hypothesis

**S2:** Calling `peft_model.merge_adapter()` once at `_enable_commit`
(after the `set_adapter("commit")` switch) and `peft_model.unmerge_adapter()`
once at `_disable_commit` reduces the per-step LLaDA forward to a single
`y = x @ W_merged.T` matmul (instead of `y = x @ W_base.T + (x @ A.T) @ B.T`)
during commit-active sub-blocks. Expected per-step wall-clock cut on
LLaDA-8B bf16 RTX 4090: **≥3%**, with **|Δaccuracy| ≤ 0.5pp** on GSM8K
dev-50 at k_steps ∈ {32, 64} and seeds ∈ {0, 1, 2}.

**S3:** When `LOGIT_SHIFT_NORM=1`, run a shadow forward at the *last*
step of each commit-active sub-block with the adapter unmerged, take the
L2 norm of `(logits_with_commit − logits_base)` over committed positions,
and write to `state.logit_shift_norm`. Expected pattern: mean
`logit_shift_norm` at sub-block 1 (first commit-active) is **≥ 2× the
mean at sub-block 3 (final commit-active)**, i.e. early commit blocks
shift logits more than the answer-formatter block. This quantifies
whether v3's `COMMIT_N_BLOCKS=3` budget is uniformly utilised.

## Conditions tested

- **Baseline (S2 off):** `CONDITION=c2c K_STEPS=64 SEED=0 LORA_PATH=…
  COMMIT_LORA_PATH=… COMMIT_N_BLOCKS=3 N_PROBLEMS=50` (current code).
- **S2 treatment:** same env, with merge-on-toggle patch active.
  No env knob — the merge happens unconditionally inside
  `_enable_commit` / `_disable_commit`.
- **S3 measurement:** baseline with `LOGIT_SHIFT_NORM=1` + a
  `step_callback` that captures the per-sub-block norm. (S3 is
  diagnostic-only; it does not change generation.)

## Success / kill criteria

| Metric | WIN | LOSS |
|---|---|---|
| S2 per-step wall-clock cut at k=32 | ≥ 3% | < 1% |
| S2 per-step wall-clock cut at k=64 | ≥ 3% | < 1% |
| S2 c2c accuracy on dev-50 (3 seeds) | within ±0.5pp of baseline | ≥ 1pp drop on any cell |
| S3 `mean(logit_shift_norm)` ratio sub-block 1 / sub-block 3 | ≥ 2 | — (descriptive) |
| S3 backcompat fixture | PASS within bf16 ε (1e-3 tolerance bumped) | byte-mismatch on legacy path |

LOSS on S2 → revert; investigate PEFT merge correctness for the
`base_lora` + `commit` two-adapter setup. LOSS on backcompat fixture
→ revert; investigate whether `merge_and_unload` mutates state we
care about beyond the targeted Linears.

## Measurement plan

1. **Wall-clock instrumentation:** the per-problem `wallclock_ms` column
   landed in S0 already; we use it as-is for the S2 comparison.
2. **Per-step wall-clock:** add a single timer around the inner per-step
   loop in `_generate` (already exists at `t0 = time.time()` line 435 —
   move out to per-step granularity, log to `state.wallclock_ms` already
   in the schema). Aggregate into per-sub-block means at the end of the
   pre-reg analysis.
3. **Accuracy:** binom-CI on N=50 dev-set (cheap subset for spike
   pre-reg); only if WIN, re-run at N=200 for the full CI confirmation.
4. **Logit-shift measurement:** the shadow forward fires only when
   env `LOGIT_SHIFT_NORM=1` is set; default code path is bit-identical.
5. **Backcompat:** `phase2/inference_viz/test_backcompat.py` against
   the locked fixture. Bump tolerance from byte-equal to `1e-3` if needed
   and document in the test docstring.

## Files modified

**Source:**
- `e4/diff_llada.py:_enable_commit` — call `merge_adapter()` after the
  `set_adapter("commit")` / `enable_adapter_layers()` line.
- `e4/diff_llada.py:_disable_commit` — call `unmerge_adapter()` before
  the `set_adapter("base_lora")` / `disable_adapter_layers()` line.
- `e4/diff_llada.py:_generate` — at the *last* step of each
  commit-active sub-block, if `LOGIT_SHIFT_NORM=1`, run a shadow
  forward with the adapter temporarily unmerged, compute L2 norm over
  committed positions, write to `state.logit_shift_norm`.
- `e4/diff_llada.py:_generate_batched` — same logit-shift hook for
  the batched path so cmajc traces also populate the slot.

**Tests:**
- `phase2/inference_viz/test_backcompat.py` — verify within-ε pass.

## Compute envelope

- Backcompat fixture: 0 GPU (mock).
- N=50 c2c × 3 seeds × 2 k_steps = 6 cells × ~1 min each ≈ 6 min ≈ **$0.05**.
- N=200 c2c WIN-confirm × 1 seed: ~25 min ≈ **$0.10**.
- Total: **≤$1**.

## Rollback / safety

- Merge/unmerge are PEFT-native; revert is one line each in
  `_enable_commit` / `_disable_commit`.
- If the backcompat fixture fails byte-equal, the bumped-tolerance test
  documents *why* (bf16 rounding) and the fixture itself becomes the
  kill criterion. If even ε-tolerance fails, revert.

## Falsification artifacts

- `phase2/spikes/lora-merge-toggle/wallclock_n50_baseline.jsonl`
- `phase2/spikes/lora-merge-toggle/wallclock_n50_s2.jsonl`
- `phase2/spikes/lora-merge-toggle/logit_shift_n50_s3.jsonl`
- `phase2/spikes/lora-merge-toggle/RESULT.md`
