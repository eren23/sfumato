# S2 + S3 — Implementation log

**Date:** 2026-05-04
**Status:** code landed; pod validation pending.

## What landed

### S2 — PEFT merge-on-toggle

- `e4/diff_llada.py:_enable_commit` (lines 678–696) — after the
  `set_adapter("commit")` / `enable_adapter_layers()` call, invoke
  `model.merge_adapter()` and set `self._commit_merged = True`. Wrapped
  in try/except so older PEFT versions or models without LoRA-on-Linear
  fall back to the un-merged path silently.
- `e4/diff_llada.py:_disable_commit` (lines 698–712) — call
  `model.unmerge_adapter()` first (when `_commit_merged` is True) so
  the base weights are restored *before* switching back to the
  `base_lora` adapter or disabling adapter layers. Resets
  `self._commit_merged = False`.
- Result: per-step LLaDA forward during commit-active sub-blocks runs
  one matmul per Linear (`y = x @ W_merged.T`) instead of two
  (`y = x @ W_base.T + (x @ A.T) @ B.T`). Numerically equivalent
  within bf16 rounding because PEFT's merge math is `W += α/r · BA`
  and unmerge is exactly the inverse `W -= α/r · BA`.

### S3 — `logit_shift_norm` populated

- `e4/diff_llada.py:_maybe_logit_shift_norm` (lines 714–752) — new
  helper. Gated behind `LOGIT_SHIFT_NORM=1` env var so production
  runs pay zero cost. When enabled and commit-LoRA is active and
  merged: temporarily un-merges the adapter, runs one extra base-only
  forward, re-merges, computes
  `||logits_with_commit[committed_positions] − logits_base[committed_positions]||₂`
  and returns the float.
- `e4/diff_llada.py:_generate` (around line 615) — at the sub-block
  boundary, when `commit_active_now` is True, calls
  `_maybe_logit_shift_norm(last_logits, x, committed_positions)` and
  writes the result to `state.logit_shift_norm` (previously hardcoded
  to `None`).
- `_generate_batched` does NOT populate the slot — `BatchStepState`
  has no `logit_shift_norm` field by design (the diagnostic targets
  the c2c/single-branch path, not cmajc).

### Backcompat

- `phase2/inference_viz/test_backcompat.py` PASS 3/3. Mock fixture
  unchanged because mock mode never goes through `_enable_commit` or
  `_disable_commit` — those are `_Real`-only.
- For real-mode the merge/unmerge math is exact within bf16; if a
  future fixture comparison is bit-exact and breaks, bump tolerance to
  `1e-3` and document.

### Mock smoke matrix

| Condition | LOGIT_SHIFT_NORM | Result |
|---|---|---|
| c2c | 0 | OK |
| c2c | 1 | OK (no shadow forward in mock — no commit_lora_path) |
| cmajc | 0 | OK |
| cmajc | 1 | OK (BatchStepState skips slot by design) |

## Env knobs

- `LOGIT_SHIFT_NORM=1` — enable S3 shadow forward + L2 measurement.
  Default off. Costs roughly **+1 forward per commit-active sub-block**
  (so for v3 / `COMMIT_N_BLOCKS=3`: 3 extra forwards/problem).

## What's pending

- **Real-mode pod validation** per `PRE_REG.md`:
  - c2c N=50 paired wallclock: S2 off vs S2 on at k_steps ∈ {32, 64},
    seeds ∈ {0, 1, 2}. Expect ≥3% per-step cut, |Δacc| ≤ 0.5pp.
  - c2c N=200 with S2 on: WIN-confirm.
  - c2c N=50 with `LOGIT_SHIFT_NORM=1`: harvest per-sub-block
    `logit_shift_norm` distributions; check ratio sub-block 1 / sub-block 3
    is ≥ 2.

  Estimated cost: ~$1 on RTX 4090 spot. Needs Crucible.

- **RESULT.md** to be written after pod runs return.
