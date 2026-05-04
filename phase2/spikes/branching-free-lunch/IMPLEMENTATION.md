# S0 + S1 — Implementation log

**Date:** 2026-05-04
**Status:** code landed; pod validation pending.

## What landed

### S0 — branch batching

- `e4/diff_llada.py`:
  - New dataclasses `BatchStepState`, `BatchStepDirective` + type alias
    `BatchStepCallback`. Mirror the per-row `StepState`/`StepDirective`
    contract but carry the full `(B, L+gen)` live tensor + per-row
    `active` flags. `_default_batch_step_callback` returns
    `BatchStepDirective.continue_all(B)`.
  - New helper `_add_gumbel_noise_batched(logits, temperature, generators)`
    — per-row `torch.Generator` keeps branch determinism across calls.
    Falls back to global RNG if generators is None.
  - `_Mock.denoise_block_batched(prompt, k_steps, seeds, ...)` — calls
    `denoise_block` per-seed in a list comprehension. Mock has no
    batched semantics to optimise; this exists so runner.py can use one
    API in `MOCK_MODELS=1` smoke.
  - `_Real.denoise_block_batched(prompt, k_steps, seeds, ...)` — builds
    one prompt, replicates across `B = len(seeds)` rows, allocates per-row
    generators, calls `_generate_batched`. Single-branch (`B==1`) defers
    to the legacy `denoise_block` for bit-identical backcompat.
  - `_Real._generate_batched(...)` — mirrors `_generate` but allocates
    `(B, L+gen)` tensor; per-row topk + scatter inside the per-step loop.
    Per-row `active` bool flag controls whether a row commits this step
    (default True; ESC flips to False on pruned rows). Forward pass and
    softmax run over the full batch.

- `e4/runner.py`:
  - `cmaj`, `cmajc`, `cmerge` rewritten to call `denoise_block_batched`
    once with `seeds = [seed*100 + b for b in range(n_branches)]`.
    Original sequential-loop code removed from these conditions; legacy
    `denoise_block` path remains intact for `c2`, `c2c`, `c3`, `c3p`,
    `c4`, etc.
  - Per-problem `wallclock_ms` added to the JSONL row dict in the main
    loop (`time.time()` around each `run_condition` call).

### S1 — ESC quorum exit

- `e4/runner.py:_make_esc_callback(diff_model, n_branches, esc_state)`
  — module-level helper. Returns `None` when `ESC` env var is not `"1"`
  or when `diff_model._tokenizer` is unavailable (mock mode). Otherwise
  returns a `BatchStepCallback` that:
  1. Skips firing until `state.sub_block >= ESC_MIN_BLOCK` (default 2).
  2. Decodes each active row's committed prefix `x[bi, L:end]` via the
     LLaDA tokenizer.
  3. Runs `grade.extract_answer` per row.
  4. Counts non-empty answers; if a quorum (`B//2 + 1`) agree, sets
     `should_stop[bi]=True` for any active row whose answer != winner,
     and stashes `trigger_block`, `winner`, `branches_pruned`,
     `partial_answers` into the caller's `esc_state` dict.
- `cmaj`/`cmajc` use the helper; if ESC fired, the cmaj winner is
  taken from `esc_state["winner"]` (overrides the post-hoc Counter
  vote), since pruned rows freeze and may extract noisy mid-CoT
  numbers if re-decoded after the call. JSONL trace gets two extra
  fields when ESC fires: `esc_trigger_block`, `esc_branches_pruned`.
- `cmerge` does NOT use ESC — final AR merger needs all candidate
  branches as input regardless of mid-flight agreement.

### Env knobs

- `BRANCHES` — number of branches (existing). Default 5 for cmaj/cmajc, 3 for cmerge.
- `ESC` — gate ESC quorum exit. Default `"0"` (off). Set `"1"` to enable.
- `ESC_MIN_BLOCK` — earliest sub-block ESC may fire. Default `"2"`
  (last 2 of 4) so the answer span has time to land.

### Backcompat

- `phase2/inference_viz/test_backcompat.py` PASS 3/3 — legacy single-row
  `_generate` path is unchanged; default-callback path is bit-identical
  against the locked fixture; stop-directive terminates cleanly.

### Mock smoke matrix

| Condition | ESC | Result |
|---|---|---|
| cmaj | 0 | OK, JSONL has 5 branches + votes + winner + wallclock_ms |
| cmaj | 1 | OK (mock has no tokenizer → cb returns None → S0-only path) |
| cmajc | 0 | OK |
| cmajc | 1 | OK |
| cmerge | 0 | OK, 3 branches, AR finalize call unchanged |
| cmerge | 1 | OK (cmerge ignores ESC by design) |

## What's pending

- **Real-mode pod validation** per `PRE_REG.md` §"Success / kill criteria":
  - N=20 paired wallclock baseline / S0 / S0+S1 — paired Wilcoxon for
    the speedup CIs.
  - N=200 cmaj accuracy (S0+S1) ∈ [78.0%, 81.0%].
  - N=200 cmajc-v3 accuracy (S0+S1) ∈ [80.0%, 85.0%].
  - ESC trigger-rate + false-prune-rate distribution.

  Estimated cost: ~$3–5 on RTX 4090 spot. Needs `mcp__crucible-fleet__run_project`
  with `LORA_PATH=eren23/sfumato-llada-prefix-robust-v3`,
  `COMMIT_LORA_PATH=eren23/sfumato-llada-commit-v3`, `BRANCHES=5`,
  `TEMP=0.7`, `K_STEPS=64`, plus `ESC=1` for the S0+S1 cell.

- **RESULT.md** to be written after the pod runs return — paired
  wallclock distributions, accuracy CIs, ESC trigger histogram.
