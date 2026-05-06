# T1.C Fast-dLLM Commit-Aware Port — RESULT

**Pre-reg:** `PRE_REG.md` (committed before run) + Phase-3 plan T1.C.
**Run date:** 2026-05-06 | **Cost:** ~$0.10 (24GB 4090 spot, ~16 min).
**Outcome:** **LOSS** — both accuracy and speedup miss thresholds.

---

## Headline numbers

| Metric | Phase-2 baseline (no FAST_DLLM) | T1.C blockwise wrapper | Pre-reg target | Verdict |
|---|---:|---:|---:|---|
| cmajc N=20 acc | 0.80 ± 0.05 | **0.45** | ≥ 0.75 | **LOSS** |
| wallclock per problem | ~22 s | ~12.8 s | ≤ 5.5 s (≥4× speedup) | **LOSS** (1.7×) |

(Source: wandb [ooh4doyi](https://wandb.ai/eren23/sfumato-e4/runs/ooh4doyi).
Total wallclock 256s for N=20.)

## Pre-registered decision rules — outcome

| Rule | Triggered? |
|---|---|
| acc ∈ [0.75, 0.85] AND speedup ≥ 4× | **NO** — both miss |
| acc ∈ [0.75, 0.85] AND speedup ∈ [2×, 4×) | **NO** |
| acc < 0.75 OR speedup < 2× | **YES** — LOSS |

## Root cause

The upstream NVlabs/Fast-dLLM v1 `generate(prompt, gen_length, ...)` is
**one-shot**: it allocates a length-`gen_length` mask suffix to the
prompt and runs the full parallel-decode loop end-to-end. The blockwise
wrapper this spike introduced calls it once per sub-block with
`gen_length=block_length=32` and feeds the previous output as the new
prompt. Two failures cascade:

1. **Mask-length mismatch.** Upstream appends `gen_length=32` *new*
   mask tokens at each call. After block 1 the shape is
   `(1, L + 32)`. The next call appends another 32 on top, so the
   sequence grows to `(1, L + 64)` instead of refining the existing
   suffix. Each "block" generates fresh content rather than continuing
   to denoise the in-flight trajectory. cmajc's expected
   commit-LoRA-on-blocks-2-4 schedule reduces to "generate 4
   independent chunks of 32 tokens with commit-LoRA toggling between
   them." That's not LLaDA's semi-AR sub-block sampler; it's a
   different generative procedure entirely. Hence acc=0.45.

2. **KV-cache reset between calls.** Upstream's KV cache is freed at
   the end of `generate()`. Calling it 4× per problem rebuilds the
   cache 4× and loses the across-block savings that drove the
   original 6.5× single-call speedup. Hence speedup=1.7× not ≥4×.

## What this kills, what it leaves open

**Killed:** the "wrap upstream `generate()` per-block" approach to
commit-LoRA-aware Fast-dLLM. The `(prompt, gen_length)` contract is
fundamentally one-shot, and re-entering it per sub-block does not
compose with commit-LoRA toggling.

**Open paths for Phase-3+ revision:**

- **A. Fork upstream Fast-dLLM** to add a per-block callback hook
  inside their parallel-decode loop. ~5–7 eng-days, $0 GPU. Anti-goal
  in the original plan ("No upstream Fast-dLLM fork — wrap in our shim
  instead"); this result is the evidence to revise that anti-goal.
- **B. Reimplement the parallel-decode primitive in sfumato** (port
  upstream's confidence-threshold commit + KV cache logic into
  `e4/diff_llada.py:_generate`). Larger eng cost (~10–15 days) but
  zero upstream dependency.
- **C. Accept separate paths**: use FAST_DLLM=1 only on c2 / c2c /
  cmaj (no commit-LoRA toggle), keep cmajc on the slower legacy path.
  This is the status quo in `eren23/sfumato` main pre-T1.C, with the
  Phase-2 result `c2 N=200 = 0.745, ~3.4 s/problem = 6.5× speedup`
  remaining the canonical FAST_DLLM productization headline. T1.C
  doesn't break this — the legacy `fast_dllm_generate()` one-shot
  path is preserved bit-identically when `commit_last_block=False`.

The honest paper-section reading is that **commit-LoRA's K2 inverted-U
finding (§3 of the draft) is mechanistically real but does not
trivially compose with KV-cached parallel-decode acceleration at the
upstream API surface.** Future work that wants both must either fork
the kernel or reimplement the parallel-decode loop with adapter
toggles threaded through.

## What stays in main

- `e4/fast_dllm_adapter.py:fast_dllm_generate_blockwise` — the
  function still works for any caller that explicitly opts into the
  per-block callback model. The unit test
  (`test_blockwise_callbacks.py`) still passes — the function does
  what it advertises; the failure was that what it advertises isn't
  what cmajc + commit-LoRA needs.
- `e4/diff_llada.py:_generate` — the FAST_DLLM=1 path now branches
  on `commit_last_block`:
  - `commit_last_block=False` → legacy one-shot `fast_dllm_generate`
    (preserves Phase-2 6.5× speedup, used by c2 / cmaj).
  - `commit_last_block=True` → the broken-by-this-spike blockwise
    path. **This branch should be considered untested** until a
    fork-based or reimplementation-based fix lands.

A safer interim is to make `commit_last_block=True AND FAST_DLLM=1`
log a warning and fall through to the legacy non-toggled fastpath
(commit-LoRA bypassed, identical to Phase-2 status quo). I'll add
that guard in a follow-up commit.

## Cost ledger

- T1.C code + unit test (commit `03e3fd1`): $0
- Pod-04 provision + bootstrap + fast_dllm_setup smoke + cmajc N=20
  verify: ~$0.10
- Total T1.C spend: ~$0.10. Of the <$1 Tier-1 GPU budget, ~$0.10
  consumed; ~$0.90 remains for T1.B substrate harvest.

## Files

- `PRE_REG.md` — pre-registered hypothesis + thresholds
- `test_blockwise_callbacks.py` — unit test (5 assertions, still passes)
- `RESULT.md` — this file (LOSS)
- wandb: https://wandb.ai/eren23/sfumato-e4/runs/ooh4doyi
