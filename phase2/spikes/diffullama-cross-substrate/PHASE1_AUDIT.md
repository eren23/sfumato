# DiffuLLaMA Phase-1 audit (AutoModel compat + sampler structure)

**Date:** 2026-05-07. **Cost:** $0 (CPU-only HF metadata fetch + upstream
script inspection). **Outcome:** **AutoModel compat = ✓, sub-block contract = ✗.**

---

## What this audit checked

The PRE_REG (commit `ce22404`) staged this spike as scaffold-only with
a Phase-1 prereq: confirm that `AutoModelForCausalLM.from_pretrained(...)`
loads `diffusionfamily/diffullama` cleanly, then (Phase 2) port the
sampler from upstream's `inf_diffullama.py` into a `_generate(...)` that
matches the LLaDA contract (sub-block boundaries + step_callback +
StepDirective).

This audit ran Phase 1 + a structural read of Phase 2 inputs.

## Phase 1 result: AutoModel compat ✓

`diffusionfamily/diffullama/config.json`:
```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "max_position_embeddings": 4096,
  "vocab_size": 32000
}
```

This is a **vanilla LLaMA-2 7B** in standard HF format. No
`auto_map`, no `trust_remote_code` required. The PRE_REG estimated
1-2 days for this Phase 1 alone — actual cost was a 30s HF metadata
fetch.

`diffusionfamily/diffullama-gsm/adapter_config.json` is also a
standard PEFT adapter (r=16, alpha=32, targets =
`{q,k,v,o,gate,up,down}_proj`, task_type=`CAUSAL_LM`). Compatible
with `peft.PeftModel.from_pretrained(...)`.

## Phase 2 blocker: sampler has no sub-block contract

Reading upstream `inf_diffullama.py` + `model.py`
(https://github.com/HKUNLP/DiffuLLaMA, MIT license):

```
for t in range(diffusion_steps - 1, 0, -1):    # T-1 down to 1
    p_to_x0 = 1 / (t + 1)
    masked_to_x0 = maskable_mask & (rand < p_to_x0)
    xt.masked_scatter_(masked_to_x0, x0[masked_to_x0])
    logits = model(xt, attention_mask)
    x0 = top_p_sample(logits, ...)
```

The sampler is a **flat random-keep schedule** over the entire
generation length: at each of T denoising steps, ~`1/(t+1)` of
currently-masked positions are randomly selected to commit (decoded
via top-p of the logits). There are no sub-block boundaries.

**Why this matters for the K2 cross-substrate claim:** sfumato's K2
inverted-U finding (`phase2/PAPER_DRAFT.md` §3) is defined per-sub-
block: with `gen_length=128` divided into 4 × 32-token sub-blocks,
the commit-LoRA toggle activates for the LAST `commit_n_blocks`
sub-blocks. The K2 lift is mechanistically tied to which sub-block
the toggle starts at:

  k=0   → no commit toggling                        = c2c baseline
  k=1   → commit ON for last 1 sub-block (final 32 tokens)
  k=2   → commit ON for last 2 sub-blocks (sub-blocks 2,3)
  k=3   → commit ON for last 3 sub-blocks (sub-blocks 1,2,3)  ← peak
  k=4   → commit ON for all 4 sub-blocks            = always-on

DiffuLLaMA has no sub-block structure. The commit-toggle's mechanistic
identity (a discrete on/off boundary at a known position-range) cannot
be replicated without one of:

1. **Force-add sub-blocks to DiffuLLaMA's sampler.** Chunk the gen
   sequence into 4 × 32 tokens and run the random-keep loop ONLY
   within each chunk before advancing. This *does* re-create a sub-
   block structure — but DiffuLLaMA was NOT trained with this
   conditioning; the per-chunk forwards see a different mask
   distribution than training. Likely produces degraded text quality
   AND a confounded K2 result (any signal would be "the chunked
   schedule helps DiffuLLaMA," not "K2 transfers").

2. **Redefine K2 for flat-schedule DLMs.** Interpret the K2 toggle as
   a fraction-of-steps milestone: commit-LoRA active for the LAST
   ~`(commit_n_blocks/4) × T` denoising steps. This is a different
   definition than sfumato's per-sub-block boundary, and the
   "inverted-U" claim becomes about a different knob.

3. **Train commit-LoRA differently.** Train it under DiffuLLaMA's
   native random-keep schedule with a "commit phase" defined by
   denoising-step fraction rather than sub-block index. New objective,
   new training cost. Drift from the LLaDA recipe makes the cross-
   substrate claim weaker (different LoRA, different schedule).

None of these is a clean apples-to-apples K2 test as the PRE_REG
defined it.

## Recommendation

**Pause this spike.** The PRE_REG's anti-goal #3 ("no sub-block
resizing within DiffuLLaMA to match LLaDA's 32-token blocks — adapt
to DiffuLLaMA's native sub-block contract") is unsatisfiable: the
native contract is "no sub-blocks." Any of the three workarounds
breaks one PRE_REG anti-goal or another.

For paper §3.5, the cleanest framing remains:

> *"K2 inverted-U is defined for sub-block-structured discrete-
> diffusion samplers (LLaDA, BD3-LMs, Planned Diffusion). Cross-
> family generalization to flat-schedule samplers (DiffuLLaMA's
> random-keep) requires redefining the toggle, which we leave to
> future work."*

This is itself a useful finding for the paper: the K2 toggle is a
property of *block-structured* mask diffusion, not all mask diffusion.

## What this kills, what it leaves open

**Killed:**
- The "DiffuLLaMA = cheaper Option B for cross-substrate K2 generality"
  hypothesis. Cost is not the bottleneck; architectural fit is.
- Any near-term cross-substrate K2 claim that doesn't first redefine
  K2 for flat-schedule samplers.

**Open:**
- **BD3-LMs (T2.C)** is now the lowest-architectural-distance option
  with a sub-block contract that matches LLaDA's. Cost stays at
  ~$40-60 + 14-21 days but the K2 test would be apples-to-apples.
- **Planned Diffusion** (Khanov et al. 2024, ICLR-26 paper) explicitly
  uses sub-block structure; another candidate for cross-substrate K2
  if a checkpoint becomes public.
- **Dream-7B** (Qwen2.5-7B diffusion-adapted): unclear if its sampler
  has sub-block structure. Same Phase-1 audit needed.

## Cost re-estimate

Original PRE_REG: $15-25 + 3-7 days. After this audit:

| Path | New cost | Why |
|---|---:|---|
| Force-chunked sampler + K2 sweep | $15-25 + 5-10 days | confounded result, low value |
| Redefine K2 + commit-LoRA retrain | $30-50 + 10-14 days | not apples-to-apples |
| **Pause this spike → BD3-LMs T2.C** | $40-60 + 14-21 days | apples-to-apples K2 |

## Files

- `PRE_REG.md` — locked at `ce22404`
- `PHASE1_AUDIT.md` — this file
- `e4/diff_diffullama.py` — scaffold (NotImplementedError on Real path,
  unchanged)
- `scripts/train_diffullama_commit_lora.py` — scaffold (placeholder)

## References

- DiffuLLaMA paper: arXiv 2410.17891 (HKU-NLP / Shansan Gong et al.,
  ICLR-25)
- Upstream code: https://github.com/HKUNLP/DiffuLLaMA (MIT)
- Upstream `inf_diffullama.py` + `model.py` snapshot: 2026-05-07
- HF: `diffusionfamily/diffullama` (LlamaForCausalLM, vanilla)
- HF: `diffusionfamily/diffullama-gsm` (PEFT LoRA r=16)
- R3 prior-art note: `phase2/research/r3_cross_substrate_priorart.md`
- Sister BD3 PRE_REG: `phase2/spikes/bd3lms-cross-substrate/PRE_REG.md`
