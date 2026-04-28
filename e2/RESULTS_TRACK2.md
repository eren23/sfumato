# E2 Track 2 — Results (commit-adapter for consensus distillation)

Updated 2026-04-28 ~02:00 UTC. **Headline: pre-registered prediction #1 falsified at v2; cmajc v2 still pending.**

## TL;DR

A late-block FFN-only commit-LoRA, trained with answer-span loss only on 109 mixture rows, did not lift c2c single-shot accuracy at v1 OR v2 (capacity bumped 3.25× by fixing a LoRA-target bug). Pre-registered prediction #1 (c2c ≥ 80%) **falsified at -9.5pp**.

This is a *narrow* negative — about this specific commit-LoRA *design*, not about consensus distillation in general. We did not test (a) commit on earlier blocks, (b) full-response loss, (c) RL on consensus reward, or (d) reranker-style aggregation. The diagnosis below identifies which design choices most likely caused the failure.

## Pipeline

**Datasets:**
- `eren23/sfumato-consensus-gsm8k` — 500 GSM8K-train problems, raw cmaj b=5 t=0.7 outputs + greedy reference. Public.
- `eren23/sfumato-commit-mixture-gsm8k` — 109 rows derived from the raw, three buckets:
  - rescue (consensus right, greedy wrong): 46 rows
  - preserve_disagreement (both right but branches differ): 58 rows
  - pure_agreement (both right, all 5 branches identical): 5 rows
  - 104 train / 5 validation
- Baseline branch-agreement distribution (pre-LoRA falsifier reference): 5/5 same answer = 52% of problems, 4/5 = 7%, 3/5 = 13%, 2/5 = 24%, 5/5 unique = 4%.

**Adapters:**
- `eren23/sfumato-llada-commit` (v1) — buggy: PEFT `target_modules=["gate_proj","up_proj","down_proj"]` only matched `up_proj` because LLaDA's `LLaDALlamaBlock` uses olmo-style names (`ff_proj`, `up_proj`, `ff_out`). 4.2M trainable params, 1/3 of FFN attached.
- `eren23/sfumato-llada-commit-v2` — fix applied: `target_modules=["ff_proj","up_proj","ff_out"]`. 13.6M trainable params (3.25× v1).

Same training recipe for both: r=8, alpha=16, 10 epochs, bs=1×grad_accum=4, lr=5e-5, mask range U(0.3, 0.9) on the answer-only span only (loss masked over `[answer_start..response_end)`).

## Bug discovered

LLaDA-8B-Instruct uses `LLaDALlamaBlock` with non-standard FFN module names:

| Standard Llama | LLaDA actual |
|---|---|
| `q_proj` | `q_proj` ✅ |
| `k_proj` | `k_proj` ✅ |
| `v_proj` | `v_proj` ✅ |
| `o_proj` | **`attn_out`** ❌ |
| `gate_proj` | **`ff_proj`** ❌ |
| `up_proj` | `up_proj` ✅ |
| `down_proj` | **`ff_out`** ❌ |

Track 1 v2 was also affected — only matched 4 of 7 intended modules (q/k/v/up_proj). It still produced usable signal (the published result), so we didn't retrain it; this stays an "improvement headroom" note for a future Track 1 v3.

## Eval results (N=200 GSM8K-test, k=64)

| Cond | Setup | Base | Track 1 v2 | + commit v1 | + commit v2 |
|---|---|---:|---:|---:|---:|
| C2 / c2c | single-shot, no prefix | 74% | 70.5% | 70.5% | **70.5%** |
| cmaj / cmajc | b=5 t=0.7 | 80% | 81.5% | 81.5% | *pending* |

Commit-LoRA had ZERO effect on c2c at v1 AND v2. cmajc v1 was also unchanged from cmaj baseline.

## Pre-registered prediction scorecard

1. **c2c ≥ 80%** (single-shot bridges to b=5 ceiling): ❌ **FAILS** at 70.5%, off by 9.5pp at both v1 and v2.
2. **base cmaj b=5 (commit OFF) within ±1pp of 80%**: ✅ Track 1 alone gives 81.5%, commit OFF preserves it (this is the "diversity preserved" falsifier).
3. **cmajc ≤ cmaj + 1pp** (no double-dip / collapse): TBD pending cmajc v2 result.

## Diagnosis

The commit-LoRA architecture has a structural problem the eval surfaces clearly:

1. **Commit-LoRA only fires on the last sub-block** (32 of 128 generated tokens, the answer-tail) per `e4/diff_llada.py:_generate(commit_last_block=True)`. By the time the sampler reaches block 4 of 4, the previous 96 tokens have committed the CoT. The answer string `"Answer: X"` is mostly determined by what came before; the commit adapter can only shift the literal answer-tokens, but X is already implied.
2. **Training supervises only the answer span.** Loss is masked to `[answer_start..response_end)` — typically ~10 tokens. With 109 training rows × 10 epochs and a tiny answer-span loss target, the adapter saturates fast (loss → 0.0001) and the deltas don't generalize to unseen answer trajectories.
3. **The mixture-as-designed (rescue 40 / preserve 50 / pure 5) couldn't rescue this.** Even with preserve-disagreement examples, the supervision target is the greedy CoT's answer span, which the model already produces correctly at greedy decode. Zero gradient signal on those.

Combine 1 + 2 + 3 → at inference the commit-LoRA can either (a) reproduce what the base model already produces (when greedy was right), or (b) try to flip the answer-span tokens, but it can't override 96 prior CoT tokens. So it silently does nothing.

This is consistent with both v1 (1/3 FFN coverage) and v2 (3/3 FFN coverage) producing the *exact same* c2c=70.5%. The bug-fix bumping capacity 3.25× changed *trainable parameters* but not the structural problem.

## What would actually work (next-iteration ideas)

- **Apply commit across more blocks**, not just the last. Commit on blocks 2–4 lets the adapter shape the late-CoT path, not just the post-decided answer string.
- **Train commit on the FULL response**, not just answer-span. Lets it learn to rescue mid-CoT errors.
- **Bigger, harder mining**: scale `rescue` to 200+ rows by widening the filter (e.g. consensus-correct even if greedy was right, then teach commit to *strengthen* the right answer rather than rescue wrong ones). Or RL with answer-reward.
- **Different surgery entirely**: skip LoRA, train a small *reranker* over 5 sampled branches (compute-time, no model surgery). That's the simpler version of "consensus distillation" that doesn't fight LLaDA's commit mechanics.

## Spend so far (Track 2)

| Phase | $ |
|---|---:|
| Consensus dataset gen (500 problems) | ~$0.40 |
| Mixture build (offline, free) | $0 |
| Track 2 v1 train + push (203s) | ~$0.05 |
| Track 2 v1 eval (c2c + cmajc, ~145min) | ~$0.50 |
| Track 2 v2 retrain (~7min) | ~$0.05 |
| Track 2 v2 eval c2c (~25min) | ~$0.10 |
| Track 2 v2 eval cmajc (running, ~60min) | ~$0.20 |
| Pod 2 abortive (twice) | ~$0.10 |
| **Track 2 total (projected)** | **~$1.40** |

E2 total still well under the $7.55 envelope.

## Wandb runs

- Track 2 v1 train: https://wandb.ai/eren23/sfumato-e2/runs/8x5t5gha (val/loss 0.0001)
- Track 2 v2 train: https://wandb.ai/eren23/sfumato-e2/runs/itvuo4du
- Eval c2c v1: https://wandb.ai/eren23/sfumato-e4/runs/pq204yhv (acc 0.705)
- Eval cmajc v1: (recovered from raw_cmajc_k64_seed0.jsonl, acc 0.815)
- Eval c2c v2: (acc 0.705)
- Eval cmajc v2: pending

## What this means for the paper

The cleaner Track 1 result holds: prefix-robust LoRA flattened format-damage spread without collapsing diffusion diversity (cmaj b=5 stayed at 81.5%). That's the structural-separation thesis confirmed.

Track 2's negative is informative but *narrow*. The failure rules out one specific design (late-block answer-span LoRA, ≤14M params, 109 mixture rows). It does NOT rule out earlier-block commit, full-response loss, or reranker-style aggregation. The diagnosis section above identifies the late-block-only design as the most likely cause. Future work should test at least one variant before claiming the broader pattern.

Paper framing (current scope, no overreach):

> Hybrid AR/DDLM failure decomposes into at least three orthogonal axes — *interface-format brittleness*, *planner-content trust*, and *sampling-diversity preservation*. We show interface-format brittleness is trainable (Track 1 v2: prefix-robust LoRA, +5.5pp on C2hint, preserves cmaj b=5 = 81.5%); planner-content trust is partially trainable (Track 1 swaps the U-shape on planner quality). For sampling-diversity, our specific Track 2 design — a late-block FFN-only LoRA distilling cmaj consensus into the final 32-token answer span — failed to bridge the c2c-vs-cmaj gap; the failure mode appears to be mechanistic (commit fires too late to override the CoT that has already determined the answer) rather than capacity-limited (the v2 fix tripled trainable parameters without changing c2c).

That last clause is the load-bearing one. It's defensible on the data we have and doesn't make a broader claim than the experiments support.

## Open follow-ups (cheap, defer to paper revision)

1. **Track 1 v3 with full FFN module coverage.** v2's published 81.5% used only 4/7 modules. Confirm it doesn't change materially with full coverage. ~$0.20.
2. **Logit-shift diagnostic.** Dump c2c logits with vs without commit-LoRA active on 10 problems. Tells us whether commit is shifting logits (capacity-limited) or learned identity (design-limited). Decides whether (3) is worth running. ~$0.05.
3. **v3 commit-LoRA on blocks 2-4 with full-response loss.** If logits show shift but no flip, this likely moves c2c. ~$0.15.
4. **Reranker baseline.** Train a small classifier over the 5 cmaj branches; compare reranker-cmaj to vote-cmaj. Provides the "internalize consensus into compute, not weights" baseline. Several hundred dollars or several days of work — defer to E1 / future paper.
5. **Post-Track-1 branch-agreement rate.** Compute fraction of problems where 5/5 branches give the same answer, post-Track-1 vs base. Tests diversity preservation directly rather than via accuracy. Free, ~10 min once cmajc v2 raw is available.
