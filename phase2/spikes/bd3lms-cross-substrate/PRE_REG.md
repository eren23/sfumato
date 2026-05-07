# Pre-registration — BD3-LMs Cross-Substrate Commit-LoRA K2

**Date:** 2026-05-06 | **Spike:** Phase-3 T2.C of
`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`.
**Status:** SCAFFOLDED, not yet running. Multi-week eng prerequisite
(see "Eng prerequisites" below).

## Hypothesis

The K2 inverted-U commit-LoRA finding has been replicated:
- **In-domain:** GSM8K (Phase-2 §3, k=3 = 0.822 peak)
- **Cross-domain (same diffusion-LM):** MATH-500 numeric N=50 (T2.A
  WIN, k=3 = 0.58 peak — commit `9c81bfd`)

This spike tests whether the K2 inverted-U **also replicates across
diffusion-LM families**. Specifically: train a fresh 14M-param commit-
LoRA on top of BD3-LMs (Block Diffusion language model, Arrelou et al.
ICLR-25), repeat the K-sweep, and check whether (a) commit-LoRA helps
on a different DLM substrate at all, and (b) the optimal sub-block
boundary is in [k=2, k=3] as in LLaDA.

If **WIN** → the commit-LoRA primitive is genuinely a "schedule-toggle
for any block-diffusion LM" rather than a LLaDA-specific calibration.
This would be the strongest §3 generality claim sfumato can ship.

## Substrate

- **Model:** BD3-LMs base (which size? — eng-prereq question; the paper
  ships ~110M and ~340M variants)
- **Substrate:** GSM8K-test idx 0..199, same as Phase-2 K2 ablation.
- **Adapter:** Fresh 14M commit-LoRA (LoRA rank 16, target modules per
  BD3-LMs architecture). Trained on GSM8K-train ~1k examples, same
  recipe as LLaDA commit-LoRA v3.
- **K-sweep:** 4 conditions (c2c k=0, cmajc k=2/3/4) × N=50 BATCHED=0
  BRANCHES=5 K_STEPS=64 TEMP=0.7 SEED=0.

## Eng prerequisites (multi-week)

This is the largest Phase-3 single spike. Cannot be one-shot dispatched
like T2.A. Required engineering:

1. **Fork + clone BD3-LMs.** Locate the official ICLR-25 repo
   (Arrelou's GitHub if public; otherwise pull from HF model card
   references). Set up local + pod environments.
2. **Audit BD3-LMs sub-block sampler.** sfumato's K-sweep depends on
   the sub-block boundary semantics. BD3-LMs may use different block
   sizes / different denoising schedule. Map their block contract to
   sfumato's COMMIT_N_BLOCKS knob.
3. **Train commit-LoRA on BD3-LMs base.** Reuse Phase-1 commit-LoRA-v3
   training recipe (LoRA target modules, k_steps schedule, GSM8K-train
   data). ~1 GPU-day on 48GB+. Produces `eren23/sfumato-bd3-commit-v1`
   (HF artifact).
4. **Adapt sfumato runner.** `e4/diff_llada.py` → fork to
   `e4/diff_bd3.py` with BD3-LMs forward pass + commit-LoRA toggle
   threading. Match StepState contract so existing trace tooling works.
5. **K-sweep run.** Once 1-4 land, this is a single 4×N=50 run
   identical to T2.A on the new substrate.

**Eng days: ~10-14.** GPU days: ~1.5 (training) + ~1 (K-sweep). Cost:
~$10-15.

## Pre-registered decision rules

| Outcome | Verdict |
|---|---|
| cmajc-best ≥ 0.30 baseline AND k-sweep produces inverted-U with peak in [k=2, k=3] | **WIN** — commit-LoRA primitive transfers across DLM families. §3 generality claim becomes "schedule-toggle for *any* mask-diffusion LM" rather than LLaDA-specific |
| cmajc-best ≥ 0.30 AND positive lift but no inverted shape (e.g., monotone, k=4 peak) | **PARTIAL** — commit-LoRA helps but the boundary calibration is family-specific |
| Any cmajc < c2c | **LOSS** — commit-LoRA is LLaDA-specific. Paper §3.5 caveat: "validated on LLaDA only, BD3 disconfirms cross-family generalization" |

## Anti-goals

- No swap of base diffusion-LM mid-spike — pin BD3-LMs version at
  whatever the ICLR-25 paper artifacts shipped.
- No retrain of commit-LoRA from scratch on cross-substrate data
  (defeats the purpose of testing transfer).
- No sub-block resizing within BD3-LMs to match LLaDA's 32-token
  blocks — adapt to BD3's native blocks. (If BD3's block size is
  different, that itself is a methodological note for the paper.)

## Cost (when this fires)

| Item | $ |
|---|---:|
| Eng days (10-14 days, no GPU during code work) | $0 |
| LoRA training on BD3 base (~1 GPU-day, 48GB on-demand) | ~$8 |
| K-sweep × 4 conditions × N=50 | ~$2 |
| Pod overhead + buffer | ~$2 |
| **Total (original estimate)** | **~$12** |

## Cost re-estimate (2026-05-07)

The 2026-05-06 PRE_REG above assumed reuse of LLaDA's `generate.py`
contract on BD3-LMs. A WebFetch + arXiv read on 2026-05-07 (during the
Phase-4 T2.C scaffold dispatch) revealed three blockers that triple the
real cost:

1. **BD3-LMs ships OWT pretrain ONLY.** The released checkpoints
   (`kuleshov-group/bd3lm-owt-block_size{4,8,16}` and `block_size1024-pretrain`
   on HF) are pure perplexity-trained on OpenWebText. There is **no math
   instruction-tuned variant**. To compare apples-to-apples with LLaDA-8B-
   Instruct, BD3 must first be SFT'd on GSM8K-train (~7.5k examples).
2. **No PEFT/LoRA infrastructure exists.** BD3's model class is a custom
   DiT, not a `transformers.AutoModel` subclass. `peft.PeftModel.from_pretrained`
   does NOT work out of the box. The LoRA target-module discovery + adapter
   wiring + serialize/load cycle has to be added to BD3's model class.
3. **Hydra-only sampler.** BD3's release entrypoint is
   `main.py mode=sample_eval` driven by Hydra config. No programmatic
   `_generate(prompt_ids, k_steps, ...)` Python entrypoint matching the
   LLaDA contract — must be lifted out of the Hydra sampler loop into a
   reusable function.

Realistic re-estimate:

| Phase | Eng-days | $ |
|---|---:|---:|
| Phase 1: fork BD3 Hydra repo + local/pod env | 3-4 | $0 |
| Phase 2: SFT BD3-base on GSM8K-train (~7.5k examples) | 2-3 | $15-25 |
| Phase 3: add PEFT/LoRA infra to BD3 model class | 3-4 | $0 |
| Phase 4: train commit-LoRA on BD3-base+SFT | 2-3 | $5-10 |
| Phase 5: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0) | 1 | $2-3 |
| Phase 6: buffer (debugging + paper write-up) | 2-3 | $10-20 |
| **Total (realistic)** | **14-21 days** | **~$40-60** |

That's **3-4× the original estimate** ($12 + 10-14 days → $40-60 + 14-21
days). The decision rules above are unchanged — only the cost amendment.

**Status (2026-05-07):** scaffolds landed at `e4/diff_bd3.py` and
`scripts/train_bd3_commit_lora.py` (Phase-4 T2.C dispatch). Both raise
NotImplementedError / print the eng-prereq table. Direction A
(schedule-RLHF) is the prioritized Phase-4 bet; T2.C stays scaffolded
but unfunded until budget approval.

## Files (when this fires)

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled after K-sweep
- `e4/diff_bd3.py` — BD3-LMs adapter (new, ~500 LOC)
- `train_bd3_commit_lora.py` — fresh commit-LoRA training (new)
- `eren23/sfumato-bd3-commit-v1` — HF artifact (new)

## Status notes

**Why not auto-dispatched in this session:** the eng prerequisites
(fork, audit, port, train) are 10-14 days of work that can't be
one-shot via `run_project`. The PRE_REG is staged so a future session
can pick it up directly. The decision rules are locked NOW so a
future-session researcher can't move the goalposts.

**Trigger condition:** start this spike when (a) a paper revision
explicitly asks for cross-substrate evidence, OR (b) a future
contributor wants to extend §3 to a 3-substrate sweep.
