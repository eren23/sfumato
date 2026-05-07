# Pre-registration — DiffuLLaMA Cross-Substrate Commit-LoRA K2

**Date:** 2026-05-07 | **Spike:** Phase-4 T2.C-light dispatch of
`/Users/eren/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`.
**Status:** SCAFFOLDED. Cheaper Option B alternative to BD3-LMs T2.C
(which sits at $40-60 + 14-21 days; see
`phase2/spikes/bd3lms-cross-substrate/PRE_REG.md`).

## Rationale (why this spike, why now)

R3's research validation note
(`phase2/research/r3_cross_substrate_priorart.md`, 2026-05-07)
reverse-ranked candidate substrates by cost-of-evidence:

| Target family | Est. compute | Wall-clock |
|---|---:|---:|
| **DiffuLLaMA** (this spike) | ~$15-25 | 3-7 days |
| Dream-7B | ~$30-50 | 7-14 days |
| BD3-LMs (T2.C) | ~$50-80 | 14-21 days |

DiffuLLaMA is an AR→DLM-converted LLaMA-2 7B model (HKU-NLP, arXiv
2410.17891, ICLR-25; checkpoint `diffusionfamily/diffullama` on HF).
Three properties make it the cheapest cross-family entry point:

1. **GSM8K-symbolic LoRA already exists** as `diffusionfamily/diffullama-gsm`
   — no SFT-from-scratch step required (vs. BD3 which ships OWT-only).
2. **PEFT/LoRA infrastructure ships in repo** (`examples/train_lora/llama2_lora_ddm-sft.yaml`,
   `lora_rank: 16`) — no PEFT-from-scratch step required (vs. BD3
   which has zero LoRA infra on its custom DiT).
3. **LLaMA-2 backbone** — likely `transformers.AutoModelForCausalLM`-
   compatible at the architecture level; only the discrete-diffusion
   sampling loop (`inf_diffullama.py`) is custom. WebFetch was silent
   on AutoModel registration explicitly — flagged TODO, not blocker.

Per R3's "Option B" recommendation: this spike + (later) Dream-7B
together give a 3-family generality claim at ~half the cost of BD3
alone.

**Note on size discrepancy:** the dispatching session described
DiffuLLaMA as "GPT-2 1B based". The HF `diffusionfamily/diffullama`
checkpoint is actually **LLaMA-2 7B based** (arXiv 2410.17891 covers
the AR→DLM conversion recipe and ships both). The 1B-class siblings
are `diffugpt-s` (0.1B) and `diffugpt-m` (0.4B) — GPT-2 family,
*sister* models with same conversion recipe but no GSM-tuned LoRA.
This spike defaults to `diffusionfamily/diffullama` (7B) because the
GSM-tuned variant is a much closer apples-to-apples comparison
against LLaDA-8B-Instruct (7B vs 8B, both LLaMA-family, both
math-instruction-tuned). Cost is in the same ~$15-25 envelope.

## Hypothesis

The K2 inverted-U commit-LoRA finding has been replicated:
- **In-domain:** GSM8K (Phase-2 §3, k=3 = 0.822 peak)
- **Cross-domain (same diffusion-LM):** MATH-500 numeric N=50 (T2.A
  WIN, k=3 = 0.58 peak — commit `9c81bfd`)

This spike tests whether the K2 inverted-U **also replicates across
diffusion-LM families**. Specifically: train a fresh 14M-param
commit-LoRA on top of DiffuLLaMA (or use the existing
`diffusionfamily/diffullama-gsm` LoRA as a Track-1 base + add a
Track-2 commit-LoRA on top), repeat the K-sweep, and check whether
(a) commit-LoRA helps on a different DLM family at all, and (b) the
optimal sub-block boundary is in [k=2, k=3] as in LLaDA.

If **WIN** → the commit-LoRA primitive is genuinely a "schedule-toggle
for any block-diffusion LM" rather than a LLaDA-specific calibration.
This unlocks R3's Option-B 3-family generality claim
(LLaDA + DiffuLLaMA + Dream-7B) at ~half the cost of T2.C alone.

## Substrate

- **Model:** `diffusionfamily/diffullama` (LLaMA-2 7B converted to
  discrete-diffusion via DiffuLLaMA recipe).
- **Substrate:** GSM8K-test idx 0..199 first (same as Phase-2 K2
  ablation). If `diffullama-gsm` LoRA proves the model handles the
  domain at baseline cmajc ≥ 0.30, **also** run MATH-500 numeric
  N=50 to mirror the LLaDA T2.A cross-domain test.
- **Adapter:** Fresh 14M commit-LoRA (LoRA rank 16, alpha 32, target
  modules per LLaMA-2 standard `gate_proj/up_proj/down_proj` — full
  module names since DiffuLLaMA wraps stock LLaMA-2 layers, unlike
  LLaDA's custom `ff_proj/up_proj/ff_out` naming). Trained on
  GSM8K-train ~1k examples, recipe forked from LLaDA commit-LoRA v3.
- **Optional Track-1 base:** `diffusionfamily/diffullama-gsm` (already
  released by HKU-NLP as a GSM8K-symbolic-tuned LoRA on
  DiffuLLaMA-base). If used, our Track-2 commit-LoRA stacks on top.
- **K-sweep:** 4 conditions (c2c k=0, cmajc k=2/3/4) × N=50 BATCHED=0
  BRANCHES=5 K_STEPS=64 TEMP=0.7 SEED=0.

## Eng prerequisites (much smaller than BD3 T2.C)

Compared to BD3's six-phase 14-21 day, $40-60 prereq stack
(`phase2/spikes/bd3lms-cross-substrate/PRE_REG.md`), DiffuLLaMA's
ladder is **3 phases, 3-7 days, $15-25**:

| Phase | Eng-days | $ |
|---|---:|---:|
| Phase 1: clone HKUNLP/DiffuLLaMA repo, audit `inf_diffullama.py`, confirm AutoModel compat | 1-2 | $0 |
| Phase 2: port sampler into a programmatic `_generate(prompt_ids, ...)` matching the LLaDA contract; map sub-block boundaries | 1-2 | $0 |
| Phase 3: train commit-LoRA on `diffullama-gsm` base (~1 GPU-day, 48GB) | 1-2 | $10-15 |
| Phase 4: K-sweep dispatch (cmajc k=2/3/4 vs c2c k=0) × N=50 | <1 | $2-3 |
| Buffer (debugging + paper write-up) | 1-2 | $3-7 |
| **Total** | **3-7 days** | **~$15-25** |

Why so much cheaper than BD3:
- **No SFT-from-scratch:** `diffullama-gsm` LoRA already exists.
  BD3 needed a $15-25 SFT step on GSM8K-train.
- **No PEFT-from-scratch:** DiffuLLaMA repo ships LoRA training
  configs for LLaMA-2-style modules. BD3 had zero LoRA infra on
  its custom DiT class.
- **Likely AutoModel-compatible:** LLaMA-2 7B backbone means
  `AutoModelForCausalLM.from_pretrained` *might* just work (TBC in
  Phase 1). BD3's custom DiT was confirmed-not-AutoModel.
- **Hydra-free:** DiffuLLaMA ships a vanilla Python
  `inf_diffullama.py` script. BD3 was Hydra-only.

## Pre-registered decision rules

(Mirrored from BD3 PRE_REG so cross-spike comparisons are clean.)

| Outcome | Verdict |
|---|---|
| cmajc-best ≥ 0.30 baseline AND k-sweep produces inverted-U with peak in [k=2, k=3] | **WIN** — commit-LoRA primitive transfers across DLM families. Combined with LLaDA's existing 2 substrates (GSM8K + MATH-500), this gives a 2-family + 2-domain claim. Add Dream-7B for a 3-family claim per R3 §3 generality bar. |
| cmajc-best ≥ 0.30 AND positive lift but no inverted shape (e.g., monotone, k=4 peak) | **PARTIAL** — commit-LoRA helps on DiffuLLaMA but the boundary calibration is family-specific. §3.5 caveat: K2 toggle is universal, but the K2 *value* needs per-family tuning. |
| Any cmajc < c2c | **LOSS** — commit-LoRA is LLaDA-specific. Paper §3.5 caveat: "validated on LLaDA only; DiffuLLaMA disconfirms cross-family generalization at the K2 setting." This would be a meaningful negative result given DiffuLLaMA is *closest* to LLaDA in the candidate set (both LLaMA-family, both 7-8B). If even DiffuLLaMA fails, BD3 (much greater architectural distance) is unlikely to succeed — the spike would inform a deprioritization of full T2.C. |

**Threshold rationale:** cmajc-best ≥ 0.30 mirrors the LLaDA-MATH-500
baseline (Phase-2 §4, T2.A cmajc-k3 = 0.41). If DiffuLLaMA's GSM8K
baseline can't clear 0.30, the model isn't math-capable enough for a
clean K-sweep signal to surface — re-run with the easier `diffullama-gsm`
LoRA active before declaring LOSS.

## Anti-goals

- No swap of base diffusion-LM mid-spike — pin to
  `diffusionfamily/diffullama` HF rev at commit-time of this PRE_REG.
- No retrain of commit-LoRA from scratch on cross-substrate data
  (defeats the purpose of testing transfer).
- No sub-block resizing within DiffuLLaMA to match LLaDA's 32-token
  blocks — adapt to DiffuLLaMA's native sub-block contract. (If
  DiffuLLaMA's block size or denoising schedule differs, that itself
  is a methodological note for the paper.)
- No silent mixing of `diffullama-gsm` LoRA on/off across K-sweep
  conditions — pin Track-1 base state and document.

## Cost (when this fires)

| Item | $ |
|---|---:|
| Eng days (3-7 days, no GPU during code work) | $0 |
| Commit-LoRA training (~1 GPU-day, 48GB on-demand) | ~$10-15 |
| K-sweep × 4 conditions × N=50 (GSM8K) | ~$2 |
| Optional MATH-500 N=50 K-sweep × 4 | ~$3 |
| Pod overhead + buffer | ~$2 |
| **Total** | **~$15-25** |

vs BD3 T2.C: ~$40-60. Difference ≈ $25-40 saved by skipping the
SFT-from-scratch + PEFT-from-scratch prereqs that DiffuLLaMA already
ships pre-built.

## Files (when this fires)

- `PRE_REG.md` — this file
- `RESULT.md` — to be filled after K-sweep
- `e4/diff_diffullama.py` — DiffuLLaMA adapter (new, scaffold landed
  this commit; ~250 LOC mirroring `e4/diff_bd3.py`)
- `scripts/train_diffullama_commit_lora.py` — commit-LoRA training
  (new, scaffold landed this commit; placeholder pending Phase-1
  AutoModel compat audit)
- `eren23/sfumato-diffullama-commit-v1` — HF artifact (new, future)

## Status notes

**Why scaffold-only:** R3 explicitly recommended Option B
(DiffuLLaMA + Dream-7B parallel) as the cost-optimal next cross-
substrate move. This dispatch lands the scaffold so a future session
can pick up Phase 1 (AutoModel compat audit) without re-doing the
PRE_REG / decision-rules work. The decision rules are locked NOW so
a future-session researcher can't move the goalposts.

**Trigger condition:** start this spike when (a) Direction A
(schedule-RLHF) ships its §3.5 marquee result and the paper still
needs cross-family generality, OR (b) a paper-revision request asks
for ≥2 DLM families, OR (c) a future contributor wants to extend §3
to a 3-substrate sweep.

**Comparison to BD3 T2.C:** if Phase 1 of this spike confirms
AutoModel compatibility, this is the strictly-cheaper path and BD3
T2.C should likely stay scaffolded indefinitely (R3 explicitly
flagged BD3 as the highest-cost lowest-yield option). If Phase 1
finds a custom model class, costs converge with BD3 (+$5-10) and the
choice between substrates becomes about architectural distance from
LLaDA — DiffuLLaMA is *closer*, BD3 is *farther*; for a generality
claim, BD3 is stronger evidence if it WINs but more likely to LOSS.

## References

- DiffuLLaMA: arXiv 2410.17891 (ICLR-25, HKU-NLP / Shansan Gong et al.)
- Repo: https://github.com/HKUNLP/DiffuLLaMA
- HF: `diffusionfamily/diffullama` (LLaMA-2 7B base, 2024-10-25)
- HF: `diffusionfamily/diffullama-gsm` (GSM8K-symbolic LoRA, 2025-02-19)
- HF: `diffusionfamily/diffugpt-{s,m}` (sister GPT-2 0.1B/0.4B variants)
- R3 prior-art note: `phase2/research/r3_cross_substrate_priorart.md`
- Sister BD3 PRE_REG: `phase2/spikes/bd3lms-cross-substrate/PRE_REG.md`
