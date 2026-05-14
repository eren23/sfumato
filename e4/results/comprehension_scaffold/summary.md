# D — comprehension scaffolding test

## Setup

- Base: `GSAI-ML/LLaDA-8B-Instruct`
- LoRA: `eren23/sfumato-llada-prefix-robust-v3`
- Commit-LoRA: `eren23/sfumato-llada-commit-v3`
- N=100, k_steps=64, T=0.0, commit_n_blocks=3
- Wall: 1446s

## Headline

| variant | n | correct | accuracy |
|---|---|---|---|
| `scaffold` | 100 | 84 | 84.0% |
| `baseline` | 100 | 75 | 75.0% |

**Δ (scaffold − baseline): +9.00pp**

## Reference numbers (from paper 1)

- LLaDA c2 (single branch, no LoRAs, raw Q): 74%
- cmajc-v3 (b=5, both LoRAs, raw Q): 82.5%
- Qwen-2.5-7B AR baseline: 86.5%

## Interpretation

- Scaffold lifts by +9.0pp (75.0% → 84.0%). Claim partially supported — meaningful comprehension component but not the whole story.

## Files

- `sonnet_setups.jsonl`, `sonnet_meta.json` — step 1 output
- `llada_outputs.jsonl` — per-problem raw
- `summary.json` — machine-readable
- `summary.md` — this document
