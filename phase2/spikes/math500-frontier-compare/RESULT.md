# MATH-500 Cross-Domain Frontier Compare — RESULT

**Date:** 2026-05-07. **Substrate:** MATH-500 numeric subset idx 0..49 (N=50).
**Cost:** ~$3.30 OpenRouter (10 models × 50 problems, no GPU).
**Outcome:** **WIN-honest** — sfumato cmajc-k3 sits **above the weak
peer 7B but below mid/strong peer 7-8B chat models** on cross-domain
math. Materially different from GSM8K story where sfumato unique-WIN'd
over peer-class. Honest data for §3.5.

---

## Headline numbers

| Tier | Model | MATH-500 numeric N=50 acc |
|---|---|---:|
| Frontier | Claude Sonnet 4.5 | **0.900** |
| Frontier | GPT-4o | 0.840 |
| Frontier | Gemini 2.5 Pro | 0.820 |
| OSS large 70B+ | Qwen 2.5 72B | 0.840 |
| OSS large 70B+ | DeepSeek Chat | 0.820 |
| OSS large 70B+ | Llama 3.3 70B | 0.760 |
| OSS mid 30B | Qwen3 30B-A3B | 0.740 |
| OSS peer 7-8B | Qwen 2.5 7B | **0.800** |
| OSS peer 7-8B | Llama 3.1 8B | 0.620 |
| OSS peer 7-8B | Mistral 7B v0.1 | 0.160 |
| **Sfumato** | cmajc-k3 ≈7B (this work) | **0.580** |

## Cross-domain narrative

**GSM8K (Phase-2 Showcase Phase C):** sfumato cmajc N=200 = 0.83;
peer-class 7-8B chat models in 0.62-0.85 range; sfumato unique-WIN
on 3 of 50 problems where NO peer model got it right.

**MATH-500 numeric (this run):** sfumato cmajc-k3 N=50 = 0.58; peer-class
7-8B in 0.16-0.80 range; sfumato BELOW Qwen 2.5 7B (0.80) and Llama 3.1
8B (0.62), only above Mistral 7B v0.1 (0.16).

**Honest read:** sfumato's GSM8K-trained adapters do not fully transfer
the peer-class advantage to MATH-500. The cross-domain K2 lift (+8.5pp
over c2c-k0 at N=200) is real and replicates, BUT the absolute
ranking against peer-class chat models flips: sfumato wins on its
training distribution (GSM8K) but loses to 2/3 strong peer 7-8B chat
on the harder cross-domain MATH numeric subset.

## What this kills, what it leaves open

**Killed:**
- Any framing that says "sfumato beats peer-class at peer scale" —
  TRUE on GSM8K, FALSE on MATH-500. Paper §3.5 must qualify the claim
  to in-domain only.

**Open (Phase-4 directions):**
- Direction A (Schedule-RLHF): could close the gap by training the
  adapter against MATH-train + GSM8K mixture. Pre-reg threshold
  was "+10pp on MATH-500 cmajc-k3" which would push sfumato to 0.51
  N=200 (still below Qwen 2.5 7B 0.80 but more competitive).
- Direction C (Base swap): if LLaDA-1.5 / Dream-7B / MMaDA closes the
  gap from architecture alone, the §3.5 framing becomes "K2 toggle
  composes with stronger DLM bases."

## Files

- `RESULT.md` — this file
- `phase2/frontier_compare/results_50_math500/{claude45,gpt4o,gemini25,deepseek_chat,qwen25_72b,llama33_70b,qwen3_30b_a3b,qwen25_7b,llama31_8b,mistral_7b_v01}.jsonl` — per-model JSONLs (50 rows each)
- 10 wandb tags: `math500-{model_tag}` not applicable (OpenRouter, not wandb)
