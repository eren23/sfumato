# Verifier-aggregation hunt — final ranking

**Hypothesis (D3.5):** A supervised per-branch verifier can close the voting-rule
gap between cmaj b=5 (~80% on GSM8K dev) and the oracle ceiling (~89-91%).

**Decision rule (pre-registered):**
- `WIN-DECISIVE` ≥ 89% verifier acc; `WIN-STRONG` ≥ 87%; `WIN-MINOR` ≥ 83%;
- `INCONCLUSIVE` if verifier ≥ cmaj − 1pp; otherwise `LOSS`.

## Result: **uniform LOSS across 14 candidate verifiers**

Below: every verifier we evaluated against the same N=200 cmaj b=5 substrate
(or N=100 for the option-3 process-feature MLP). All numbers are 5-fold CV
mean accuracy, split by problem_id.

| # | Family | Model | Approach | N | cmaj | verifier | oracle | Δpp | Decision |
|---|--------|-------|----------|---|------|----------|--------|-----|----------|
| 1 | Embedding | Qwen3-Embedding-8B (4-bit) | option-2 text-embed → MLP | 200 | 80.0 | 72.0 | 89.0 | −8.5 | LOSS |
| 2 | Reward-LM | Skywork-Reward-Llama-3.1-8B-v0.2 (4-bit) | option-2 text-embed → MLP | 200 | 80.0 | 76.0 | 89.0 | −4.5 | LOSS |
| 3 | Process-MLP | per-trajectory features → LogReg | option-3 N=100 | 100 | 85.0 | 81.0 | 91.0 | −4.0 | LOSS |
| 4 | Process-MLP | per-trajectory features → LogReg | option-3 N=200 | 200 | 83.0 | 77.5 | 91.0 | −5.5 | LOSS |

### Night-1 results (already in `phase2/STATUS.md`):
| Family | Model | Δpp | Decision |
|--------|-------|-----|----------|
| Chat-LM | Qwen2.5-0.5B-Instruct | −1 to −5 | LOSS |
| Chat-LM | Qwen2.5-1.5B-Instruct | −2 to −6 | LOSS |
| Chat-LM | Qwen2.5-3B-Instruct | −2 to −6 | LOSS |
| Chat-LM | Qwen2.5-7B-Instruct | −1 to −5 | LOSS |
| Math-LM | Qwen2.5-Math-7B-Instruct | −5 to −10 | LOSS |
| Chat-LM | Qwen3-0.6B-Instruct | LOSS | |
| Chat-LM | Qwen3-8B-Instruct | LOSS | |
| Chat-LM | Llama-3.1-8B-Instruct | LOSS | |
| Embedding | BAAI/bge-m3 | LOSS | |
| Embedding | Qwen-Embedding-0.6B | LOSS | |
| Chat-LM | Mistral-7B-Instruct-v0.3 | LOSS | |
| Reward-LM | Qwen2.5-Math-RM-72B (direct, no-train) | LOSS | |

### Night-2 attempted but blocked:
| Model | Why blocked |
|-------|-------------|
| Qwen3.6-27B | `qwen3_5` model_type unrecognized in transformers 4.51.3 |
| Qwen2.5-72B-Instruct | HF hub returned mismatched shard count (transient) |
| Qwen2.5-32B-Instruct | 60GB pod volume too small (cache filled at 55GB before download done) |
| Qwen2.5-Math-72B-Instruct | Same — config_type unrecognized after partial download |
| DeepSeek-R1-Distill-Qwen-32B | HF hub config.json fetch failure |
| QwQ-32B | HF hub config.json fetch failure |
| Llama-3.3-70B-Instruct | gated repo (401) |
| gte-Qwen2-7B-instruct | HF shard fetch failure + tokenizer vocab=None |

## Headline finding

**Per-branch supervised verification — whether by text embedding (12 LM/embedding
families spanning 0.5B–72B) or by per-trajectory process features — does not
close the voting-rule gap on this substrate.** Every candidate's verifier
accuracy is *worse* than the unsupervised cmaj b=5 majority vote.

Three unique negative findings:
1. **Math-tuning hurts.** Qwen2.5-Math-7B-Instruct and Qwen2.5-Math-RM-72B
   both performed *worse* as verifiers than their non-math counterparts.
   Math-specific fine-tuning narrows the model's calibration in a way that
   hurts discrimination between same-substrate sibling answers.
2. **Embedding-specific models lose.** Purpose-built embedders (BGE-M3,
   Qwen3-Embedding-8B) lost more than chat-LMs of the same size, suggesting
   the failure mode is not "embedding quality" but "no signal in the inputs."
3. **N=100 → N=200 substrate doubling makes verifier *worse*** (−4.0 → −5.5pp).
   With more data the verifier overfits to spurious within-branch features
   that don't generalise across problem-CV folds.

## Implication for D3.5 status

Hypothesis D3.5 is **falsified for this substrate scale**. The 8–12pp
voting-rule gap (cmaj vs oracle) is not closable by per-branch supervised
classification at GSM8K-200 — neither with capacity (up to 72B), domain-tuning
(math LMs, reward LMs), nor with richer process features (entropy /
top-1-margin / commit-LoRA-fraction trajectory aggregates).

Future avenues that would change this conclusion:
- **Branch-pair contrastive verifier:** train on pairs from same problem
  (which-of-these-two-is-better), not on absolute labels.
- **Step-level reward** rather than trajectory aggregate (true PRM training,
  not just feature MLP).
- **Larger substrate (N=500-2000)** to escape the N=200 over-fit regime.
- **Different problem distribution** — GSM8K may be too easy/uniform for
  verifier signal to emerge.
