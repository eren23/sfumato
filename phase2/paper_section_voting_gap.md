# Paper-section draft — Voting-Rule Gap (Phase 2)

This is a Phase-2 paper-section draft suitable for inclusion in `sfumato_paper`
(the sibling LaTeX repo). All numbers come from `phase2/spikes/temperature-diversity-falsifier/`
and `phase2/spikes/verifier-aggregation/` artifacts on this branch. Convert to
LaTeX prose + a single figure when integrating.

---

## §X. The voting-rule gap and two negative results

### X.1 Headline observation

Across all `cmaj` configurations evaluated in this paper, the **oracle ceiling
exceeds majority-vote accuracy by 8–12 percentage points** on GSM8K-dev:

| Configuration | $N$ | $a_b$ (cmaj b=5) | Oracle | Gap |
|---|---:|---:|---:|---:|
| base LLaDA-8B-Instruct, $\tau=0.7$ | 50 | 78.0% | 90.0% | 12.0 pp |
| Track-1-v3 LoRA, $\tau=0.7$ | 200 | 79.5% | 88.0% | 8.5 pp |
| Track-1-v3 + commit-v3, seed=1, $\tau=0.7$ | 100 | 82.0% | 90.0% | 8.0 pp |
| Track-1-v3 + commit-v3, seed=2, $\tau=0.7$ | 100 | 82.0% | 90.0% | 8.0 pp |

In the median problem the right answer **is** present in at least one of the
five sampled branches; majority voting simply discards it. This holds across
the base model, the v3 prefix-robust LoRA stack, and the v3 + commit-v3 stack
— suggesting the gap is a **structural property of cmaj aggregation** rather
than a model-quality artifact addressable by further LoRA training.

### X.2 Multi-seed variance bar (Phase-1 paper-hardening)

To rule out that our Phase-1 headline cmaj-with-commit (cmajc) accuracy of
**82.5%** (seed=0, $N=200$, Track-1-v3 + commit-v3) was a seed-dependent
artifact, we re-ran cmajc at $N=100$ for two additional seeds:

- seed=1: $a_b = 82.0\%$
- seed=2: $a_b = 82.0\%$

Three-seed mean **82.2%, sample standard deviation 0.29 pp** ($\sigma$ tighter
than the W&B Wilson interval at $N=100$). The Phase-1 headline is robust to
seed selection.

### X.3 Two negative results on per-branch verifier aggregation

The 8–12 pp gap immediately suggests training a per-branch verifier that
re-ranks the 5 branches at inference, replacing majority vote with verifier
top-1. We pre-registered two architectures with success threshold $\geq$ 83%
mean accuracy on held-out problems (5-fold CV split by `problem_id` to prevent
leakage). **Both lost.**

#### Architecture 1 — TF-IDF + Logistic Regression (text-only)

Baseline architecture: `sklearn` TF-IDF $1$–$2$-grams over branch text
(`max_features=10\,000`), logistic regression with class-weight balancing,
trained on $1750$ labelled branches (Phase-1 across $\tau \in \{0.3, 0.7,
1.0\}$ plus our Phase-2 substrate at $\tau = 0.7$ with v3 LoRA, $N=200$).
Compute cost: $0$ (CPU, $\sim 30$ s training).

| Setting | cmaj | TF-IDF verifier | $\Delta_{\text{pp}}$ | Oracle |
|---|---:|---:|---:|---:|
| Train Phase-1, eval Phase-2 substrate | 79.0% | 69.5% | **−9.5** | 88.0% |
| 5-fold CV on combined $N=1750$ | 80.5% | 66.5% | **−14.0** | 89.5% |

**Verdict: LOSS.** The verifier picks fluent-but-wrong branches at the cost of
correct-but-poorly-formatted ones — surface text features (length,
math-formatting) do not separate correct from incorrect arithmetic.

#### Architecture 2 — Qwen-encoder + MLP head

Hypothesis: a 0.5B-parameter language-model encoder should provide
arithmetic-aware features that TF-IDF lacks. We use Qwen2.5-0.5B-Instruct as a
feature extractor with the prompt template `"Problem: {q}\n\nSolution:\n{branch_text}"`,
mean-pool the last-layer hidden states (after attention masking), and train a
two-layer MLP head ($896 \to 256 \to 1$) with BCE loss on the same $1750$-branch
dataset. Embedding extraction takes 11 s on a single RTX 4090; per-fold MLP
training is $\sim 100$ s.

| Fold | $N$ (problems) | cmaj | Qwen-encoder verifier | $\Delta_{\text{pp}}$ |
|---:|---:|---:|---:|---:|
| 0 | 40 | 80.0% | 72.5% | $-7.5$ |
| 1 | 40 | 80.0% | 67.5% | $-12.5$ |
| 2 | 40 | 75.0% | 62.5% | $-12.5$ |
| 3 | 40 | 82.5% | 77.5% | $-5.0$ |
| 4 | 40 | 85.0% | 80.0% | $-5.0$ |
| **Mean** | **40** | **80.5%** | **72.0%** | **$-8.5$** |

**Verdict: LOSS** (gap-closure $-94.4\%$, i.e. the verifier moves *away* from
the oracle ceiling rather than toward it).

### X.4 What we learn from the two falsifiers

The voting-rule gap is real and reproducible across LoRA configurations and
seeds, but is **not closable** by per-branch supervised classifiers at this
dataset scale ($N=200$ problems, $1750$ labelled branches) using either
text-only features or 0.5B-parameter encoder embeddings. This is consistent
with the verifier literature: Cobbe et al. \cite{cobbe2110_14168} used a
fine-tuned GPT-3 for their GSM8K verifier; Lightman et al. \cite{lightman2305_20050}
used step-level human supervision (PRM800K) to outperform outcome-reward
classifiers at scale.

### X.5 Future work

Two paths remain open within the voting-gap research direction:

1. **Process-reward verifier** consuming per-step trajectory features
(per-position entropy, commit-LoRA logit shifts, AR/DDLM mechanism source) via
the JSONL trace schema defined in Workstream C of our Phase-2 protocol. This
is option-3 of the D3.5 proposal in `phase2/proposals/verifier-based-aggregation.md`
and is gated on producing 5–10 manual real-mode traces from the visualizer
substrate before training is justified.

2. **Larger encoder** (e.g. Qwen2.5-7B-Instruct or a math-tuned variant). At
the 0.5B scale we appear to be encoder-limited; a 7B encoder costs roughly
$10\times$ more per spike but could plausibly close the gap to $> 85\%$ if the
gap closure is monotonic in encoder capacity.

Both are deferred to Phase 3, beyond this paper's scope. We document the gap
and the two negative results here to flag the direction as productive for
follow-up rather than closed.

### X.6 Reproduction

All four cmaj/cmajc runs in §X.1, the multi-seed sweep in §X.2, and both
verifier negative results in §X.3 are reproducible via:

```bash
# Generate substrate (one-time, ~60 min on A6000 spot, ~$0.30):
crucible run experiment --preset screen \
  --set CONDITION=cmaj,N_PROBLEMS=200,BRANCHES=5,TEMP=0.7,SEED=0,\
LORA_PATH=eren23/sfumato-llada-prefix-robust-v3

# Multi-seed cmajc (~30 min each):
for SEED in 1 2; do
  crucible run experiment --preset screen --set CONDITION=cmajc,SEED=$SEED,...
done

# Verifier negative results (CPU + GPU, ~5 min):
python phase2/spikes/verifier-aggregation/train_verifier.py
python phase2/spikes/verifier-aggregation/train_verifier_option2.py
```

Phase-2 cumulative compute spend through these results: **\$2.27** on RunPod
spot RTX 4090 / A6000.
