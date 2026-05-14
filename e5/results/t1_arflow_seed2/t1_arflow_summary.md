# T1 — toy AR + continuous-flow composite, seed=2

## Headline

| | accuracy | n_params |
|--|---|---|
| **composite (paired)** | **6.0%** | 66.3M |
| **B3 paired_separate** | **2.0%** | 73.6M |
| **Δ** | **+4.00pp** | |

Verdict: **within_noise_positive**

## All modes

| variant | mode | acc |
|---|---|---|
| composite | ar | 6.0% |
| composite | paired (self) | 6.0% |
| ar_only | ar | 2.0% |
| flow_only | flow | 0.0% |
| paired_separate | paired (cross) | 2.0% |

## Caveats

- 60M-class model on ~4M GSM8K tokens, 3k steps, single seed.
- Continuous-flow decoding via cosine-NN lookup (weaker than ELF's shared-weight decoder).
- ODE: Euler with 32 steps. Higher-order or more steps may lift acc.
