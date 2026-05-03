# Pre-Registration — Temperature Diversity Falsifier (Spike for D3)

**Author:** Workstream B subagent | **Pre-registered at:** 2026-05-02 ~14:55 UTC
**Spike for proposal:** D3 — diversity-as-objective fine-tuning (`phase2/proposals/diversity-as-objective.md`)
**Budget cap:** $5.00 (target ≤ $2.00)

## Hypothesis under test

Branch diversity, as a function of sampling temperature τ, is a causally
meaningful predictor of cmaj b=5 accuracy on GSM8K-dev with the existing
`eren23/sfumato-prefix-robust-gsm8k-v3` adapter. Specifically: there exists
a τ where mean branch-agreement rate `p_maj(τ)` is meaningfully *lower*
than at τ=0.7 (Phase-1 default) **without** dropping per-branch single-shot
accuracy by more than the diversity gain on cmaj.

If true → diversity is a free lever; D3's training-time diversity reward
is worth a $5+ Phase-2 follow-up.
If false → diversity is bound to per-branch correctness; D3 dies cheaply
and we publish the negative result.

## Procedure (committed before any run)

- **Adapter:** `eren23/sfumato-prefix-robust-gsm8k-v3` (HF hub, public, commit hash unspecified — use latest at run time).
- **Base model:** `GSAI-ML/LLaDA-8B-Instruct`.
- **Dataset:** `e4/data/gsm8k_dev_200.json` (frozen Phase-1 dev set).
- **N = 20 problems**, indices 0..19 (deterministic via `e4/runner.py`'s `load_problems` slicing on `N_PROBLEMS=20`).
- **k = 64 LLaDA steps**.
- **Branches per problem b = 5**.
- **Temperature sweep:** τ ∈ {0.5, 0.7, 1.0, 1.3} → 4 runs.
- **CONDITION = `cmaj`** (single-LoRA, no commit-LoRA, branch-then-vote).
- **Seed:** 0 for all 4 runs.

## Metrics committed (computed from Phase-1 raw_cmaj jsonl)

For each τ:
1. `bar_a1(τ)` — mean per-branch single-shot accuracy = sum(branch_correct) / (N × b)
2. `a_b(τ)` — cmaj b=5 accuracy = fraction of problems where `mode(branch_answers) == gold`
3. `bar_p_maj(τ)` — mean of per-problem `max(count(answer)/b)`
4. `oracle_ceiling(τ)` — fraction of problems where ≥1 branch hit gold
5. 95% Clopper-Pearson CI on `a_b(τ)` and `oracle_ceiling(τ)` via `scripts/binom_ci.py`

## Decision rules (pre-committed)

| Outcome | Decision |
|---|---|
| ∃ τ ≠ 0.7 with `a_b(τ) - a_b(0.7) ≥ +1.5pp` AND `bar_p_maj(τ) - bar_p_maj(0.7) ≤ -0.05` | **WIN** — D3 graduates to Phase-2 training experiment. |
| `a_b(τ)` monotone decreasing in τ across {0.5, 0.7, 1.0, 1.3} AND `oracle_ceiling(τ)` flat-or-decreasing | **LOSS** — D3 falsified cheaply, publish negative result, do NOT graduate. |
| 95% CIs on `a_b(τ)` overlap across all τ | **INCONCLUSIVE** — recommend N=200 graduating run only if D3 still ranks #1 in `RANKING.md`. |
| Anything else | Inconclusive; written up as "needs higher N." |

## Auxiliary diagnostic (not gating)

Compute `oracle_ceiling - a_b` per τ (the "diversity gap" — how much cmaj
leaves on the table when at least one branch had the right answer). If
this gap **widens** with τ even if `a_b` does not improve, that's
suggestive evidence the cmaj voting rule is suboptimal at high τ — would
motivate a separate proposal on better aggregation.

## Crucible dispatch payload

```python
mcp__crucible-fleet__run_project(
    project_name="sfumato_e4",
    overrides={
        "CONDITION": "cmaj",
        "K_STEPS": "64",
        "N_PROBLEMS": "20",
        "BRANCHES": "5",
        "TEMP": "<sweep value>",  # 0.5, 0.7, 1.0, 1.3
        "SEED": "0",
        "HF_LORA_REPO": "eren23/sfumato-prefix-robust-gsm8k-v3",
        "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",  # not used in cmaj path but spec requires it
        "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
        "MOCK_MODELS": "0",
        "WANDB_RUN_NAME": "spike-tau-<sweep value>",
    },
)
```

4 dispatches × ~10 min/run on RTX 4090 spot ≈ 40 min wallclock × 1 GPU × $0.34/h ≈ **$0.23 estimated**.
Buffer for provision/bootstrap overhead: $1.00. **Total cap: $2.00.**

## Pre-reg integrity statement

I, the Workstream B subagent, commit to publishing the spike result in
`RESULT.md` regardless of which decision rule fires, and will NOT alter
the win/loss/inconclusive thresholds above after seeing any spike data.
Hash of this file at commit time goes into `RESULT.md` as audit trail.
