# G — trace feature sanity check (n=7 deterministic traces)

## Verdict

**`signal_plausible`** — based on decision rule: any feature has |Spearman ρ| ≥ 0.5 AND permutation p ≤ 0.1 on n=7 traces (10000 perms).

### Hits

- `min_entropy` vs `trace_correct`: ρ=-0.791, perm-p=0.0971

## Sample

n = 7 deterministic Workstream-C traces, one per problem idx in {10, 20, 30, 40, 50, 60, 70}, each containing 4 sub-block records.

- **trace_correct** label: extracted from `make_real_traces_summary.json`'s `final_text_tail` via `Answer:\s*N` regex, compared to GSM8K-test gold. 5/7 correct.
- **cmaj_correct** label: the parent cmaj-b=5 vote's correctness from `raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`. 6/7 correct.

The two label axes differ because the traces use various directive sequences (`all_llada`, `ar_handoff`, `cmaj_branch`, `mid_ar_handoff`, `early_cmaj`) — they are deterministic single-branch runs, not standard cmaj-b=5 generations.

## Top features (by |ρ| vs trace_correct)

| feature | ρ | perm-p | mean(correct) | mean(incorrect) |
|---------|---|--------|---------------|-----------------|
| `min_entropy` | -0.7906 | 0.0971 | 0.00031 | 0.0038 |
| `sb0_entropy_mean` | -0.6325 | 0.1981 | 0.74741 | 1.29856 |
| `sb3_entropy_mean` | -0.6325 | 0.1972 | 0.07127 | 0.13642 |
| `sb0_entropy_min` | -0.6325 | 0.1998 | 0.01915 | 0.05108 |
| `sb0_topk_entropy_mean` | -0.6325 | 0.1962 | 0.45179 | 0.70612 |
| `mean_entropy` | -0.4743 | 0.3862 | 0.4875 | 0.66898 |
| `max_entropy` | -0.4743 | 0.3907 | 2.93246 | 3.41311 |
| `std_entropy` | -0.4743 | 0.3907 | 0.64498 | 0.83348 |

## Per-trace details

| problem_idx | tag | trace_answer | gold | trace_correct | cmaj_correct |
|-------------|-----|--------------|------|---------------|--------------|
| p10 | all_llada | 366 | 366 | 1 | 1 |
| p20 | all_llada | 18 | 15 | 0 | 0 |
| p30 | ar_handoff | 109 | 109 | 1 | 1 |
| p40 | cmaj_branch | 8 | 8 | 1 | 1 |
| p50 | mid_ar_handoff | 294 | 294 | 1 | 1 |
| p60 | all_llada | 17 | 17 | 1 | 1 |
| p70 | early_cmaj | 74250 | 7425 | 0 | 1 |

## Caveats (must read before acting on the verdict)

- **n=7 is too small for any rigorous statistical claim.** This is a hint-of-signal probe, not an evaluation. A `no_signal` result here may be type-II error (false negative).
- **Existing traces are deterministic replays** with pre-planned directive sequences, not natural cmaj branches. Feature distributions may not represent what E4 would actually see at full scale.
- **Heterogeneous trace tags.** The 7 traces span 5 different directive configurations. Cross-trace feature comparisons mix conditions, which inflates feature variance and reduces correlation power.
- **Mechanism diversity is near-degenerate.** Most traces are predominantly `llada` mechanism; the `mechanism_diversity` feature has very little dynamic range at this small n.
- **logit_shift_norm is null** in these traces (commit-LoRA was not active for any of the 7). The richest single feature from the STATUS schema is therefore unmeasured here. A follow-up should re-generate a few traces with `LOGIT_SHIFT_NORM=1` and re-run G.

## Interpretation alongside E2

E2 says ≥91% of cmaj-disagreement problems first diverge at sub-block 0 (per `e4/results/cmaj_divergence/summary.md`). Combined with this G result:

- If `signal_plausible` == `no_signal`: **Cell 4** of the fork — both legs of per-sub-block routing fail. Strong stop signal for Direction A's current per-sub-block form. Move to discussion.tex's other Phase-3 paths or finalize paper 2 (B) and D.
- If `signal_plausible` == `signal_plausible`: **Cell 3** — features carry information but the decision point is at sub-block 0 (not mid-trajectory). Per-sub-block routing collapses to "pilot-and-route after sub-block 0". Direction A needs redesign.

In either branch, **E1/E3/E4/E5 in their original forms are not the right next step**. D (comprehension scaffolding) and B (paper 2 write-up) remain the highest-leverage remaining items.

## Files

- `univariate_table.csv` — full per-feature stats
- `summary.md` — this document
- `../../scripts/trace_feature_sanity.py` — script

## Follow-up

To strengthen G's `no_signal` result before fully deferring E4:

1. Re-generate 20–30 deterministic traces using `make_real_traces.py` with **mixed directive sequences** and **`LOGIT_SHIFT_NORM=1`** enabled. Cost ~$1 on a 4090 spot.
2. Re-run this script with the larger sample; permutation power tightens by ~3×.
3. If still no signal, treat E4 as deferred indefinitely.
