# E2 — cmaj branch divergence histogram

## Verdict

**Outcome band: `sb0_geq80`** — ≥80% of cmaj-disagreement problems first diverge at sub-block 0. The per-sub-block routing thesis from `paper/sections/discussion.tex:239-243` is **empirically weakened** by this substrate.

## Numbers (N=108 disagreement problems / 200 total, cmaj-b=5, GSM8K-test)

Token-level first-divergence (edit-distance ≥3 on 32-token sub-block slices):

| sub-block      | %    |
|----------------|------|
| sb 0           | 91.7 |
| sb 1           |  8.3 |
| sb 2           |  0.0 |
| sb 3           |  0.0 |
| no divergence  |  0.0 |
| **≥ sb 1**     |  **8.3** |

Semantic cross-check (TF-IDF cosine ≤ 0.7 on decoded sub-block text):

| sub-block      | %    |
|----------------|------|
| sb 0           | 88.9 |
| sb 1           |  9.3 |
| sb 2           |  0.9 |
| sb 3           |  0.0 |
| no divergence  |  0.9 |
| **≥ sb 1**     | **10.2** |

**Token–semantic agreement rate: 95.4%.** The token-level result is not a tokenization artifact; the semantic check confirms the same story.

## Interpretation

The 5 cmaj branches share only the prompt prefix. Once LLaDA starts generating (sub-block 0, positions 0–32), branches immediately diverge in ≥91% of problems where the final vote disagrees. There is essentially **no mid-trajectory decision point** for a per-sub-block router to operate on.

This places us in **Cell 3 or Cell 4** of the post-E2+G fork:

- **Cell 3** (G says trajectory features carry signal): per-sub-block routing collapses but features extracted *after* a probe sub-block 0 may still be useful. Implies a "pilot-and-route" architecture (one decision after sub-block 0, not at every boundary). Pause Direction A in its current per-sub-block form; consider redesign.
- **Cell 4** (G says no feature signal): both legs of the per-sub-block routing thesis fail. Strong signal to stop Direction A. Move to discussion.tex's other Phase-3 paths (32B+ encoder, PRM800K-style step supervision) or accept the negative.

## Direct implications

1. **Direction A's premise is weakened.** GRPO-trained per-sub-block policy is learning over an action space where the actions don't have distinguishable conditioning information (branches already differ at sub-block 0; sub-block 1+ are mostly downstream extensions of an already-committed choice).
2. **E4 conditional value drops.** Even if trajectory features carry signal at small N (G's verdict pending), they likely carry it at sub-block 0 only — the same level where the decision is already made by sampling stochasticity.
3. **E1 (re-diffuse-last-K) likely dead.** Re-diffusing sub-blocks 1–3 conditioned on a wrong sub-block 0 will tend to converge to the same wrong setup. Backtracking would have to target sub-block 0 itself, which is just re-sampling — and that's what cmaj b=5 already does.
4. **E5 (AR-extend mid-gen) likely dead.** Switching to AR after sub-block 0 propagates the committed wrong setup. The action only helps if there's a mid-trajectory decision point to re-route, which E2 says there isn't.

## What this finding strengthens

- The paper-1 framing that the failure is **upstream of generation** (problem-comprehension, axis-2 planner-content trust) — this is exactly what E2 shows mechanistically. The setup is determined at sub-block 0; if the setup is wrong, the rest of the trajectory propagates the error.
- The case for **D (comprehension-scaffolding test)**: if scaffolding the setup (Sonnet-extracted) lifts accuracy meaningfully, you have a deployable recipe that targets the actual upstream bottleneck.
- The case for **B (paper 2 write-up)**: the voting-gap negative result (no trained verifier beats majority) maps cleanly onto E2's finding (branches diverge at sub-block 0, so there's no trajectory information for a verifier to use beyond what's already in the decoded text — and a frontier judge with comprehension is exactly the right tool).

## Robustness

- The disagreement subset is 54% of N=200 (108 problems) — large enough for clean histogram.
- 95.4% token–semantic agreement makes the result robust to the choice of divergence metric.
- Sample stratification: of the 108 disagreement problems, cmaj is correct on 69 (63.9%) and incorrect on 39. Both strata show the same sub-block-0-dominant pattern (see histogram.png for breakdown).

## Files

- `histogram.png`, `histogram.pdf` — the headline figure
- `summary.json` — machine-readable summary
- `per_problem.jsonl` — per-disagreement-problem details (idx, votes, branch lengths, first-div sub-block, semantic first-div sub-block)
- `../../scripts/cmaj_divergence_histogram.py` — script

## Caveats

- Edit-distance threshold 3 (token IDs) is sensitive to tokenizer quirks at low values. Threshold 5 or 7 would likely give similar sb-0-dominant results; threshold 0 (any token differs) would be trivially sb 0.
- TF-IDF cosine threshold 0.7 is somewhat arbitrary but agrees with token-level at 95.4%, so the band classification is stable to reasonable threshold choices.
- The substrate is a *single* cmaj batch (seed=0). Multi-seed would tighten the band-classification but not change the headline (≥91% sub-block-0 is far enough from any plausible noise band).
