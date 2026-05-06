# Pre-registration — Frontier-vs-sfumato GSM8K compare (Phase A pilot)

**Phase:** Phase 2 follow-up / showcase enrichment
**Spike ID:** frontier-compare-A
**Author:** background research session
**Date:** 2026-05-05
**Plan source:** `~/.claude/plans/bro-bro-bro-bro-prancy-volcano.md`

## Hypothesis

Frontier API models, asked to solve GSM8K-test problems zero-shot CoT through
OpenRouter with a strict `#### N` final-answer marker, produce parseable
final-answer markers on **≥96%** of attempts and achieve aggregate accuracy
within published bounds:

- GPT-4o (`openai/gpt-4o`): ≥0.85
- Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`): ≥0.88
- Gemini 2.5 Pro (`google/gemini-2.5-pro`): ≥0.80

The pilot uses the first **N=50** problems of the HuggingFace `gsm8k`/`main`/
`test` split (idx 0..49) — the same indices already covered by the sfumato
cmajc-v3 N=100 substrate (`e4/results/raw_cmajc_k64_seed1_b5_v3LoRA_N100.jsonl`),
enabling per-problem joins.

## Conditions tested

For each model in `{gpt-4o, claude-sonnet-4.5, gemini-2.5-pro}`:

- **Single chat completion** per problem, `temperature=0.0`, `max_tokens=1024`.
- **Prompt** (zero-shot CoT, asking for `#### N` terminator):

  > Solve the following grade-school math problem. Show your reasoning step
  > by step. End your response with the final numeric answer in exactly this
  > format on its own line: `#### N` (replace N with the number).
  >
  > Problem: {question}

- **Extractor:** `e4.grade.extract_final_answer` (strict `#### N` / `Answer: N`),
  with a fallback to `e4.grade.extract_answer` (last-numeric-token) only if the
  strict extractor returns empty. The fallback is *additive* — a successful
  strict parse always wins; we just don't want to throw away a good answer
  that emitted "The answer is 42" instead of "#### 42".
- **Concurrency:** asyncio semaphore = 8 across requests.
- **Retry:** up to 2 attempts per (idx, model) on transient errors (HTTP non-200,
  timeout, connection error). After 2 failures, mark `pred=null`,
  `correct=null`, `error=<reason>`.

## Success / kill criteria

| Metric (per model) | WIN | LOSS | INCONCLUSIVE |
|---|---|---|---|
| Parsing rate (rows with non-null `pred`) | ≥96% | <90% | [90%, 96%) |
| GPT-4o accuracy | ≥0.85 | <0.425 | [0.425, 0.85) |
| Claude-4.5 accuracy | ≥0.88 | <0.44 | [0.44, 0.88) |
| Gemini-2.5 accuracy | ≥0.80 | <0.40 | [0.40, 0.80) |

- **WIN (overall):** all three models hit parsing≥96% AND accuracy≥published
  bound → proceed to N=200 scale (Phase B), pending human go-ahead.
- **LOSS:** any model parsing rate <90% OR accuracy below half its lower bound
  → fix prompt/extractor, re-run pilot, do not scale.
- **PARTIAL:** anything in between (parsing 90–96% or accuracy below bound but
  ≥half-bound) → escalate to human; record verdict; default to no-scale.

## Measurement plan

1. **Substrate:** HF `gsm8k`/`main`/`test` idx 0..49.
2. **Per-row JSONL** under `phase2/frontier_compare/results_50/{tag}.jsonl`
   with fields `{idx, question, gold, pred, correct, model, raw_response,
   latency_ms, attempts, usage, cost_estimate_usd, error}`. Idempotent: re-runs
   skip rows already present for a given `idx`.
3. **Aggregates** computed in `PILOT_REPORT.md`: parsing rate, accuracy, mean
   latency_ms, total `cost_estimate_usd`.
4. **Sfumato comparator:** accuracy on the same idx 0..49 from
   `e4/results/raw_cmajc_k64_seed1_b5_v3LoRA_N100.jsonl`.
5. **Per-problem WIN/LOSS counts:** how many of 50 sfumato-correct where ALL 3
   frontier missed; how many all-3-frontier-correct where sfumato missed; how
   many unanimous (all 4 correct OR all 4 wrong).

## Compute / cost envelope

- 50 problems × 3 models = 150 requests.
- Approx 200 input tokens + 600 completion tokens per request → **<$0.30**
  total estimated, with a hard cap at **$0.50** in the runner; runner stops if
  the running estimate exceeds the cap.

## Falsification artifacts

- `phase2/frontier_compare/results_50/gpt4o.jsonl`
- `phase2/frontier_compare/results_50/claude45.jsonl`
- `phase2/frontier_compare/results_50/gemini25.jsonl`
- `phase2/frontier_compare/PILOT_REPORT.md`

## Notes / scope guards

- **No GPU work touched.** This is API-only.
- **No mutations** to `e4/runner.py`, `e4/diff_llada.py`, the existing
  showcase, or `examples.json`. Only `e4/grade.py` is *read* (re-imported as
  `e4.grade.extract_final_answer`).
- **No auto-scale.** Even on WIN, the pilot ends with "Awaiting human
  go-ahead for Phase B (N=200)."
