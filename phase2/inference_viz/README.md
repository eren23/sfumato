# Sfumato — step-by-step inference visualizer

Workstream C, Phase 2. Watch the LLaDA + commit-LoRA + AR-extend system
generate token-by-token, with the ability to see at each step which
mechanism is producing which tokens, and to manually intervene at
sub-block boundaries.

This is partly research instrumentation and partly the substrate for the
**E1 mode router** that Workstream B's `phase2/proposals/D1_*` cites: the
JSONL traces this app emits are the training data for the router's
offline-replay bandit.

## What you get

- A **4×32 token grid** showing the four LLaDA sub-blocks of `gen_length=128`,
  one cell per committed token, color-coded by per-position Shannon entropy
  (cool blue = confident, warm amber = uncertain, matching the
  Workstream A palette).
- **Hover any cell** for token id, decoded string, entropy, top-5
  alternatives, commit-LoRA on/off, and which mechanism produced it
  (`llada` / `ar_extend` / `cmaj_branch`).
- **Manual-mode buttons** at every sub-block boundary: Continue LLaDA,
  AR-extend (slider for N tokens to graft), Cmaj branch (slider for b),
  or Stop.
- A live trace pane on the right.
- A "Save trace" button that writes the current session's JSONL to
  `phase2/inference_viz/traces/`.

## Run it locally (no GPU needed)

```bash
# one-shot launcher — spawns FastAPI server + Gradio app
MOCK_MODELS=1 python3 phase2/inference_viz/launch.py --problem-idx 42
```

That starts the server on `127.0.0.1:8765` and the Gradio app on
`127.0.0.1:7860`. Open the Gradio URL in your browser, click **Start
session**, then drive the generation one sub-block at a time with the
directive buttons.

The mock backend exercises the full HTTP / queue / callback / trace
pipeline using deterministic synthetic states, so you can demo the UI
flow exactly the way the real model would behave — minus the actual
LLaDA forward passes.

### Required Python packages

- `fastapi`, `uvicorn`, `pydantic` (server)
- `gradio`, `requests` (frontend)
- `torch`, `transformers`, `peft` (only for **real** mode — not needed for `MOCK_MODELS=1`)

```bash
pip install fastapi uvicorn pydantic gradio requests
```

## Run it against a Crucible-provisioned RunPod

```bash
# 1) On the pod, run the server with the v3 LoRAs.
#    Crucible orchestration: mcp__crucible-fleet__run_project("sfumato_e4", overrides={
#        "command": "python3 phase2/inference_viz/server.py --host 0.0.0.0 --port 8765",
#        "env": {
#            "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
#            "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
#            "COMMIT_N_BLOCKS": "3",
#            "MOCK_MODELS": "0",
#        }
#    })

# 2) From your laptop, SSH-tunnel the server port:
ssh -L 8765:localhost:8765 user@pod.ip

# 3) Launch the local Gradio frontend pointing at the tunnel:
python3 phase2/inference_viz/launch.py --remote --backend http://127.0.0.1:8765 --problem-idx 42
```

Budget: ≤ $2 for development across all sessions (RTX 4090 spot at
$0.20/hr → ~10 hours total). See `phase2/COST_LEDGER.md`.

## Architecture

```
┌──────────────────┐  HTTP    ┌────────────────────────┐  callback   ┌────────────────────┐
│  Gradio (local)  │ ───────▶ │  FastAPI server (pod)  │ ──────────▶ │  e4/diff_llada     │
│  app.py          │ ◀─────── │  server.py             │ ◀────────── │  _Real._generate   │
└──────────────────┘  JSONL   └────────────────────────┘  StepState  └────────────────────┘
                                       │                                      ▲
                                       │ ar_qwen.extend_cot                  │
                                       ▼                                      │
                              ┌────────────────────────┐                      │
                              │  e4/ar_qwen._Real      │  graft tokens into   │
                              │  Qwen2.5-Instruct      │  x_handle ────────── ┘
                              └────────────────────────┘
```

The LLaDA sampler runs in a worker thread inside the FastAPI process.
Two `queue.Queue` objects per session pump `StepState` snapshots out
(`state_q`) and `StepDirective` decisions in (`directive_q`). The HTTP
`/step` handler is a long-poll: it forwards the client's directive then
blocks on `state_q.get()` for the next sub-block boundary.

The `step_callback` hook in `_generate` is the only modification to
`e4/diff_llada.py`. **Default is None / no-op**, so all existing
`runner.py` callers (c1..c4, c2c, cmaj, cmajc, ...) stay bit-identical.
A regression test (`test_backcompat.py`) locks this against a fixture.

## Files

- `server.py` — FastAPI backend (in-process session dict + worker thread).
- `app.py` — Gradio frontend, talks to the backend over plain HTTP.
- `launch.py` — one-shot launcher (spawns server + app together).
- `test_backcompat.py` — locks the bit-identical-default-behavior contract.
- `_make_example_traces.py` — drives the live server to produce the three
  canonical example traces (used by Workstream B for E1 schema validation).
- `traces/` — JSONL traces. Schema is pinned in `phase2/STATUS.md` under
  Workstream C → "Trace JSONL schema".
- `fixtures/` — backcompat fixture for the regression test.

## Trace JSONL schema (also pinned in STATUS.md)

One record per sub-block boundary. See `phase2/STATUS.md` →
"Workstream C → Trace JSONL schema" for the full annotated definition.
The three checked-in example traces in `traces/` are the canonical
specimens:

| File                         | What it demonstrates                                   |
|------------------------------|--------------------------------------------------------|
| `trace_all_llada.jsonl`      | Default flow: LLaDA at every block, commit-LoRA fires on blocks 1-3 (commit_n_blocks=3) |
| `trace_mid_ar_handoff.jsonl` | AR-extend at block 0 (n_tokens=6), then LLaDA resumes  |
| `trace_cmaj_branching.jsonl` | LLaDA blocks 0-2, then b=5 cmaj branch at block 3      |

To regenerate them after editing the schema, restart the mock server and
run:

```bash
MOCK_MODELS=1 python3 phase2/inference_viz/server.py --port 8765 &
python3 phase2/inference_viz/_make_example_traces.py
```

## Backcompat regression

```bash
python3 phase2/inference_viz/test_backcompat.py
```

Locks three invariants against a JSON fixture:
1. `denoise_block(...)` with **no callback** matches the fixture byte-for-byte.
2. `denoise_block(..., step_callback=lambda s: continue_llada())` is
   bit-equivalent to (1).
3. A callback that returns `StepDirective.stop()` mid-flight terminates
   without crashing.

If any of these breaks, the diff_llada refactor leaked behavior into
`runner.py`'s existing conditions — STOP and triage before proceeding.

## Real-mode validation (2026-05-02)

A `c2c` condition run with v3 + commit-v3 LoRAs on real models was completed
against a Crucible-provisioned RTX 4090 pod. Result: **acc 1.0** on GSM8K
problem 0 (Janet's ducks), gold=18 / pred=18, FLOPs ≈ 1.31×10¹⁴.

Trace JSONL pulled to:
`phase2/inference_viz/traces/trace_c2c_real_mode_v3commit_problem42.jsonl`

(Note: `SEED=42` was set but `runner.load_problems()` selects by `N_PROBLEMS=1`
prefix, so problem 0 was used — that's a runner-side detail, not a viz-side
issue. The trace is from the real diff_llada call with adapters loaded.)

This closes the "real-pod test deferred" item from C's day-1 report. The trace
format here is the runner.py `raw_*.jsonl` schema (single-record, no per-step
detail) — not the per-sub-block JSONL schema pinned in `phase2/STATUS.md`
"Workstream C → Trace JSONL schema", which requires the FastAPI server's
step-callback hook to capture per-step entropy / commit-LoRA toggling. To
generate a STATUS-schema-compliant trace from real models, the next session
should: provision a pod, sync the visualizer's `server.py` over, and run
through the Gradio app's manual mode driving a single problem to completion.

## Known limitations / out-of-scope

- The Gradio app currently waits for a directive at every sub-block
  boundary (no auto-play). This is deliberate — research instrumentation
  should be deliberate, not racy.
- AR-extend grafts tokens into the next masked positions in the current
  block; it doesn't yet support overwriting already-committed positions.
- Real-mode cmaj branching falls back to running `denoise_block` from
  scratch with `b` different seeds (matches `runner.py:cmaj`); we don't
  yet deep-copy the in-flight `x` for true mid-generation forking.
- WebSocket streaming was considered and rejected in favor of
  long-polling — the backend is GPU-bound, not network-bound, and a
  per-session HTTP request fits the manual-mode interaction model
  perfectly. Re-add WebSockets only if we move to autoplay.

## Screenshots

_(captured locally with `MOCK_MODELS=1`; placeholder list — drop PNGs
into `phase2/inference_viz/screenshots/` when running the real demo)_

- `screenshots/grid_initial.png` — empty grid before Start
- `screenshots/grid_after_block_0.png` — first sub-block committed
- `screenshots/grid_with_ar_handoff.png` — AR-extend marker on block 1
- `screenshots/grid_with_cmaj.png` — cmaj branch marker on block 3
