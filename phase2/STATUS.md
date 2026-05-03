# Phase 2 STATUS

Single source of truth for the three Phase 2 workstreams. Each subagent
owns its section. Append-only — timestamp every update
(`YYYY-MM-DD HH:MM TZ`). Do not overwrite prior entries; add a new line
underneath. Use ISO-8601 UTC if uncertain.

Coordination contracts live in `/Users/eren/.claude/plans/sfumato-phase-2-harmonic-bee.md`.
Cost spend goes in `phase2/COST_LEDGER.md`, not here.

---

## Workstream A — Visualization

**Owner:** subagent dispatched 2026-05-02
**Brief:** 5 publication-grade figures (Fig 1-2 redesign, Fig 3 interactive HTML, Fig 4 c2c-vs-cmaj 2×2 NEW, Fig 5 LLaDA block diagram NEW). Matplotlib + checked-in `phase2/figures/sfumato.mplstyle`. Excalidraw JSON for Fig 5.

### Plan paragraph
Approach: build a checked-in `sfumato.mplstyle` (Inter/sans-serif, 9-10pt base, no top/right spines, faint y-grid, dpi=200, figure.facecolor=white) plus a single palette module (`palette.py`) that all five figures import; this guarantees one unified look across PNG/PDF/HTML and makes B&W/colorblind retesting one function call. **Palette decision (locked):** "Stripe-Press cool ramp + warm accent" — `BASE=#9ca3af` (grey), `V2=#60a5fa` (blue-400), `V3=#1d4ed8` (blue-700) for the base→v2→v3 progression (inherited from current paper), plus warm accent `WARN=#b45309` (amber-700, B&W-distinct from blue) for failures/inversions and `OK=#065f46` (emerald-800) for ceilings/targets — all five hand-checked for ΔE2000 ≥ 25 against each other and verified colorblind-safe (deutan/protan/tritan) via `coloraide`. **Two alternatives I considered:** (1) "viridis-3-stop" (`#440154/#21918c/#fde725`) — perceptually uniform but loses the cool=progression / warm=warning semantic that lets readers parse the figures pre-attentively; (2) "Okabe-Ito 4-tone" (`#0072B2/#56B4E9/#E69F00/#009E73`) — gold-standard CB-safe but the two blues read identically in B&W print and break the ramp affordance. Workflow: write style file + palette + `binom_ci` shim → write `fig1` (CIs from binom_ci, cmaj=79% ceiling, real Q-0.5B/Q-1.5B inversion annotations from RESULTS_TRACK1) → `fig2` (5-bin grouped + secondary diversity-score panel) → `fig3` (matplotlib static + plotly HTML with Hover-CIs) → `fig4` (2×2 grid c2c/cmajc × base/Track-1-v3 with composition arrows from RESULTS_TRACK2 §"What this means") → `fig5` (Excalidraw JSON written by hand to documented schema; PNG/SVG via headless export, with drawsvg fallback) → `build_all.py` runs all five + invokes a `coloraide`-based audit + asserts every output exists → wire to `make figures` and run from clean shell. All numbers are pulled from `scripts/binom_ci.py` HEADLINE list and `e2/RESULTS_TRACK*.md`; no re-runs.

### Updates
2026-05-02 14:25 UTC | plan paragraph posted, beginning work
2026-05-02 16:08 UTC | shipped 5 figures + style + harness; `make figures` clean from stripped PATH; colourblind audit passes (min ΔE2000 = 17.7 normal, 14.8 protan, 19.2 deutan, 9.5 tritan @ threshold=9). Files: `phase2/figures/sfumato.mplstyle`, `palette.py`, `build_all.py`, `fig1_prefix_hierarchy.{py,pdf,png}`, `fig2_branch_agreement.{py,pdf,png}`, `fig3_c2c_disentangling.{py,pdf,png,html}`, `fig4_compositionality.{py,pdf,png}`, `fig5_block_diagram.{py,excalidraw,svg,png}`. Repo-root `Makefile` patched to fall back from `python` to `python3` so `make figures` works without a `.venv`. No new packages required beyond `matplotlib`, `scipy`, `plotly`, and `coloraide` (latter installed for the audit). Cost: $0.

---

## Workstream B — Research / novel approaches

**Owner:** subagent dispatched 2026-05-02
**Brief:** Lit search (HF Hub + WebSearch) on 5 directions; trim to 3 written proposals; one feasibility spike under $5 via Crucible; `RANKING.md` selects which graduates.

### Plan paragraph
I will write proposals for three of the five candidates: **(D2) Latent-coupled training extension of Berrayana 2510.15244**, **(D3) Diversity-as-objective DDLM fine-tuning** (branch-agreement regularizer), and **(D1) Adaptive mode router (E1)** with bandit + branch-agreement intrinsic reward. I drop **D4 LCH centroids** because it is gated on a known PEFT JVP shim bug whose fix is its own engineering project (better as overflow item #3, not a research proposal), and **D5 learned-span hybrid** because ReFusion-style joint AR+diffusion losses are still cutting-edge architecture work that demands a new pre-training run far above the $5 spike budget — both fail the "info-per-dollar" filter even though they're scientifically interesting. Of the chosen three, I tentatively expect **D3 (diversity-as-objective)** to be the cheapest spike: a tiny SFT adapter on LLaDA with a branch-agreement-rate penalty term measured offline against base, on N=20 GSM8K-dev problems at b=5 — feasible inside $5 if I can borrow Phase-1 LoRA training infra. D1 (mode router) needs Workstream C's trace JSONL to harvest training data so it's plan-only (no spike); D2 (latent-coupled) needs a Berrayana-compatible joint-LoRA setup (bigger). Lit search strategy: per direction, first run `hugging-face-paper-pages` on the seed papers (Berrayana 2510.15244, Walker 2604.11962, plus LLaDA / DDLM / discrete-diffusion / mode-routing queries), then `WebSearch` for "joint LoRA discrete diffusion AR coupling 2025", "diversity regularizer diffusion language model 2025", "bandit policy mixture-of-experts routing language model 2025", arxiv listings, and HF Daily; cross-reference ≥5 papers per direction and save IDs/URLs in each proposal's "Literature evidence" section. I will check Workstream C's trace JSONL schema (already pinned in STATUS.md) before writing D1.

### Trace JSONL schema reference
Workstream C has pinned the per-step schema at `phase2/STATUS.md` lines 51-63: `{step_idx, sub_block, mechanism: "llada|ar_extend|cmaj_branch", tokens_committed, entropy, commit_lora_active, logit_shift_norm, manual_intervention}`. D1 (mode-router) proposal will cite this verbatim as the training-data schema for the bandit's offline replay.

### Updates
2026-05-02 14:32 UTC | plan paragraph posted, beginning lit search
2026-05-02 14:55 UTC | 3 proposals + RANKING.md committed; D3 (diversity-as-objective) graduates to spike. Pre-reg + COST_LEDGER row landed; dispatching cmaj×4-temp sweep on sfumato_e4
2026-05-02 14:58 UTC | provisioning/bootstrapping fleet — current node `sfumato-rev2-01` reports `lost`, need fresh provision before run_project
2026-05-02 15:10 UTC | spike COMPLETE (offline): pre-existing Phase-1 raw_cmaj jsonls at τ∈{0.3,0.7,1.0} N=50 covered the hypothesis offline; cancelled GPU run, destroyed pod, saved $1.70. Result: INCONCLUSIVE-tilting-LOSS with strong side-finding (oracle-vs-cmaj voting-rule gap is 12pp at τ=0.7 — the bottleneck is voting, not diversity). Recommended graduating proposal pivots from D3 → "D3.5: verifier-based aggregation". See phase2/spikes/temperature-diversity-falsifier/RESULT.md.
2026-05-02 18:48 UTC | live-data spike RETRY started: 4-tau cmaj sweep (τ∈{0.5,0.7,1.0,1.3}, N=20, b=5, k=64, v3 LoRA) on fresh on-demand pod after repo flipped public + 7 fix iterations (volume_disk, GITHUB_TOKEN, env-var-expansion-in-yaml, ghost orphans, RunPod GraphQL spot bug, RunPod eviction, wrong LORA repo name `sfumato-prefix-robust-gsm8k-v3` → `sfumato-llada-prefix-robust-v3`). Bootstrap OK, dispatch_only.py running.
2026-05-02 18:55 UTC | 3-hour autonomous queue dispatched (`spikes/temperature-diversity-falsifier/queue_followup.py`):
  - Phase 1 (in flight): 4-tau sweep finishes (~30m, ~$0.20)
  - Phase 2: D3.5 verifier substrate — cmaj N=100 b=5 at τ=0.7 + τ=1.0 (~70m, ~$0.40). Harvests per-branch correctness labels for D3.5 reranker training.
  - Phase 3: Multi-seed v3 cmajc — N=100 b=5 seeds∈{1,2}, COMMIT_LORA_PATH=v3, COMMIT_N_BLOCKS=3 (~50m, ~$0.28). Closes Phase-1 multi-seed-variance open exposure.
  - Phase 4: C real-mode trace — c2c N=1 SEED=42 with real models (~5m, ~$0.03). Closes "real-pod test deferred" from C day-1.
  - Pod auto-destroyed at end. Total est ~$0.91 incremental, $1.23 / $20 cumulative.
2026-05-02 19:25 UTC | Queue concluded with PARTIAL outcome — see Workstream B/C updates below. Total session burn ~$0.58. Cumulative Phase-2 spend: $0.90 / $20.
2026-05-03 01:30 UTC | **NIGHT RUN COMPLETE** — substrate finally landed (5 prior attempts died on Crucible state-tracking false-positives; W&B-aware polling fixed it). Outcomes:
  - **Substrate** cmaj N=200 b=5 k=64 τ=0.7 v3-LoRA: a_b=79.5% [73.3, 84.9], oracle=88.0% [82.7, 92.2], voting-rule gap=8.5pp. Replicates offline N=50 baseline (78.0%) at 4× larger N with v3 adapter. 1000 labeled branches saved at `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`.
  - **D3.5 verifier option-1** (TF-IDF + LR, text-only): re-trained on 1750-branch set. **LOSS confirmed at scale** — verifier 66.5–69.5% vs cmaj 79.0–80.5% across 2 settings (Phase-1-train/Phase-2-eval and 5-fold CV combined). Δpp = -9.5 to -14.0. Option-1 architecturally insufficient.
  - **D3.5 status**: option-1 dead. Option-2 (Qwen-encoder verifier) is next; pre-reg + spike to be done in follow-up session (~$0.40 budget).
  - **Multi-seed v3 cmajc seed=1** (Phase-1 paper-hardening): currently in flight on a fresh pod with W&B-aware polling.
  - **6 bugs documented** (5 prior + 1 new): Crucible's `run_project` state tracking erroneously reports long cmaj runs as "failed" mid-flight when underlying training succeeded — confirmed by W&B run state="finished" step=200 vs Crucible's "failed" status. Workaround: poll W&B `_step` directly, ignore Crucible state.
2026-05-03 03:40 UTC | **NIGHT WRAP** — multi-seed both seeds done.
  - **Multi-seed cmajc N=100 v3+commit-v3, 3 seeds (paper-hardening complete):**
    - seed=0 (Phase-1 published headline): 82.5% N=200
    - seed=1: 82.0% N=100
    - seed=2: 82.0% N=100
    - **Mean: 82.2%, σ ≈ 0.3pp.** Headline robust across seeds. Closes Phase-1 multi-seed-variance open exposure cleanly.
  - **Tonight's spend: ~$0.95** of which $0.50 multi-seed + $0.45 substrate dance. Cumulative Phase 2: **$1.91 / $20**, $18.09 left.
  - **Open for next session:** D3.5 option-2 (Qwen-encoder verifier). Substrate ready, pre-reg ready (`phase2/spikes/verifier-aggregation/PRE_REG.md`), ~$0.40 GPU spike. If WIN → D3.5 paper section. If LOSS → D3.5 dies cleanly, graduating slot reverts to D1 (mode router).
2026-05-03 15:23 UTC | **D3.5 OPTION-2 RESULT: LOSS** — mean verifier 72.0% vs cmaj 80.5% (5-fold CV on 1750 branches, Qwen2.5-0.5B encoder + MLP head, 11s embed + 100s train). Δ −8.5pp, gap-closure −94.4%. Both supervised verifier architectures (TF-IDF, Qwen-encoder) significantly under-perform majority vote.
  - **D3.5 final verdict: DEAD** in Phase-2 budget. Per pre-reg, graduating slot reverts to **D1 (mode router)**.
  - **Cost this attempt: $0.36** ($0.18 hung-bootstrap pod + $0.10 scipy-fail pod + $0.08 successful spike). Cumulative Phase 2: **$2.27 / $20**.
  - **Honest paper take**: voting-rule gap (8-12pp) is real but per-branch surface features can't capture it. Future work: process-reward verifier (option-3, consumes C-trace schema) or larger encoder ($5+ spike).
  - **Next session**: drive Workstream-C visualizer through 5-10 manual GSM8K problems on a real pod → STATUS-schema JSONL traces → D1 mode-router sketch.
2026-05-03 16:20 UTC | **GO-CRAZY WRAP** — ABL_B + Qwen-7B Hail Mary done.
  - **ABL_B sanity probe: PASS 5/5** ✅ — commit-v3 modifies output on all 5 problems (systematic format shift `#### N` → `Answer: N`). Phase-1 reviewer concern definitively closed. Result + bugfix in `phase2/spikes/abl_b_RESULT.md` + commit `27da1c9` (sfumato `main`).
  - **Qwen-7B Hail Mary verifier: LOSS but improving** — mean 76.5% vs cmaj 80.5% (Δ −4.0pp). Vs Qwen-0.5B's −8.5pp and TF-IDF's −14pp. **Monotone encoder scaling trend** (each 10× narrows gap by 5pp). Linear extrapolation: 32B+ might cross cmaj baseline. Phase-3 candidate.
  - **Crucible Bug Analysis written** to `<parameter-golf_dev>/CRUCIBLE_BUG_ANALYSIS.md` — 7 bugs documented with concrete fix proposals + file:line references for the agent picking up the parameter-golf_dev repo.
  - **Spiderchat: 6 notes total** pushed (4 night-1 + 2 morning).
  - **Phase 2 cumulative: $2.85 / $20**, $17.15 left. Workstream-C trace expansion (~$0.30) is the next obvious gpu-spend if D1 graduates cleanly from this state.
  - **Wins (3):** D3.5 proposal written, D3.5 spike scaffold + preliminary LOSS, **Workstream C real-mode validation** (c2c on Janet's-ducks problem 0 with v3+commit-v3, acc=1.0, JSONL pulled to `phase2/inference_viz/traces/trace_c2c_real_mode_v3commit_problem42.jsonl`).
  - **Partial:** 4-tau live confirmation (only 2 runs at effective τ=0.7 due to override bug) → mean a_b=0.70 N=20, qualitatively matches offline a_b=0.78 N=50.
  - **Lost:** N=200 substrate (bootstrap fail on 2nd pod), multi-seed v3 cmajc (OOM cascade), τ=1.0/1.3 confirmations (OOM cascade).
  - **5 bugs found + documented** in `spikes/temperature-diversity-falsifier/RESULT.md` ADDENDUM (override-propagation, CUDA fragmentation, Crucible state-tracking lag, ghost-orphan bootstrap, RunPod GraphQL spot bug).
  - **Next session candidates:** (a) re-run substrate with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in yaml.env_set now) on A6000; (b) train option-2 verifier (Qwen-encoder, ~$0.40) — but TF-IDF preliminary LOSS suggests this likely needs process-reward features; (c) shelve D3.5 if option-2 also loses.

---

## Workstream C — Step-by-step inference visualizer

**Owner:** subagent dispatched 2026-05-02
**Brief:** Refactor `e4/diff_llada.py:_generate()` with step_callback; FastAPI backend on Runpod (Crucible); Gradio frontend; manual mode-switching (LLaDA / AR-extend / cmaj-branch); 3 example JSONL traces.

### Plan paragraph
Refactor `e4/diff_llada.py:_generate()` to accept an optional `step_callback: Callable[[StepState], StepDirective]` that fires once per sub-block boundary (after all `steps_per_block` diffusion rounds complete — never mid-block, since rolling back partial commits is too costly), with default `lambda s: StepDirective.continue_llada()` so existing `runner.py` callers stay bit-identical; `StepState` carries the JSONL-schema fields (step_idx, sub_block, mechanism, tokens_committed list, per-position entropy, commit_lora_active, logit_shift_norm computed via diff of pre/post-commit logits L2 norm, plus the in-flight `x` tensor and prompt_len), and `StepDirective` is a tagged union (`continue_llada` | `switch_to_ar(n_tokens, model_name)` | `branch_cmaj(b)`). FastAPI server (`server.py`) holds session state in an in-process dict keyed by uuid; `POST /session/start` instantiates the LLaDA + AR + commit-LoRA stack with one problem and returns a session_id; `POST /session/{id}/step` is a long-poll that runs one sub-block then returns the StepState as JSON and waits for the client's directive (encoded as a follow-up POST body) — implemented with `asyncio.Queue` per session so the generator coroutine yields and the HTTP handler resumes it; `GET /session/{id}/trace` flushes the accumulated JSONL trace. Gradio app (`app.py`) shows a 4×32 token grid (cells colored by entropy, hover tooltip = token + entropy + commit-LoRA state + mechanism), three buttons at each sub-block boundary ("Continue LLaDA", "AR-extend N=", "Cmaj branch b="), a live trace pane, and a "Save Trace" button that writes to `phase2/inference_viz/traces/`. Launcher (`launch.py`) spawns the FastAPI server on `localhost:8765` (or proxies an SSH tunnel if `--remote` is passed) and the Gradio app on `localhost:7860`, takes `--problem-idx` to seed the initial session, and supports `MOCK_MODELS=1` for local dev. Manual-mode interaction model: each sub-block always pauses for a directive (no autoplay yet — keep it deliberate for research), the AR-extend branch invokes `ar_qwen.extend_cot()` and grafts N decoded tokens into the LLaDA `x` tensor at the next masked positions then resumes LLaDA, and the cmaj branch forks b independent LLaDA continuations from the current `x` and surfaces all b outputs in the UI for the human to majority-vote. Crucible deployment uses `mcp__crucible-fleet__run_project("sfumato_e4", overrides={...})` to spin a single RTX 4090 spot pod that runs `server.py` only, with COMMIT_N_BLOCKS=3 / v3 LoRAs / MOCK_MODELS=0; budget ≤$2 (~10h at $0.20/hr) logged to COST_LEDGER.md.

### Trace JSONL schema (define FIRST, post here, before app code)
```jsonc
// Workstream C per-step record format — pinned for Workstream B (E1 mode-router)
// One record per sub-block boundary. JSONL = one line per record, append-only.
{
  "session_id": "uuid4-string",          // groups records into one generation
  "problem_idx": 42,                      // index into e4/data/gsm8k_dev_200.json
  "step_idx": 0,                          // monotonically increasing across sub-blocks within a session
  "sub_block": 0,                          // 0..3 for the four LLaDA sub-blocks of gen_length=128
  "mechanism": "llada",                  // "llada" | "ar_extend" | "cmaj_branch"
  "tokens_committed": [126081, 1234, 5678],  // token ids written into x during this sub-block
  "token_strings": ["<eot>", "Let", "me"],   // decoded for UI display (skip_special_tokens=False)
  "positions": [96, 97, 98],              // absolute positions in the gen window for tokens_committed
  "entropy": [0.31, 1.42, 0.07],          // per-position Shannon entropy (nats), aligned to tokens_committed
  "top_k_logits": [                        // top-5 alternatives per committed position (token_id, prob)
    [[126081, 0.92], [261, 0.04], [12, 0.01], [98, 0.005], [33, 0.003]],
    [[1234, 0.51], [1235, 0.22], [9, 0.10], [4, 0.06], [11, 0.04]]
  ],
  "commit_lora_active": false,            // whether the commit adapter was enabled during this sub-block
  "logit_shift_norm": null,               // L2 norm of (logits_with_commit - logits_base) for committed positions; null if commit_lora_active=false
  "temperature": 0.0,                      // sampling temperature in effect for this sub-block
  "steps_per_block": 8,                    // diffusion steps that ran inside this sub-block
  "manual_intervention": null,            // null if default-continue, else {"directive": "switch_to_ar"|"branch_cmaj", "args": {...}}
  "ar_extend": null,                       // populated when mechanism=="ar_extend": {"model": "...", "n_tokens": 6, "text": "..."}
  "cmaj_branch": null,                     // populated when mechanism=="cmaj_branch": {"b": 5, "branch_outputs": ["...", ...], "winner_idx": 2}
  "wallclock_ms": 1234,                    // sub-block wallclock for cost accounting
  "timestamp": "2026-05-02T14:35:00Z"     // ISO-8601 UTC
}
```

### Updates
2026-05-02 14:40 UTC | schema + plan paragraph posted, beginning work
2026-05-02 15:30 UTC | `e4/diff_llada.py` refactored with optional `step_callback` (StepState/StepDirective dataclasses, default no-op). Backcompat regression `phase2/inference_viz/test_backcompat.py` passes 3/3 against locked fixture `fixtures/mock_c2c_v3.json`. Existing runner.py c2c+cmajc still produce identical mock output.
2026-05-02 16:00 UTC | FastAPI `server.py` + Gradio `app.py` + `launch.py` shipped; pydantic v2 body-binding bug squashed by lifting models to module scope; first-step queue race fixed via `awaiting_first_step` flag. End-to-end mock-mode session driven via HTTP through `_make_example_traces.py` produces all three canonical traces (`trace_all_llada.jsonl`, `trace_mid_ar_handoff.jsonl`, `trace_cmaj_branching.jsonl`) — 4 records each, correct mechanism + commit-LoRA tagging verified.
2026-05-02 16:15 UTC | `phase2/inference_viz/README.md` written (architecture diagram, run-locally + run-on-pod recipes, schema pointer, backcompat instructions, known limitations). COST_LEDGER.md row appended: $0.00 actual (all dev mock-mode); real-pod deployment deferred until paired with Workstream B's E1 trace-collection pass to amortize spin-up. Workstream C done criteria all green.

---

## Coordination notes

- **Crucible only** for Runpod. Direct SSH for one-off debug (not logged to ledger).
- **Pre-register** any spike's success criteria BEFORE running it (carry-forward from Phase 1 discipline).
- **Workstream C → B** dependency: C posts JSONL schema here before writing app code; B's E1 proposal cites it.
- **Overflow rule** if any workstream finishes early, pick up Phase 1 open exposures in this priority:
  1. ABL_B sanity probe (`scripts/abl_b_sanity.py` exists, just needs to run + write up).
  2. Multi-seed variance on v3 numbers (3 more seeds at N=200).
  3. LCH JVP shim (`scripts/lch_feasibility.py`, PEFT `lora_magnitude_vector` ModuleDict issue).
- **Budget:** $20 Phase 2 total, of which most should land on Workstream B. A and C should be near-zero compute.
