# Phase 2 — Next Session Handoff

**Last session ended:** 2026-05-03 ~15:25 UTC (~$2.27 / $20 spent, $17.73 remaining) — NIGHT-1+morning wrap

## Morning addendum

**D3.5 is DEAD.** Both supervised verifier options tested and lost:
- Option 1 (TF-IDF + LR): −9.5 to −14pp vs cmaj
- Option 2 (Qwen2.5-0.5B encoder + MLP, 5-fold CV on 1750 branches): mean verifier 72.0% vs cmaj 80.5% (Δ −8.5pp, gap-closure −94.4%)

**Per pre-reg, graduating slot reverts to D1 (mode router).** D1 is gated on Workstream-C trace data — we have 1 real-mode trace, need 5-10 to make D1's bandit-on-replay sketch concrete enough to spike.



## Night-1 wins (vs prior handoff)

- **D3.5 substrate landed**: `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl` (200 problems × 5 branches = 1000 labeled). a_b=79.5%, oracle=88%, voting-gap=8.5pp on N=200 with v3 LoRA. Replicates offline N=50 finding.
- **D3.5 verifier option-1 (TF-IDF)**: definitively LOSS at N=1750 — verifier 66.5–69.5% vs cmaj 79.0–80.5%. Architecturally insufficient. Option-2 (Qwen-encoder) is the only remaining D3.5 path.
- **Multi-seed v3 cmajc closed** (Phase-1 overflow): 3 seeds (0/1/2) × N=100-200 → mean 82.2%, σ≈0.3pp. Phase-1 headline robust. Saved to `e4/results/raw_cmajc_k64_seed{1,2}_b5_v3LoRA_N100.jsonl`.

## Recommended next session sequence (in order)

1. **D3.5 option-2 spike — Qwen-encoder verifier** (~$0.40, ~30 min)
   - Use Qwen2.5-0.5B-Instruct as feature extractor, mean-pool last-layer hidden states from (problem + branch_text), MLP head → P(correct)
   - Train on Phase-1 N=750 + substrate N=1000 = 1750 labeled branches, 5-fold CV by problem
   - Pre-reg threshold (binding): mean verifier accuracy ≥ 83% on held-out folds (cmaj baseline 79–80%, oracle 88–90%)
   - **WIN** (≥83%) → D3.5 graduates to paper section; **LOSS** (<78%) → D3.5 dies, RANKING reverts to D1

2. If option-2 LOSS: pivot to D1 (mode router) — needs more Workstream C real-mode traces. Provision pod, SSH in, drive `phase2/inference_viz/server.py` through Gradio with manual interventions on 5-10 problems → STATUS-schema JSONL traces.

3. Phase-1 remaining overflow (low priority):
   - ABL_B sanity probe (`scripts/abl_b_sanity.py` — script exists, needs to run on pod)
   - LCH JVP shim (`scripts/lch_feasibility.py` — code-debug, no GPU)

## Locked-in mitigations from night-1

- `parameter-golf_dev/.crucible/projects/sfumato_e4.yaml` has `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `volume_disk: 60`
- `gpu_type` fallback list prefers 48GB GPUs (A6000/L40S/RTX-6000-Ada) over 24GB RTX 4090
- env_set values are all quoted strings (sync.py crashes on yaml ints — silent failure)
- W&B-aware polling pattern (in `phase2/spikes/temperature-diversity-falsifier/run_via_ssh.py` and the inline scripts that worked tonight) — Crucible's run_project state tracking is unreliable for long cmaj runs; poll W&B `_step` directly
- Repo is PUBLIC again (substrate run cloned successfully)



## State you'll inherit

**Done:**
- Workstream A — 5 publication-grade figures (`phase2/figures/*`), `make figures` runs clean.
- Workstream B day-1 — 3 proposals + RANKING (`phase2/proposals/*`), offline temperature-diversity-falsifier spike with INCONCLUSIVE-tilting-LOSS verdict + 12pp voting-gap side-finding.
- Workstream B day-2 — D3.5 verifier-based-aggregation proposal (`phase2/proposals/verifier-based-aggregation.md`), spike scaffold (`phase2/spikes/verifier-aggregation/`), preliminary LOSS on TF-IDF option (mean −14pp vs cmaj on N=750).
- Workstream C — visualizer code, 3 mock traces, **real-mode validation** (c2c on Janet's-ducks problem 0 with v3+commit-v3, acc=1.0, JSONL at `phase2/inference_viz/traces/trace_c2c_real_mode_v3commit_problem42.jsonl`).
- Live confirmation of voting-gap finding — partial: 2 N=20 cmaj runs at effective τ=0.7 returned mean a_b=0.70, consistent with offline N=50 a_b=0.78 within Wilson CIs.

**Open / queued:**
- D3.5 substrate harvest (cmaj N=200 b=5 τ=0.7 v3) — bootstrap failed on retry.
- D3.5 verifier option 2 (Qwen-encoder verifier, ~$0.40 GPU) — pending substrate.
- Multi-seed v3 cmajc variance bars — Phase-1 overflow exposure, queue Phase 3 failed.
- Workstream C STATUS-schema-compliant trace (per-sub-block JSONL) from real-mode pod — needs server.py SSH'd over.

## 5 documented bugs (read before re-attempting GPU work)

See `phase2/spikes/temperature-diversity-falsifier/RESULT.md` ADDENDUM for details. Summary:

1. **Override-propagation broken**: Crucible `run_project({"overrides": {...}})` does NOT propagate TEMP/SEED to runner's `os.environ`. Workaround: hardcode in `spec.env_set` of yaml, or use direct SSH (blocked by safety hook in this session — would need pre-approval).
2. **CUDA fragmentation OOM cascade**: cmaj b=5 runs leak GPU memory across attempts. **Mitigation already applied**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is in `parameter-golf_dev/.crucible/projects/sfumato_e4.yaml` `env_set`. Plus `gpu_type` reordered to prefer 48GB GPUs (A6000, L40S, RTX 6000 Ada).
3. **Crucible state-tracking lag**: runs report "failed" when wandb finalize exits non-zero, even when training completed. Use `phase2/spikes/temperature-diversity-falsifier/analyze_live.py` to read W&B summary as ground truth.
4. **Ghost-orphan bootstrap**: stale pods in inventory → bootstrap_project iterates them → SSH disconnect mid-run. Mitigation: `crucible fleet destroy --node <name>` BEFORE provisioning new pod with same project name.
5. **RunPod GraphQL spot bug**: deprecated `PodFindAndDeployInterruptableInput` type. Crucible falls back to REST (slower but works on COMMUNITY tier). Filed against parameter-golf_dev / Crucible repo, not sfumato.

## Recommended next-session sequence

1. **Diagnose bootstrap failure** before next provision. The latest A6000 attempt died at `install_wandb` step or just after (setup phase). Run `crucible fleet bootstrap --skip-data` against an existing pod, watch full output. Likely fix: `python scripts/freeze_gsm8k.py` setup needs venv activation in the same shell as the install steps; check `parameter-golf_dev/src/crucible/fleet/bootstrap.py` for setup-step env handling.
2. Once bootstrap is reliable, re-run substrate (cmaj N=200 b=5 τ=0.7 v3) → ~30 min, ~$0.18 on A6000 spot.
3. With substrate landed, re-run TF-IDF verifier (`phase2/spikes/verifier-aggregation/train_verifier.py`) — see if N=1750 changes the LOSS verdict.
4. If TF-IDF still LOSS, kick option 2 (Qwen-encoder verifier) — ~$0.40 on RTX 4090.
5. Multi-seed v3 cmajc variance — N=100 b=5 seed=1 (workaround override bug by editing yaml.env_defaults SEED=1, then SEED=2).

## Files of interest

- `phase2/STATUS.md` — full timeline
- `phase2/COST_LEDGER.md` — every spend
- `phase2/spikes/temperature-diversity-falsifier/RESULT.md` — primary spike (offline + live ADDENDUM)
- `phase2/spikes/verifier-aggregation/RESULT.md` — D3.5 spike preliminary (LOSS on TF-IDF)
- `phase2/proposals/verifier-based-aggregation.md` — D3.5 proposal
- `phase2/proposals/RANKING.md` — D3.5 supersedes original D3
- `phase2/inference_viz/traces/trace_c2c_real_mode_v3commit_problem42.jsonl` — C real-mode validation

## Budget remaining: $19.42 / $20
