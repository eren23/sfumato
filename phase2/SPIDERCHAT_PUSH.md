# Spider Chat KB push — sfumato Phase 2 night-1

Spider Chat MCP server disconnected mid-session, so this file is a structured
dump you can paste as separate notes/memories. Each H2 = one note. Memories
listed at the bottom.

**Note: 4 main notes were successfully pushed earlier when MCP was alive (IDs
69f6c5adf693abc21be4481f, 69f6c5aef693abc21be44826, 69f6c5aff693abc21be4482c,
69f6c5b1f693abc21be44832). The "D3.5 verdict update" note below should be
upserted on next reconnect — it MODIFIES the architecture-decision note's
conclusion (option 2 also lost).**

---

## Sfumato — D3.5 FINAL — both verifier options lost (2026-05-03)

**Tags:** `#sfumato` `#findings` `#decision`

D3.5 is dead in Phase-2 budget. Both supervised verifier architectures tested:

- **Option 1 (TF-IDF + LR text-only)**: at N=1750, verifier 66.5–69.5% vs cmaj 79.0–80.5% (Δ −9.5 to −14pp).
- **Option 2 (Qwen2.5-0.5B encoder mean-pool + MLP head, 5-fold CV)**: verifier 72.0% vs cmaj 80.5% (Δ −8.5pp, gap-closure −94.4%, pre-reg threshold ≥83% missed by 11pp).

**Per pre-reg, graduating slot reverts to D1 (mode router)**, gated on Workstream-C trace expansion (need 5-10 manual GSM8K runs through the Gradio visualizer to produce STATUS-schema JSONL traces for D1's bandit-on-replay).

**Honest paper take**: voting-rule gap (oracle 88-90% vs cmaj 79-82%, 8-12pp) is the most actionable Phase-2 finding, but per-branch surface features at this dataset size (200 problems, 1750 branches) are insufficient to capture the signal. Two negative results document the limitation cleanly. Future work: process-reward verifier (option 3, consumes per-step trajectory features) or larger encoder (Qwen-7B+, costs $5+ per spike).

Cumulative Phase-2 spend: $2.27 / $20.

### Related

- [[Sfumato — D3.5 Verifier Architecture Decision]] (this update OVERRIDES the option-2 "PENDING" status)
- [[Sfumato — Voting Rule Gap as Phase 2 Headline]]

---

## Sfumato Phase 2 — Night-1 Wrap (2026-05-03)

**Tags:** `#sfumato` `#phase2` `#decision` `#findings`

After 3 days of work and ~$1.91 of $20 budget, Phase 2 has graduated into a clean handoff state with 4 wins, 1 graduating proposal, 1 dead direction, and 6 documented orchestration bugs.

### Wins
1. **D3.5 verifier substrate harvested**: cmaj N=200 b=5 τ=0.7 with v3 LoRA. a_b=79.5% [73.3, 84.9], oracle=88.0% [82.7, 92.2], voting-rule gap = **8.5pp**. 1000 labeled branches saved at `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`.
2. **Phase-1 multi-seed variance closed**: cmajc N=100 v3+commit-v3, seeds {0, 1, 2} → mean 82.17%, σ ≈ 0.29pp. Phase-1 published headline (82.5% seed=0 N=200) is rock-solid.
3. **Workstream C real-mode validated**: c2c condition with v3+commit-v3 LoRAs on real models, GSM8K problem 0 (Janet's ducks), acc=1.0. JSONL trace pulled.
4. **Workstream A figures**: 5 publication-grade figures + style file + harness, `make figures` clean.

### Dead direction
- **D3 (diversity-as-objective)**: spike showed cmaj a_b is monotone-decreasing in τ. Killed.
- **D3.5 verifier option-1 (TF-IDF + LR text-only)**: at full N=1750 verifier 66.5–69.5% vs cmaj 79.0–80.5%. Architecturally insufficient — text features can't discriminate correct from incorrect arithmetic without embedding-level signal.

### Graduating proposal
**D3.5 — Verifier-based branch aggregation** (option-2 form). The 8–12pp voting-rule gap (oracle 88–90% vs cmaj 79–82%) is the most actionable finding from Phase 1+2. Option-2 (Qwen-encoder verifier on per-branch hidden states) is the remaining viable path. Pre-reg: ≥83% on N=200 held-out. Compute estimate: ~$0.40 GPU.

### Related
- [[Sfumato — Voting Rule Gap as Phase 2 Headline]]
- [[Sfumato — D3.5 Verifier Architecture Decision]]
- [[Sfumato — Crucible Orchestration Bug Catalog]]

---

## Sfumato — Voting Rule Gap as Phase 2 Headline

**Tags:** `#sfumato` `#findings` `#paper`

The single most interesting unexpected finding in Phase 2 was the **voting-rule gap**: across multiple cmaj configurations, oracle ceiling exceeds majority-vote accuracy by 8–12pp.

| Config | N | a_b (cmaj) | Oracle | Gap |
|---|---|---|---|---|
| Offline base LLaDA τ=0.7 | 50 | 78.0% | 90.0% | 12pp |
| Substrate v3-LoRA τ=0.7 | 200 | 79.5% | 88.0% | 8.5pp |
| cmajc v3+commit-v3 seed=1 | 100 | 82.0% | 90.0% | 8.0pp |
| cmajc v3+commit-v3 seed=2 | 100 | 82.0% | 90.0% | 8.0pp |

**Interpretation**: in the median problem, the right answer IS in some branch — majority vote just throws it away. This holds across base LLaDA, v3 prefix-robust LoRA, and v3+commit-v3 stacks. The gap is roughly invariant to LoRA upgrades, suggesting it's a **structural property of cmaj aggregation**, not a model-quality artifact.

**Implication for D3.5**: a verifier that even partially reranks branches could lift cmaj by up to 8pp without any further training of the generation model. Free lift if the verifier can be built.

**Caveat**: the TF-IDF verifier we tried can't discriminate (LOSS at N=1750). Embedding-based verifier (Qwen-encoder, option 2) is the remaining open architecture.

### Related
- [[Sfumato Phase 2 — Night-1 Wrap (2026-05-03)]]
- [[Sfumato — D3.5 Verifier Architecture Decision]]

---

## Sfumato — D3.5 Verifier Architecture Decision

**Tags:** `#sfumato` `#decision` `#architecture`

D3.5 has 3 architecture options, ranked by info-per-dollar. Night-1 tested option 1 (LOSS); option 2 is the next session's task.

**Option 1 — TF-IDF + Logistic Regression (text only)**: KILLED night-1.
- Trains on (problem, branch_text) → P(correct) using sklearn TF-IDF n-grams + LR
- $0 cost, runs on CPU in 30s
- N=1750 result: verifier 66.5–69.5% vs cmaj 79.0–80.5%. **−9.5pp to −14pp.**
- Why it failed: surface text features (length, math-formatting) don't separate correct from incorrect arithmetic chains-of-thought.

**Option 2 — Qwen-encoder verifier**: GRADUATING CANDIDATE for next session.
- Use Qwen2.5-0.5B-Instruct as feature extractor; mean-pool last-layer hidden states from (problem, branch_text); MLP head → P(correct)
- Train with BCE on 1750 branches, 5-fold CV by problem
- Compute: ~$0.40 GPU on RTX 4090 spot, ~30 min
- Pre-reg threshold (binding): mean verifier accuracy ≥ 83% on held-out folds
- WIN → D3.5 paper section; LOSS → D3.5 dies, RANKING reverts to D1

**Option 3 — Process-reward verifier**: PHASE-3 candidate.
- Consume Workstream-C JSONL trace schema (per-step entropy, commit-LoRA logit shifts, mechanism source)
- Per-step features rather than per-answer
- Requires more real-mode traces from Workstream C visualizer first
- Defer until option 2 is concluded

### Related
- [[Sfumato — Voting Rule Gap as Phase 2 Headline]]
- [[Sfumato — Crucible Orchestration Bug Catalog]]

---

## Sfumato — Crucible Orchestration Bug Catalog

**Tags:** `#sfumato` `#crucible` `#debugging` `#infrastructure`

6 distinct bugs hit during Phase 2 Crucible/RunPod orchestration. All have documented mitigations.

1. **`run_project` overrides don't propagate to runner env**: TEMP/SEED/etc. passed via `overrides` dict get lost; runner.py reads `os.environ.get("TEMP", "0.7")` and falls back to the hardcoded default. Verified by inspecting per-row `temperature` field in W&B-pulled artifacts. **Workaround**: set values in `spec.env_set` of yaml (sealed at bootstrap, not at run dispatch); rewrite yaml between runs OR run one config per provision cycle.

2. **`sync.py:write_remote_env` shlex.quote crashes on int values**: `for k,v in env_set.items(): shlex.quote(v)` raises TypeError silently when v is an integer (yaml ints like `K_STEPS: 64`). The env_set entry is silently skipped → pod runs with defaults. **Workaround**: ALWAYS wrap env_set values in quotes in yaml (`K_STEPS: "64"`).

3. **CUDA fragmentation OOM cascade across cmaj b=5 runs**: first 1-2 runs succeed, third dies with "reserved but unallocated memory is large" message at LLaDA load. **Mitigation**: `env_set.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in yaml. Plus prefer 48GB GPUs (A6000 etc.) over 24GB RTX 4090 in `gpu_type` fallback list.

4. **`run_project` state-tracking mismarks long cmaj runs as "failed" mid-flight**: The actual training process is still alive on the pod and W&B shows `state=running` with monotonically-increasing `_step`, but Crucible reports `status=failed`. Orchestrator scripts that trust this destroy the pod, killing the in-flight runner. Lost ≥3 substrate runs to this. **Workaround**: poll W&B `_step` directly, ignore Crucible state. Only declare failure if W&B `_step` plateaus AND state goes terminal.

5. **Ghost-orphan pods cause dual-bootstrap**: stale pods named `parameter-golf__sfumato_e4-01` accumulate across attempts; Crucible's `bootstrap_project` iterates them all, doubling install time and risking SSH disconnects. **Mitigation**: clean inventory (`crucible fleet destroy --node ...`) and verify zero pods on RunPod side BEFORE provisioning a new pod with the same project name.

6. **RunPod GraphQL spot allocator uses deprecated type**: `PodFindAndDeployInterruptableInput` 400s on every spot request; should be `PodRentInterruptableInput`. Crucible falls back to REST POST `/pods` which works on COMMUNITY tier but slower. Bug is in Crucible's `runpod` provider module, not in sfumato.

### Related
- [[Sfumato Phase 2 — Night-1 Wrap (2026-05-03)]]

---

## Memories (one fact per entry)

- The sfumato repo (`https://github.com/eren23/sfumato`) was made PUBLIC again on 2026-05-02 to enable Crucible fleet bootstrap without GITHUB_TOKEN forwarding. (was temporarily PRIVATE during Phase 1 paper-prep and broke our first GPU spike attempt)
- Sfumato uses Crucible Fleet on a single RunPod RTX 4090 spot (or A6000 fallback) for all GPU experiments. Direct SSH for one-off debugging.
- The Phase 2 budget was $20; through 2026-05-03 ~03:40 UTC we've spent $1.91. Next-session option-2 verifier estimated +$0.40.
- LoRA artifacts on HF: `eren23/sfumato-llada-prefix-robust-{v2,v3}` (Track 1) and `eren23/sfumato-llada-commit-{v2,v3}` (Track 2). All public.
- Frozen GSM8K dev set indices (200 problems) at `e4/data/gsm8k_dev_200.json` — used for all Phase 1+2 evaluations.
- W&B project: `eren23/sfumato-e4` is the source of truth for run state; Crucible's `get_project_run_status` is unreliable for long cmaj runs.
- TODO: Train Qwen-encoder verifier (D3.5 option 2) on 1750 labeled branches at `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl` + Phase-1 raw_cmaj jsonls. Pre-reg threshold ≥83%. Pod budget ~$0.40.
- TODO: If D3.5 option 2 fails, RANKING reverts to D1 (mode router). D1 needs more Workstream C real-mode traces — drive `phase2/inference_viz/server.py` through Gradio with manual interventions on 5-10 problems.
- Phase 1 paper headline cmajc v3+commit-v3 (seed=0 N=200) = 82.5%. Multi-seed (1, 2) at N=100 give 82.0% each — variance σ ≈ 0.3pp. Headline robust.
- Voting-rule gap is invariant across LoRA configs (8-12pp across base, v3, v3+commit-v3). It's a structural property of cmaj majority-vote aggregation.
