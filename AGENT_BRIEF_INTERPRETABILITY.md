# Sfumato Interpretability — Agent Brief

> Copy-paste this into a fresh agent (Claude Code with the Crucible MCP
> server connected). It is self-contained: project context, tool
> choices, staged plan, file paths, success criteria. The agent should
> work in this directory (`/Users/eren/Documents/ai/sfumato`) plus
> `/Users/eren/Documents/ai/sfumato_paper`.

## Context

Sfumato is a hybrid autoregressive + discrete-diffusion language model.
Single backbone, two heads (`head_ar` for next-token, `head_diff` for
mask-fill). Scales tested: 50M → 305M params on GPT-2 BPE vocab.
Trained on GSM8K + FineWeb-Edu. Currently 89+ checkpoints across
variants in `e5/results/`.

The empirical paper (TMLR-readiness draft at
`sfumato_paper/paper_C/`) characterises three things:

1. **Joint-training advantage** — composite beats compute-matched pure
   diffusion by −0.69 NLL at 200M.
2. **Data-efficiency crossover** at ≈1000 GSM8K problems (composite
   wins below, loses above).
3. **Mode-switching routing (Phase K)** — at 305M, diffusion drafts
   in parallel + AR refills bottom-K% confidence tokens delivers
   −0.58 NLL/token over pure AR.

**The paper is empirical. There is zero mechanistic interpretability
work yet.** This brief covers exactly that gap. The unique question
Sfumato can answer (and no one else has): *do AR and diff modes share
features inside the backbone, or specialise? What features drive the
mode-switching router's per-token decisions?*

This is publishable on its own as a follow-up paper, and the
interp scaffolding strengthens the existing TMLR submission's
limitations section.

## Goal

Build a mechanistic-interpretability layer on top of Sfumato:

- Train sparse autoencoders on the shared backbone, on each head, and
  on the routing decision pre-activations.
- Generate Neuronpedia-style attribution graphs for representative
  prompts (a GSM8K problem, a FineWeb completion, a mode-switch
  decision token).
- Produce a feature dashboard the lab can browse — local first, then
  optionally upload to Neuronpedia.
- Answer the headline question: "How much do AR and diff features
  overlap?" with a concrete metric (cosine similarity matrix of
  decoder columns + Jaccard on top-activating tokens).

Stretch: discover a real circuit driving Phase K mode-switching
decisions (which features predict "this token gets re-filled by AR"?).

## Recommended tooling — chosen for fit, not popularity

Sfumato is a **custom PyTorch model**, NOT a HuggingFace model. This
constrains choices:

| Need | Pick | Why for Sfumato |
|---|---|---|
| Hook framework | **nnsight** (via `nnterp`) | TransformerLens requires per-architecture conversion code; Sfumato is `CompositeLM`, not in their model list. nnsight wraps arbitrary PyTorch nn.Modules with a tracing API. nnterp (NeurIPS 2025) is a lightweight standardiser layer. |
| SAE training | **EleutherAI sparsify** | TopK activation (cleaner sparsity than L1), on-the-fly activations (no 100s of GB of cached residuals), custom hookpoints via pattern matching. SAELens is the popular alternative but assumes HF models + caches to disk. Try sparsify first; fall back to SAELens if you hit a corner case. |
| Attribution graphs | **circuit-tracer (Decode Research / Anthropic Fellows)** with `backend='nnsight'` | Native CLT (cross-layer transcoder) support, plus the experimental nnsight backend that handles non-HF architectures. Output is `.pt` + JSON consumable by Neuronpedia frontend. |
| Visualisation | **sae_dashboard / SAE-vis** locally first; **Neuronpedia self-hosted** later | sae_dashboard renders to static HTML in <1h, lets the lab browse without infra. Neuronpedia self-host needs Postgres + S3 — defer until features are interesting enough to share publicly. |
| Steering / validation | **repeng** (Theia Vogel) | Tiny dependency, just builds steering vectors. Use to verify SAE features actually causally matter: steer with the SAE-derived direction, see if mode-switch behaviour shifts. |
| Evals | **SAEBench** | 8 metrics over 200+ SAEs, lets you compare your SAE against known baselines on the same model class (where applicable). Probably skip for v1 — adds a week. |

### Explicit "do NOT use" calls

- **TransformerLens directly** — requires writing a `convert_composite_lm_weights` function, weeks of yak-shaving for a model that won't graduate to the public model list.
- **dictionary_learning** — fine library but slower-moving than sparsify, and its tutorials assume HF models. Use sparsify.
- **Penzai** — Google's tool, JAX-first. Sfumato is PyTorch. Wrong substrate.
- **pyvene** — strong on intervention but the SAE training story is thinner.

## Staged execution plan

Each phase has a Crucible MCP-tool invocation pattern. Phases are
sized so each one earns the next one's compute budget — don't try to
do all five at once.

### Phase 0 — Baseline + tooling sanity (1 day, no GPU)

**Goal:** install everything, verify model loads under nnsight, dump a
single layer's activations for one prompt.

1. `cd /Users/eren/Documents/ai/sfumato`
2. Create a `.crucible/recipes/interp-phase-0-bootstrap.yaml` that
   pins versions: `nnsight>=0.4`, `eai-sparsify>=0.3`,
   `circuit-tracer`, `repeng`. Use `crucible recipe save` to
   persist it.
3. Write `e5/interp/load_model.py` — a thin function that returns an
   `nnsight.NNsight` wrapper around `CompositeLM` loaded from a
   checkpoint path. Follow the pattern in `e5/score_probes.py:56–79`.
4. Write `e5/interp/dump_activations.py` — given a checkpoint and a
   prompt, dump residual-stream activations from every layer + the
   pre-`head_ar` / pre-`head_diff` logits to a `.pt` file under
   `e5/interp/cache/`.
5. Smoke test: run on the smallest checkpoint
   (`results/f1_fineweb/.../model.pt`) and a single 32-token prompt.
   File should land in <1MB.

Crucible tools: `recipe_save`, `note_add` to record per-step
findings, `runs_search` to query what you've done.

**Exit criteria:** activations dumped, files load back cleanly,
shapes match the architecture spec.

### Phase 1 — Train SAEs on the backbone (3-5 days, $30-50 GPU)

**Goal:** one SAE per layer of the F10 305M backbone. TopK k=64,
expansion factor 16x (so 512 d_model → 8192 features).

1. `crucible mcp call code_mutation_list` — verify Crucible's mutation
   surface is wired (you may need it later for sparsify config sweeps).
2. Pick the F10 checkpoint: `e5/results/f10_*/model.pt`. Confirm
   `config["d_model"] == 512`, `n_layers == 8`.
3. Write `e5/interp/train_saes.py` driving `eai-sparsify`'s
   `Trainer` with custom hookpoint patterns:
   `["model.layers.*.residual_post"]` (adapt to actual Sfumato module
   names — check `model_composite.py`).
4. Crucible-ise it: `crucible project new sfumato-saes --template generic`,
   set `env_set.CHECKPOINT=...`, `LAYER=...`, `EXPANSION=16`,
   `TOPK=64`. Use `provision_nodes count=2 gpu_type="RTX 4090"` for
   $0.40/hr spot capacity. 2 pods × 4 layers = 8 SAEs in parallel.
5. Tier the experiments: `smoke` first (1000 steps, sanity), then
   `screen` (50k steps, real training). Use Crucible's `dispatch_experiments`
   and `collect_results`.
6. Eval each SAE on the standard SAEBench 3-metric subset:
   reconstruction loss (delta loss on held-out FineWeb), L0 (sparsity),
   feature diversity (Jaccard on top-activating tokens across feature pairs).
7. `crucible mcp call context_push_finding` for any SAE with
   delta-loss < 0.1 and average L0 < 100 — those are the keepers.

Crucible tools: `provision_nodes`, `bootstrap_nodes`,
`design_enqueue_batch`, `dispatch_experiments`, `collect_results`,
`get_leaderboard`, `context_push_finding`, `note_add`.

**Exit criteria:** 8 trained SAEs in `e5/interp/saes/layer_{0..7}/`,
each scoring within 10% of the reconstruction-loss target.

### Phase 2 — Per-head SAEs + cross-head feature overlap (2-3 days, $20 GPU)

**Goal:** the headline scientific result for the interp follow-up
paper. Train a separate SAE on each head's pre-output activations.
Compare features.

1. Repeat Phase 1's training loop, but hookpoint =
   `model.head_ar.input` and `model.head_diff.input`. Same TopK,
   same expansion factor.
2. Compute the **cross-head feature overlap matrix**: cosine similarity
   between every AR-SAE decoder column and every diff-SAE decoder column.
   Save as `e5/interp/overlap_ar_diff.npy`.
3. Compute the **top-activating-token Jaccard**: for each (AR feature,
   diff feature) pair, what fraction of their top-100 activating tokens
   overlap?
4. Headline number: **what fraction of AR features have a >0.7 cosine
   sibling in the diff SAE, and vice versa?** Hypotheses:
   - >0.8 overlap → backbone learned one shared feature basis, modes
     differ only in head weights. Validates the "single representation,
     two readouts" design.
   - <0.4 overlap → modes specialise. Validates the "two-axis benefit"
     claim in the paper but suggests the backbone is bigger than needed
     (could distil into two smaller models).
   - 0.4-0.8 → partial sharing. The interesting case. Identify which
     concepts are shared (likely syntax) vs specialised (likely
     planning / arithmetic for AR, denoising / smoothing for diff).

Crucible tools: same as Phase 1, plus `context_push_finding` with
`scope="track"` to promote the overlap result.

**Exit criteria:** the overlap matrix exists, the headline number is
in `e5/interp/results/cross_head_overlap.md`, and a Crucible note is
written to `note_add` describing the finding.

### Phase 3 — Attribution graphs on representative prompts (2-3 days, $20 GPU)

**Goal:** for 3 representative prompts, generate a circuit-tracer
attribution graph that traces which features drive the output token.

Prompts:
- **G** — a GSM8K problem, run in AR mode end-to-end.
- **F** — a FineWeb completion, AR mode.
- **K** — a Phase K mode-switch token (find one where diff drafted
  X and AR re-filled to Y). This is the unique-to-Sfumato prompt.

1. Install circuit-tracer:
   `pip install git+https://github.com/decoderesearch/circuit-tracer`.
2. Build a custom `ReplacementModel` subclass for Sfumato.
   Use `backend='nnsight'` per the circuit-tracer docs (the
   experimental backend is the only path for non-HF models).
   - You'll need to map your trained SAEs from Phase 1 into
     circuit-tracer's "transcoder" format. There's a converter
     utility (`save_transcoders_to_cache`) — check the repo's
     `examples/` for the schema.
3. Run `circuit-tracer attribute --prompt "..." --transcoder_set ./sfumato_saes
   --slug prompt_G --graph_file_dir e5/interp/graphs/`.
4. For prompt K (the mode-switch token), produce TWO graphs: one with
   the AR head active, one with diff. The diff between them is the
   mode-switching circuit.
5. View locally via `circuit-tracer attribute ... --server` (spins up a
   local Neuronpedia-compatible viewer at localhost:8050).

Crucible tools: `enqueue_experiment` per graph generation (each is
~10 min on one pod), `collect_results`, `note_add` to record what
each graph shows.

**Exit criteria:** 3 graphs in `e5/interp/graphs/`, each renders in
the local viewer, each has a 1-paragraph reading in
`e5/interp/results/graph_readings.md`.

### Phase 4 — Causal validation via steering (1-2 days, no big GPU)

**Goal:** prove the features Phase 2/3 surfaced actually matter
causally, not just correlationally.

1. For the top-5 cross-head specialised AR features (lowest cosine
   sibling in diff SAE), build a steering vector with `repeng`:
   take the SAE decoder column, normalise, prepare to add to the
   residual stream.
2. Generate from F10 with and without steering. Compare AR vs diff
   mode outputs. If the feature truly drives "AR-mode-only" behaviour,
   steering with +k * direction should NOT change diff outputs but
   SHOULD change AR outputs.
3. For the Phase K routing decision: identify the top-3 features that
   the attribution graph identified as the "switch driver". Steer with
   them; measure whether the mode-switch rate changes on a held-out
   GSM8K batch.
4. The clean result: "steering with feature X shifts mode-switch rate
   by Y%, validating it causally drives routing."

Crucible tools: `enqueue_experiment` for the steering sweeps,
`runs_search` to compare runs, `context_push_finding` for any
causally-validated feature.

**Exit criteria:** at least one causally-validated feature with a
quantitative effect size, written into
`e5/interp/results/causal_validation.md`.

### Phase 5 — Paper section + Neuronpedia upload (1 day)

**Goal:** write the interpretability section for the next paper
revision (Sfumato paper_D? or a standalone follow-up).

1. `crucible mcp call note_generate_paper_draft track_name=sfumato-interp`
   — gathers findings + leaderboard + notes into a prompt envelope.
2. Run the orchestrator's LLM against that envelope (claude opus 4.7
   recommended). Submit via `note_generate_paper_draft action=submit`.
3. Markdown lands in `.crucible/notes/`. Manual polish into LaTeX
   under `sfumato_paper/paper_C/sections/interpretability.tex` (new file).
4. **Optional** — self-host Neuronpedia (docker compose from
   `github.com/hijohnnylin/neuronpedia`), import the SAEs +
   graphs locally, share read-only URL with collaborators.

Crucible tools: `note_generate_paper_draft`,
`hf_publish_findings` (if you want the SAE weights public),
`hf_publish_recipes` (for the recipe that reproduces this).

**Exit criteria:** a 2-3 page interpretability section in the paper
repo, citing the SAE training recipe + attribution graph URLs.

## Cost & timeline summary

| Phase | Wall-clock | GPU $ | Orchestrator $ |
|---|---|---|---|
| 0 | 1 day | $0 | $5 |
| 1 | 3-5 days | $30-50 | $10 |
| 2 | 2-3 days | $15-25 | $10 |
| 3 | 2-3 days | $15-25 | $20 (graphs eat tokens) |
| 4 | 1-2 days | $5-10 | $10 |
| 5 | 1 day | $0 | $20 (paper draft) |
| **Total** | **~2 weeks** | **$65-110** | **~$75** |

## Anti-patterns to avoid

- **Don't train one giant SAE on every layer concatenated.** Per-layer
  SAEs are the field standard for a reason — attribution graphs assume
  per-layer features.
- **Don't skip Phase 0.** If nnsight can't wrap CompositeLM cleanly
  the entire plan collapses. Fail-fast on Day 1.
- **Don't run circuit-tracer before SAEs are trained.** The tool needs
  pretrained transcoders; without them, the attribution edges have no
  feature labels and the graph is unreadable.
- **Don't try to upload to neuronpedia.org's public instance.** They
  only accept models with significant community interest. Self-host
  for your own lab.
- **Don't conflate this work with re-running the Sfumato paper's main
  results.** Use the existing F10 checkpoint, don't retrain.
- **Don't burn GPU on Phase 2 before Phase 1 SAEs converge.** Cheap
  validation per SAE: delta-loss < 0.1 on FineWeb held-out. Reject
  and retrain anything that misses that bar.

## Files this brief will create / modify

```
sfumato/
  e5/interp/                         # NEW — all interp code lives here
    __init__.py
    load_model.py                    # nnsight wrapper around CompositeLM
    dump_activations.py              # phase 0 sanity dump
    train_saes.py                    # phase 1+2 SAE training
    cross_head_overlap.py            # phase 2 analysis
    attribution_graphs.py            # phase 3 driver
    steering.py                      # phase 4 causal validation
    cache/                           # activation cache
    saes/                            # trained SAE weights
    graphs/                          # attribution graph .pt + JSON
    results/                         # markdown writeups per phase
  .crucible/recipes/
    interp-phase-0-bootstrap.yaml
    interp-saes-per-layer.yaml
    interp-cross-head.yaml
    interp-attribution.yaml
    interp-steering.yaml

sfumato_paper/
  paper_C/sections/interpretability.tex   # NEW — phase 5 output
```

## References

Tooling:
- nnsight: https://nnsight.net + https://arxiv.org/abs/2511.14465 (nnterp wrapper)
- eai-sparsify: https://github.com/EleutherAI/sparsify
- circuit-tracer: https://github.com/decoderesearch/circuit-tracer
- Neuronpedia self-host: https://github.com/hijohnnylin/neuronpedia
- SAELens (fallback): https://github.com/decoderesearch/SAELens
- repeng: https://github.com/vgel/repeng
- SAEBench: https://arxiv.org/abs/2503.09532
- Attribution-graph methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Sfumato:
- Model: `/Users/eren/Documents/ai/sfumato/e5/model_composite.py` (CompositeLM)
- Loading pattern: `/Users/eren/Documents/ai/sfumato/e5/score_probes.py:56-79`
- Existing probes: `/Users/eren/Documents/ai/sfumato/e5/score_probes.py`
- Paper: `/Users/eren/Documents/ai/sfumato_paper/paper_C/`
- Project CLAUDE.md hard rules: `/Users/eren/Documents/ai/sfumato/CLAUDE.md`

Crucible MCP tools to use (each is `crucible mcp call <name>` or via
your MCP client):
- Recipe persistence: `recipe_save`, `recipe_get`, `recipe_list`
- Project lifecycle: `provision_nodes`, `bootstrap_nodes`, `dispatch_experiments`, `collect_results`, `destroy_nodes`
- Experiment design: `design_enqueue_batch`, `enqueue_experiment`
- Knowledge: `note_add`, `note_search`, `context_push_finding`, `finding_promote`
- Analysis: `get_leaderboard`, `runs_search`, `get_research_briefing`
- Tournament (use for SAE arch comparison): `hypothesis_tournament_create/pair/submit/rank`
- Paper draft: `note_generate_paper_draft`
- Publish (optional): `hf_publish_findings`, `hf_publish_recipes`

## How to start

Drop this brief into a fresh Claude Code session in
`/Users/eren/Documents/ai/sfumato/` with the Crucible MCP server
connected. Begin Phase 0. Use `note_add` after every meaningful
step so your future self (and your collaborators) can replay the
work. Do not skip phases. Ask for review before moving from
Phase N to N+1 — each phase's exit criteria gates the next.
