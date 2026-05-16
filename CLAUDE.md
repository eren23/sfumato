# Sfumato — project notes for future Claude sessions

## What this repo is

Composite AR + discrete-diffusion language-model training from scratch.
Main active area: `e5/` (composite training, FineWeb / GSM8K substrates).
Inactive but committed: `e2/`, `e3/`, `e4/` (inference-bandage Phase 1-3
work on frozen LLaDA + Qwen).

Paper drafts live in a separate repo: `/Users/eren/Documents/ai/sfumato_paper/`
(paper_A, paper_B, paper_C).

## Hard rules for training scripts (`e5/train.py` and any new driver)

### 1. ALWAYS save optimizer state + step + lr-schedule state — not just model weights.

Current `train_one()` only saves `{config, state_dict, variant, n_params,
max_steps}` at the very end. **This is a bug** for any non-trivial run:
- mid-training crash = all progress lost
- "continue training" not actually possible (Adam moments + LR cosine
  position not preserved)

For new training scripts, save a full checkpoint dict periodically (every
~1000–5000 steps, depending on cost) AND at end:
```python
torch.save({
    "config": cfg.__dict__,
    "model_state_dict": model.state_dict(),
    "optim_state_dict": optim.state_dict(),
    "step": step,
    "peak_lr": peak_lr,
    "max_steps": max_steps,
    "variant": variant,
    "n_params": n_params,
    "rng_torch": torch.get_rng_state(),
    "rng_numpy": np.random.get_state(),
    # plus any sample/val/metric state if reproducibility matters
}, ckpt_path)
```

And accept a `resume_from=Path|None` arg in the training loop that:
- loads model state_dict
- loads optim state_dict
- sets `start_step = ckpt["step"] + 1`
- replays the lr schedule from that step

Without this, every long run is one OOM/network-blip away from total
restart, and stopping early to save cost / pivot is impossible.

### 2. Up-front `wandb.init()` for visibility during tokenization.

Don't bury wandb init inside `train_one()` — by then tokenization has
been running silently for 30+ min. Driver scripts should do a short
`wandb.init(job_type="driver")` at the top of `main()` so the run shows
in the dashboard immediately. Then `train_one()` opens its own per-variant
run later. See `e5/scripts/f7_1b_emerge.py` and
`e5/scripts/f8_paramgolf_small.py` for the pattern.

### 3. Log val NLL + sample generations to wandb during training, not only at end.

`train_one()` takes optional `val_problems`, `sample_prompts`,
`val_every`, `sample_every`, `tokenizer_for_samples` kwargs. Pass them
from any driver — gives loss curves AND text samples in wandb as the
run progresses.

### 4. `WANDB_MODE=offline` is a trap.

Was set in `.crucible/projects/sfumato_e5.yaml` originally — caused
hours of "why isn't wandb showing anything?" debugging. Always
`WANDB_MODE=online` for live runs.

### 5. Hard-coded compute estimates lie.

- FineWeb-Edu streaming from HF is ~1.3M tok/sec CPU-bound — for 6B
  tokens this is 4.5 hr just to tokenize, not the 60 min I estimated.
- A40 + 300M model + BS=32 T=1024 + bf16 = **OOM** (38 GB used, 6 GB
  free, dies on backward). Use BS≤16 at that scale.
- A40 at 1B model + BS=4 T=1024 = ~0.5 sec/step. **Not** 1–2 sec.
- "Param count from arch" requires measuring (e.g. via `model.num_params()`),
  not napkin math. F7 advertised as "1B" was actually 760M because I
  estimated wrong.

### 6. Pod gotchas

- RunPod sometimes has zero 4090 / A100 availability. The spec
  (`.crucible/projects/sfumato_e5.yaml`) puts A40 first because it's
  reliably available.
- After `bootstrap_project`, wandb's API key lives in
  `/workspace/sfumato/.env` (8 lines, format quirky). Source via
  `set -a; source /workspace/sfumato/.env; set +a` before launching.
  Don't try to parse it line-by-line with grep — bash `source` is the
  reliable path.
- `pip install ... --break-system-packages` required (Python 3.12 +
  PEP 668 externally-managed-environment).
- Default `bootstrap_project` may NOT install our specific deps. Install
  `transformers==4.46.3 datasets huggingface_hub numpy wandb` explicitly
  after bootstrap.

### 7. Disk / cache layout on pods

- FineWeb token cache: `~/.cache/sfumato_e5/fineweb_gpt2_{N_TOKENS}.npy`
  - 6B tokens = 12 GB cached file (uint16)
  - Survives `bootstrap_project` re-runs if pod is reused
- Training output: `/workspace/sfumato/e5/results/{run_name}/{variant}/`
  contains `model.pt + summary.json + train_log.jsonl + samples.md +
  score.json`
- model.pt files NOT committed to git (see `.gitignore` excludes
  `e5/results/**/*.pt`).

### 8. Pod cost discipline

- 1 A40 ≈ $0.44/hr on-demand. 1 A100 80GB ≈ $1.39/hr.
- After ANY experiment finishes, destroy the pod immediately
  (`mcp__crucible-fleet__destroy_nodes node_names=[...]`).
- Don't leave idle pods overnight — burned $5+ multiple times.

### 9. Don't lie to the user about ETAs

When in doubt: report a wider range. "ETA 2-4 hr" beats "ETA 1 hr" that
turns out to be 4.5 hr. The user can plan around uncertainty; they can't
plan around confidently-wrong estimates.

## Canonical results doc

`e5/results/T0_PROBES_FINAL.md` is the authoritative summary of all
Phase D / E / F / G findings. Update it (don't replace) when adding new
experiments.

Paper class verdict (locked at end of Phase F + G): **TMLR with
caveats / strong workshop**. Headline: composite training has a real
AR-axis tax (+0.12–0.19 NLL consistent across 8M → 760M) and a real
diff-axis advantage (−0.35–0.74 NLL vs compute-matched pure-diff,
grows with scale up to 300M then plateaus).
