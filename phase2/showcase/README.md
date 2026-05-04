# Sfumato showcase

A static HTML query interface for browsing sfumato's GSM8K validation
traces. Lets a non-expert visitor see, per problem, the question, the
gold answer, all 5 LLaDA branch reasoning traces, the vote tally, and
which (if any) branches the commit-LoRA repaired.

Pure read-only. No backend, no API keys, no ongoing cost. Designed to
be served from GitHub Pages.

## Layout

```
phase2/showcase/
├── README.md            (this file)
├── build_examples.py    # one-shot builder: JSONL → static/examples.json
└── static/              # GitHub-Pages-deployable, self-contained
    ├── index.html       # browse / detail / about pages
    ├── style.css        # palette imported from phase2/figures/palette.py
    ├── app.js           # vanilla JS, no framework
    └── examples.json    # generated artifact (~1 MB)
```

## Build the examples bank

```bash
python3 phase2/showcase/build_examples.py
```

Reads:
- `e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`
- `e4/results/raw_cmajc_k64_seed{1,2}_b5_v3LoRA_N100.jsonl`

Joins question text from `gsm8k`/`main`/`test` via the `datasets`
library, tags every example with one or more category labels, and
writes `examples.json` consumed by the frontend.

Tags currently emitted:

| Tag | Meaning |
|---|---|
| `unanimous_correct` | all 5 branches voted same correct answer |
| `unanimous_wrong` | all 5 agreed but on a wrong answer |
| `clear_majority_correct` / `_wrong` | 4-of-5 |
| `near_tie_correct` / `_wrong` | 3-of-5 quorum |
| `redundancy_save` | branches dissented, vote still correct |
| `commit_lora_repair` | cmaj wrong on this idx, cmajc correct |
| `commit_lora_active` | cmajc condition (commit-LoRA enabled) |
| `esc_early_trigger` | (post-S1 only) ESC quorum fired before block 3 |
| `no_extractable_answer` | regex extraction returned empty |

## Preview locally

```bash
cd phase2/showcase/static
python3 -m http.server 8080
```

Open `http://localhost:8080/` in a browser. `examples.json` is colocated
with `index.html` so all paths are relative.

## Deploy to GitHub Pages

The `static/` dir is fully self-contained (~1 MB total) and ready to
serve as-is. Two deploy paths:

1. **From this repo** — enable Pages with the source set to the
   `static/` subdirectory: Settings → Pages → "Deploy from a branch"
   → branch=`main` → folder=`/phase2/showcase/static`.

2. **From the paper repo (`sfumato_paper`)** — copy
   `phase2/showcase/static/*` into `docs/showcase/` on the paper repo.
   The paper repo is the public-facing artifact; sfumato (this repo)
   stays for source.

## Schema (one record)

```json
{
  "source": "cmajc_n100_v3LoRA_seed1",
  "id": "0",
  "idx": 0,
  "condition": "cmajc",
  "k_steps": 64,
  "seed": 1,
  "lora_path": "eren23/sfumato-llada-prefix-robust-v3",
  "commit_lora_path": "eren23/sfumato-llada-commit-v3",
  "n_branches": 5,
  "question": "Janet's ducks lay 16 eggs per day. ...",
  "gold": "18",
  "gold_rationale": "Janet sells 16 - 3 - 4 = 9 ...\n#### 18",
  "pred": "18",
  "correct": true,
  "flops": 656998400000000,
  "wallclock_ms": null,
  "branches": ["...trace 0...", "...trace 1...", ..., "...trace 4..."],
  "votes_str": "18 | 18 | 18 | 18 | 18",
  "winner": "18",
  "esc_trigger_block": null,
  "esc_branches_pruned": null,
  "tags": ["unanimous_correct", "commit_lora_active"]
}
```

## Future passes

- **v0.5 (post-S0+S1):** extend `build_examples.py` to read the post-spike
  JSONLs (`*_S0S1.jsonl`); the showcase will then surface
  `wallclock_ms` and `esc_trigger_block` automatically because the
  schema already accommodates them.
- **v1 (post-S4):** add a per-problem speedup bar chart to the detail
  page using paired pre/post wallclock columns.
- **v2 (optional):** mirror as a Gradio Space on Hugging Face for
  live-querying with a hosted model.
