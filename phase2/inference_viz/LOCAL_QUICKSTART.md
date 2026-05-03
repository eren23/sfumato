# Local Quickstart — Watch sfumato inference live

3 ways to see the LLaDA + commit-LoRA + AR-extend system generate token-by-token.

## Option A — Mock mode (no GPU, instant)

For exploring the UI / interaction model without paying for a GPU.

```bash
cd /Users/eren/Documents/AI/sfumato

# One-shot launcher: spawns FastAPI backend + Gradio frontend
MOCK_MODELS=1 python3 phase2/inference_viz/launch.py --problem-idx 42
```

Open the printed Gradio URL (`http://127.0.0.1:7860`) in a browser. Click **Start session**, then drive generation one sub-block at a time with the **Continue / AR-extend / Cmaj-branch / Stop** buttons.

The 4×32 token grid colors entropy (cool blue = confident, warm amber = uncertain). Hover any cell for the token id, decoded string, top-5 alternatives, commit-LoRA status, and which mechanism produced it.

Mock mode synthesizes 4 deterministic sub-blocks with fake tokens and entropy values — useful for grokking the data flow without a 16GB model load.

## Option B — Real models via SSH tunnel to a pod

For watching ACTUAL LLaDA-8B + Qwen + commit-v3 generation live. Requires:
- A pod provisioned via Crucible (or any cloud GPU with ≥24GB VRAM)
- Local SSH client + the Crucible SSH key

### Step 1 — provision a pod

```bash
cd /Users/eren/Documents/AI/parameter-golf_dev
.venv/bin/python -c "
import sys, os, pathlib
for f in [pathlib.Path('.env'), pathlib.Path('.env.runpod.local')]:
    if f.exists():
        for line in f.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k,v=line.split('=',1); os.environ.setdefault(k.strip(), v.strip().strip('\"').strip(\"'\"))
sys.path.insert(0, 'src')
from crucible.mcp.tools import provision_project
print(provision_project({'project_name':'sfumato_e4','count':1,'interruptible':False}))
"
```

Wait ~60s for SSH ready. Get the pod's `host:port` from `crucible fleet status`.

### Step 2 — manual bootstrap (Crucible's project bootstrap is wedged for fastapi deps; one-liner does it)

```bash
HOST=...   # from fleet status
PORT=...

ssh -i ~/.ssh/id_ed25519_runpod -p $PORT root@$HOST '
  set -e && cd /workspace && rm -rf sfumato 2>/dev/null || true &&
  git clone --depth=1 https://github.com/eren23/sfumato.git sfumato && cd sfumato &&
  python3 -m venv .venv &&
  .venv/bin/pip install --quiet --upgrade pip &&
  .venv/bin/pip install --quiet torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 &&
  .venv/bin/pip install --quiet transformers==4.46.3 peft accelerate datasets huggingface_hub wandb numpy fastapi uvicorn requests pydantic &&
  echo BOOTSTRAP_DONE'
```

### Step 3 — start the server on the pod (foreground in a `tmux` session or via `ssh -f`)

```bash
# In one terminal: start a tmux session on the pod that runs the server
ssh -t -i ~/.ssh/id_ed25519_runpod -p $PORT root@$HOST '
  tmux new-session -d -s viz "cd /workspace/sfumato && \
    HF_TOKEN=$YOUR_HF_TOKEN HUGGINGFACE_HUB_TOKEN=$YOUR_HF_TOKEN \
    /workspace/sfumato/.venv/bin/python phase2/inference_viz/server.py --port 8765"'

# Verify it came up
ssh -i ~/.ssh/id_ed25519_runpod -p $PORT root@$HOST 'curl -s http://127.0.0.1:8765/healthz'
# Expect: {"ok": true, ...}
```

### Step 4 — tunnel the server's port to localhost

```bash
ssh -N -L 8765:127.0.0.1:8765 -i ~/.ssh/id_ed25519_runpod -p $PORT root@$HOST &
```

### Step 5 — run the Gradio frontend locally pointing at the tunnel

```bash
cd /Users/eren/Documents/AI/sfumato
SFUMATO_VIZ_BACKEND=http://127.0.0.1:8765 python3 phase2/inference_viz/app.py
```

Open `http://127.0.0.1:7860` — same UI as Option A but generation is REAL.

Cost: pod runs at $0.34/h on-demand RTX 4090. A 1-hour exploration session is ~$0.34.

## Option C — Headless trace generation (if you don't need the UI)

Already used in this session. Generates 7 STATUS-schema JSONL traces over varied directive sequences:

```bash
# On a pod with the bootstrap above
/workspace/sfumato/.venv/bin/python phase2/inference_viz/make_real_traces.py
```

Output lands in `phase2/inference_viz/traces/trace_real_p*_*.jsonl`. Each trace has 4 sub-block records with full StepState (entropy, top-k logits, commit-LoRA status, planned mechanism intervention, etc.). See `phase2/inference_viz/traces/make_real_traces_summary.json` for the index.

## Troubleshooting

- **"could not read Username for github.com"** — repo went private at some point; was made public again 2026-05-02. Confirm with `gh repo view eren23/sfumato --json visibility`.
- **`bash: line 1: /workspace/sfumato/.venv/bin/python: No such file or directory`** — bootstrap died at install_torch or install_uv. Crucible's auto-bootstrap is flaky; use the manual recipe in Step 2.
- **Gradio app loads but `Start session` errors with "connection refused"** — SSH tunnel died. Re-run Step 4.
- **Server `state=running` but `/healthz` 404** — wrong port. Server defaults to 8765. Check `--port` arg.
- **GPU OOM at LLaDA load** — pod has <24GB VRAM. Use RTX 4090 (24GB) or A6000 (48GB). The yaml fallback list at `parameter-golf_dev/.crucible/projects/sfumato_e4.yaml` already prefers 48GB.
