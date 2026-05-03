"""3-hour autonomous queue: wait for 4-tau sweep → D3.5 substrate → multi-seed → C real-mode → destroy.

Run alongside dispatch_only.py. Triggers after results_live.json appears (signals 4-tau done).
Reuses the alive pod. Destroys pod at end.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SPIKE_DIR = pathlib.Path(__file__).parent
PHASE2_DIR = SPIKE_DIR.parent.parent
PROJECT = "sfumato_e4"
RESULTS_LIVE = SPIKE_DIR / "results_live.json"
QUEUE_RESULTS = PHASE2_DIR / "spikes" / "queue_followup_results.json"


def load_env(path: pathlib.Path) -> None:
    if not path.exists(): return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env(PG_ROOT / ".env")
load_env(PG_ROOT / ".env.runpod.local")
sys.path.insert(0, str(PG_ROOT / "src"))
os.chdir(PG_ROOT)

from crucible.mcp.tools import (  # noqa: E402
    run_project,
    get_project_run_status,
    destroy_nodes,
    get_fleet_status,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_pod() -> str | None:
    fs = get_fleet_status({})
    for n in fs.get("nodes", []):
        if n.get("name", "").startswith("parameter-golf__sfumato_e4-") and n.get("state") in ("running", "ready"):
            return n["name"]
    return None


def run_one(name: str, config: dict, max_wait_s: int = 1500) -> dict:
    """Dispatch one experiment, poll to terminal, return result dict."""
    log(f"  dispatching {name} cfg={config}")
    r = run_project({"project_name": PROJECT, "overrides": config})
    if "error" in r:
        log(f"    DISPATCH FAIL: {r['error']}")
        return {"name": name, "rid": None, "state": "dispatch_error", "error": r["error"]}
    rid = r.get("run_id") or r.get("launch_id")
    log(f"    enqueued {rid}")
    deadline = time.time() + max_wait_s
    last_state = None
    while time.time() < deadline:
        time.sleep(20)
        st = get_project_run_status({"project_name": PROJECT, "run_id": rid})
        state = st.get("status")
        if state != last_state:
            log(f"    {rid} -> {state}")
            last_state = state
        if state in ("succeeded", "completed", "failed", "cancelled", "error"):
            metrics = st.get("metrics") or {}
            return {"name": name, "rid": rid, "state": state, "metrics": metrics, "log_tail": st.get("log_tail", "")[-800:]}
    log(f"    {rid} TIMEOUT after {max_wait_s}s")
    return {"name": name, "rid": rid, "state": "timeout"}


def main() -> None:
    # 1. wait for 4-tau sweep done
    log("waiting for 4-tau sweep (results_live.json)...")
    t0 = time.time()
    while not RESULTS_LIVE.exists():
        if time.time() - t0 > 3600:
            log("FAIL: 4-tau sweep didn't complete in 1h, abandoning")
            return
        time.sleep(30)
    log(f"  4-tau sweep done ({(time.time()-t0)/60:.1f}m wait)")

    # 2. find alive pod
    pod = find_pod()
    if not pod:
        log("FAIL: no alive sfumato_e4 pod found; aborting")
        return
    log(f"  using pod {pod}")

    queue_results = []
    try:
        # ---- Phase 2: D3.5 substrate (τ=0.7 and τ=1.0, N=100) ----
        for tau in ["0.7", "1.0"]:
            cfg = {
                "CONDITION": "cmaj",
                "K_STEPS": "64",
                "N_PROBLEMS": "100",
                "BRANCHES": "5",
                "TEMP": tau,
                "SEED": "0",
                "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
                "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
                "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
                "MOCK_MODELS": "0",
                "WANDB_RUN_NAME": f"d35-substrate-tau-{tau}-N100",
            }
            res = run_one(f"d35-substrate-tau-{tau}", cfg, max_wait_s=2400)
            queue_results.append(res)

        # ---- Phase 3: Multi-seed v3 cmajc (N=100, seeds 1,2) ----
        for seed in ["1", "2"]:
            cfg = {
                "CONDITION": "cmajc",
                "K_STEPS": "64",
                "N_PROBLEMS": "100",
                "BRANCHES": "5",
                "TEMP": "0.7",
                "SEED": seed,
                "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
                "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
                "COMMIT_N_BLOCKS": "3",
                "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
                "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
                "MOCK_MODELS": "0",
                "WANDB_RUN_NAME": f"v3-cmajc-N100-seed{seed}",
            }
            res = run_one(f"cmajc-v3-seed-{seed}", cfg, max_wait_s=2400)
            queue_results.append(res)

        # ---- Phase 4: C real-mode trace ----
        # Run a single c2c (commit on last N blocks) on problem-idx 42 with N=1, save trace.
        cfg = {
            "CONDITION": "c2c",
            "K_STEPS": "64",
            "N_PROBLEMS": "1",
            "BRANCHES": "1",
            "TEMP": "0.0",
            "SEED": "42",
            "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
            "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
            "COMMIT_N_BLOCKS": "3",
            "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
            "MOCK_MODELS": "0",
            "WANDB_RUN_NAME": "phase2-c-real-mode-trace",
        }
        res = run_one("c-real-mode-trace", cfg, max_wait_s=600)
        queue_results.append(res)

    finally:
        QUEUE_RESULTS.write_text(json.dumps(queue_results, indent=2, default=str))
        log(f"wrote {QUEUE_RESULTS}")
        log(f"destroying pod {pod}...")
        destroy_nodes({"node_names": [pod]})
        log("done")


if __name__ == "__main__":
    main()
