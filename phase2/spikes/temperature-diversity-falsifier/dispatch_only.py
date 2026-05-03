"""Re-dispatch sweep against an already-bootstrapped pod (no provision/bootstrap/destroy).

Use after the pod is alive and the sweep_spec was updated. Cancels any in-flight runs
for sfumato_e4 first to free the node.
"""
from __future__ import annotations
import json, os, pathlib, sys, time

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SPIKE_DIR = pathlib.Path(__file__).parent
PROJECT = "sfumato_e4"

def load_env(path: pathlib.Path) -> None:
    if not path.exists():
        return
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
    cancel_experiment,
    get_fleet_status,
)

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main() -> None:
    runs = json.loads((SPIKE_DIR / "sweep_spec.json").read_text())
    log(f"loaded {len(runs)} runs")

    # Cancel any active runs on the project (in-flight tau=1.0 from the killed script)
    fs = get_fleet_status({})
    for r in fs.get("active_project_runs", []):
        if r.get("project") == PROJECT and r.get("status") in ("running", "launched", "launching"):
            rid = r.get("run_id")
            log(f"  cancelling stuck run {rid}")
            cancel_experiment({"run_id": rid})

    # Sequential dispatch
    results = []
    for run in runs:
        log(f"dispatching {run['name']} TEMP={run['config']['TEMP']} LORA={run['config']['LORA_PATH']}")
        r = run_project({"project_name": PROJECT, "overrides": run["config"]})
        if "error" in r:
            log(f"  FAIL dispatch: {r['error']}")
            results.append((run['name'], None, "dispatch_error"))
            continue
        rid = r.get("run_id") or r.get("launch_id")
        log(f"  enqueued run_id={rid}")

        for _ in range(80):  # ~20 min
            time.sleep(15)
            st = get_project_run_status({"project_name": PROJECT, "run_id": rid})
            state = st.get("status")
            metrics = st.get("metrics") or {}
            log(f"    {rid} -> {state} acc={metrics.get('accuracy', '?')}")
            if state in ("succeeded", "completed", "failed", "cancelled", "error"):
                results.append((run['name'], rid, state))
                if state in ("failed", "error"):
                    tail = st.get("log_tail", "")[-500:]
                    log(f"    log tail: {tail}")
                break
        else:
            log(f"  WARN: {rid} timeout — moving on")
            results.append((run['name'], rid, "timeout"))

    # Final summary
    log("=== sweep complete ===")
    for name, rid, state in results:
        log(f"  {name:18s} {state:12s} {rid}")

    out = SPIKE_DIR / "results_live.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")

if __name__ == "__main__":
    main()
