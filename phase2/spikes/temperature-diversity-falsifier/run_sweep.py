"""End-to-end Phase-2 spike runner: provision → bootstrap → 4-tau sweep → collect → destroy.

Calls Crucible's internal MCP-tool functions directly because the MCP server is offline
mid-session and the CLI doesn't expose the project-spec-aware provision path.

Run from any cwd; loads parameter-golf_dev/.env first to get RUNPOD_API_KEY etc.
"""
from __future__ import annotations
import json
import os
import pathlib
import sys
import time

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SPIKE_DIR = pathlib.Path(__file__).parent
PROJECT = "sfumato_e4"

# 1. Load env from parameter-golf_dev/.env (RUNPOD_API_KEY etc.)
def load_env(path: pathlib.Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env(PG_ROOT / ".env")
load_env(PG_ROOT / ".env.runpod.local")

# 2. Make Crucible importable + cwd matters for load_config
sys.path.insert(0, str(PG_ROOT / "src"))
os.chdir(PG_ROOT)

from crucible.mcp.tools import (  # noqa: E402
    provision_project,
    bootstrap_project_tool,
    fleet_refresh,
    get_fleet_status,
    run_project,
    get_project_run_status,
    destroy_nodes,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def must(result: dict, label: str) -> dict:
    if "error" in result:
        log(f"FAIL {label}: {result['error']}")
        sys.exit(2)
    return result


def main() -> None:
    spec_path = SPIKE_DIR / "sweep_spec.json"
    runs = json.loads(spec_path.read_text())
    log(f"loaded {len(runs)} runs from {spec_path.name}")

    # ---- 1. provision spot pod ----
    log("provisioning spot pod via project spec...")
    # Spot tried first; falls back to on-demand if spot capacity unavailable / Crucible GraphQL bug.
    res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": True})
    if "error" in res or res.get("created", 0) == 0:
        log(f"  spot failed ({res.get('error') or 'created=0'}); falling back to on-demand")
        res = must(provision_project({"project_name": PROJECT, "count": 1, "interruptible": False}), "provision-ondemand")
    new_node = res["new_nodes"][0]["name"]
    log(f"  pod {new_node} requested")

    try:
        # ---- 2. wait for SSH (8 min budget, evict-tolerant) ----
        last_state = None
        evict_count = 0
        for _ in range(32):  # 32 * 15s = 8 min
            time.sleep(15)
            r = fleet_refresh({})
            n = next((x for x in r.get("nodes", []) if x.get("name") == new_node), None)
            cur = n.get('state') if n else '?'
            log(f"  refresh: {cur} env={n.get('env_ready') if n else '?'} ssh={n.get('ssh_host') if n else '-'}")
            if cur == "stopped" and last_state in ("running", "starting"):
                evict_count += 1
                log(f"  EVICTION detected (#{evict_count}) — RunPod killed on-demand pod")
                if evict_count >= 2:
                    log("FAIL: RunPod evicted pod twice; abandoning this attempt")
                    sys.exit(3)
            last_state = cur
            if n and n.get("state") == "running" and n.get("ssh_host"):
                log(f"  ssh ready @ {n['ssh_host']}:{n['ssh_port']}")
                break
        else:
            log("FAIL: pod never reached running+ssh state in 8 minutes")
            sys.exit(3)

        # ---- 3. bootstrap project ----
        log("bootstrapping project (clone + venv + install + setup)...")
        res = bootstrap_project_tool({"project_name": PROJECT, "node_names": [new_node]})
        if res.get("bootstrapped", 0) != 1:
            err = res.get("nodes", [{}])[0].get("error", "unknown")
            log(f"FAIL bootstrap: {err}")
            sys.exit(4)
        log("  bootstrap OK")

        # ---- 4. dispatch each tau, wait per-run ----
        run_ids = []
        for run in runs:
            log(f"dispatching {run['name']} TEMP={run['config']['TEMP']}...")
            r = must(run_project({"project_name": PROJECT, "overrides": run["config"]}), f"run {run['name']}")
            rid = r.get("run_id") or r.get("launch_id")
            log(f"  enqueued run_id={rid}")
            run_ids.append((run["name"], rid))

            # Poll status — run sequentially since 1 node
            for _ in range(80):  # ~20 min max per run
                time.sleep(15)
                rstatus = get_project_run_status({"project_name": PROJECT, "run_id": rid})
                state = rstatus.get("status") or rstatus.get("state")
                log(f"    {rid} -> {state}")
                if state in ("succeeded", "completed", "failed", "cancelled", "error"):
                    break
            else:
                log(f"  WARN: {rid} timeout — continuing anyway")

        # ---- 5. collect results to local ----
        log("results collection: results live in pod /workspace/sfumato/e4/results/ + W&B")
        log(f"  WandB project: sfumato-e4 (pull via wandb CLI later)")
        log(f"  run_ids: {run_ids}")

    finally:
        # ---- 6. destroy pod (always) ----
        log(f"destroying pod {new_node}...")
        destroy_nodes({"node_names": [new_node]})
        log("done")


if __name__ == "__main__":
    main()
