"""Single-shot substrate harvest: provision A6000 + bootstrap + ONE cmaj N=200 b=5 τ=0.7 run + destroy.

Targets the D3.5 verifier substrate. No batch, no per-run dispatch — one run-project call.
Override mechanism is broken (TEMP/SEED don't propagate), but defaults give us TEMP=0.7 SEED=0
which IS what we want for substrate. So no override needed.

PYTORCH_CUDA_ALLOC_CONF set in yaml.env_set, A6000 first in gpu_type fallback list.
"""
from __future__ import annotations
import json, os, pathlib, sys, time

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SPIKE_DIR = pathlib.Path(__file__).parent
PROJECT = "sfumato_e4"


def load_env() -> None:
    for f in [PG_ROOT / ".env", PG_ROOT / ".env.runpod.local"]:
        if not f.exists(): continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
sys.path.insert(0, str(PG_ROOT / "src"))
os.chdir(PG_ROOT)

from crucible.mcp.tools import (  # noqa: E402
    provision_project,
    bootstrap_project_tool,
    fleet_refresh,
    run_project,
    get_project_run_status,
    destroy_nodes,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("provisioning A6000 spot/on-demand pod...")
    res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": True})
    if "error" in res or res.get("created", 0) == 0:
        log(f"  spot fail; on-demand fallback")
        res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": False})
        if "error" in res or res.get("created", 0) == 0:
            log(f"FAIL: {res.get('error')}")
            sys.exit(2)
    new_node = res["new_nodes"][0]["name"]
    log(f"  pod {new_node} requested")

    try:
        for _ in range(32):
            time.sleep(15)
            r = fleet_refresh({})
            n = next((x for x in r.get("nodes", []) if x.get("name") == new_node), None)
            if n and n.get("state") == "running" and n.get("ssh_host"):
                log(f"  ssh ready @ {n['ssh_host']}:{n['ssh_port']}")
                break
            log(f"  refresh: {n.get('state') if n else '?'}")
        else:
            log("FAIL: pod never ready in 8min")
            sys.exit(3)

        log("bootstrapping...")
        b = bootstrap_project_tool({"project_name": PROJECT, "node_names": [new_node]})
        if b.get("bootstrapped", 0) != 1:
            log(f"FAIL bootstrap: {b.get('nodes', [{}])[0].get('error', '?')}")
            sys.exit(4)
        log("  bootstrap OK")

        log("dispatching substrate-N200-tau0.7...")
        cfg = {
            "CONDITION": "cmaj",
            "K_STEPS": "64",
            "N_PROBLEMS": "200",
            "BRANCHES": "5",
            "TEMP": "0.7",
            "SEED": "0",
            "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
            "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
            "MOCK_MODELS": "0",
            "WANDB_RUN_NAME": "phase2-substrate-N200-tau07-seed0",
        }
        r = run_project({"project_name": PROJECT, "overrides": cfg})
        if "error" in r:
            log(f"FAIL dispatch: {r['error']}")
            return
        rid = r.get("run_id") or r.get("launch_id")
        log(f"  enqueued {rid}")
        for _ in range(160):  # ~40 min
            time.sleep(15)
            st = get_project_run_status({"project_name": PROJECT, "run_id": rid})
            state = st.get("status")
            if state in ("succeeded", "completed", "failed", "cancelled", "error"):
                tail = (st.get("log_tail") or "")[-400:]
                log(f"  terminal: {state}")
                log(f"  log tail: {tail}")
                (SPIKE_DIR / "substrate_run_status.json").write_text(json.dumps({"rid": rid, "state": state, "tail": tail}, indent=2))
                break
        else:
            log("  substrate run timed out (40m)")

    finally:
        log(f"destroying pod {new_node}...")
        destroy_nodes({"node_names": [new_node]})
        log("done")


if __name__ == "__main__":
    main()
