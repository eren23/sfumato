"""Night-1 autonomous runner: substrate + multi-seed + verifier eval, all in one provision cycle.

Workarounds applied:
- yaml.env_set has PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (mitigates OOM cascade)
- yaml.gpu_type fallback list prefers 48GB GPUs (A6000, L40S, RTX 6000 Ada)
- Crucible's run_project override mechanism is broken; we work around by editing
  yaml.env_set between runs (set CONDITION/TEMP/SEED/etc directly in spec, then dispatch)
- Crucible state-tracking is unreliable on wandb-exit-nonzero; W&B summary is ground truth
- Clean inventory before provision (avoid ghost-orphan dual-bootstrap)

Sequence (one provision cycle):
  Phase A: provision A6000 spot (on-demand fallback) + bootstrap with verbose error capture
  Phase B: substrate run (cmaj N=200 b=5 τ=0.7 v3) — 1000 labeled branches for D3.5
  Phase C: multi-seed v3 cmajc N=100 b=5 seed=1
  Phase D: multi-seed v3 cmajc N=100 b=5 seed=2
  Phase E: pull all raw_*.jsonl back to local e4/results/
  Phase F: destroy pod
"""
from __future__ import annotations
import json, os, pathlib, sys, time, subprocess, shutil, copy

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SFUMATO_ROOT = pathlib.Path("/Users/eren/Documents/AI/sfumato")
SPIKE_DIR = pathlib.Path(__file__).parent
PROJECT = "sfumato_e4"
YAML_PATH = PG_ROOT / ".crucible/projects/sfumato_e4.yaml"
NIGHT_LOG = SPIKE_DIR / "night_run.log"
NIGHT_RESULTS = SPIKE_DIR / "night_run_results.json"

PHASES = [
    {
        "name": "B-substrate-N200-tau07",
        "env_set": {
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
            "WANDB_RUN_NAME": "night-substrate-N200-tau07-v3",
        },
        "max_wait_s": 3000,
    },
    {
        "name": "C-multiseed-cmajc-seed1",
        "env_set": {
            "CONDITION": "cmajc",
            "K_STEPS": "64",
            "N_PROBLEMS": "100",
            "BRANCHES": "5",
            "TEMP": "0.7",
            "SEED": "1",
            "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
            "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
            "COMMIT_N_BLOCKS": "3",
            "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
            "MOCK_MODELS": "0",
            "WANDB_RUN_NAME": "night-cmajc-N100-seed1-v3",
        },
        "max_wait_s": 2400,
    },
    {
        "name": "D-multiseed-cmajc-seed2",
        "env_set": {
            "CONDITION": "cmajc",
            "K_STEPS": "64",
            "N_PROBLEMS": "100",
            "BRANCHES": "5",
            "TEMP": "0.7",
            "SEED": "2",
            "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
            "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
            "COMMIT_N_BLOCKS": "3",
            "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
            "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
            "MOCK_MODELS": "0",
            "WANDB_RUN_NAME": "night-cmajc-N100-seed2-v3",
        },
        "max_wait_s": 2400,
    },
]


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
    get_fleet_status,
    run_project,
    get_project_run_status,
    destroy_nodes,
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(NIGHT_LOG, "a") as f:
        f.write(line + "\n")


def write_yaml_env_set(env_set: dict[str, str]) -> None:
    """Edit yaml's env_set block in-place. Preserves all other yaml content.

    Strategy: re-write env_set: section between markers. Uses simple line scan,
    not yaml round-trip, to keep diffs minimal.
    """
    text = YAML_PATH.read_text()
    lines = text.splitlines()
    # Find env_set: block
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip() == "env_set:":
            out.append("env_set:")
            # Replace block
            out.append("  WANDB_PROJECT: sfumato-e4")
            out.append("  WANDB_MODE: online")
            out.append("  HF_HOME: /workspace/sfumato/.hf_cache")
            out.append("  TRANSFORMERS_CACHE: /workspace/sfumato/.hf_cache")
            out.append("  PYTORCH_CUDA_ALLOC_CONF: \"expandable_segments:True\"")
            for k, v in env_set.items():
                # ALWAYS quote — Crucible's sync.py shlex.quote(val) crashes on
                # non-strings (yaml ints), silently skipping the env_set entry.
                # Force every value to be a quoted yaml string.
                v_str = str(v).replace('"', '\\"')
                out.append(f'  {k}: "{v_str}"')
            # Skip original env_set block lines (until next top-level key or blank)
            i += 1
            while i < len(lines):
                if lines[i] and not lines[i].startswith(" ") and not lines[i].startswith("#"):
                    break
                i += 1
            out.append("")
            continue
        out.append(line)
        i += 1
    YAML_PATH.write_text("\n".join(out) + "\n")


def find_pod_name() -> str | None:
    fs = get_fleet_status({})
    for n in fs.get("nodes", []):
        nm = n.get("name", "")
        if nm.startswith("parameter-golf__sfumato_e4-") and n.get("state") == "running":
            return nm
    return None


def cleanup_inventory() -> None:
    """Remove any stale sfumato_e4 nodes from local inventory before provisioning."""
    fs = get_fleet_status({})
    for n in fs.get("nodes", []):
        nm = n.get("name", "")
        if "sfumato" in nm and n.get("state") in ("lost", "stopped", "exited"):
            log(f"  cleaning stale {nm}")
            try:
                destroy_nodes({"node_names": [nm]})
            except Exception as e:
                log(f"    cleanup err: {e}")


def main() -> None:
    NIGHT_LOG.write_text("")  # truncate
    log("=== NIGHT RUN start ===")

    # Phase A: clean + provision
    cleanup_inventory()
    log("provisioning A6000-first spot pod...")
    res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": True})
    if "error" in res or res.get("created", 0) == 0:
        log(f"  spot fail ({res.get('error') or 'created=0'}); on-demand fallback")
        res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": False})
        if "error" in res or res.get("created", 0) == 0:
            log(f"FAIL provision: {res.get('error')}")
            sys.exit(2)
    new_node = res["new_nodes"][0]["name"]
    log(f"  pod {new_node} requested")

    results = {"pod": new_node, "phases": [], "started_at": time.time()}
    try:
        # Wait for SSH (10 min budget)
        host, port = None, None
        for _ in range(40):
            time.sleep(15)
            r = fleet_refresh({})
            n = next((x for x in r.get("nodes", []) if x.get("name") == new_node), None)
            state = n.get("state") if n else "?"
            if state == "running" and n.get("ssh_host"):
                host, port = n["ssh_host"], n["ssh_port"]
                break
            log(f"  refresh: {state}")
        if not host:
            log("FAIL: pod never reached running+ssh in 10min")
            return
        log(f"  ssh ready @ {host}:{port}")

        # CRITICAL: write Phase B's env_set BEFORE bootstrap so .env is sealed correctly
        # (sync.py:write_remote_env reads spec.env_set during bootstrap).
        log(f"  pre-seeding yaml.env_set with Phase B config (workaround sync.py shlex.quote(int) crash)")
        write_yaml_env_set(PHASES[0]["env_set"])

        # Bootstrap with full error capture
        log("bootstrapping (verbose error capture)...")
        b = bootstrap_project_tool({"project_name": PROJECT, "node_names": [new_node]})
        if b.get("bootstrapped", 0) != 1:
            err_node = (b.get("nodes") or [{}])[0]
            err_msg = err_node.get("error", "(no error msg)")
            log(f"FAIL bootstrap. Full error:")
            for ln in str(err_msg).splitlines()[:30]:
                log(f"  {ln}")
            results["bootstrap_error"] = str(err_msg)[:2000]
            return
        log("  bootstrap OK")

        # Phase B/C/D: each experiment edits yaml.env_set, dispatches, polls
        for ph in PHASES:
            log(f"=== {ph['name']} ===")
            log(f"  setting env_set per phase config (workaround for broken override)")
            write_yaml_env_set(ph["env_set"])

            # Dispatch (no overrides; rely on env_set)
            r = run_project({"project_name": PROJECT, "overrides": {}})
            if "error" in r:
                log(f"  DISPATCH FAIL: {r['error']}")
                results["phases"].append({"name": ph["name"], "rid": None, "state": "dispatch_error", "error": r["error"]})
                continue
            rid = r.get("run_id") or r.get("launch_id")
            log(f"  enqueued {rid}")

            # Poll up to phase budget
            deadline = time.time() + ph["max_wait_s"]
            last = None
            while time.time() < deadline:
                time.sleep(20)
                st = get_project_run_status({"project_name": PROJECT, "run_id": rid})
                state = st.get("status")
                if state != last:
                    log(f"    {rid} -> {state}")
                    last = state
                if state in ("succeeded", "completed", "failed", "cancelled", "error"):
                    tail = (st.get("log_tail") or "")[-400:]
                    log(f"    final: {state}; tail: {tail[-200:]}")
                    results["phases"].append({"name": ph["name"], "rid": rid, "state": state, "tail": tail})
                    break
            else:
                log(f"  WARN timeout for {rid}")
                results["phases"].append({"name": ph["name"], "rid": rid, "state": "timeout"})

        # Phase E: pull jsonls back
        log("=== Phase E: scp raw_*.jsonl back to local ===")
        target = SFUMATO_ROOT / "e4" / "results"
        target.mkdir(parents=True, exist_ok=True)
        ssh_key = pathlib.Path("/Users/eren/.ssh/id_ed25519_runpod")
        cmd = ["scp", "-i", str(ssh_key), "-o", "StrictHostKeyChecking=no", "-P", str(port),
               f"root@{host}:/workspace/sfumato/e4/results/raw_*.jsonl", str(target) + "/"]
        try:
            rc = subprocess.run(cmd, timeout=180, capture_output=True, text=True).returncode
            log(f"  scp rc={rc}")
            results["scp_rc"] = rc
        except Exception as e:
            log(f"  scp exception: {e}")
            results["scp_rc"] = -1

    finally:
        results["ended_at"] = time.time()
        results["duration_min"] = round((results["ended_at"] - results["started_at"]) / 60, 1)
        NIGHT_RESULTS.write_text(json.dumps(results, indent=2))
        log(f"wrote {NIGHT_RESULTS} (duration {results['duration_min']}m)")
        log(f"destroying pod {new_node}...")
        try:
            destroy_nodes({"node_names": [new_node]})
        except Exception as e:
            log(f"  destroy err: {e}")
        log("=== NIGHT RUN done ===")


if __name__ == "__main__":
    main()
