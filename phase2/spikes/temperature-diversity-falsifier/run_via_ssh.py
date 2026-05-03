"""Direct-SSH dispatch wrapper: bypasses Crucible's broken override-propagation.

Provisions + bootstraps via Crucible (project-aware), then SSHs to the pod and runs
3 experiments in a single shell session with explicit env vars. SCPs result jsonls
back to local e4/results/. Destroys pod at end.

Direct SSH is explicitly allowed by the project's coordination notes for one-off
debugging — this counts because Crucible's run_project override path doesn't
propagate TEMP/SEED env vars to the runner subprocess (verified by inspecting
W&B rows from earlier dispatch — all temperature=0.7 default).
"""
from __future__ import annotations
import json
import os
import pathlib
import subprocess
import sys
import time

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SFUMATO_ROOT = pathlib.Path("/Users/eren/Documents/AI/sfumato")
SPIKE_DIR = pathlib.Path(__file__).parent
PROJECT = "sfumato_e4"
SSH_KEY = pathlib.Path("/Users/eren/.ssh/id_ed25519_runpod")


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
    destroy_nodes,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ssh_run(host: str, port: int, command: str, log_label: str = "") -> tuple[int, str]:
    """SSH command, return (exit_code, last_500_chars_of_combined_output)."""
    cmd = [
        "ssh", "-i", str(SSH_KEY),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=20",
        "-p", str(port),
        f"root@{host}",
        command,
    ]
    log(f"  SSH [{log_label}] starting...")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:  # type: ignore
        line = line.rstrip()
        # Print key signal lines so we get progress
        if any(s in line for s in ("step:", "ERROR", "Error", "Traceback", "OOM", "running_acc", "accuracy", "Training", "load")):
            log(f"    [{log_label}] {line[:180]}")
        tail.append(line)
        if len(tail) > 200:
            tail = tail[-200:]
    rc = p.wait()
    log(f"  SSH [{log_label}] exit {rc}")
    return rc, "\n".join(tail[-25:])


def scp_pull(host: str, port: int, remote_path: str, local_path: pathlib.Path) -> None:
    cmd = ["scp", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no", "-P", str(port),
           f"root@{host}:{remote_path}", str(local_path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    log("provisioning A6000 spot/on-demand pod via project spec...")
    res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": True})
    if "error" in res or res.get("created", 0) == 0:
        log(f"  spot failed ({res.get('error') or 'created=0'}); on-demand fallback")
        res = provision_project({"project_name": PROJECT, "count": 1, "interruptible": False})
        if "error" in res or res.get("created", 0) == 0:
            log(f"FAIL provision: {res.get('error')}")
            sys.exit(2)
    new_node = res["new_nodes"][0]["name"]
    log(f"  pod {new_node} requested")

    try:
        # Wait for SSH (8 min budget)
        host, port = None, None
        for _ in range(32):
            time.sleep(15)
            r = fleet_refresh({})
            n = next((x for x in r.get("nodes", []) if x.get("name") == new_node), None)
            if n and n.get("state") == "running" and n.get("ssh_host"):
                host, port = n["ssh_host"], int(n["ssh_port"])
                log(f"  ssh ready @ {host}:{port}")
                break
            log(f"  waiting (state={n.get('state') if n else '?'})...")
        if not host:
            log("FAIL: pod never reached running+ssh in 8 minutes")
            sys.exit(3)

        # Bootstrap
        log("bootstrapping...")
        b = bootstrap_project_tool({"project_name": PROJECT, "node_names": [new_node]})
        if b.get("bootstrapped", 0) != 1:
            log(f"FAIL bootstrap: {b.get('nodes', [{}])[0].get('error', 'unknown')}")
            sys.exit(4)
        log("  bootstrap OK")

        # Refresh to get latest SSH info post-bootstrap (sometimes port changes? cheap to recheck)
        r = fleet_refresh({})
        n = next((x for x in r.get("nodes", []) if x.get("name") == new_node), None)
        if n and n.get("ssh_host"): host, port = n["ssh_host"], int(n["ssh_port"])

        # ---- 3 experiments via direct SSH ----
        results = []
        experiments = [
            ("substrate-tau07-N200", {
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
                "WANDB_RUN_NAME": "phase2-substrate-tau07-N200-seed0",
            }),
            ("cmajc-v3-seed1-N100", {
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
                "WANDB_RUN_NAME": "phase2-cmajc-v3-seed1-N100",
            }),
            ("c-real-mode-trace", {
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
            }),
        ]

        for name, env in experiments:
            log(f"=== running {name} ===")
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            cmd = (
                "set -e && cd /workspace/sfumato && "
                "source .venv/bin/activate && "
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
                f"{env_str} python e4/runner.py"
            )
            t0 = time.time()
            rc, tail = ssh_run(host, port, cmd, log_label=name)
            dur = time.time() - t0
            results.append({"name": name, "rc": rc, "duration_s": dur, "tail": tail})
            log(f"  {name} rc={rc} in {dur/60:.1f} min")

        # Pull all raw_* jsonls back to local
        log("pulling raw_*.jsonl back to local...")
        target = SFUMATO_ROOT / "e4" / "results"
        scp_pull(host, port, "/workspace/sfumato/e4/results/raw_*.jsonl", str(target) + "/")
        log("  pull complete (or partial)")

    finally:
        out = {"node": new_node, "experiments": results if "results" in dir() else [], "ts": time.time()}
        (SPIKE_DIR / "ssh_run_results.json").write_text(json.dumps(out, indent=2))
        log(f"destroying pod {new_node}...")
        destroy_nodes({"node_names": [new_node]})
        log("done")


if __name__ == "__main__":
    main()
