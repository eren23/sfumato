"""One-shot launcher: start the FastAPI server in a child process, then
launch the Gradio app in the foreground. Ctrl-C tears both down.

Usage
-----
    # local end-to-end with mocks (no GPU)
    MOCK_MODELS=1 python3 phase2/inference_viz/launch.py --problem-idx 42

    # talk to a remote pod (assumes you SSH-tunneled 8765 -> pod:8765)
    python3 phase2/inference_viz/launch.py --remote --problem-idx 42

The launcher does the simplest possible thing — no IPC games, no
WebSocket plumbing. The Gradio app talks to the server over plain HTTP.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--problem-idx", type=int, default=42)
    p.add_argument("--server-port", type=int, default=8765)
    p.add_argument("--app-port", type=int, default=7860)
    p.add_argument(
        "--remote",
        action="store_true",
        help="Don't spawn a local server; assume one is already reachable "
             "at --backend (e.g. via SSH tunnel from a Crucible pod).",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Override backend URL (defaults to http://127.0.0.1:<server-port>).",
    )
    args = p.parse_args()

    backend = args.backend or f"http://127.0.0.1:{args.server_port}"

    server_proc: subprocess.Popen | None = None
    if not args.remote:
        if _port_open("127.0.0.1", args.server_port):
            print(f"[launch] server already running on :{args.server_port}, reusing")
        else:
            print(f"[launch] spawning FastAPI server on :{args.server_port} ...")
            env = os.environ.copy()
            server_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(THIS_DIR / "server.py"),
                    "--host", "127.0.0.1",
                    "--port", str(args.server_port),
                ],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            if not _wait_for_port("127.0.0.1", args.server_port, timeout_s=30):
                print("[launch] server failed to come up within 30s", file=sys.stderr)
                if server_proc:
                    server_proc.terminate()
                return 1
            print(f"[launch] server ready at {backend}")

    print(f"[launch] starting Gradio app on :{args.app_port} (backend={backend})")
    env = os.environ.copy()
    env["BACKEND_URL"] = backend
    app_proc = subprocess.Popen(
        [
            sys.executable,
            str(THIS_DIR / "app.py"),
            "--backend", backend,
            "--problem-idx", str(args.problem_idx),
            "--port", str(args.app_port),
        ],
        env=env,
    )

    def _cleanup(*_: object) -> None:
        for proc in (app_proc, server_proc):
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    signal.signal(signal.SIGINT, lambda *a: _cleanup() or sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *a: _cleanup() or sys.exit(0))

    try:
        app_proc.wait()
    finally:
        _cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
