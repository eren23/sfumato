"""Verify e4/data/gsm8k_dev_200.json against the actual HF dataset.

Run on the pod (or any machine with network) to seal the indices:
    python scripts/freeze_gsm8k.py

If integrity_sha256 is absent in the JSON, the script writes it. If present,
the script verifies it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(spec["dataset"], spec["config"], split=spec["split"])
    h = hashlib.sha256()
    for i in spec["indices"]:
        h.update(ds[i]["question"].encode("utf-8"))
        h.update(b"|")
        h.update(ds[i]["answer"].encode("utf-8"))
        h.update(b"\n")
    digest = h.hexdigest()

    existing = spec.get("integrity_sha256")
    if existing is None:
        spec["integrity_sha256"] = digest
        SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"[freeze] wrote integrity_sha256={digest}")
        return 0

    if existing != digest:
        print(
            f"[freeze] MISMATCH: pinned={existing} actual={digest}",
            file=sys.stderr,
        )
        return 1

    print(f"[freeze] OK integrity_sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
