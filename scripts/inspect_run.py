"""Inspect raw_*.jsonl outputs from a run."""

from __future__ import annotations

import json
import sys


def main(path: str, full: bool = False) -> int:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            print(f"--- idx={r['idx']}")
            print(f"  gold={r['gold']!r}")
            print(f"  pred={r['pred']!r}  correct={r['correct']}")
            if full and "output_text" in r:
                print("  --- output_text ---")
                print(r["output_text"])
    return 0


if __name__ == "__main__":
    full = "--full" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0], full=full))
