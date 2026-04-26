"""Print full trace for one row of a raw_*.jsonl file."""

from __future__ import annotations

import json
import sys


def main(path: str, idx: int = 0) -> int:
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if idx >= len(rows):
        print(f"only {len(rows)} rows", file=sys.stderr)
        return 1
    r = rows[idx]
    print(f"=== row {idx} ===")
    print(f"  ar_model:   {r.get('ar_model')}")
    print(f"  diff_model: {r.get('diff_model')}")
    print(f"  k_steps:    {r['k_steps']}")
    print(f"  gold:       {r['gold']!r}")
    print(f"  pred:       {r['pred']!r}  correct={r['correct']}")
    for k, v in r.get("trace", {}).items():
        print()
        print(f"### {k} ({len(v)} chars)")
        print(v[:1200])
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    idx = int(args[1]) if len(args) > 1 else 0
    sys.exit(main(path, idx))
