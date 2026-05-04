"""Dump cmaj-failed problems (where majority of branches got it wrong) for
human inspection. Question we want to answer: is the right answer
distinguishable from the wrong ones in the branch text alone?
"""
from __future__ import annotations
import json
import os
import pathlib
from collections import Counter, defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RICH = REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl"
if not DEFAULT_RICH.exists():
    DEFAULT_RICH = pathlib.Path("/tmp/rich_substrate_n500.jsonl")
RICH_PATH = pathlib.Path(os.environ.get("RICH_PATH", str(DEFAULT_RICH)))
OUT_PATH = REPO_ROOT / "phase2/spikes/option3-process-reward/cmaj_failures_inspection.md"


def main():
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["problem_id"]].append(r)

    cmaj_fail_pids = []
    for pid, items in by_pid.items():
        gold = items[0]["gold"]
        votes = [r["extracted"] for r in items]
        cmaj_pick = Counter(votes).most_common(1)[0][0]
        oracle_hit = any(v == gold for v in votes)
        if cmaj_pick != gold and oracle_hit:
            cmaj_fail_pids.append(pid)

    print(f"[inspect] {len(cmaj_fail_pids)}/{len(by_pid)} problems where cmaj fails but oracle could win")

    lines = ["# cmaj-failures inspection (recoverable losses only)\n"]
    lines.append(f"Total problems in N=500 substrate: **{len(by_pid)}**")
    lines.append(f"cmaj fails AND oracle could recover: **{len(cmaj_fail_pids)}** problems\n")
    lines.append("These are the only problems where any verifier could matter.")
    lines.append("Read 10 of them below — is the right branch distinguishable from the wrong ones?\n")
    lines.append("---\n")

    # Sort by problem_id and take first 10
    for pid in sorted(cmaj_fail_pids)[:10]:
        items = by_pid[pid]
        gold = items[0]["gold"]
        question = items[0].get("problem_question", "(question text not in substrate)")
        lines.append(f"## problem_id={pid}  gold=`{gold}`\n")
        lines.append(f"_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_\n")
        votes = [r["extracted"] for r in items]
        vote_counter = Counter(votes)
        cmaj_pick, cmaj_n = vote_counter.most_common(1)[0]
        n_correct = sum(1 for v in votes if v == gold)
        lines.append(f"votes: `{dict(vote_counter)}` | cmaj_pick=`{cmaj_pick}` ({cmaj_n}/5) | "
                     f"correct branches: **{n_correct}/5**\n")
        for i, r in enumerate(items):
            mark = "✅" if r["correct"] else "❌"
            text = r["branch_text"].strip()
            lines.append(f"### branch_{i} {mark}  extracted=`{r['extracted']}`\n")
            lines.append(f"```\n{text}\n```\n")
        lines.append("---\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"[inspect] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
