"""T3.C: Temporal Self-Consistency × commit-LoRA aggregator.

Schedule-weighted vote across (T sub-blocks × B branches) of partial-answer
extracts from cmajc k=3 trace dumps. Sub-blocks where commit-LoRA is active
(t in {1,2,3} for the K=3 schedule) get weight w_active=1.5; the inactive
prefix (t=0) gets weight w_inactive=1.0.

If this beats vanilla cmajc-vote, it's the first sfumato result where a
non-trivial aggregator beats the 5-branch majority — directly answering the
unified-negative diagnostic story raised in §2 with a positive.

================================================================
SUBSTRATE DEPENDENCY (READ THIS BEFORE REAL EVAL)
================================================================
Real evaluation requires a fresh harvest with EMIT_PARTIAL_PREDS=1, the
runner extension landed at commit e3824ae. That extension teaches
e4/runner.py:_make_trace_dump_callback to decode the committed prefix at
each sub-block boundary and emit `partial_answer_strict` /
`partial_answer_loose` fields into each sidecar JSONL row.

The existing T1.B-redux substrate at
  e4/results/traces/cmajc-prm-v2-N100-seed0-shifted/
was harvested BEFORE that runner extension and does NOT have the
partial_answer fields. Loading it through this aggregator will yield
all-None partial cells and degenerate to the cmajc-vote fallback.

Cost to re-harvest: 48GB on-demand pod x ~70min x $0.33/hr ~= $0.45.
Trigger: TRACE_STEPS=1 EMIT_PARTIAL_PREDS=1 BATCHED=0 BRANCHES=5
         CONDITION=cmajc N_PROBLEMS=100 SEED=0 K=3 ...

Pre-reg thresholds locked in PRE_REG.md (commit 9d12950):
  Delta >= +6pp  -> WIN
  +2pp..+6pp     -> PARTIAL
  Delta <  +2pp  -> LOSS
================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from e4 import grade  # noqa: E402

N_BLOCKS = 4
N_BRANCHES = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_sidecars(traces_dir: Path) -> dict[tuple[int, int], list[dict]]:
    """Read branch_<b>_idx_<i>.jsonl files. Returns {(b, i): [sub_block rows]}."""
    out: dict[tuple[int, int], list[dict]] = {}
    for path in sorted(traces_dir.glob("branch_*_idx_*.jsonl")):
        parts = path.stem.split("_")
        b, i = int(parts[1]), int(parts[3])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: r["sub_block"])
        out[(b, i)] = rows
    return out


def _load_outcome(outcome_path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in outcome_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["idx"]] = rec
    return out


def load_partial_traces(traces_dir: Path, outcome_path: Path) -> dict:
    """Join sidecars with outcome JSONL.

    Returns dict with:
      partial[(b, i)]              -> length-4 list of partial_answer_strict
                                      (or None if missing)
      partial_loose[(b, i)]        -> length-4 list of partial_answer_loose
                                      (or None if missing)
      subblock_active[(b, i, t)]   -> bool from `commit_lora_active` field
      final_pred[i]                -> cmajc winner string
      gold[i]                      -> gold answer string
      votes[i]                     -> list of 5 final-branch answer strings
    """
    sidecars = _load_sidecars(traces_dir)
    outcomes = _load_outcome(outcome_path)

    partial: dict[tuple[int, int], list[Optional[str]]] = {}
    partial_loose: dict[tuple[int, int], list[Optional[str]]] = {}
    subblock_active: dict[tuple[int, int, int], bool] = {}
    final_pred: dict[int, str] = {}
    gold: dict[int, str] = {}
    votes: dict[int, list[str]] = {}

    for i, rec in outcomes.items():
        gold[i] = rec.get("gold", "")
        trace = rec.get("trace", {}) or {}
        final_pred[i] = trace.get("winner", "") or rec.get("pred", "")
        v_str = trace.get("votes", "") or ""
        votes[i] = [a.strip() for a in v_str.split(" | ")] if v_str else []

        for b in range(N_BRANCHES):
            rows = sidecars.get((b, i))
            if rows is None:
                continue
            strict = [None] * N_BLOCKS
            loose = [None] * N_BLOCKS
            for r in rows:
                t = r.get("sub_block")
                if t is None or not (0 <= t < N_BLOCKS):
                    continue
                strict[t] = r.get("partial_answer_strict")
                loose[t] = r.get("partial_answer_loose")
                subblock_active[(b, i, t)] = bool(r.get("commit_lora_active", False))
            partial[(b, i)] = strict
            partial_loose[(b, i)] = loose

    return {
        "partial": partial,
        "partial_loose": partial_loose,
        "subblock_active": subblock_active,
        "final_pred": final_pred,
        "gold": gold,
        "votes": votes,
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def temporal_sc_aggregate(
    partial: dict[tuple[int, int], list[Optional[str]]],
    votes: dict[int, list[str]],
    subblock_active: dict[tuple[int, int, int], bool],
    *,
    w_active: float = 1.5,
    w_inactive: float = 1.0,
    use_loose: bool = False,
    partial_loose: Optional[dict[tuple[int, int], list[Optional[str]]]] = None,
) -> dict[int, str]:
    """For each problem i, build a weighted vote across (T x B) partial answers.

    weight[t,b] = w_active if subblock_active[(b,i,t)] else w_inactive
    score[a] = sum over (t,b) of weight[t,b] * 1[partial[(b,i)][t] == a]
    final_answer[i] = argmax_a score[a]   (fall back to votes[i][0] if empty)
    """
    problem_ids = {i for (_, i) in partial.keys()} | set(votes.keys())
    out: dict[int, str] = {}
    for i in sorted(problem_ids):
        scores: Counter[str] = Counter()
        for b in range(N_BRANCHES):
            cells = partial.get((b, i))
            if cells is None:
                continue
            loose_cells = (partial_loose or {}).get((b, i)) if use_loose else None
            for t in range(N_BLOCKS):
                a = cells[t] if t < len(cells) else None
                if (a is None or a == "") and use_loose and loose_cells is not None:
                    a = loose_cells[t] if t < len(loose_cells) else None
                if a is None or a == "":
                    continue
                w = w_active if subblock_active.get((b, i, t), False) else w_inactive
                scores[a] += w
        if scores:
            out[i] = scores.most_common(1)[0][0]
        else:
            v = votes.get(i, [])
            out[i] = v[0] if v else ""
    return out


# ---------------------------------------------------------------------------
# Eval + compare
# ---------------------------------------------------------------------------

def evaluate(final_answer: dict[int, str], gold: dict[int, str]) -> float:
    if not final_answer:
        return 0.0
    n_correct = sum(1 for i, a in final_answer.items() if grade.is_correct(a, gold.get(i, "")))
    return n_correct / len(final_answer)


def _verdict(delta_pp: float) -> str:
    if delta_pp >= 6.0:
        return "WIN"
    if delta_pp >= 2.0:
        return "PARTIAL"
    return "LOSS"


def compare(traces_dir: Path, outcome_path: Path, *, use_loose: bool = False) -> None:
    data = load_partial_traces(traces_dir, outcome_path)
    cmajc_acc = evaluate(data["final_pred"], data["gold"])
    tsc = temporal_sc_aggregate(
        data["partial"], data["votes"], data["subblock_active"],
        use_loose=use_loose, partial_loose=data["partial_loose"],
    )
    tsc_acc = evaluate(tsc, data["gold"])
    delta = tsc_acc - cmajc_acc
    delta_pp = delta * 100.0
    print(f"cmajc-vote:  {cmajc_acc:.4f}")
    print(f"temporal-SC: {tsc_acc:.4f}")
    print(f"Delta = {delta_pp:+.2f}pp -> verdict: {_verdict(delta_pp)}")


# ---------------------------------------------------------------------------
# CPU smoke
# ---------------------------------------------------------------------------

def _build_synthetic_substrate() -> dict:
    """Synthesize 10 problems x 5 branches x 4 sub-blocks.

    Construction (gold = problem_idx as string for simplicity):
      - Problems 0..5  (6 of 10): final-block majority is correct.
                                  cmajc-vote gets these right.
      - Problems 6,7   (2 of 10): final-block majority is WRONG (drift),
                                  but sub-block-2 majority across branches
                                  is correct. cmajc-vote=wrong;
                                  temporal-SC=correct (the 2 extras).
      - Problems 8,9   (2 of 10): all sub-blocks across all branches wrong;
                                  both methods fail.

    Goal: cmajc-vote = 6/10 = 0.60, temporal-SC = 8/10 = 0.80.
    """
    partial: dict[tuple[int, int], list[Optional[str]]] = {}
    partial_loose: dict[tuple[int, int], list[Optional[str]]] = {}
    subblock_active: dict[tuple[int, int, int], bool] = {}
    final_pred: dict[int, str] = {}
    gold: dict[int, str] = {}
    votes: dict[int, list[str]] = {}

    # Schedule: t=0 inactive, t in {1,2,3} active (K=3 schedule).
    schedule = {0: False, 1: True, 2: True, 3: True}

    for i in range(10):
        good = str(i)
        bad = f"{i}99"  # arbitrary distractor
        bad2 = f"{i}77"
        gold[i] = good
        if i <= 5:
            # cmajc-vote gets these right: final block has 5/5 = good.
            branch_finals = [good] * 5
            # Sub-blocks: drift early, lock onto good by final block.
            for b in range(5):
                cells = [bad, bad, good, good]  # t=0,1 wrong; t=2,3 correct
                partial[(b, i)] = cells
                partial_loose[(b, i)] = list(cells)
                for t in range(4):
                    subblock_active[(b, i, t)] = schedule[t]
            final_pred[i] = good
            votes[i] = branch_finals
        elif i in (6, 7):
            # cmajc-vote WRONG: final block has 3/5 = bad (winner=bad).
            # But sub-block 2 across branches is mostly correct.
            # 3 of 5 branches: final=bad, but sub-block-1,2 = good.
            # 2 of 5 branches: final=good (consistent throughout).
            # Final-block tally:    bad x3, good x2  -> cmajc winner = bad.
            # Temporal-SC tally (weights: t=0 -> 1.0, t=1,2,3 -> 1.5):
            #   "good": from 3 branches (t=1,2 active) + from 2 branches (t=0..3 all)
            #         = 3*(1.5 + 1.5) + 2*(1.0 + 1.5 + 1.5 + 1.5)
            #         = 3*3.0 + 2*5.5
            #         = 9.0 + 11.0 = 20.0
            #   "bad":  from 3 branches (t=0 inactive, t=3 active)
            #         = 3*(1.0 + 1.5) = 7.5
            #   "bad2": from 2 distractor branches at t=0 only
            #         = 2*1.0 = 2.0   (irrelevant)
            #   -> good wins. Temporal-SC gets it right.
            branch_finals = [bad, bad, bad, good, good]
            # Branches 0..2: drift back to bad at the end.
            for b in range(3):
                cells = [bad, good, good, bad]
                partial[(b, i)] = cells
                partial_loose[(b, i)] = list(cells)
                for t in range(4):
                    subblock_active[(b, i, t)] = schedule[t]
            # Branches 3..4: consistent good throughout.
            for b in range(3, 5):
                cells = [good, good, good, good]
                partial[(b, i)] = cells
                partial_loose[(b, i)] = list(cells)
                for t in range(4):
                    subblock_active[(b, i, t)] = schedule[t]
            final_pred[i] = bad  # cmajc winner = majority of branch finals
            votes[i] = branch_finals
        else:
            # Problems 8, 9: both methods fail (all branches wrong everywhere).
            branch_finals = [bad] * 5
            for b in range(5):
                cells = [bad, bad2, bad, bad2]
                partial[(b, i)] = cells
                partial_loose[(b, i)] = list(cells)
                for t in range(4):
                    subblock_active[(b, i, t)] = schedule[t]
            final_pred[i] = bad
            votes[i] = branch_finals

    return {
        "partial": partial,
        "partial_loose": partial_loose,
        "subblock_active": subblock_active,
        "final_pred": final_pred,
        "gold": gold,
        "votes": votes,
    }


def smoke() -> None:
    data = _build_synthetic_substrate()
    cmajc_acc = evaluate(data["final_pred"], data["gold"])
    tsc = temporal_sc_aggregate(
        data["partial"], data["votes"], data["subblock_active"],
        w_active=1.5, w_inactive=1.0,
    )
    tsc_acc = evaluate(tsc, data["gold"])
    delta = tsc_acc - cmajc_acc
    delta_pp = delta * 100.0
    # Pre-reg thresholds apply to N=100 real eval; on the 10-problem
    # synthetic we just verify the aggregator math by reporting Delta.
    print(f"cmajc-vote: {cmajc_acc:.2f}")
    print(f"temporal-SC: {tsc_acc:.2f}")
    print(f"Delta = {delta_pp:+.2f}pp")
    tag = "synthetic-WIN" if delta_pp >= 6.0 else (
        "synthetic-PARTIAL" if delta_pp >= 2.0 else "synthetic-LOSS"
    )
    print(tag)
    # Hard-fail the smoke if the math is broken so CI/devs notice.
    assert abs(cmajc_acc - 0.60) < 1e-9, f"cmajc-vote expected 0.60, got {cmajc_acc}"
    assert abs(tsc_acc - 0.80) < 1e-9, f"temporal-SC expected 0.80, got {tsc_acc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Run synthetic-substrate smoke.")
    ap.add_argument("--traces-dir", type=Path, default=None)
    ap.add_argument("--outcome", type=Path, default=None)
    ap.add_argument("--use-loose", action="store_true",
                    help="Fall back to partial_answer_loose when strict is None.")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if not args.traces_dir or not args.outcome:
        print("Provide --traces-dir and --outcome, or use --smoke.")
        print("NOTE: real eval requires EMIT_PARTIAL_PREDS=1 re-harvest "
              "(commit e3824ae); existing T1.B-redux trace dumps lack "
              "partial_answer_* fields.")
        sys.exit(2)
    if not args.traces_dir.exists() or not args.outcome.exists():
        print(f"missing inputs: traces_dir={args.traces_dir.exists()} "
              f"outcome={args.outcome.exists()}")
        sys.exit(2)
    compare(args.traces_dir, args.outcome, use_loose=args.use_loose)


if __name__ == "__main__":
    main()
