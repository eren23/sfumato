"""Symbolic arithmetic verifier — bypass ML entirely.

For each (problem, branch) in the rich substrate:
  1. Regex-extract arithmetic statements `A op B = C` from branch_text
     (also picks up `A op B op C = D` for chains).
  2. Safely re-compute the right-hand side via AST whitelist (no Python's
     built-in dangerous expression-execution function).
  3. Branch score = fraction of recovered statements whose computed RHS
     matches the stated RHS, with a small bonus for "more statements" so a
     branch with 4-of-4 right beats a branch with 0-of-0.
  4. Per problem: pick branch with highest score. Compare vs cmaj b=5.

Pre-reg threshold (binding, same as the other spikes):
  WIN-DECISIVE >=89%, WIN-STRONG >=87%, WIN-MINOR >=83%,
  INCONCLUSIVE if >=cmaj-1pp, else LOSS.

Reads:  rich_substrate_n500.jsonl  (or RICH_PATH env var; default /tmp copy)
Writes: phase2/spikes/symbolic-verifier/results.json
"""
from __future__ import annotations
import ast
import json
import operator
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RICH = REPO_ROOT / "phase2/spikes/option3-process-reward/rich_substrate_n500.jsonl"
if not DEFAULT_RICH.exists():
    DEFAULT_RICH = pathlib.Path("/tmp/rich_substrate_n500.jsonl")
RICH_PATH = pathlib.Path(os.environ.get("RICH_PATH", str(DEFAULT_RICH)))
OUT_PATH = REPO_ROOT / "phase2/spikes/symbolic-verifier/results.json"

# ----------------- safe arithmetic computation (AST whitelist, no exec/eval) ----------------
ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_compute(expr: str):
    """Compute a numeric arithmetic expression by AST-walking a whitelist."""
    try:
        node = ast.parse(expr, mode="eval").body
        return _walk(node)
    except Exception:
        return None


def _walk(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric literal")
    if isinstance(node, ast.BinOp):
        op = ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"op {type(node.op).__name__} not allowed")
        return op(_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp):
        op = ALLOWED_UNARY.get(type(node.op))
        if op is None:
            raise ValueError(f"unary {type(node.op).__name__} not allowed")
        return op(_walk(node.operand))
    raise ValueError(f"node {type(node).__name__} not allowed")


# ----------------- statement extractor ----------------
NUM_RE = r"[-+]?\d+(?:[\.,]\d+)?"
EXPR_CHARS = r"[\d\s\+\-\*\/\(\)\.\^]"
STMT_RE = re.compile(
    rf"({EXPR_CHARS}{{2,}}?)\s*=\s*({NUM_RE})", re.MULTILINE
)


def _normalize(s: str) -> str:
    return s.replace(",", "").replace("^", "**").strip()


def extract_statements(text: str):
    """Return list of (lhs_expr, expected_rhs_float)."""
    out = []
    for m in STMT_RE.finditer(text):
        lhs_raw, rhs_raw = m.group(1), m.group(2)
        lhs = _normalize(lhs_raw)
        rhs = _normalize(rhs_raw)
        if not any(op in lhs for op in "+-*/^"):
            continue
        try:
            rhs_val = float(rhs)
        except ValueError:
            continue
        out.append((lhs, rhs_val))
    return out


def score_branch(branch_text: str):
    """Returns (score, n_correct, n_total). score = correct/total + 0.05*sqrt(total)."""
    stmts = extract_statements(branch_text)
    if not stmts:
        return (0.0, 0, 0)
    correct = 0
    for lhs, rhs in stmts:
        val = safe_compute(lhs)
        if val is None:
            continue
        if abs(val - rhs) < 1e-6 or (rhs != 0 and abs((val - rhs) / rhs) < 1e-4):
            correct += 1
    score = (correct / max(1, len(stmts))) + 0.05 * (len(stmts) ** 0.5)
    return (score, correct, len(stmts))


# ----------------- evaluation ----------------
def cv_split(rows, k=5, seed=0):
    rng = np.random.default_rng(seed)
    pids = sorted({r["problem_id"] for r in rows})
    rng.shuffle(pids)
    fold_size = len(pids) // k
    by_pid_idx = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid_idx[r["problem_id"]].append(i)
    folds = []
    for f in range(k):
        s, e = f*fold_size, (f+1)*fold_size if f < k-1 else len(pids)
        test_pids = set(pids[s:e])
        ti, tri = [], []
        for pid, idxs in by_pid_idx.items():
            (ti if pid in test_pids else tri).extend(idxs)
        folds.append((np.array(tri), np.array(ti)))
    return folds


def score_all(rows):
    """Score every branch and pick argmax per problem; report 5-fold stats."""
    by_pid = defaultdict(list)
    branch_scores = {}
    for i, r in enumerate(rows):
        sc, nc, nt = score_branch(r.get("branch_text", ""))
        branch_scores[i] = (sc, nc, nt)
        by_pid[r["problem_id"]].append(i)

    folds = cv_split(rows, k=5, seed=0)
    fold_results = []
    for f, (_, te) in enumerate(folds):
        test_pids = {rows[i]["problem_id"] for i in te}
        cmaj_correct = ver_correct = oracle_correct = recovers = 0
        n = 0
        for pid in test_pids:
            n += 1
            items = [rows[i] for i in by_pid[pid]]
            gold = items[0]["gold"]
            votes = [r["extracted"] for r in items]
            cmaj_pick = Counter(votes).most_common(1)[0][0]
            cmaj_hit = (cmaj_pick == gold)
            if cmaj_hit: cmaj_correct += 1
            if any(v == gold for v in votes): oracle_correct += 1
            scores = [branch_scores[by_pid[pid][k]][0] for k in range(len(items))]
            ver_pick = items[int(np.argmax(scores))]["extracted"]
            if ver_pick == gold: ver_correct += 1
            if not cmaj_hit and ver_pick == gold: recovers += 1
        fold_results.append({
            "n_problems": n,
            "cmaj_acc": cmaj_correct/n,
            "verifier_acc": ver_correct/n,
            "oracle_acc": oracle_correct/n,
            "verifier_recovers": recovers,
            "delta_pp": (ver_correct - cmaj_correct)/n*100,
        })
        print(f"  fold {f}: cmaj={cmaj_correct/n:.1%} verifier={ver_correct/n:.1%} "
              f"(d={(ver_correct-cmaj_correct)/n*100:+.1f}pp) oracle={oracle_correct/n:.1%} "
              f"recovers={recovers}")

    valid = fold_results
    mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
    mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
    mean_oracle = float(np.mean([r["oracle_acc"] for r in valid]))
    dpp = (mean_ver - mean_cmaj) * 100
    if mean_ver >= 0.89: dec = "WIN-DECISIVE"
    elif mean_ver >= 0.87: dec = "WIN-STRONG"
    elif mean_ver >= 0.83: dec = "WIN-MINOR"
    elif mean_ver >= mean_cmaj - 0.01: dec = "INCONCLUSIVE"
    else: dec = "LOSS"
    return {
        "model_id": "symbolic-arithmetic-verifier",
        "rich_path": str(RICH_PATH),
        "n_branches": len(rows),
        "n_problems": len({r["problem_id"] for r in rows}),
        "fold_results": fold_results,
        "mean_cmaj": mean_cmaj,
        "mean_verifier": mean_ver,
        "mean_oracle": mean_oracle,
        "delta_pp_vs_cmaj": dpp,
        "decision": dec,
        "n_branches_with_statements": int(sum(1 for v in branch_scores.values() if v[2] > 0)),
        "mean_statements_per_branch": float(np.mean([v[2] for v in branch_scores.values()])),
        "mean_correct_statements_per_branch": float(np.mean([v[1] for v in branch_scores.values()])),
    }


def main():
    if not RICH_PATH.exists():
        print(f"[symbolic] FATAL - substrate {RICH_PATH} missing"); sys.exit(1)
    rows = [json.loads(l) for l in RICH_PATH.read_text().splitlines() if l.strip()]
    print(f"[symbolic] loaded {len(rows)} branches from {RICH_PATH.name}", flush=True)

    # Sanity check: print extraction on first 2 branches
    for i in range(min(2, len(rows))):
        sc, nc, nt = score_branch(rows[i].get("branch_text", ""))
        stmts = extract_statements(rows[i].get("branch_text", ""))
        print(f"  branch[{i}] correct={rows[i]['correct']} score={sc:.3f} "
              f"({nc}/{nt} stmts checked): {stmts[:3]}")

    res = score_all(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(res, indent=2))
    print(f"\n[symbolic] === MEAN: cmaj={res['mean_cmaj']:.1%} "
          f"verifier={res['mean_verifier']:.1%} oracle={res['mean_oracle']:.1%} ===")
    print(f"[symbolic] dpp={res['delta_pp_vs_cmaj']:+.2f}  "
          f">>> DECISION: {res['decision']} <<<")
    print(f"[symbolic] {res['n_branches_with_statements']}/{res['n_branches']} "
          f"branches had >=1 verifiable statement; mean {res['mean_statements_per_branch']:.1f} stmts/branch")
    print(f"[symbolic] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
