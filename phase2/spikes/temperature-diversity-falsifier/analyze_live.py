"""Live analyzer: pulls cmaj sweep results from W&B (Crucible's status reporting is unreliable
when the runner exits non-zero on wandb finalize), recomputes spike metrics on live N=20
data, then merges with offline N=50 Phase-1 jsonl data into a unified RESULT_LIVE.md.

Run after queue_followup.py finishes (or anytime live runs have completed in W&B).
"""
from __future__ import annotations
import json
import os
import pathlib
import sys
import tempfile
from collections import Counter

# Reuse the offline analyzer's helpers
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from analyze import cp_ci, parse_votes, analyze_one  # noqa: E402

PG_ROOT = pathlib.Path("/Users/eren/Documents/AI/parameter-golf_dev")
SPIKE_DIR = pathlib.Path(__file__).parent
WANDB_PROJECT = "eren23/sfumato-e4"


def load_env() -> None:
    for f in [PG_ROOT / ".env", PG_ROOT / ".env.runpod.local"]:
        if not f.exists(): continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch_run_jsonls(run_name_substring: str = "cmaj-k64") -> dict[float, list[dict]]:
    """For each W&B run matching the substring, download its 'rows' artifact and parse jsonl.

    Returns dict[temperature -> list of row dicts]. Multiple runs at same TEMP get concatenated.
    """
    load_env()
    import wandb
    api = wandb.Api()

    runs = list(api.runs(WANDB_PROJECT, order="-created_at", per_page=50))
    # Filter to recent finished cmaj runs from today's spike
    today = "2026-05-02"
    candidates = [
        r for r in runs
        if (r.state == "finished")
        and (r.created_at or "").startswith(today)
        and run_name_substring in r.name
        and (r.summary or {}).get("accuracy") is not None
    ]
    print(f"Found {len(candidates)} finished candidate runs for substring '{run_name_substring}' on {today}")

    by_tau: dict[float, list[dict]] = {}
    for r in candidates:
        tau = None
        # Try to recover tau from artifact rows themselves (config is often empty)
        for art in r.logged_artifacts():
            if art.type != "rows": continue
            with tempfile.TemporaryDirectory() as tmp:
                d = pathlib.Path(art.download(root=tmp))
                jsonls = list(d.glob("*.jsonl"))
                if not jsonls:
                    continue
                rows = []
                for jp in jsonls:
                    for line in jp.read_text().splitlines():
                        if not line.strip(): continue
                        rec = json.loads(line)
                        rows.append(rec)
                if rows:
                    tau = float(rows[0].get("temperature", 0.7))
                    by_tau.setdefault(tau, []).extend(rows)
                    print(f"  {r.name} created={r.created_at} tau={tau} N={len(rows)} acc={r.summary.get('accuracy')}")
            break  # one artifact per run is enough
    return by_tau


def metrics_from_rows(rows: list[dict]) -> dict:
    """Compute spike metrics directly from a list of per-problem rows.

    Each row should have {gold, trace.{branch_*, votes, winner}}.
    Mirrors analyze.analyze_one() logic.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "error": "empty"}
    a_b_correct = 0
    bar_a1_num = 0
    bar_a1_den = 0
    p_maj_sum = 0.0
    oracle_correct = 0
    for r in rows:
        gold = str(r.get("gold", "")).strip()
        trace = r.get("trace") or {}
        votes_str = trace.get("votes", "")
        votes = parse_votes(votes_str) if votes_str else []
        if not votes: continue
        b = len(votes)
        winner = (trace.get("winner") or "").strip()
        if winner == gold: a_b_correct += 1
        per_branch_correct = sum(1 for v in votes if v == gold)
        bar_a1_num += per_branch_correct; bar_a1_den += b
        cnt = Counter(votes); p_maj_sum += max(cnt.values()) / b
        if any(v == gold for v in votes): oracle_correct += 1
    a_b = a_b_correct / n
    bar_a1 = bar_a1_num / max(1, bar_a1_den)
    bar_p_maj = p_maj_sum / n
    oracle = oracle_correct / n
    return {
        "n": n,
        "a_b": a_b, "a_b_correct": a_b_correct, "a_b_ci": cp_ci(a_b_correct, n),
        "bar_a1": bar_a1, "bar_p_maj": bar_p_maj,
        "oracle": oracle, "oracle_ci": cp_ci(oracle_correct, n),
        "diversity_gap_pp": (oracle - a_b) * 100,
    }


def main() -> None:
    print(f"\n=== Live spike data from W&B {WANDB_PROJECT} ===")
    by_tau = fetch_run_jsonls("cmaj-k64")
    if not by_tau:
        print("\nNo live runs found. Falling back to offline-only.")
        return

    print(f"\n{'tau':>5} | {'N':>4} | {'a_b':>7} (95%CI)        | {'bar_a1':>7} | {'p_maj':>6} | {'oracle':>7} | {'gap':>6}")
    print("-" * 95)
    live_metrics = {}
    for tau in sorted(by_tau.keys()):
        m = metrics_from_rows(by_tau[tau])
        live_metrics[tau] = m
        if "error" in m:
            print(f"{tau:>5.2f} | -- | (no data)")
            continue
        print(
            f"{tau:>5.2f} | {m['n']:>4} | {m['a_b']*100:>5.1f}%  [{m['a_b_ci'][0]*100:>4.1f}, {m['a_b_ci'][1]*100:>4.1f}] | "
            f"{m['bar_a1']*100:>5.1f}% | {m['bar_p_maj']:>6.3f} | "
            f"{m['oracle']*100:>5.1f}%  | {m['diversity_gap_pp']:>5.1f}pp"
        )

    out = {f"tau_{tau}": {k: (list(v) if isinstance(v, tuple) else v) for k, v in m.items()}
           for tau, m in live_metrics.items()}
    out_path = SPIKE_DIR / "results_live.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
