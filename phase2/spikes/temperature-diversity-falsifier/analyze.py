"""Offline analyzer for temperature-diversity-falsifier spike (D3).

Reads existing Phase-1 raw_cmaj jsonls at multiple temperatures and computes
the four metrics committed in PRE_REG.md:

1. bar_a1(tau) — mean per-branch single-shot accuracy
2. a_b(tau)  — cmaj b=5 accuracy (fraction correct)
3. bar_p_maj(tau) — mean per-problem majority share
4. oracle_ceiling(tau) — fraction of problems where ANY branch hit gold
plus 95% Clopper-Pearson CIs on a_b and oracle_ceiling.

Inputs are existing JSONLs:
  raw_cmaj_k64_seed0_b5_t0.3.jsonl   (tau=0.3)
  raw_cmaj_k64_seed0_b5.jsonl        (tau=0.7, default)
  raw_cmaj_k64_seed0_b5_t1.0.jsonl   (tau=1.0)

Pre-reg committed tau ∈ {0.5, 0.7, 1.0, 1.3}; we substitute the
existing-data temperatures (0.3, 0.7, 1.0). Documented as a deviation
in RESULT.md.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("/Users/eren/Documents/AI/sfumato/e4/results")
SPIKE_DIR = Path("/Users/eren/Documents/AI/sfumato/phase2/spikes/temperature-diversity-falsifier")

JSONLS = {
    0.3: RESULTS_DIR / "raw_cmaj_k64_seed0_b5_t0.3.jsonl",
    0.7: RESULTS_DIR / "raw_cmaj_k64_seed0_b5.jsonl",
    1.0: RESULTS_DIR / "raw_cmaj_k64_seed0_b5_t1.0.jsonl",
}


def cp_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson 95% CI (two-sided). Returns (low, high) in [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        low = 0.0
    else:
        from scipy.stats import beta
        low = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        high = 1.0
    else:
        from scipy.stats import beta
        high = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(low), float(high))


def parse_votes(votes_str: str) -> list[str]:
    """The 'votes' field is a pipe-separated string like '18 | 18 | 18 | 18 | 18'."""
    return [v.strip() for v in votes_str.split("|")]


def analyze_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = len(rows)
    a_b_correct = 0           # problems where cmaj winner = gold
    bar_a1_num = 0            # sum of per-branch correctness (num)
    bar_a1_den = 0            # number of (problem, branch) pairs
    p_maj_sum = 0.0
    oracle_correct = 0
    per_branch_acc = []
    p_majs = []

    for r in rows:
        gold = str(r["gold"]).strip()
        votes = parse_votes(r["trace"]["votes"])
        b = len(votes)
        # cmaj
        winner = r["trace"]["winner"].strip()
        if winner == gold:
            a_b_correct += 1
        # per-branch single-shot
        per_branch_correct = sum(1 for v in votes if v == gold)
        bar_a1_num += per_branch_correct
        bar_a1_den += b
        per_branch_acc.append(per_branch_correct / b)
        # majority share
        cnt = Counter(votes)
        p_maj = max(cnt.values()) / b
        p_maj_sum += p_maj
        p_majs.append(p_maj)
        # oracle ceiling
        if any(v == gold for v in votes):
            oracle_correct += 1

    a_b = a_b_correct / n
    bar_a1 = bar_a1_num / bar_a1_den
    bar_p_maj = p_maj_sum / n
    oracle = oracle_correct / n
    a_b_ci = cp_ci(a_b_correct, n)
    oracle_ci = cp_ci(oracle_correct, n)
    return {
        "n": n,
        "a_b": a_b,
        "a_b_correct_count": a_b_correct,
        "a_b_ci_low": a_b_ci[0],
        "a_b_ci_high": a_b_ci[1],
        "bar_a1": bar_a1,
        "bar_p_maj": bar_p_maj,
        "oracle": oracle,
        "oracle_ci_low": oracle_ci[0],
        "oracle_ci_high": oracle_ci[1],
        "diversity_gap": oracle - a_b,
        "per_branch_acc_per_problem": per_branch_acc,
        "p_maj_per_problem": p_majs,
    }


def main() -> None:
    results = {}
    for tau, path in JSONLS.items():
        if not path.exists():
            print(f"missing: {path}")
            continue
        results[tau] = analyze_one(path)

    # Pretty-print
    print(f"\n{'tau':>5} | {'N':>3} | {'a_b':>6} (95% CI)        | {'bar_a1':>6} | {'bar_p_maj':>9} | {'oracle':>6} | {'div_gap':>7}")
    print("-" * 90)
    for tau, r in sorted(results.items()):
        print(
            f"{tau:>5.2f} | {r['n']:>3} | {r['a_b']*100:>5.1f}%  [{r['a_b_ci_low']*100:>4.1f}, {r['a_b_ci_high']*100:>4.1f}] | "
            f"{r['bar_a1']*100:>5.1f}% | {r['bar_p_maj']:>9.3f} | {r['oracle']*100:>5.1f}% | {r['diversity_gap']*100:>6.1f}pp"
        )

    # Decision rule check (vs τ=0.7 baseline)
    base = results.get(0.7)
    if base is None:
        print("\nNo τ=0.7 baseline; cannot apply decision rule.")
        return

    print("\nDecision-rule check (per pre-reg):")
    for tau, r in sorted(results.items()):
        if tau == 0.7:
            continue
        delta_a_b_pp = (r["a_b"] - base["a_b"]) * 100
        delta_p_maj = r["bar_p_maj"] - base["bar_p_maj"]
        win = (delta_a_b_pp >= 1.5) and (delta_p_maj <= -0.05)
        print(
            f"  τ={tau:.2f}: Δa_b = {delta_a_b_pp:+.2f}pp | Δp_maj = {delta_p_maj:+.3f}  → {'WIN' if win else 'no-win'}"
        )

    # Monotone check
    taus_sorted = sorted(results.keys())
    a_b_seq = [results[t]["a_b"] for t in taus_sorted]
    oracle_seq = [results[t]["oracle"] for t in taus_sorted]
    monotone_dec_ab = all(a_b_seq[i] >= a_b_seq[i + 1] for i in range(len(a_b_seq) - 1))
    monotone_dec_oracle = all(oracle_seq[i] >= oracle_seq[i + 1] for i in range(len(oracle_seq) - 1))
    print(f"\n  Monotone-decreasing a_b in τ? {monotone_dec_ab}")
    print(f"  Monotone-decreasing oracle in τ? {monotone_dec_oracle}")

    # Write JSON dump
    out = {tau: {k: v for k, v in r.items() if k not in {"per_branch_acc_per_problem", "p_maj_per_problem"}}
           for tau, r in results.items()}
    out_path = SPIKE_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
