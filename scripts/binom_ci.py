"""Exact (Clopper-Pearson) binomial 95% CI helper for paper tables.

Usage: python scripts/binom_ci.py 158 200
Or:    python scripts/binom_ci.py --table   (regenerates every CI in the paper)
"""
import sys
from scipy.stats import binomtest


def ci(k: int, n: int) -> tuple[float, float, float]:
    r = binomtest(k, n).proportion_ci(method="exact")
    return 100 * k / n, 100 * r.low, 100 * r.high


def fmt(k: int, n: int, label: str = "") -> str:
    p, lo, hi = ci(k, n)
    head = f"{label}: " if label else ""
    return f"{head}{k}/{n} = {p:.1f}% [{lo:.1f}, {hi:.1f}]"


# Every percentage that appears in a headline body table, with the
# integer k count we believe matches it (k = round(N * p / 100)).
HEADLINE = [
    # Table 1 base hierarchy
    ("Tab1 C2 (74%)", 148, 200),
    ("Tab1 C2hint (68%)", 136, 200),
    ("Tab1 C2empty (66%)", 132, 200),
    ("Tab1 C3p Q-0.5B (64%)", 128, 200),
    ("Tab1 C3p Q-1.5B (60%)", 120, 200),
    # Table 2 Track 1 v2
    ("Tab2 C2 v2 (70.5%)", 141, 200),
    ("Tab2 C2hint v2 (73.5%)", 147, 200),
    ("Tab2 C2empty v2 (73.0%)", 146, 200),
    ("Tab2 C3p Q-0.5B v2 (60.0%)", 120, 200),
    ("Tab2 C3p Q-1.5B v2 (67.0%)", 134, 200),
    ("Tab2 cmaj v2 dev (81.5%)", 163, 200),
    # Table 3 v2 vs v3 (v3 column only; v2 already covered)
    ("Tab3 C2 v3 (73.0%)", 146, 200),
    ("Tab3 C2hint v3 (73.5%)", 147, 200),
    ("Tab3 C2empty v3 (74.0%)", 148, 200),
    ("Tab3 C3p Q-0.5B v3 (65.0%)", 130, 200),
    ("Tab3 C3p Q-1.5B v3 (54.0%)", 108, 200),
    ("Tab3 cmaj v3 test (79.5%)", 159, 200),
    # Table 4 branch-agreement bin fractions (binomial)
    ("Tab4 base 5/5 same (51.5%)", 103, 200),
    ("Tab4 base 4/5 unique (6.0%)", 12, 200),
    ("Tab4 base 3/5 unique (11.5%)", 23, 200),
    ("Tab4 base 2/5 unique (27.5%)", 55, 200),
    ("Tab4 base 5/5 unique (3.5%)", 7, 200),
    ("Tab4 v2 5/5 same (47.5%)", 95, 200),
    ("Tab4 v2 4/5 unique (8.5%)", 17, 200),
    ("Tab4 v2 3/5 unique (18.0%)", 36, 200),
    ("Tab4 v2 2/5 unique (19.5%)", 39, 200),
    ("Tab4 v2 5/5 unique (6.5%)", 13, 200),
    # Table 5 distill main
    ("Tab5 Track1 v2 alone c2c (70.5%)", 141, 200),
    ("Tab5 +commit v1 c2c (70.5%)", 141, 200),
    ("Tab5 +commit v1 cmajc (81.5%)", 163, 200),
    ("Tab5 +commit v2 c2c (70.5%)", 141, 200),
    ("Tab5 +commit v2 cmajc (82.0%)", 164, 200),
    ("Tab5 Track1 v3 alone c2c (73.0%)", 146, 200),
    ("Tab5 +commit v3 c2c (79.0%)", 158, 200),
    # Table 6 disentangle
    ("Tab6 v3 alone c2c (73.0%)", 146, 200),
    ("Tab6 ABL_A c2c (77.0%)", 154, 200),
    ("Tab6 ABL_B c2c (73.0%)", 146, 200),
    ("Tab6 v3 full c2c (79.0%)", 158, 200),
    # self_consistency
    ("Qwen-SC (40.5%)", 81, 200),
]


def emit_table() -> None:
    for label, k, n in HEADLINE:
        print(fmt(k, n, label))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--table":
        emit_table()
    elif len(sys.argv) == 3:
        print(fmt(int(sys.argv[1]), int(sys.argv[2])))
    else:
        sys.exit("usage: python binom_ci.py K N  |  python binom_ci.py --table")
