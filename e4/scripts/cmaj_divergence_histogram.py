"""E2 — cmaj branch divergence histogram.

For the 200-problem cmaj-b=5 substrate, ask: at which sub-block do the 5
branches first diverge?

Input:  e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl
Output: e4/results/cmaj_divergence/{histogram.png,histogram.pdf,summary.json}

Method:
  1. Filter to disagreement problems (>=2 unique extracted answers across 5).
  2. Re-tokenize each branch with the LLaDA-8B-Instruct tokenizer.
  3. Pad each branch to 128 tokens with the tokenizer's pad/eos id.
  4. Split into 4 sub-blocks of 32 tokens (matches LLaDA semi-AR structure).
  5. For each sub-block, compute pairwise (10 pairs) Levenshtein over token ids.
     Mark sub-block "diverged" if any pair has distance >= 3.
  6. First-divergence-sub-block = min diverged sub-block per problem; "none"
     if no sub-block diverged.
  7. Histogram stratified by cmaj-correct vs cmaj-incorrect.
  8. Cross-check: TF-IDF cosine on decoded sub-block text. Mark sub-block
     "semantically diverged" if any pair has cosine <= 0.7. Compare to
     token-level result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL = REPO_ROOT / "e4" / "results" / "raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl"
OUT_DIR = REPO_ROOT / "e4" / "results" / "cmaj_divergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = REPO_ROOT / "phase2" / "figures"
sys.path.insert(0, str(FIG_DIR))
from palette import PALETTE  # noqa: E402

plt.style.use(str(FIG_DIR / "sfumato.mplstyle"))

GEN_LEN = 128
SUB_BLOCK_LEN = 32
N_SUB_BLOCKS = GEN_LEN // SUB_BLOCK_LEN  # 4
EDIT_THRESHOLD = 3
COSINE_THRESHOLD = 0.7


def levenshtein(a: list[int], b: list[int]) -> int:
    """Token-id Levenshtein distance, length-tolerant. O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ai != bj),
            )
        prev = cur
    return prev[-1]


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    return tok


def pad_to(ids: list[int], target: int, pad_id: int) -> list[int]:
    if len(ids) >= target:
        return ids[:target]
    return ids + [pad_id] * (target - len(ids))


def sub_block_slice(ids: list[int], s: int) -> list[int]:
    return ids[s * SUB_BLOCK_LEN : (s + 1) * SUB_BLOCK_LEN]


def first_divergence_sub_block(branches_ids: list[list[int]]) -> int | None:
    """Return min sub-block index s where any pair (i<j) has edit distance
    >= EDIT_THRESHOLD on sub-block s. None if no sub-block diverged."""
    for s in range(N_SUB_BLOCKS):
        slices = [sub_block_slice(b, s) for b in branches_ids]
        for i, j in combinations(range(len(slices)), 2):
            if levenshtein(slices[i], slices[j]) >= EDIT_THRESHOLD:
                return s
    return None


def semantic_first_divergence(branches_text: list[str], tok) -> int | None:
    """TF-IDF cosine across decoded sub-block text. Returns min sub-block s
    with any pair cosine <= COSINE_THRESHOLD. None if no divergence."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Re-tokenize each branch, pad, then decode each sub-block slice back
    # so the TF-IDF basis is the sub-block text.
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        pad_id = 0
    padded_ids = [pad_to(tok.encode(t, add_special_tokens=False), GEN_LEN, pad_id) for t in branches_text]

    for s in range(N_SUB_BLOCKS):
        texts = [tok.decode(b[s * SUB_BLOCK_LEN : (s + 1) * SUB_BLOCK_LEN], skip_special_tokens=True) for b in padded_ids]
        # Edge case: all empty strings (e.g., late sub-block on short branches)
        if all(not t.strip() for t in texts):
            continue
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
            sims = cosine_similarity(vec)
            n = len(texts)
            for i, j in combinations(range(n), 2):
                if sims[i, j] <= COSINE_THRESHOLD:
                    return s
        except ValueError:
            # TF-IDF can fail on empty vocab; treat as not-diverged here.
            continue
    return None


def main() -> None:
    rows = [json.loads(l) for l in open(JSONL)]
    print(f"loaded {len(rows)} problems from {JSONL.name}")

    tok = load_tokenizer()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        pad_id = 0
    print(f"tokenizer pad_id={pad_id}, eos={tok.eos_token_id}")

    disagreement = []
    for r in rows:
        votes_raw = r["trace"]["votes"]
        votes = [v.strip() for v in votes_raw.split(" | ")]
        if len(set(votes)) >= 2:
            disagreement.append(r)
    print(f"disagreement subset: {len(disagreement)} / {len(rows)} ({100*len(disagreement)/len(rows):.1f}%)")

    results = []
    for r in disagreement:
        trace = r["trace"]
        branches_text = [trace[f"branch_{i}"] for i in range(5)]
        branches_ids = [pad_to(tok.encode(t, add_special_tokens=False), GEN_LEN, pad_id) for t in branches_text]
        token_lengths = [len(tok.encode(t, add_special_tokens=False)) for t in branches_text]

        tok_div = first_divergence_sub_block(branches_ids)
        sem_div = semantic_first_divergence(branches_text, tok)

        results.append({
            "idx": r["idx"],
            "id": r["id"],
            "cmaj_correct": bool(r["correct"]),
            "winner": trace["winner"],
            "gold": r["gold"],
            "votes": [v.strip() for v in trace["votes"].split(" | ")],
            "branch_token_lengths": token_lengths,
            "first_div_token": tok_div,
            "first_div_semantic": sem_div,
        })

    # ---- aggregate ----
    cats = list(range(N_SUB_BLOCKS)) + ["none"]
    cat_label = {0: "sb 0", 1: "sb 1", 2: "sb 2", 3: "sb 3", "none": "no div."}

    counts_correct = {c: 0 for c in cats}
    counts_incorrect = {c: 0 for c in cats}
    counts_sem_correct = {c: 0 for c in cats}
    counts_sem_incorrect = {c: 0 for c in cats}

    for r in results:
        tk = r["first_div_token"] if r["first_div_token"] is not None else "none"
        sk = r["first_div_semantic"] if r["first_div_semantic"] is not None else "none"
        if r["cmaj_correct"]:
            counts_correct[tk] += 1
            counts_sem_correct[sk] += 1
        else:
            counts_incorrect[tk] += 1
            counts_sem_incorrect[sk] += 1

    n = len(results)
    def pct(d):
        return {str(k): (100 * v / n if n else 0.0) for k, v in d.items()}

    def both(d_corr, d_inc):
        return {str(k): d_corr[k] + d_inc[k] for k in cats}

    pct_token_total = pct(both(counts_correct, counts_incorrect))
    pct_sem_total = pct(both(counts_sem_correct, counts_sem_incorrect))

    # Agreement rate token-vs-semantic
    n_agree = sum(1 for r in results if r["first_div_token"] == r["first_div_semantic"])

    summary = {
        "input_jsonl": str(JSONL),
        "n_total_problems": len(rows),
        "n_disagreement_problems": n,
        "pct_disagreement": round(100 * n / len(rows), 2),
        "cmaj_acc_on_disagreement": round(100 * sum(r["cmaj_correct"] for r in results) / n, 2) if n else 0.0,
        "token_first_divergence_pct": {k: round(v, 2) for k, v in pct_token_total.items()},
        "semantic_first_divergence_pct": {k: round(v, 2) for k, v in pct_sem_total.items()},
        "pct_diverge_sb_geq1_token": round(sum(pct_token_total[str(s)] for s in (1, 2, 3)), 2),
        "pct_diverge_sb_geq1_semantic": round(sum(pct_sem_total[str(s)] for s in (1, 2, 3)), 2),
        "pct_diverge_sb0_token": round(pct_token_total["0"], 2),
        "pct_diverge_sb0_semantic": round(pct_sem_total["0"], 2),
        "token_first_div_by_correctness": {
            "cmaj_correct": {str(k): v for k, v in counts_correct.items()},
            "cmaj_incorrect": {str(k): v for k, v in counts_incorrect.items()},
        },
        "semantic_first_div_by_correctness": {
            "cmaj_correct": {str(k): v for k, v in counts_sem_correct.items()},
            "cmaj_incorrect": {str(k): v for k, v in counts_sem_incorrect.items()},
        },
        "token_semantic_agreement_rate": round(100 * n_agree / n, 2) if n else 0.0,
        "outcome_band": classify_band(pct_token_total),
        "edit_threshold": EDIT_THRESHOLD,
        "cosine_threshold": COSINE_THRESHOLD,
        "gen_len": GEN_LEN,
        "sub_block_len": SUB_BLOCK_LEN,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "per_problem.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")

    print()
    print("token-level first divergence (% of disagreement problems):")
    for c in cats:
        print(f"  {cat_label[c]:>8}: {pct_token_total[str(c)]:5.1f}%")
    print(f"  >= sb 1 (mid-traj):   {summary['pct_diverge_sb_geq1_token']:5.1f}%")
    print(f"  sb 0 only:            {summary['pct_diverge_sb0_token']:5.1f}%")
    print()
    print("semantic (TF-IDF cos) first divergence:")
    for c in cats:
        print(f"  {cat_label[c]:>8}: {pct_sem_total[str(c)]:5.1f}%")
    print(f"  >= sb 1 (mid-traj):   {summary['pct_diverge_sb_geq1_semantic']:5.1f}%")
    print()
    print(f"token vs semantic agreement: {summary['token_semantic_agreement_rate']:.1f}%")
    print(f"outcome band: {summary['outcome_band']}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(cats))
    width = 0.38

    corr_vals = [counts_correct[c] for c in cats]
    inc_vals = [counts_incorrect[c] for c in cats]

    bars_corr = ax.bar(
        x - width / 2,
        corr_vals,
        width,
        label="cmaj correct",
        color=PALETTE.v3,
        edgecolor=PALETTE.ink,
        linewidth=0.4,
    )
    bars_inc = ax.bar(
        x + width / 2,
        inc_vals,
        width,
        label="cmaj incorrect",
        color=PALETTE.warn,
        edgecolor=PALETTE.ink,
        linewidth=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([cat_label[c] for c in cats])
    ax.set_xlabel("first sub-block at which any pair of branches has token edit-distance ≥ 3")
    ax.set_ylabel("number of disagreement problems")
    ax.set_title(
        f"E2 — cmaj-b=5 branch divergence by sub-block (N={n} disagreement problems / {len(rows)})"
    )
    ax.legend(loc="upper right")

    # value labels above bars
    for bars in (bars_corr, bars_inc):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.6,
                    str(int(h)),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=PALETTE.sub,
                )

    # annotation: outcome band
    band_text = {
        "mid_traj_geq40": "≥40% diverge at sub-block ≥1: mid-trajectory routing thesis supported",
        "sb0_geq80": "≥80% diverge at sub-block 0: per-sub-block routing thesis weakens",
        "mixed": "mixed (40–80% mid-traj): inconclusive at this N",
        "low_disagreement": "low disagreement: thesis testable only with more problems",
    }
    ax.text(
        0.01,
        0.97,
        band_text.get(summary["outcome_band"], summary["outcome_band"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=PALETTE.sub,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE.rule, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "histogram.png", dpi=200)
    fig.savefig(OUT_DIR / "histogram.pdf")
    plt.close(fig)

    print()
    print(f"wrote {OUT_DIR/'histogram.png'}")
    print(f"wrote {OUT_DIR/'histogram.pdf'}")
    print(f"wrote {OUT_DIR/'summary.json'}")
    print(f"wrote {OUT_DIR/'per_problem.jsonl'}")


def classify_band(pct: dict) -> str:
    mid = sum(pct[str(s)] for s in (1, 2, 3))
    sb0 = pct["0"]
    if mid >= 40:
        return "mid_traj_geq40"
    if sb0 >= 80:
        return "sb0_geq80"
    if 0 < mid < 40 and sb0 < 80:
        return "mixed"
    return "low_disagreement"


if __name__ == "__main__":
    main()
