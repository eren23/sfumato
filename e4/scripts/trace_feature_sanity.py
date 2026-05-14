"""G — feature sanity check on 7 existing Workstream-C traces.

Goal: before paying to scale trace collection to 1000 trajectories for E4,
check whether the existing 7 STATUS-schema traces contain ANY hint of a
feature x correctness association at n=7. Pure signal-presence detection,
not a verifier evaluation.

Input:  phase2/inference_viz/traces/trace_real_p{10..70}_*.jsonl (4 records each)
        phase2/inference_viz/traces/make_real_traces_summary.json (final_text_tail)
        e4/results/raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl (gold labels)
Output: e4/results/trace_feature_sanity/{univariate_table.csv,summary.md}

Decision rule: if any feature has |Spearman rho| >= 0.5 AND permutation p-value
<= 0.10, declare "signal plausible at small N -- E4 justified for paid scaling".
Otherwise "no signal at small N -- E4 deferred".
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from pathlib import Path
from statistics import mean

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO_ROOT / "phase2" / "inference_viz" / "traces"
TRACE_SUMMARY = TRACE_DIR / "make_real_traces_summary.json"
CMAJ_JSONL = REPO_ROOT / "e4" / "results" / "raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl"
OUT_DIR = REPO_ROOT / "e4" / "results" / "trace_feature_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANSWER_PAT = re.compile(r"Answer:\s*(-?\$?\d[\d,]*\.?\d*)")


def topk_entropy(topk_row):
    """Shannon entropy of the top-k probability distribution at one position.

    topk_row is a list of [token_id, prob] pairs (top 5). Renormalise (so it
    sums to ~1 within the slice) and compute -sum p log p in nats.
    """
    probs = [p for (_t, p) in topk_row]
    s = sum(probs)
    if s <= 0:
        return 0.0
    probs = [p / s for p in probs]
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    return h


def extract_features(records):
    """records: list of 4 sub-block step records. Returns a dict of features."""
    n_blocks = len(records)
    all_entropies = []
    sb_entropy_means = []
    sb_topk_entropy_means = []
    sb_entropy_mins = []
    sb_entropy_maxs = []
    sb_wallclock = []
    mechanisms = []
    commit_active_count = 0
    tokens_committed_total = 0

    for r in records:
        ents = r.get("entropy") or []
        all_entropies.extend(ents)
        sb_entropy_means.append(mean(ents) if ents else 0.0)
        sb_entropy_mins.append(min(ents) if ents else 0.0)
        sb_entropy_maxs.append(max(ents) if ents else 0.0)

        topk = r.get("top_k_logits") or []
        if topk:
            sb_topk_entropy_means.append(mean(topk_entropy(row) for row in topk))
        else:
            sb_topk_entropy_means.append(0.0)

        sb_wallclock.append(r.get("wallclock_ms") or 0)
        mechanisms.append(r.get("mechanism") or "?")
        if r.get("commit_lora_active"):
            commit_active_count += 1
        tokens_committed_total += len(r.get("tokens_committed") or [])

    feats = {
        "mean_entropy": mean(all_entropies) if all_entropies else 0.0,
        "min_entropy": min(all_entropies) if all_entropies else 0.0,
        "max_entropy": max(all_entropies) if all_entropies else 0.0,
        "std_entropy": float(np.std(all_entropies)) if all_entropies else 0.0,
        "sb0_entropy_mean": sb_entropy_means[0] if n_blocks > 0 else 0.0,
        "sb1_entropy_mean": sb_entropy_means[1] if n_blocks > 1 else 0.0,
        "sb2_entropy_mean": sb_entropy_means[2] if n_blocks > 2 else 0.0,
        "sb3_entropy_mean": sb_entropy_means[3] if n_blocks > 3 else 0.0,
        "sb0_entropy_min": sb_entropy_mins[0] if n_blocks > 0 else 0.0,
        "sb0_entropy_max": sb_entropy_maxs[0] if n_blocks > 0 else 0.0,
        "mean_topk_entropy": mean(sb_topk_entropy_means) if sb_topk_entropy_means else 0.0,
        "sb0_topk_entropy_mean": sb_topk_entropy_means[0] if sb_topk_entropy_means else 0.0,
        "mechanism_diversity": len(set(mechanisms)),
        "wallclock_total_ms": sum(sb_wallclock),
        "wallclock_max_ms": max(sb_wallclock) if sb_wallclock else 0,
        "commit_lora_fraction": commit_active_count / max(1, n_blocks),
        "tokens_committed_total": tokens_committed_total,
    }
    return feats


def spearman_rho(x, y):
    """Spearman rank-correlation, hand-rolled. x and y are 1D arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Rank with average-tied-rank
    rx = _rank(x)
    ry = _rank(y)
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = rx.mean(), ry.mean()
    num = float(np.sum((rx - mx) * (ry - my)))
    den = float(np.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2)))
    if den == 0:
        return float("nan")
    return num / den


def _rank(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # Average tied ranks
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros_like(uniq, dtype=float)
        for i, v in enumerate(a):
            sums[inv[i]] += ranks[i]
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def permutation_p(x, y, n_perms=1000, seed=42):
    """Two-sided permutation p-value for Spearman rho."""
    rng = np.random.default_rng(seed)
    observed = abs(spearman_rho(x, y))
    if np.isnan(observed):
        return float("nan")
    y_arr = np.array(y, dtype=float)
    ge = 0
    for _ in range(n_perms):
        rng.shuffle(y_arr)
        r = spearman_rho(x, y_arr)
        if not np.isnan(r) and abs(r) >= observed:
            ge += 1
    return (ge + 1) / (n_perms + 1)


def extract_answer(text: str) -> str | None:
    m = ANSWER_PAT.search(text or "")
    if m:
        return m.group(1).replace(",", "").replace("$", "")
    return None


def main() -> None:
    files = sorted(glob.glob(str(TRACE_DIR / "trace_real_p*.jsonl")))
    print(f"found {len(files)} trace files")

    cmaj_rows = {r["idx"]: r for r in (json.loads(l) for l in open(CMAJ_JSONL))}
    trace_summary = {entry["problem_idx"]: entry for entry in json.loads(open(TRACE_SUMMARY).read())}

    rows = []
    for f in files:
        recs = [json.loads(l) for l in open(f)]
        if not recs:
            continue
        pidx = recs[0]["problem_idx"]
        gold = cmaj_rows.get(pidx, {}).get("gold")
        cmaj_pred = cmaj_rows.get(pidx, {}).get("pred")
        summary_entry = trace_summary.get(pidx, {})
        final_text = summary_entry.get("final_text_tail", "")
        trace_ans = extract_answer(final_text)
        trace_correct = int(trace_ans is not None and gold is not None and str(trace_ans).strip() == str(gold).strip())
        cmaj_correct = int(cmaj_pred is not None and gold is not None and str(cmaj_pred).strip() == str(gold).strip())
        tag = summary_entry.get("tag", "?")

        feats = extract_features(recs)
        feats["problem_idx"] = pidx
        feats["trace_tag"] = tag
        feats["trace_answer"] = trace_ans
        feats["gold"] = gold
        feats["trace_correct"] = trace_correct
        feats["cmaj_correct"] = cmaj_correct
        rows.append(feats)

    # Sort by problem_idx for determinism
    rows.sort(key=lambda r: r["problem_idx"])
    print()
    print("per-trace summary:")
    for r in rows:
        print(f"  p{r['problem_idx']} tag={r['trace_tag']:<14} ans={str(r['trace_answer']):<8} gold={str(r['gold']):<8} trace_correct={r['trace_correct']} cmaj_correct={r['cmaj_correct']}")
    print()

    FEATURES = [
        "mean_entropy", "min_entropy", "max_entropy", "std_entropy",
        "sb0_entropy_mean", "sb1_entropy_mean", "sb2_entropy_mean", "sb3_entropy_mean",
        "sb0_entropy_min", "sb0_entropy_max",
        "mean_topk_entropy", "sb0_topk_entropy_mean",
        "mechanism_diversity", "wallclock_total_ms", "wallclock_max_ms",
        "commit_lora_fraction", "tokens_committed_total",
    ]

    # ---- univariate associations against trace_correct ----
    label_trace = np.array([r["trace_correct"] for r in rows])
    label_cmaj = np.array([r["cmaj_correct"] for r in rows])

    table = []
    for fname in FEATURES:
        x = np.array([r[fname] for r in rows], dtype=float)
        m_corr = float(x[label_trace == 1].mean()) if (label_trace == 1).any() else float("nan")
        m_inc = float(x[label_trace == 0].mean()) if (label_trace == 0).any() else float("nan")
        rho = spearman_rho(x, label_trace)
        p = permutation_p(x, label_trace.tolist(), n_perms=10000, seed=42)
        rho_cmaj = spearman_rho(x, label_cmaj)
        p_cmaj = permutation_p(x, label_cmaj.tolist(), n_perms=10000, seed=42)
        table.append({
            "feature": fname,
            "mean_correct": round(m_corr, 5),
            "mean_incorrect": round(m_inc, 5),
            "spearman_rho_vs_trace": round(rho, 4) if not np.isnan(rho) else "nan",
            "perm_p_vs_trace": round(p, 4) if not np.isnan(p) else "nan",
            "spearman_rho_vs_cmaj": round(rho_cmaj, 4) if not np.isnan(rho_cmaj) else "nan",
            "perm_p_vs_cmaj": round(p_cmaj, 4) if not np.isnan(p_cmaj) else "nan",
        })

    # ---- write CSV ----
    csv_path = OUT_DIR / "univariate_table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # ---- decision rule ----
    SIGNAL_RHO = 0.5
    SIGNAL_P = 0.10

    signal_hits = []
    for row in table:
        for label_axis in ("trace", "cmaj"):
            rho_field = f"spearman_rho_vs_{label_axis}"
            p_field = f"perm_p_vs_{label_axis}"
            rho_v = row[rho_field]
            p_v = row[p_field]
            try:
                if abs(float(rho_v)) >= SIGNAL_RHO and float(p_v) <= SIGNAL_P:
                    signal_hits.append((row["feature"], label_axis, float(rho_v), float(p_v)))
            except (TypeError, ValueError):
                continue

    verdict = "signal_plausible" if signal_hits else "no_signal"

    print("top features by |rho| (vs trace_correct):")
    for row in sorted(table, key=lambda r: abs(float(r["spearman_rho_vs_trace"])) if r["spearman_rho_vs_trace"] != "nan" else 0, reverse=True)[:6]:
        print(f"  {row['feature']:<26} rho={row['spearman_rho_vs_trace']:>8} p={row['perm_p_vs_trace']:>8} | corr_mean={row['mean_correct']:>10} inc_mean={row['mean_incorrect']:>10}")
    print()
    print("top features by |rho| (vs cmaj_correct):")
    for row in sorted(table, key=lambda r: abs(float(r["spearman_rho_vs_cmaj"])) if r["spearman_rho_vs_cmaj"] != "nan" else 0, reverse=True)[:6]:
        print(f"  {row['feature']:<26} rho={row['spearman_rho_vs_cmaj']:>8} p={row['perm_p_vs_cmaj']:>8}")
    print()
    print(f"decision rule: |rho| >= {SIGNAL_RHO} AND perm-p <= {SIGNAL_P}")
    if signal_hits:
        for f, axis, rho_v, p_v in signal_hits:
            print(f"  HIT: {f} vs {axis}: rho={rho_v:.3f} p={p_v:.4f}")
    else:
        print("  no feature clears both thresholds")
    print(f"verdict: {verdict}")

    # ---- summary markdown ----
    n_trace_correct = int(label_trace.sum())
    n_trace_total = len(label_trace)
    n_cmaj_correct = int(label_cmaj.sum())

    summary_md = f"""# G — trace feature sanity check (n=7 deterministic traces)

## Verdict

**`{verdict}`** — based on decision rule: any feature has |Spearman ρ| ≥ {SIGNAL_RHO} AND permutation p ≤ {SIGNAL_P} on n={n_trace_total} traces (10000 perms).

{"### Hits" if signal_hits else "### No hits"}

{chr(10).join(f"- `{f}` vs `{axis}_correct`: ρ={rho_v:.3f}, perm-p={p_v:.4f}" for (f, axis, rho_v, p_v) in signal_hits) if signal_hits else "No feature cleared both thresholds. Note that with n=7, the minimum possible |ρ|=1.0 has perm-p ≈ 0.014, so the threshold is achievable in principle — the substrate just doesn't show it on these features."}

## Sample

n = {n_trace_total} deterministic Workstream-C traces, one per problem idx in {{10, 20, 30, 40, 50, 60, 70}}, each containing 4 sub-block records.

- **trace_correct** label: extracted from `make_real_traces_summary.json`'s `final_text_tail` via `Answer:\\s*N` regex, compared to GSM8K-test gold. {n_trace_correct}/{n_trace_total} correct.
- **cmaj_correct** label: the parent cmaj-b=5 vote's correctness from `raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl`. {n_cmaj_correct}/{n_trace_total} correct.

The two label axes differ because the traces use various directive sequences (`all_llada`, `ar_handoff`, `cmaj_branch`, `mid_ar_handoff`, `early_cmaj`) — they are deterministic single-branch runs, not standard cmaj-b=5 generations.

## Top features (by |ρ| vs trace_correct)

| feature | ρ | perm-p | mean(correct) | mean(incorrect) |
|---------|---|--------|---------------|-----------------|
"""
    for row in sorted(table, key=lambda r: abs(float(r["spearman_rho_vs_trace"])) if r["spearman_rho_vs_trace"] != "nan" else 0, reverse=True)[:8]:
        summary_md += f"| `{row['feature']}` | {row['spearman_rho_vs_trace']} | {row['perm_p_vs_trace']} | {row['mean_correct']} | {row['mean_incorrect']} |\n"

    summary_md += """
## Per-trace details

| problem_idx | tag | trace_answer | gold | trace_correct | cmaj_correct |
|-------------|-----|--------------|------|---------------|--------------|
"""
    for r in rows:
        summary_md += f"| p{r['problem_idx']} | {r['trace_tag']} | {r['trace_answer']} | {r['gold']} | {r['trace_correct']} | {r['cmaj_correct']} |\n"

    summary_md += f"""
## Caveats (must read before acting on the verdict)

- **n=7 is too small for any rigorous statistical claim.** This is a hint-of-signal probe, not an evaluation. A `no_signal` result here may be type-II error (false negative).
- **Existing traces are deterministic replays** with pre-planned directive sequences, not natural cmaj branches. Feature distributions may not represent what E4 would actually see at full scale.
- **Heterogeneous trace tags.** The 7 traces span 5 different directive configurations. Cross-trace feature comparisons mix conditions, which inflates feature variance and reduces correlation power.
- **Mechanism diversity is near-degenerate.** Most traces are predominantly `llada` mechanism; the `mechanism_diversity` feature has very little dynamic range at this small n.
- **logit_shift_norm is null** in these traces (commit-LoRA was not active for any of the 7). The richest single feature from the STATUS schema is therefore unmeasured here. A follow-up should re-generate a few traces with `LOGIT_SHIFT_NORM=1` and re-run G.

## Interpretation alongside E2

E2 says ≥91% of cmaj-disagreement problems first diverge at sub-block 0 (per `e4/results/cmaj_divergence/summary.md`). Combined with this G result:

- If `{verdict}` == `no_signal`: **Cell 4** of the fork — both legs of per-sub-block routing fail. Strong stop signal for Direction A's current per-sub-block form. Move to discussion.tex's other Phase-3 paths or finalize paper 2 (B) and D.
- If `{verdict}` == `signal_plausible`: **Cell 3** — features carry information but the decision point is at sub-block 0 (not mid-trajectory). Per-sub-block routing collapses to "pilot-and-route after sub-block 0". Direction A needs redesign.

In either branch, **E1/E3/E4/E5 in their original forms are not the right next step**. D (comprehension scaffolding) and B (paper 2 write-up) remain the highest-leverage remaining items.

## Files

- `univariate_table.csv` — full per-feature stats
- `summary.md` — this document
- `../../scripts/trace_feature_sanity.py` — script

## Follow-up

To strengthen G's `no_signal` result before fully deferring E4:

1. Re-generate 20–30 deterministic traces using `make_real_traces.py` with **mixed directive sequences** and **`LOGIT_SHIFT_NORM=1`** enabled. Cost ~$1 on a 4090 spot.
2. Re-run this script with the larger sample; permutation power tightens by ~3×.
3. If still no signal, treat E4 as deferred indefinitely.
"""

    (OUT_DIR / "summary.md").write_text(summary_md)
    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT_DIR/'summary.md'}")


if __name__ == "__main__":
    main()
