"""Aggregate E3a/E3b/E3c results and emit the Phase-E paper-class verdict.

Pulls from:
  e5/results/e3a_probe5_n8/summary.json
  e5/results/e3b_multiscale_d2/summary.json
  e5/results/e3c_d3_crossover/summary.json

Emits:
  e5/results/T0_PHASE_E_VERDICT.md — the paper-class decision per the
    Phase-D decision tree, with the new numbers.

Decision rule (per the Phase E plan):
  | E3a n=8 mode-switch        | E3b multi-scale       | E3c crossover     | Class |
  | ≥+3 pp, SEM ≤ 1pp          | comp wins at 3/3      | localised         | TMLR  |
  | ≥+2 pp                     | comp wins at 2/3      | consistent        | Strong workshop / TMLR |
  | ≤+1 pp                     | mixed                 | noisy             | Workshop |
"""

from __future__ import annotations

import json
import math
from pathlib import Path

E5 = Path("/Users/eren/Documents/ai/sfumato/e5/results")


# Existing composite-3k diff-NLL at each scale (from T0_PROBES_FINAL).
COMPOSITE_3K_NLL = {"60M": 5.74, "120M": 5.68, "200M": 5.42, "300M": 5.35}


def load(name):
    p = E5 / name / "summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def e3a_aggregate(data):
    """Per-mode mean / SEM across E3a seeds for composite and ar_only."""
    if not data:
        return None
    modes = ("ar_only", "mode_switch_96_32", "mode_switch_64_32", "paired_64_64")
    out = {"n_seeds": len(data)}
    for variant in ("composite", "ar_only"):
        out[variant] = {}
        for m in modes:
            accs = [row[variant][m]["accuracy"]
                    for row in data
                    if variant in row and m in row[variant] and "accuracy" in row[variant][m]]
            if not accs:
                continue
            mean = sum(accs) / len(accs)
            std = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs))
            sem = std / math.sqrt(len(accs))
            out[variant][m] = {"mean": mean, "sem": sem, "n": len(accs),
                               "per_seed": [round(a*100, 1) for a in accs]}
    return out


def e3b_aggregate(data):
    """Per-scale pure-diff-6k mean NLL + delta vs existing composite-3k."""
    if not data:
        return None
    by_scale = {}
    for r in data:
        by_scale.setdefault(r["scale"], []).append(r["avg_nll"])
    out = {}
    for scale, vals in by_scale.items():
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
        comp_3k = COMPOSITE_3K_NLL.get(scale)
        # Composite wins if pd6k > composite_3k. We report (pd6k - comp_3k)
        # so a positive value means composite is better.
        composite_lead = (m - comp_3k) if comp_3k is not None else None
        out[scale] = {"pure_diff_6k_mean": round(m, 3), "std": round(s, 3),
                      "n": len(vals), "composite_3k": comp_3k,
                      "composite_lead_nll": round(composite_lead, 3) if composite_lead is not None else None}
    return out


def e3c_aggregate(data):
    """Per-data-size composite-mean vs ar_mean."""
    if not data:
        return None
    by = {}
    for r in data:
        by.setdefault((r["size"], r["variant"]), []).append(r["avg_nll"])
    sizes = sorted({s for (s, _) in by})
    out = {}
    for size in sizes:
        comp = by.get((size, "composite"), [])
        ar = by.get((size, "ar_only"), [])
        if comp and ar:
            comp_m = sum(comp) / len(comp)
            ar_m = sum(ar) / len(ar)
            out[size] = {"composite_mean": round(comp_m, 3),
                         "ar_mean": round(ar_m, 3),
                         "delta": round(comp_m - ar_m, 3),
                         "n_seeds": len(comp)}
    return out


def decision(e3a, e3b, e3c):
    """Apply the Phase E decision rule."""
    notes = []
    # E3a
    if e3a and "composite" in e3a and "mode_switch_64_32" in e3a["composite"]:
        c = e3a["composite"]["mode_switch_64_32"]
        a = e3a["ar_only"]["mode_switch_64_32"]
        delta_pp = (c["mean"] - a["mean"]) * 100
        sem_pp = math.sqrt(c["sem"]**2 + a["sem"]**2) * 100
        if delta_pp >= 3 and sem_pp <= 1:
            e3a_class = "TMLR"
        elif delta_pp >= 2:
            e3a_class = "strong_workshop"
        elif delta_pp <= 1:
            e3a_class = "workshop"
        else:
            e3a_class = "ambiguous"
        notes.append(f"E3a: composite_ms64_32 = {c['mean']*100:.2f}±{c['sem']*100:.2f}% vs ar_only {a['mean']*100:.2f}±{a['sem']*100:.2f}%, Δ = {delta_pp:+.2f}pp → {e3a_class}")
    else:
        e3a_class = "incomplete"
        notes.append("E3a: incomplete")

    # E3b
    if e3b:
        wins = sum(1 for s, r in e3b.items()
                   if r.get("composite_lead_nll") is not None
                   and r["composite_lead_nll"] > 0.2)
        total = len([r for r in e3b.values() if r.get("composite_lead_nll") is not None])
        if total == 0:
            e3b_class = "incomplete"
        elif wins == total and total >= 3:
            e3b_class = "TMLR"
        elif wins >= 2:
            e3b_class = "strong_workshop"
        else:
            e3b_class = "workshop"
        notes.append(f"E3b: composite wins at {wins}/{total} scales (>0.2 NLL) → {e3b_class}")
    else:
        e3b_class = "incomplete"
        notes.append("E3b: incomplete")

    # E3c
    if e3c:
        deltas = [r["delta"] for r in e3c.values()]
        monotone = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))
        crossover = any(d < 0 for d in deltas) and any(d > 0 for d in deltas)
        if monotone and crossover:
            e3c_class = "TMLR"
        elif crossover:
            e3c_class = "strong_workshop"
        else:
            e3c_class = "workshop"
        notes.append(f"E3c: deltas={deltas}, monotone={monotone}, crossover={crossover} → {e3c_class}")
    else:
        e3c_class = "incomplete"
        notes.append("E3c: incomplete")

    # Overall
    classes = [e3a_class, e3b_class, e3c_class]
    if all(c == "TMLR" for c in classes):
        overall = "TMLR (all three positive)"
    elif "incomplete" in classes:
        overall = f"INCOMPLETE — pending: {[c for c in classes if c == 'incomplete']}"
    elif classes.count("TMLR") >= 2 and "workshop" not in classes:
        overall = "TMLR with caveats / strong workshop"
    else:
        overall = "Workshop"
    notes.append(f"\nOVERALL: {overall}")
    return overall, notes


def main():
    e3a_raw = load("e3a_probe5_n8")
    e3b_raw = load("e3b_multiscale_d2")
    e3c_raw = load("e3c_d3_crossover")

    e3a = e3a_aggregate(e3a_raw)
    e3b = e3b_aggregate(e3b_raw)
    e3c = e3c_aggregate(e3c_raw)

    overall, notes = decision(e3a, e3b, e3c)

    lines = ["# T0 / Phase E — TMLR upgrade verdict\n"]
    lines.append(f"## Overall: {overall}\n")
    for n in notes:
        lines.append(f"- {n}")

    if e3a:
        lines.append("\n## E3a: probe-5 mode-switching (multi-seed)\n")
        lines.append(f"n_seeds = {e3a['n_seeds']}")
        for variant in ("composite", "ar_only"):
            lines.append(f"\n**{variant}**:")
            for mode, st in e3a[variant].items():
                lines.append(f"  - {mode}: mean={st['mean']*100:.2f}%, SEM={st['sem']*100:.2f}%, n={st['n']}, per-seed={st['per_seed']}")

    if e3b:
        lines.append("\n## E3b: D2 multi-scale compute-matched control\n")
        lines.append("| Scale | composite-3k NLL | pure-diff-6k NLL | composite lead (pd6k − comp_3k) | n |")
        lines.append("|---|---|---|---|---|")
        for scale in sorted(e3b):
            r = e3b[scale]
            lines.append(f"| {scale} | {r['composite_3k']} | {r['pure_diff_6k_mean']} ± {r['std']} | "
                         f"{r['composite_lead_nll']:+} | {r['n']} |")

    if e3c:
        lines.append("\n## E3c: D3 crossover refinement\n")
        lines.append("| Data size | composite | ar_only | Δ (composite − ar) | n |")
        lines.append("|---|---|---|---|---|")
        for size in sorted(e3c):
            r = e3c[size]
            lines.append(f"| {size}p | {r['composite_mean']} | {r['ar_mean']} | {r['delta']:+.3f} | {r['n_seeds']} |")

    out_path = E5 / "T0_PHASE_E_VERDICT.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path)
    print(f"\n{overall}")
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
