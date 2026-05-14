"""Aggregate d4_d5_raw.json by scale × kind."""
import json
from collections import defaultdict
from pathlib import Path

d = json.loads(Path("e5/results/d4_d5_raw.json").read_text())

def get_scale(path):
    if "200M" in path: return "200M"
    if "120M" in path: return "120M"
    if "250M" in path: return "250M"
    if "300M" in path: return "300M"
    if "t1_arflow" in path: return "T1-60M"
    if "60M" in path or "t0_3k_seed" in path or "toy_composite_seed0" in path: return "60M"
    return "?"

buckets = defaultdict(lambda: defaultdict(list))
for path, res in d.items():
    if "error" in res: continue
    scale = get_scale(path)
    parts = path.split("/")
    if len(parts) < 2: continue
    kind = parts[-2]
    if len(parts) >= 3 and parts[-2] in ("ar_only", "diff_only", "flow_only") and parts[-3] == "paired":
        kind = "paired_" + parts[-2]

    ood = res.get("ood", {})
    if isinstance(ood, dict) and "error" not in ood:
        for split, v in ood.items():
            if isinstance(v, dict) and v.get("avg_nll") is not None:
                buckets[(scale, kind)]["ood_" + split].append(v["avg_nll"])

    ece = res.get("ece", {})
    if isinstance(ece, dict) and "error" not in ece:
        if ece.get("ece") is not None:
            buckets[(scale, kind)]["ece"].append(ece["ece"])
        if ece.get("mean_conf") is not None:
            buckets[(scale, kind)]["conf"].append(ece["mean_conf"])
        if ece.get("mean_acc") is not None:
            buckets[(scale, kind)]["acc"].append(ece["mean_acc"])


def avg(vs):
    return sum(vs) / len(vs) if vs else None

def fmt(v, decimals=3):
    if v is None: return "—"
    return f"%.{decimals}f" % v

# Focus on the central question: composite vs ar_only at each scale
# on OOD axes and ECE.
print("{:<10} {:<22} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
    "scale", "kind", "n", "wikitext", "owt", "ECE", "conf", "acc"
))
print("-" * 90)
for (scale, kind), vals in sorted(buckets.items()):
    n = max((len(v) for v in vals.values()), default=0)
    print("{:<10} {:<22} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        scale, kind, n,
        fmt(avg(vals.get("ood_wikitext2", [])), 3),
        fmt(avg(vals.get("ood_openwebtext_holdout", [])), 3),
        fmt(avg(vals.get("ece", [])), 4),
        fmt(avg(vals.get("conf", [])), 3),
        fmt(avg(vals.get("acc", [])), 3),
    ))
