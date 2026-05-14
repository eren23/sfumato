"""Aggregate probes_raw.json by scale × checkpoint kind."""
import json
import sys
from collections import defaultdict
from pathlib import Path

d = json.loads(Path("e5/results/probes_raw.json").read_text())

def get_scale(path):
    if "200M" in path: return "200M"
    if "120M" in path: return "120M"
    if "250M" in path: return "250M"
    if "300M" in path: return "300M"
    if "t1_arflow" in path: return "T1-60M"
    if "60M" in path: return "60M"
    if "t0_3k_seed" in path or "toy_composite_seed0" in path: return "60M"
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

    p1 = res.get("probe1_diff_axis", {}) or {}
    p2 = res.get("probe2_fim", {}) or {}
    p3 = res.get("probe3_ood", {}) or {}
    p4 = res.get("probe4_pos_strat", {}) or {}
    p7 = res.get("probe7_calibration", {}) or {}

    if p1.get("avg_nll") is not None:
        buckets[(scale, kind)]["diff_nll"].append(p1["avg_nll"])
    if p2.get("fim_diff_avg_nll") is not None:
        buckets[(scale, kind)]["fim_nll"].append(p2["fim_diff_avg_nll"])
    if p3.get("ood_avg_nll") is not None:
        buckets[(scale, kind)]["ood_nll"].append(p3["ood_avg_nll"])
    if p7.get("mean_entropy_nats") is not None:
        buckets[(scale, kind)]["entropy"].append(p7["mean_entropy_nats"])
        buckets[(scale, kind)]["top1_acc"].append(p7["top1_accuracy"])
    for i in range(4):
        v = p4.get("bin" + str(i) + "_avg_nll")
        if v is not None:
            buckets[(scale, kind)]["bin" + str(i)].append(v)


def avg(vs):
    return sum(vs) / len(vs) if vs else None

def fmt(v):
    return ("%.3f" % v) if v is not None else "—"

header = "{:<10} {:<22} {:>3} {:>9} {:>9} {:>9} {:>9} {:>7} {:>22}".format(
    "scale", "kind", "n", "diff_NLL", "fim_NLL", "OOD_NLL", "entropy", "top1", "AR position [b0 b1 b2 b3]"
)
print(header)
print("-" * len(header))
for (scale, kind), vals in sorted(buckets.items()):
    n = max((len(v) for v in vals.values()), default=0)
    bins_str = " ".join(fmt(avg(vals.get("bin" + str(i), []))) for i in range(4))
    print("{:<10} {:<22} {:>3} {:>9} {:>9} {:>9} {:>9} {:>7} [{}]".format(
        scale, kind, n,
        fmt(avg(vals.get("diff_nll", []))),
        fmt(avg(vals.get("fim_nll", []))),
        fmt(avg(vals.get("ood_nll", []))),
        fmt(avg(vals.get("entropy", []))),
        fmt(avg(vals.get("top1_acc", []))),
        bins_str,
    ))
