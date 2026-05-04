"""Upload all night-1/2/3 verifier result JSONs to W&B as one-shot runs.

Each result JSON becomes a W&B run with:
  - config: model_id, n_branches, n_problems, cv_folds
  - summary metrics: cmaj_acc, verifier_acc, oracle_acc, delta_pp_vs_cmaj, decision
  - artifact: the JSON itself

Run with: python phase2/scripts/upload_verifier_results_wandb.py
"""
from __future__ import annotations
import glob
import json
import pathlib
import sys

import wandb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT = "sfumato-e4"

OPTION2_PATTERN = REPO_ROOT / "phase2/spikes/verifier-aggregation/option2_results_*.json"
OPTION3_PATTERNS = [
    REPO_ROOT / "phase2/spikes/option3-process-reward/option3_results_*.json",
    REPO_ROOT / "phase2/spikes/option3-process-reward/option3_pair_results.json",
    REPO_ROOT / "phase2/spikes/option3-process-reward/option3_step_prm_results.json",
]


def upload_one(path: pathlib.Path, group: str):
    data = json.loads(path.read_text())
    cmaj = data.get("mean_cmaj") or data.get("cmaj_acc") or 0.0
    ver = data.get("mean_verifier") or data.get("verifier_acc") or 0.0
    oracle = data.get("mean_oracle") or data.get("oracle_acc") or 0.0
    dpp = data.get("delta_pp_vs_cmaj") or data.get("dpp_vs_cmaj") or (ver - cmaj) * 100
    decision = data.get("pre_reg_decision") or data.get("decision") or "?"
    name = path.stem
    config = {
        "model_id": data.get("model_id"),
        "n_branches": data.get("n_branches"),
        "n_problems": data.get("n_problems"),
        "cv_folds": data.get("cv_folds"),
        "rich_path": data.get("rich_path"),
        "approach": group,
        "result_file": path.name,
    }
    run = wandb.init(project=PROJECT, name=name, group=group, job_type="verifier-result",
                     reinit=True, config={k: v for k, v in config.items() if v is not None})
    run.summary["cmaj_acc"] = cmaj
    run.summary["verifier_acc"] = ver
    run.summary["oracle_acc"] = oracle
    run.summary["delta_pp_vs_cmaj"] = dpp
    run.summary["decision"] = decision
    art = wandb.Artifact(name=f"result-{name}", type="verifier-result-json")
    art.add_file(str(path))
    run.log_artifact(art)
    run.finish()
    print(f"  uploaded {name}: cmaj={cmaj:.1%} ver={ver:.1%} dpp={dpp:+.2f} {decision}")


def main():
    files = []
    for f in sorted(glob.glob(str(OPTION2_PATTERN))):
        files.append((pathlib.Path(f), "option2-text-embed"))
    for pattern in OPTION3_PATTERNS:
        for f in sorted(glob.glob(str(pattern))):
            p = pathlib.Path(f)
            if "pair" in p.name:
                grp = "option3-branchpair"
            elif "step_prm" in p.name:
                grp = "option3-step-prm"
            else:
                grp = "option3-process-mlp"
            files.append((p, grp))
    print(f"uploading {len(files)} verifier results to W&B project '{PROJECT}'")
    for p, grp in files:
        try:
            upload_one(p, grp)
        except Exception as e:
            print(f"  FAIL {p.name}: {e}")


if __name__ == "__main__":
    main()
