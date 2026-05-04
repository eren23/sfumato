"""Upload comparison GIFs to W&B as job_type=comparison-trace runs.

Each GIF gets its own short run with the GIF as wandb.Image and a caption
explaining what the comparison shows.
"""
from __future__ import annotations
import pathlib

import wandb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
COMP_DIR = REPO_ROOT / "phase2/inference_viz/traces/comparisons"
PROJECT = "sfumato-e4"


def caption_for(name: str) -> str:
    if "substrate_p" in name and "correct_vs_wrong" in name:
        pid = name.split("substrate_p")[1].split("_")[0]
        return (f"Same problem (id={pid}), two LLaDA branches: ✅ correct (left) vs "
                f"❌ wrong (right). Both have 100% correct arithmetic — the wrong "
                f"branch fails on problem decomposition, not computation. "
                f"Visualizes the night-3 diagnosis: failure mode is comprehension.")
    if "all_llada" in name and "ar_at" in name:
        return ("all-LLaDA generation (left) vs hybrid AR-into-LLaDA splicing (right). "
                "Cells with green borders = AR-grafted tokens (Qwen2.5-0.5B-injected), "
                "around which LLaDA continues denoising. Demonstrates the mid-stream "
                "AR splicing mechanism that's the sfumato architecture's novelty.")
    return name


def main():
    gifs = sorted(COMP_DIR.glob("compare_*.gif"))
    print(f"uploading {len(gifs)} comparison GIFs to W&B project {PROJECT!r}")
    for g in gifs:
        cap = caption_for(g.stem)
        run = wandb.init(project=PROJECT, name=f"comparison-{g.stem}",
                         job_type="comparison-trace", reinit=True,
                         config={"gif_file": g.name, "caption": cap})
        run.log({"comparison": wandb.Image(str(g), caption=cap)})
        art = wandb.Artifact(name=f"comparison-{g.stem}", type="comparison-gif")
        art.add_file(str(g))
        run.log_artifact(art)
        run.finish()
        print(f"  uploaded {g.name}")


if __name__ == "__main__":
    main()
