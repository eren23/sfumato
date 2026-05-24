"""Phase P.1 + P.2 — train TopK SAEs on Sfumato activations.

Trains one SAE for a single hookpoint at a time. Activations are
collected on-the-fly via forward hooks on the loaded CompositeLM,
trained for N steps, evaluated on a held-out batch.

Hookpoint scheme:
  "block.{i}.ar"       — residual after block i, AR mode
  "block.{i}.diff"     — residual after block i, diff mode
  "ln_f.ar"            — post-ln_f, AR mode  (input to AR head)
  "ln_f.diff"          — post-ln_f, diff mode (input to diff head's
                         optional projection)
  "head_diff_proj"     — output of head_diff_proj

ENV (see file).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from e5.interp.load_model import load_composite_for_interp
from e5.interp.topk_sae import TopKSAE, topk_sae_loss


def attach_hook(raw_model, hookpoint: str, store: dict):
    def _cap(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        store["act"] = output
    if hookpoint == "ln_f":
        return raw_model.ln_f.register_forward_hook(_cap)
    if hookpoint == "head_diff_proj":
        return raw_model.head_diff_proj.register_forward_hook(_cap)
    if hookpoint.startswith("block."):
        idx = int(hookpoint.split(".")[1])
        return raw_model.blocks[idx].register_forward_hook(_cap)
    raise ValueError(f"unknown hookpoint {hookpoint!r}")


def resolve_hookpoint(hookpoint_spec: str):
    parts = hookpoint_spec.split(".")
    if parts[-1] in ("ar", "diff"):
        mode = parts[-1]
        target = ".".join(parts[:-1])
        return target, mode
    return hookpoint_spec, "ar"


def collect_activations(raw_model, tokens, hook_target, mode, n_batches,
                        batch, T, device):
    store = {"act": None}
    handle = attach_hook(raw_model, hook_target, store)
    try:
        rng = np.random.default_rng(0)
        for _ in range(n_batches):
            starts = rng.integers(0, len(tokens) - T, size=batch)
            windows = np.stack([tokens[s:s+T] for s in starts]).astype(np.int64)
            idx = torch.from_numpy(windows).to(device)
            with torch.no_grad():
                raw_model(idx, mode=mode)
            act = store["act"].detach().reshape(-1, store["act"].shape[-1]).float()
            yield act
    finally:
        handle.remove()


def main():
    ckpt = os.environ.get("CKPT",
        str(REPO_ROOT / "e5/results/f10_mixed/composite/model_slim_final.pt"))
    hookpoint = os.environ.get("HOOKPOINT", "block.10.ar")
    out_dir = Path(os.environ.get("OUT_DIR",
        str(REPO_ROOT / "e5/interp/saes" / hookpoint.replace(".", "_"))))
    tokens_path = os.environ.get("TOKENS_PATH",
        str(Path.home() / ".cache/sfumato_e5/fineweb_gpt2_2850000000.npy"))
    d_features = int(os.environ.get("D_FEATURES", "16384"))
    k = int(os.environ.get("TOPK", "64"))
    steps = int(os.environ.get("STEPS", "8000"))
    batch = int(os.environ.get("BATCH", "8"))
    T = int(os.environ.get("T", "256"))
    lr = float(os.environ.get("LR", "3e-4"))
    eval_every = int(os.environ.get("EVAL_EVERY", "500"))
    renorm_every = int(os.environ.get("RENORM_EVERY", "200"))

    device = os.environ.get("DEVICE",
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[sae] device={device}  ckpt={ckpt}  hookpoint={hookpoint}", flush=True)
    print(f"[sae] cfg: d_feat={d_features} k={k} steps={steps} batch={batch} T={T} lr={lr}",
          flush=True)

    # wandb init (silent skip if no key or wandb missing)
    wb = None
    try:
        if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "disabled":
            import wandb as wb_lib  # noqa
            label = hookpoint.replace(".", "_")
            wb = wb_lib.init(
                project=os.environ.get("WANDB_PROJECT", "sfumato-interp"),
                name=os.environ.get("WANDB_RUN_NAME", f"sae-{label}"),
                group=os.environ.get("WANDB_GROUP", "p1-saes"),
                job_type="sae-train",
                config={
                    "hookpoint": hookpoint, "d_features": d_features, "k": k,
                    "steps": steps, "batch": batch, "T": T, "lr": lr,
                    "ckpt": str(ckpt), "device": device,
                },
                reinit=True,
            )
            print(f"[wandb] init OK: {wb.url}", flush=True)
    except Exception as e:
        print(f"[wandb] init failed: {e!s:.200}", flush=True)
        wb = None

    nn_model, raw_model, cfg = load_composite_for_interp(ckpt, device=device)
    d_in = cfg.d_model
    print(f"[sae] d_in={d_in} from CompositeLM", flush=True)

    hook_target, mode = resolve_hookpoint(hookpoint)
    print(f"[sae] resolved: target={hook_target} mode={mode}", flush=True)

    sae = TopKSAE(d_in=d_in, d_features=d_features, k=k).to(device)
    print(f"[sae] SAE params: {sae.num_params()/1e6:.1f}M", flush=True)
    optim = torch.optim.AdamW(sae.parameters(), lr=lr, betas=(0.9, 0.999),
                               weight_decay=0.0)

    if Path(tokens_path).exists():
        tokens = np.load(tokens_path, mmap_mode="r")
        print(f"[sae] tokens: {len(tokens):,} from {tokens_path}", flush=True)
    else:
        print(f"[sae] WARN: tokens file not found, falling back to random", flush=True)
        tokens = np.random.randint(0, 50256, size=(10_000_000,), dtype=np.uint16)

    n_total = len(tokens)
    heldout_start = int(n_total * 0.99)
    train_tokens = tokens[:heldout_start]
    heldout_tokens = tokens[heldout_start:]

    log_rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_eval_chunks = 8
    train_iter = collect_activations(raw_model, train_tokens, hook_target,
        mode=mode, n_batches=steps, batch=batch, T=T, device=device)

    step = 0
    for acts in train_iter:
        recon, z = sae(acts)
        loss, recon_loss, sparsity = topk_sae_loss(acts, recon, z)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optim.step()
        if step > 0 and step % renorm_every == 0:
            sae.renormalize_decoder()
        if step % eval_every == 0 or step == steps - 1:
            sae.train(False)
            with torch.no_grad():
                heldout_iter = collect_activations(raw_model, heldout_tokens,
                    hook_target, mode=mode, n_batches=n_eval_chunks,
                    batch=batch, T=T, device=device)
                he_recon_losses = []
                he_baseline_losses = []
                he_l0s = []
                for he_acts in heldout_iter:
                    he_recon, he_z = sae(he_acts)
                    he_recon_losses.append(float((he_acts - he_recon).pow(2).mean()))
                    he_baseline_losses.append(float(he_acts.pow(2).mean()))
                    he_l0s.append(float((he_z != 0).sum(-1).float().mean()))
            sae.train(True)
            avg_recon = float(np.mean(he_recon_losses))
            avg_baseline = float(np.mean(he_baseline_losses))
            delta_loss = avg_recon / max(avg_baseline, 1e-9)
            avg_l0 = float(np.mean(he_l0s))
            row = {
                "step": step, "train_loss": float(loss.item()),
                "train_recon": float(recon_loss.item()),
                "heldout_recon": avg_recon,
                "heldout_baseline": avg_baseline,
                "delta_loss_ratio": delta_loss,
                "heldout_L0": avg_l0,
                "wall_s": time.time() - t0,
            }
            log_rows.append(row)
            print(f"[sae] step {step:5d}/{steps}  loss={row['train_loss']:.4f}  "
                  f"he_recon={avg_recon:.4f}  delta_ratio={delta_loss:.3f}  "
                  f"L0={avg_l0:.1f}  wall={row['wall_s']:.0f}s", flush=True)
            if wb is not None:
                try:
                    wb.log({k: v for k, v in row.items() if k != "step"}, step=step)
                except Exception:
                    pass
        step += 1

    sae.renormalize_decoder()
    sae.train(False)
    torch.save({
        "state_dict": sae.state_dict(),
        "config": {
            "d_in": d_in, "d_features": d_features, "k": k,
            "hookpoint": hookpoint, "ckpt": str(ckpt),
            "steps": steps, "batch": batch, "T": T, "lr": lr,
        },
    }, out_dir / "sae.pt")
    log_path = out_dir / "train_log.jsonl"
    with open(log_path, "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")

    if log_rows:
        last = log_rows[-1]
        print(f"\n[sae] DONE delta_ratio={last['delta_loss_ratio']:.3f}  "
              f"L0={last['heldout_L0']:.1f}  wall={last['wall_s']:.0f}s", flush=True)
        if wb is not None:
            try:
                wb.summary["final_delta_loss_ratio"] = last["delta_loss_ratio"]
                wb.summary["final_L0"] = last["heldout_L0"]
                wb.summary["wall_s"] = last["wall_s"]
                wb.finish()
            except Exception:
                pass
    print(f"[sae] artefact: {out_dir / 'sae.pt'}", flush=True)

    # Push to HF immediately (so partial-run progress is durable)
    if os.environ.get("HF_PUSH_REPO") and os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
            label = hookpoint.replace(".", "_")
            for fn in ("sae.pt", "train_log.jsonl"):
                src = out_dir / fn
                if src.exists():
                    api.upload_file(path_or_fileobj=str(src),
                                    path_in_repo=f"interp/saes/{label}/{fn}",
                                    repo_id=os.environ["HF_PUSH_REPO"],
                                    repo_type="model")
                    print(f"[sae] pushed {label}/{fn} to HF", flush=True)
        except Exception as e:
            print(f"[sae] HF push warn: {e!s:.200}", flush=True)


if __name__ == "__main__":
    main()
