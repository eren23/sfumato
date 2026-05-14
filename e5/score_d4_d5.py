"""Phase D probes #4 (multi-OOD) and #5 (ECE).

D4: AR-mode perplexity on multiple OOD distributions:
  - WikiText-2 raw (general English prose) — already scored in Phase C
  - The Pile arxiv split (math/technical text, closest to GSM8K domain)
  - OpenWebText held-out (broad web text)

If composite is uniformly worse → generic OOD penalty.
If composite ≤ ar_only on math/tech but worse on prose → structural transfer.

D5: proper Expected Calibration Error.
  - Bin AR-mode predictions by top-1 confidence (10 bins, 0.0–1.0)
  - For each bin compute mean confidence vs mean accuracy
  - ECE = sum over bins of (bin_weight * |confidence - accuracy|)
  - Lower ECE = better calibrated

Runs locally on MPS. Scores composite + ar_only + paired/ar_only at every scale.
Output: e5/results/d4_d5_raw.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_composite(path: Path, device: str):
    from e5.model_composite import CompositeConfig, CompositeLM
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    m = CompositeLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "composite"


def _load_arflow(path: Path, device: str):
    from e5.model_arflow import ARFlowConfig, ARFlowLM
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ARFlowConfig(**ck["config"])
    m = ARFlowLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "arflow"


def load_checkpoint(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vsize = ck["config"].get("vocab_size", 50258)
    if vsize == 50258:
        return _load_composite(path, device)
    return _load_arflow(path, device)


def get_ood_chunks_multi(n_chunks_per_split: int = 50, chunk_len: int = 200):
    """Three OOD distributions: wikitext-2 raw, the_pile arxiv, openwebtext."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = {}

    # WikiText-2 raw (test split)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 300]
        chunks = []
        for t in texts[:n_chunks_per_split]:
            ids = tok.encode(t, add_special_tokens=False)[:chunk_len]
            if len(ids) >= 32:
                chunks.append(ids)
        out["wikitext2"] = chunks
    except Exception as e:
        print(f"  wikitext2 load failed: {e}")
        out["wikitext2"] = []

    # The Pile arxiv (closer to GSM8K's math/technical domain)
    try:
        from datasets import load_dataset
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        chunks = []
        seen = 0
        for ex in ds:
            if "meta" in ex and ex["meta"].get("pile_set_name") == "ArXiv":
                txt = ex["text"]
                ids = tok.encode(txt, add_special_tokens=False)[:chunk_len]
                if len(ids) >= 32:
                    chunks.append(ids)
                seen += 1
            if len(chunks) >= n_chunks_per_split:
                break
            if seen > 5000:
                break
        out["pile_arxiv"] = chunks
    except Exception as e:
        print(f"  pile_arxiv load failed: {e}")
        out["pile_arxiv"] = []

    # OpenWebText held-out (broad web text; use eos-separated docs)
    try:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        chunks = []
        for i, ex in enumerate(ds):
            if i < 10000:  # skip the head to get "held-out" content
                continue
            ids = tok.encode(ex["text"], add_special_tokens=False)[:chunk_len]
            if len(ids) >= 32:
                chunks.append(ids)
            if len(chunks) >= n_chunks_per_split:
                break
            if i > 11000:
                break
        out["openwebtext_holdout"] = chunks
    except Exception as e:
        print(f"  openwebtext load failed: {e}")
        out["openwebtext_holdout"] = []

    for k, v in out.items():
        print(f"  {k}: {len(v)} chunks")
    return out


def get_gsm8k_holdout(n_problems: int = 100, indices_offset: int = 7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(indices_offset, min(indices_offset + n_problems, len(ds))):
        row = ds[idx]
        prompt = tok.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
        answer = tok.encode(" " + row["answer"], add_special_tokens=False)
        out.append((prompt, answer))
    return out


@torch.no_grad()
def score_ood_chunks(model, chunks, device, max_len=256):
    if not chunks:
        return {"avg_nll": None, "perplexity": None, "n_tokens": 0, "n_chunks": 0}
    total_nll = 0.0
    total_count = 0
    for ids in chunks:
        if len(ids) > max_len:
            ids = ids[:max_len]
        if len(ids) < 2:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        pred = logits[0, :-1].float()
        target = idx[0, 1:]
        log_p = torch.log_softmax(pred, dim=-1)
        nlls = -log_p.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    avg = total_nll / max(1, total_count)
    return {
        "avg_nll": round(avg, 4),
        "perplexity": round(math.exp(avg), 2),
        "n_tokens": total_count,
        "n_chunks": len(chunks),
    }


@torch.no_grad()
def score_ece(model, problems, device, max_len=256, n_bins=10):
    """ECE on AR-mode predictions on GSM8K answer-region tokens.
    bin top-1 confidence into n_bins (uniform width), compute |conf - acc| weighted by bin size.
    """
    confs = []
    correct = []
    for prompt, answer in problems:
        if len(prompt) + len(answer) + 1 > max_len:
            continue
        full = prompt + answer
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt)
        pred_logits = logits[0, ans_start - 1 : ans_start - 1 + len(answer), :].float()
        probs = F.softmax(pred_logits, dim=-1)
        top1_p, top1_id = probs.max(dim=-1)
        target = torch.tensor(answer, dtype=torch.long, device=device)
        is_correct = (top1_id == target)
        confs.extend(top1_p.cpu().numpy().tolist())
        correct.extend(is_correct.cpu().numpy().astype(float).tolist())

    confs = np.array(confs)
    correct = np.array(correct)
    if len(confs) == 0:
        return {"ece": None, "n_tokens": 0}

    # ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            bins.append({"bin": [round(lo, 2), round(hi, 2)], "n": 0, "conf": None, "acc": None})
            continue
        bin_conf = confs[mask].mean()
        bin_acc = correct[mask].mean()
        bin_weight = mask.sum() / len(confs)
        ece += bin_weight * abs(bin_conf - bin_acc)
        bins.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "n": int(mask.sum()),
            "conf": round(float(bin_conf), 4),
            "acc": round(float(bin_acc), 4),
        })

    return {
        "ece": round(float(ece), 4),
        "mean_conf": round(float(confs.mean()), 4),
        "mean_acc": round(float(correct.mean()), 4),
        "n_tokens": int(len(confs)),
        "bins": bins,
    }


def score_one(ckpt, gsm8k_problems, ood_splits, device):
    try:
        model, cfg, kind = load_checkpoint(ckpt, device)
    except Exception as e:
        return {"error": f"load_failed: {e!s:.150}"}
    out = {"kind": kind, "n_params": sum(p.numel() for p in model.parameters())}
    try:
        out["ood"] = {name: score_ood_chunks(model, chunks, device, max_len=cfg.block_size)
                      for name, chunks in ood_splits.items()}
    except Exception as e:
        out["ood"] = {"error": str(e)[:150]}
    try:
        out["ece"] = score_ece(model, gsm8k_problems, device, max_len=cfg.block_size)
    except Exception as e:
        out["ece"] = {"error": str(e)[:150]}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device={device}")

    n_gsm = int(os.environ.get("N_GSM", "100"))
    n_ood = int(os.environ.get("N_OOD", "50"))

    print(f"loading GSM8K held-out ({n_gsm} problems)...")
    gsm = get_gsm8k_holdout(n_problems=n_gsm)
    print(f"  loaded {len(gsm)} problems")

    print(f"loading OOD splits ({n_ood} chunks each)...")
    ood = get_ood_chunks_multi(n_chunks_per_split=n_ood)

    results_root = REPO_ROOT / "e5" / "results"
    out = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        for ckpt in sorted(d.rglob("model.pt")):
            rel = str(ckpt.relative_to(results_root))
            print(f"\n--- {rel} ---", flush=True)
            res = score_one(ckpt, gsm, ood, device)
            out[rel] = res
            if "error" in res:
                print(f"  ERR {res['error']}")
            else:
                ood_data = res.get("ood", {})
                if "error" not in ood_data:
                    parts = []
                    for k, v in ood_data.items():
                        if v.get("avg_nll") is not None:
                            parts.append(f"{k}={v['avg_nll']:.3f}")
                    print(f"  OOD: {' '.join(parts)}")
                ece = res.get("ece", {})
                if "error" not in ece and ece.get("ece") is not None:
                    print(f"  ECE={ece['ece']:.4f} conf={ece['mean_conf']:.3f} acc={ece['mean_acc']:.3f}")

    out_path = results_root / "d4_d5_raw.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
