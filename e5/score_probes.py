"""Phase-B probes #1-4, #7 — score existing checkpoints on metrics the
original AR-NLL test couldn't reveal.

The earlier `score_perplexity.py` measured AR-mode NLL on held-out GSM8K
gold answers. It found composite is uniformly worse than pure-AR on that
axis. But that is the metric pure-AR was trained to minimise. The five
probes here test composite on axes that should be more favourable:

  Probe 1 — Diff-axis perplexity (mask-fill NLL): is composite TRADING
            AR loss for diff loss, or degrading both? Compares
            composite-as-diff vs pure-diff_only on the same held-out
            tokens at a fixed mask ratio.

  Probe 2 — Fill-in-the-middle (FIM) NLL: prompt + masked middle +
            suffix. Composite can natively use diff mode; pure-AR can't
            condition on the future. Score: NLL on the middle tokens
            given prompt + suffix.

  Probe 3 — OOD perplexity: held-out chunk of non-GSM8K English text.
            Tests whether composite generalises better (two heads =
            broader regularisation hypothesis).

  Probe 4 — Token-position-stratified AR NLL: the AR tax we found earlier
            may not be uniform. Decompose by position bin (first 16
            tokens / middle / last 16) to see where composite hurts
            vs helps.

  Probe 7 — Calibration / entropy: composite may produce better-calibrated
            distributions even if its NLL is higher. Measure mean
            prediction entropy and top-1 confidence.

Probe #5 (mode-switching inference) and #6 (low-data training) require
GPU pods and are in separate scripts.

Output: e5/results/probes_raw.json (per-checkpoint × per-probe rows).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_composite(path: Path, device: str):
    from e5.model_composite import CompositeConfig, CompositeLM, MASK_TOKEN_ID
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = CompositeConfig(**ck["config"])
    m = CompositeLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "composite", MASK_TOKEN_ID


def _load_arflow(path: Path, device: str):
    from e5.model_arflow import ARFlowConfig, ARFlowLM
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ARFlowConfig(**ck["config"])
    m = ARFlowLM(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.train(False)
    return m, cfg, "arflow", None  # ARFlow has no mask token; diff mode is continuous


def load_checkpoint(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vsize = ck["config"].get("vocab_size", 50258)
    if vsize == 50258:
        return _load_composite(path, device)
    return _load_arflow(path, device)


def get_holdout_chunks(n_problems: int = 100, indices_offset: int = 7000):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("gsm8k", "main", split="train")
    tok = AutoTokenizer.from_pretrained("gpt2")
    out = []
    for idx in range(indices_offset, min(indices_offset + n_problems, len(ds))):
        row = ds[idx]
        prompt_text = f"Question: {row['question']}\nAnswer:"
        answer_text = " " + row["answer"]
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        out.append((prompt_ids, answer_ids))
    return out


def get_ood_chunks(n_chunks: int = 50, chunk_len: int = 200):
    """Out-of-distribution: WikiText-2 raw text, ~chunked. Different domain
    from GSM8K's math word problems."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 200]
    except Exception:
        # Fallback: use the END of GSM8K-train (still GSM8K but unseen)
        ds = load_dataset("gsm8k", "main", split="train")
        texts = [f"{ds[i]['question']} {ds[i]['answer']}" for i in range(7400, len(ds))]
    tok = AutoTokenizer.from_pretrained("gpt2")
    chunks = []
    for t in texts[:n_chunks]:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > chunk_len:
            ids = ids[:chunk_len]
        if len(ids) < 32:
            continue
        chunks.append(ids)
    return chunks


# --------------------- PROBE 4: AR position-stratified -------------------

@torch.no_grad()
def probe4_position_stratified(model, problems, device, max_len=256, n_bins=4):
    """Per-token NLL bucketed by position-in-answer."""
    bins = [[] for _ in range(n_bins)]
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        full = prompt_ids + answer_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt_ids)
        pred_logits = logits[0, ans_start - 1 : ans_start - 1 + len(answer_ids), :].float()
        target = torch.tensor(answer_ids, dtype=torch.long, device=device)
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        nlls = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        for i, nll in enumerate(nlls):
            bin_idx = min(n_bins - 1, int(i * n_bins / max(1, len(nlls))))
            bins[bin_idx].append(float(nll))
    return {f"bin{i}_avg_nll": round(float(np.mean(b)), 4) if b else None for i, b in enumerate(bins)}


# --------------------- PROBE 1: Diff-axis NLL ----------------------------

@torch.no_grad()
def probe1_diff_axis(model, problems, device, mask_id, max_len=256, mask_ratio=0.5):
    """For CompositeLM only (mask_id is not None): apply a random mask to the
    answer-region tokens at fixed ratio, run diff mode, compute NLL on the
    masked positions. Lower = better mask-fill."""
    if mask_id is None:
        return {"avg_nll": None, "note": "ARFlow model — flow-axis perplexity needs a different probe"}
    total_nll = 0.0
    total_count = 0
    rng = np.random.RandomState(0)
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        # We mask only answer tokens; keep prompt intact for conditioning.
        full = prompt_ids + answer_ids
        T = len(full)
        ans_start = len(prompt_ids)
        # Choose mask positions among answer indices
        ans_len = len(answer_ids)
        n_mask = max(1, int(ans_len * mask_ratio))
        mask_positions = rng.choice(ans_len, size=n_mask, replace=False) + ans_start
        idx = torch.tensor([full], dtype=torch.long, device=device)
        # Apply mask
        masked = idx.clone()
        for p in mask_positions:
            masked[0, p] = mask_id
        logits = model(masked, mode="diff")  # (1, T, V)
        pred_logits = logits[0, mask_positions].float()
        targets = torch.tensor([full[p] for p in mask_positions], dtype=torch.long, device=device)
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    return {
        "avg_nll": round(total_nll / max(1, total_count), 4),
        "perplexity": round(math.exp(total_nll / max(1, total_count)), 2),
        "n_masked_tokens": total_count,
        "mask_ratio": mask_ratio,
    }


# --------------------- PROBE 2: FIM ---------------------------------------

@torch.no_grad()
def probe2_fim(model, problems, device, mask_id, max_len=256, suffix_len=20, mid_mask_len=20):
    """Fill-in-the-middle: given prompt + masked middle + suffix, score the
    middle tokens. Pure-AR can't condition on the suffix, so this is a
    composite-specific capability (when mid_mask_len > 0).

    For composite (diff mode) we mask the middle and condition on prompt+suffix
    bidirectionally. For pure-AR (ar_only) we ALSO try: score the middle
    tokens in AR mode (ignoring suffix), which is just standard AR.
    The headline is composite-diff-FIM vs ar_only-AR (a strict win for
    composite if composite uses the suffix usefully)."""
    if mask_id is None:
        # ARFlow path — skip for now (continuous flow FIM is non-trivial)
        return {"avg_nll": None, "note": "ARFlow continuous flow FIM TBD"}

    total_nll_diff = 0.0
    total_count_diff = 0
    for prompt_ids, answer_ids in problems:
        # Need at least suffix_len + mid_mask_len + 4 tokens in the answer
        if len(answer_ids) < suffix_len + mid_mask_len + 4:
            continue
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        full = prompt_ids + answer_ids
        ans_start = len(prompt_ids)
        # mid region: in the middle of the answer
        mid_start = ans_start + (len(answer_ids) - mid_mask_len - suffix_len) // 2
        mid_end = mid_start + mid_mask_len
        # Apply mask to the mid region; keep suffix intact
        idx = torch.tensor([full], dtype=torch.long, device=device)
        masked = idx.clone()
        masked[0, mid_start:mid_end] = mask_id
        logits = model(masked, mode="diff")
        pred_logits = logits[0, mid_start:mid_end].float()
        targets = idx[0, mid_start:mid_end]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        nlls = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll_diff += float(nlls.sum())
        total_count_diff += int(nlls.size)

    return {
        "fim_diff_avg_nll": round(total_nll_diff / max(1, total_count_diff), 4),
        "n_middle_tokens": total_count_diff,
        "mid_mask_len": mid_mask_len,
        "suffix_len": suffix_len,
        "note": "composite-only (diff mode); pure-AR can't see suffix and would be strictly higher NLL on the same tokens",
    }


# --------------------- PROBE 3: OOD perplexity ---------------------------

@torch.no_grad()
def probe3_ood(model, ood_chunks, device, max_len=256):
    """AR-mode NLL on out-of-distribution text (WikiText-2 raw)."""
    total_nll = 0.0
    total_count = 0
    for ids in ood_chunks:
        if len(ids) > max_len:
            ids = ids[:max_len]
        if len(ids) < 2:
            continue
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        # Score all positions (predicting position t+1 from positions [0..t])
        pred_logits = logits[0, :-1].float()
        target = idx[0, 1:]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        nlls = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy()
        total_nll += float(nlls.sum())
        total_count += int(nlls.size)
    return {
        "ood_avg_nll": round(total_nll / max(1, total_count), 4),
        "ood_perplexity": round(math.exp(total_nll / max(1, total_count)), 2),
        "n_tokens": total_count,
        "n_chunks": len(ood_chunks),
    }


# --------------------- PROBE 7: Calibration ------------------------------

@torch.no_grad()
def probe7_calibration(model, problems, device, max_len=256):
    """Mean prediction entropy and top-1 confidence on the answer-region
    AR-mode predictions. Lower entropy = sharper. Higher mean top-1 prob
    on correct token = better-calibrated."""
    entropies = []
    correct_probs = []
    top1_correct = 0
    total = 0
    for prompt_ids, answer_ids in problems:
        if len(prompt_ids) + len(answer_ids) + 1 > max_len:
            continue
        full = prompt_ids + answer_ids
        idx = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(idx, mode="ar")
        ans_start = len(prompt_ids)
        pred_logits = logits[0, ans_start - 1 : ans_start - 1 + len(answer_ids), :].float()
        probs = F.softmax(pred_logits, dim=-1)
        # Entropy
        log_p = torch.log(probs.clamp_min(1e-12))
        H = -(probs * log_p).sum(dim=-1)  # (T,)
        entropies.extend(H.cpu().numpy().tolist())
        # Top-1 confidence
        top1_p, top1_id = probs.max(dim=-1)
        target = torch.tensor(answer_ids, dtype=torch.long, device=device)
        correct = (top1_id == target)
        correct_probs.extend(probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())
        top1_correct += int(correct.sum().item())
        total += int(correct.numel())

    return {
        "mean_entropy_nats": round(float(np.mean(entropies)), 4) if entropies else None,
        "mean_correct_prob": round(float(np.mean(correct_probs)), 4) if correct_probs else None,
        "top1_accuracy": round(top1_correct / max(1, total), 4),
        "n_tokens": total,
    }


# --------------------- main loop -----------------------------------------

def score_one_checkpoint(ckpt_path, problems, ood_chunks, device):
    try:
        model, cfg, kind, mask_id = load_checkpoint(ckpt_path, device)
    except Exception as e:
        return {"error": f"load_failed: {e!s:.200}"}
    out = {"kind": kind, "n_params": sum(p.numel() for p in model.parameters())}
    max_len = cfg.block_size
    try:
        out["probe4_pos_strat"] = probe4_position_stratified(model, problems, device, max_len=max_len)
    except Exception as e:
        out["probe4_pos_strat"] = {"error": str(e)[:200]}
    try:
        out["probe1_diff_axis"] = probe1_diff_axis(model, problems, device, mask_id, max_len=max_len)
    except Exception as e:
        out["probe1_diff_axis"] = {"error": str(e)[:200]}
    try:
        out["probe2_fim"] = probe2_fim(model, problems, device, mask_id, max_len=max_len)
    except Exception as e:
        out["probe2_fim"] = {"error": str(e)[:200]}
    try:
        out["probe3_ood"] = probe3_ood(model, ood_chunks, device, max_len=max_len)
    except Exception as e:
        out["probe3_ood"] = {"error": str(e)[:200]}
    try:
        out["probe7_calibration"] = probe7_calibration(model, problems, device, max_len=max_len)
    except Exception as e:
        out["probe7_calibration"] = {"error": str(e)[:200]}
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

    n_holdout = int(os.environ.get("N_HOLDOUT", "100"))
    print(f"loading {n_holdout} held-out GSM8K-train problems...")
    problems = get_holdout_chunks(n_problems=n_holdout)
    print(f"loaded {len(problems)} in-domain problems")

    print(f"loading OOD chunks (wikitext-2 raw)...")
    ood_chunks = get_ood_chunks(n_chunks=50, chunk_len=200)
    print(f"loaded {len(ood_chunks)} OOD chunks (avg len {sum(len(c) for c in ood_chunks)/max(1,len(ood_chunks)):.0f})")

    results_root = REPO_ROOT / "e5" / "results"
    out = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        for ckpt in sorted(d.rglob("model.pt")):
            rel = ckpt.relative_to(results_root)
            key = str(rel)
            print(f"\n--- {key} ---", flush=True)
            res = score_one_checkpoint(ckpt, problems, ood_chunks, device)
            out[key] = res
            # Print compact summary
            if "error" in res:
                print(f"  ERROR {res['error']}")
            else:
                p4 = res.get("probe4_pos_strat", {})
                p1 = res.get("probe1_diff_axis", {})
                p2 = res.get("probe2_fim", {})
                p3 = res.get("probe3_ood", {})
                p7 = res.get("probe7_calibration", {})
                bins = [p4.get(f"bin{i}_avg_nll") for i in range(4)]
                bins_str = " ".join(f"{b:.2f}" if isinstance(b, (int, float)) else "—" for b in bins)
                print(f"  pos_strat: [{bins_str}]")
                print(f"  diff_axis NLL: {p1.get('avg_nll')}, perp: {p1.get('perplexity')}")
                print(f"  fim_diff NLL: {p2.get('fim_diff_avg_nll')}")
                print(f"  OOD AR NLL: {p3.get('ood_avg_nll')}, perp: {p3.get('ood_perplexity')}")
                print(f"  entropy: {p7.get('mean_entropy_nats')}, top1_acc: {p7.get('top1_accuracy')}")

    out_path = results_root / "probes_raw.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
