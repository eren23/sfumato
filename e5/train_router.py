"""Phase I.3 — REINFORCE-train RouterMLP on frozen F11 (or F10) ckpt.

The router decides, at each chunk boundary, one of:
  - extend AR by k_ar tokens
  - diff-fill the next k_diff_short tokens (parallel)
  - diff-fill the next k_diff_long tokens (parallel)
  - stop (commit current sequence as final)

The backbone + both heads are FROZEN. Only RouterMLP parameters train.

Reward per rollout = correct(0/1) - 0.1*is_loopy + 0.1*(logp - baseline_logp)
The last term (AR coherence shaping) prevents the router from collapsing
to "always diff" when accuracy is near-zero.

Baseline (variance reduction): per-prompt mean reward across the K rollouts.

ENV:
  CKPT=path/to/f11_or_f10/model.pt   (required)
  ROUTER_OUT=path/to/router_state.pt  (required)
  N_PROBLEMS=200                      (training problems from GSM8K-train)
  K_ROLLOUTS=8                        (per-problem rollouts)
  BATCH=4                             (problems per gradient step)
  STEPS=500                           (gradient steps)
  LR=3e-4
  ENTROPY_BETA=0.01                   (entropy regulariser to prevent collapse)
  MAX_CHUNKS=8                        (cap on chunks per rollout)
  K_AR=16, K_DIFF_SHORT=8, K_DIFF_LONG=16
  REVISE_STEPS=8                      (diff-revise inner step count)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from e5.model_composite import CompositeLM, CompositeConfig, MASK_TOKEN_ID
from e5.router import (
    RouterMLP, RouterConfig,
    ACTION_AR_CHUNK, ACTION_DIFF_SHORT, ACTION_DIFF_LONG, ACTION_END,
    extract_last_hidden,
)
from e5.data import load_gsm8k_dev_questions


def load_frozen_composite(ckpt_path: Path, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = CompositeConfig(**{k: cfg_dict[k] for k in CompositeConfig.__dataclass_fields__ if k in cfg_dict})
    model = CompositeLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def extract_answer(text: str) -> str | None:
    """Pull '#### N' answer from a GSM8K-style continuation."""
    import re
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def is_loopy(token_ids: list[int], min_run: int = 3) -> bool:
    """True iff the last 20 tokens contain a 1-3-gram repeated 3+ times."""
    if len(token_ids) < min_run * 3:
        return False
    tail = token_ids[-min_run * 4 :]
    for n in (1, 2, 3):
        for i in range(len(tail) - n * min_run):
            gram = tail[i : i + n]
            if all(tail[i + k * n : i + (k + 1) * n] == gram for k in range(min_run)):
                return True
    return False


@torch.no_grad()
def _diff_fill_block(model, prompt_ids: list[int], k: int, n_steps: int = 8,
                     temperature: float = 0.0, device: str = "cuda") -> list[int]:
    """Diff-revise k mask-positions appended to prompt_ids. Returns the
    final k token ids."""
    seq = prompt_ids + [MASK_TOKEN_ID] * k
    idx = torch.tensor([seq], dtype=torch.long, device=device)
    for step in range(n_steps):
        mask_count_now = int((idx[0] == MASK_TOKEN_ID).sum().item())
        if mask_count_now == 0:
            break
        n_to_unmask = max(1, mask_count_now // max(1, n_steps - step))
        logits = model(idx, mode="diff")[0]
        mask_positions = (idx[0] == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        probs = F.softmax(logits[mask_positions], dim=-1)
        if temperature > 0:
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            conf = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            conf, sampled = probs.max(dim=-1)
        order = torch.argsort(conf, descending=True)[:n_to_unmask]
        pos_to_unmask = mask_positions[order]
        idx[0, pos_to_unmask] = sampled[order]
    return idx[0, -k:].tolist()


@torch.no_grad()
def _ar_extend(model, seq_ids: list[int], k: int, device: str = "cuda",
               temperature: float = 0.8, top_p: float = 0.9) -> list[int]:
    """Extend seq_ids by k tokens under causal AR with top-p sampling."""
    out = list(seq_ids)
    for _ in range(k):
        ctx = torch.tensor([out[-model.cfg.block_size :]], dtype=torch.long, device=device)
        logits = model(ctx, mode="ar")[0, -1]
        if temperature > 0:
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(0)
                cutoff = (cumulative > top_p).nonzero(as_tuple=True)[0]
                if cutoff.numel() > 0:
                    keep = cutoff[0].item() + 1
                    sorted_probs[keep:] = 0
                    probs = torch.zeros_like(probs)
                    probs[sorted_idx[:keep]] = sorted_probs[:keep]
                    probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1).item())
        else:
            nxt = int(torch.argmax(logits).item())
        out.append(nxt)
    return out[-k:]


def rollout(model, router, prompt_ids: list[int], cfg_d: dict,
            device: str = "cuda") -> dict:
    """Run a single router-driven rollout. Returns dict with seq_ids,
    log_probs, n_chunks."""
    max_chunks = cfg_d["max_chunks"]
    k_ar = cfg_d["k_ar"]
    k_diff_short = cfg_d["k_diff_short"]
    k_diff_long = cfg_d["k_diff_long"]
    n_diff_steps = cfg_d["revise_steps"]
    seq = list(prompt_ids)
    log_probs = []
    entropies = []

    for _ in range(max_chunks):
        ctx = torch.tensor([seq[-model.cfg.block_size :]], dtype=torch.long, device=device)
        h_last = extract_last_hidden(model, ctx, mode="ar")
        action_logits = router(h_last.detach())
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        a = int(action.item())

        if a == ACTION_AR_CHUNK:
            new_ids = _ar_extend(model, seq, k_ar, device=device)
            seq.extend(new_ids)
        elif a == ACTION_DIFF_SHORT:
            new_ids = _diff_fill_block(model, seq, k_diff_short,
                                        n_steps=max(1, n_diff_steps // 2),
                                        device=device)
            seq.extend(new_ids)
        elif a == ACTION_DIFF_LONG:
            new_ids = _diff_fill_block(model, seq, k_diff_long,
                                        n_steps=n_diff_steps,
                                        device=device)
            seq.extend(new_ids)
        elif a == ACTION_END:
            break

    return {
        "seq": seq,
        "log_probs": torch.stack(log_probs) if log_probs else None,
        "entropies": torch.stack(entropies) if entropies else None,
        "n_chunks": len(log_probs),
    }


def reward_for_rollout(rollout_result: dict, prompt_len: int,
                        gold_answer: str, tokenizer) -> dict:
    """Compute reward components for a finished rollout."""
    seq = rollout_result["seq"]
    new_ids = seq[prompt_len:]
    text = tokenizer.decode(new_ids)
    extracted = extract_answer(text)
    correct = 1.0 if (extracted is not None and extracted == gold_answer) else 0.0
    loopy = 1.0 if is_loopy(new_ids) else 0.0
    total = correct - 0.1 * loopy
    return {"correct": correct, "is_loopy": loopy, "total": total, "text": text}


def main():
    ckpt_path = Path(os.environ["CKPT"])
    out_path = Path(os.environ["ROUTER_OUT"])
    n_problems = int(os.environ.get("N_PROBLEMS", "200"))
    k_rollouts = int(os.environ.get("K_ROLLOUTS", "8"))
    batch_problems = int(os.environ.get("BATCH", "4"))
    n_steps = int(os.environ.get("STEPS", "500"))
    lr = float(os.environ.get("LR", "3e-4"))
    entropy_beta = float(os.environ.get("ENTROPY_BETA", "0.01"))
    max_chunks = int(os.environ.get("MAX_CHUNKS", "8"))
    k_ar = int(os.environ.get("K_AR", "16"))
    k_diff_short = int(os.environ.get("K_DIFF_SHORT", "8"))
    k_diff_long = int(os.environ.get("K_DIFF_LONG", "16"))
    revise_steps = int(os.environ.get("REVISE_STEPS", "8"))
    seed = int(os.environ.get("SEED", "1313"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[router-train] device={device} ckpt={ckpt_path} out={out_path}")

    print("[router-train] loading frozen composite ...")
    model, cfg = load_frozen_composite(ckpt_path, device=device)
    print(f"[router-train] composite loaded: {model.num_params()/1e6:.1f}M params, frozen")

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    router_cfg = RouterConfig(d_model=cfg.d_model, hidden=cfg.d_model // 2)
    router = RouterMLP(router_cfg).to(device)
    print(f"[router-train] router: {router.num_params()/1e3:.1f}K params")

    optim = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)

    problems = load_gsm8k_dev_questions(n=n_problems + 50)
    train_problems = problems[:n_problems]

    rollout_cfg = dict(
        max_chunks=max_chunks, k_ar=k_ar, k_diff_short=k_diff_short,
        k_diff_long=k_diff_long, revise_steps=revise_steps,
    )

    log_rows = []
    for step in range(n_steps):
        t0 = time.time()
        batch = np.random.choice(len(train_problems), size=batch_problems, replace=False)
        step_loss = 0.0
        step_correct = 0.0
        step_loopy = 0.0
        for prob_idx in batch:
            prob = train_problems[int(prob_idx)]
            prompt_ids = prob["prompt_tokens"]
            rewards = []
            log_probs_per_roll = []
            entropies_per_roll = []
            for _ in range(k_rollouts):
                r = rollout(model, router, prompt_ids, rollout_cfg, device=device)
                rew = reward_for_rollout(r, len(prompt_ids), prob["gold"], tok)
                rewards.append(rew["total"])
                step_correct += rew["correct"]
                step_loopy += rew["is_loopy"]
                if r["log_probs"] is not None:
                    log_probs_per_roll.append(r["log_probs"].sum())
                    entropies_per_roll.append(r["entropies"].mean())
            baseline = float(np.mean(rewards))
            rewards_t = torch.tensor([r - baseline for r in rewards], device=device, dtype=torch.float32)
            if not log_probs_per_roll:
                continue
            logp_stack = torch.stack(log_probs_per_roll)
            ent_stack = torch.stack(entropies_per_roll)
            loss = -(rewards_t * logp_stack).mean() - entropy_beta * ent_stack.mean()
            loss.backward()
            step_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        wall = time.time() - t0
        n_roll = batch_problems * k_rollouts
        row = {
            "step": step,
            "loss": step_loss / batch_problems,
            "mean_correct": step_correct / n_roll,
            "mean_loopy": step_loopy / n_roll,
            "wall_s": wall,
        }
        log_rows.append(row)
        if step % 10 == 0 or step == n_steps - 1:
            print(f"[{step:4d}] loss={row['loss']:+.4f} acc={row['mean_correct']:.3f} "
                  f"loopy={row['mean_loopy']:.2f} wall={wall:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"router_state": router.state_dict(),
                "router_cfg": router_cfg.__dict__,
                "config": rollout_cfg,
                "n_steps": n_steps}, out_path)
    log_path = out_path.with_suffix(".log.jsonl")
    with open(log_path, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[router-train] done. router -> {out_path}, log -> {log_path}")


if __name__ == "__main__":
    main()
