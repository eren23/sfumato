"""Direct reward-model verifier: use Qwen2.5-Math-RM-72B as-is (no training).

Math-RM-72B is a purpose-built math reward model — outputs a scalar reward per
(problem, solution) pair. We use it AS-IS (no MLP head, no fine-tune) to
score each of the 5 cmaj branches per problem, then pick the argmax-reward
branch as the verifier prediction.

If THIS doesn't beat cmaj, no off-the-shelf supervised verifier will.

Usage:
    python phase2/spikes/verifier-aggregation/run_direct_rm.py \
        --model-id Qwen/Qwen2.5-Math-RM-72B --load-in-4bit
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "phase2/spikes/verifier-aggregation"))
from load_branches import load_phase1, load_substrate  # noqa: E402

OUT_PATH = REPO_ROOT / "phase2/spikes/verifier-aggregation/direct_rm_results.json"


@torch.no_grad()
def score_branches(rows, model_id, max_len=2048, batch_size=2, load_in_4bit=True):
    """Score each (problem, branch) with the reward model. Returns scores array."""
    from transformers import AutoModel, AutoTokenizer
    print(f"[rm] loading {model_id} (4bit={load_in_4bit})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # AutoModel (not AutoModelForCausalLM) — RM models use custom auto_map registrations
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModel.from_pretrained(model_id, quantization_config=bnb,
                                           device_map="auto", trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16,
                                           device_map="cuda", trust_remote_code=True)
    model.train(False)
    print(f"[rm] model loaded; scoring {len(rows)} branches", flush=True)

    scores = np.zeros(len(rows), dtype=np.float32)
    n_batches = math.ceil(len(rows) / batch_size)
    t0 = time.time()
    for b_idx in range(n_batches):
        batch = rows[b_idx*batch_size:(b_idx+1)*batch_size]
        # Qwen2.5-Math-RM expects a chat-style format with the problem as user msg
        # and the candidate solution as assistant msg.
        chats = []
        for r in batch:
            messages = [
                {"role": "user", "content": f"Solve this math problem:\n\n{r.problem or '(problem unavailable)'}"},
                {"role": "assistant", "content": r.branch_text},
            ]
            chats.append(messages)
        try:
            input_ids = tok.apply_chat_template(chats, tokenize=True, return_tensors="pt",
                                                 padding=True, truncation=True, max_length=max_len).to("cuda")
        except TypeError:
            # Fallback: tokenize each separately if batch chat-template fails
            texts = [tok.apply_chat_template(c, tokenize=False) for c in chats]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
            input_ids = enc.input_ids
        out = model(input_ids=input_ids)
        # Qwen2.5-Math-RM-72B: output is `out.logits` of shape [B, T, 1] — scalar
        # reward per token. Take the LAST non-pad position's reward as the
        # candidate score (matches the model card example).
        if hasattr(out, "logits"):
            # logits could be [B, T, 1] (RM) or [B, T, V] (LM); shape disambiguates
            if out.logits.shape[-1] == 1:
                # RM: take last token's reward scalar
                batch_scores = out.logits[:, -1, 0].float().cpu().numpy()
            else:
                # LM fallback: mean of last-token logits
                batch_scores = out.logits[:, -1, :].mean(dim=-1).float().cpu().numpy()
        elif hasattr(out, "rewards"):
            batch_scores = out.rewards.squeeze(-1).float().cpu().numpy()
        else:
            batch_scores = out[0].squeeze(-1).float().cpu().numpy()
        for j, _ in enumerate(batch):
            scores[b_idx*batch_size + j] = float(batch_scores[j])
        if b_idx % 20 == 0:
            done = (b_idx+1) * batch_size
            elapsed = time.time() - t0
            print(f"[rm] {min(done, len(rows))}/{len(rows)} ({elapsed:.0f}s)", flush=True)
    print(f"[rm] scoring done in {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return scores


def evaluate(rows, scores):
    """Compute cmaj baseline + RM-rerank verifier accuracy + oracle ceiling."""
    by_problem = defaultdict(list)
    for r, s in zip(rows, scores):
        by_problem[r.problem_id].append((r, float(s)))
    cmaj_correct = ver_correct = oracle_correct = recovers = n = 0
    by_problem_results = []
    for pid, items in by_problem.items():
        n += 1
        gold = items[0][0].gold
        votes = [r.extracted for r, _ in items]
        cmaj_pick = Counter(votes).most_common(1)[0][0]
        cmaj_hit = (cmaj_pick == gold)
        if cmaj_hit: cmaj_correct += 1
        if any(v == gold for v in votes): oracle_correct += 1
        ver_pick = max(items, key=lambda x: x[1])[0].extracted
        if ver_pick == gold: ver_correct += 1
        if not cmaj_hit and ver_pick == gold: recovers += 1
        by_problem_results.append({
            "problem_id": pid, "gold": gold, "cmaj_pick": cmaj_pick,
            "verifier_pick": ver_pick, "cmaj_hit": cmaj_hit, "verifier_hit": ver_pick == gold,
            "scores": [float(s) for _, s in items],
            "votes": votes,
        })
    return {
        "n_problems": n,
        "cmaj_acc": cmaj_correct/n, "verifier_acc": ver_correct/n, "oracle_acc": oracle_correct/n,
        "verifier_recovers": recovers,
        "delta_pp": (ver_correct - cmaj_correct)/n*100,
        "gap_closure": (ver_correct - cmaj_correct) / max(1e-9, oracle_correct - cmaj_correct),
        "by_problem": by_problem_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Math-RM-72B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit quant")
    args = ap.parse_args()
    if args.no_4bit: args.load_in_4bit = False

    rows = load_phase1() + load_substrate()
    print(f"[rm] {len(rows)} branches across {len({r.problem_id for r in rows})} problems", flush=True)

    scores = score_branches(rows, model_id=args.model_id, max_len=args.max_len,
                             batch_size=args.batch_size, load_in_4bit=args.load_in_4bit)
    res = evaluate(rows, scores)
    print(f"\n[rm] === MEAN: cmaj={res['cmaj_acc']:.1%} verifier={res['verifier_acc']:.1%} oracle={res['oracle_acc']:.1%} ===")
    print(f"[rm] dpp_vs_cmaj = {res['delta_pp']:+.2f}")
    print(f"[rm] gap-closure = {res['gap_closure']:.1%}")
    print(f"[rm] recovers = {res['verifier_recovers']} (cmaj-failed problems where RM picked correct branch)")

    # Pre-reg decision (same thresholds as option-2)
    v = res["verifier_acc"]
    if v >= 0.89: dec = "WIN-DECISIVE"
    elif v >= 0.87: dec = "WIN-STRONG"
    elif v >= 0.83: dec = "WIN-MINOR"
    elif v >= res["cmaj_acc"] - 0.01: dec = "INCONCLUSIVE"
    else: dec = "LOSS"
    print(f"\n[rm] >>> DIRECT-RM DECISION: {dec} (verifier={v:.1%}, cmaj={res['cmaj_acc']:.1%}) <<<")

    OUT_PATH.write_text(json.dumps({
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "n_branches": len(rows),
        **{k: v for k, v in res.items() if k != "by_problem"},
        "decision": dec,
        "by_problem": res["by_problem"][:30],  # cap to avoid huge file
    }, indent=2))
    print(f"[rm] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
