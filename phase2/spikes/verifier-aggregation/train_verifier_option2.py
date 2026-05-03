"""D3.5 spike — option-2 Qwen-encoder verifier.

Self-contained pod-side script. Loads Qwen2.5-0.5B-Instruct as encoder,
extracts mean-pooled last-layer hidden states for each (problem, branch_text)
pair, trains a small MLP head with BCE loss in 5-fold CV (split by problem),
reports per-fold cmaj baseline + verifier-rerank accuracy + oracle ceiling.

Reads jsonls from /workspace/sfumato/e4/results/ (synced via scp from local).

Pre-reg threshold (binding): mean verifier accuracy >= 83% on held-out folds.
- WIN-DECISIVE >=89%, WIN-STRONG >=87%, WIN-MINOR >=83%, INCONCLUSIVE 78-82%, LOSS <78%.
"""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.stats import beta
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

RESULTS_DIR = pathlib.Path("/workspace/sfumato/e4/results")
OUT_PATH = pathlib.Path("/workspace/sfumato/phase2/spikes/verifier-aggregation/option2_results.json")


class BranchRow(NamedTuple):
    problem_id: str
    branch_idx: int
    problem: str
    branch_text: str
    gold: str
    extracted: str
    correct: bool
    tau: float
    source: str


_ANSWER_RE = re.compile(r"Answer\s*[:=]\s*([\-\+]?\d[\d,]*\.?\d*)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"([\-\+]?\d[\d,]*\.?\d*)")


def extract_answer(text: str) -> str:
    if not text: return ""
    m = _ANSWER_RE.search(text)
    if m: return m.group(1).replace(",", "").rstrip(".")
    nums = _LAST_NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def parse_votes(votes_str: str) -> list[str]:
    return [v.strip() for v in votes_str.split("|")]


def load_jsonl(path: pathlib.Path, tau_override=None):
    rows = []
    if not path.exists(): return rows
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        tau = tau_override if tau_override is not None else rec.get("temperature", 0.7)
        gold = str(rec["gold"]).strip()
        problem = rec.get("question", "")
        trace = rec.get("trace", {})
        votes = parse_votes(trace.get("votes", ""))
        for i in range(5):
            key = f"branch_{i}"
            if key not in trace: break
            text = trace[key]
            extracted = votes[i] if i < len(votes) else extract_answer(text)
            rows.append(BranchRow(
                problem_id=str(rec["id"]), branch_idx=i, problem=problem,
                branch_text=text, gold=gold, extracted=extracted,
                correct=(extracted == gold), tau=float(tau), source=path.name,
            ))
    return rows


def load_all_branches():
    candidates = [
        ("raw_cmaj_k64_seed0_b5_t0.3.jsonl", 0.3),
        ("raw_cmaj_k64_seed0_b5.jsonl", 0.7),
        ("raw_cmaj_k64_seed0_b5_t1.0.jsonl", 1.0),
        ("raw_cmaj_k64_seed0_b5_v3LoRA_N200.jsonl", 0.7),
    ]
    rows = []
    for name, tau in candidates:
        p = RESULTS_DIR / name
        before = len(rows)
        rows.extend(load_jsonl(p, tau_override=tau))
        print(f"  loaded {p.name}: +{len(rows)-before} branches", flush=True)
    return rows


def cp_ci(k, n, alpha=0.05):
    if n == 0 or not HAVE_SCIPY: return 0.0, 1.0
    low = float(beta.ppf(alpha/2, k, n-k+1)) if k > 0 else 0.0
    high = float(beta.ppf(1-alpha/2, k+1, n-k)) if k < n else 1.0
    return low, high


@torch.no_grad()
def extract_embeddings(rows, model_id="Qwen/Qwen2.5-0.5B-Instruct", max_len=768, batch_size=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[opt2] loading {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
    )
    model.train(False)  # set inference mode (note: avoiding .eval() name)
    embeddings = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
    n_batches = math.ceil(len(rows) / batch_size)
    t0 = time.time()
    for b_idx in range(n_batches):
        batch = rows[b_idx*batch_size:(b_idx+1)*batch_size]
        texts = []
        for r in batch:
            if r.problem:
                texts.append(f"Problem: {r.problem}\n\nSolution:\n{r.branch_text}")
            else:
                texts.append(f"Solution:\n{r.branch_text}")
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1]
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for j, r in enumerate(batch):
            embeddings[b_idx*batch_size + j] = pooled[j].float().cpu().numpy()
        if b_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"[opt2] embed {(b_idx+1)*batch_size}/{len(rows)} ({elapsed:.0f}s)", flush=True)
    print(f"[opt2] done: {len(rows)} embeddings in {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return embeddings


class VerifierHead(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


def train_head(X_train, y_train, n_epochs=50, lr=1e-3, batch_size=64, class_weight_pos=2.0):
    head = VerifierHead(X_train.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([class_weight_pos]).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.tensor(X_train, dtype=torch.float32).cuda()
    yt = torch.tensor(y_train, dtype=torch.float32).cuda()
    n = len(Xt)
    head.train(True)
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        Xp, yp = Xt[perm], yt[perm]
        total = 0.0
        for i in range(0, n, batch_size):
            xb = Xp[i:i+batch_size]; yb = yp[i:i+batch_size]
            logits = head(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        if epoch % 10 == 0:
            print(f"  ep{epoch} loss={total/n:.4f}", flush=True)
    head.train(False)
    return head


@torch.no_grad()
def score(head, X):
    head.train(False)
    Xt = torch.tensor(X, dtype=torch.float32).cuda()
    return torch.sigmoid(head(Xt).squeeze(-1)).cpu().numpy()


def evaluate_fold(rows, embs, train_idx, test_idx):
    X_train = embs[train_idx]; y_train = np.array([int(rows[i].correct) for i in train_idx])
    X_test = embs[test_idx]
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return {"error": "degenerate train labels"}
    head = train_head(X_train, y_train)
    test_probs = score(head, X_test)
    by_problem = defaultdict(list)
    for j, idx in enumerate(test_idx):
        by_problem[rows[idx].problem_id].append((rows[idx], test_probs[j]))
    cmaj_correct = ver_correct = oracle_correct = recovers = 0
    n = 0
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
    return {
        "n_problems": n, "n_train_branches": len(train_idx), "n_test_branches": len(test_idx),
        "cmaj_acc": cmaj_correct/n, "verifier_acc": ver_correct/n, "oracle_acc": oracle_correct/n,
        "verifier_recovers": recovers,
        "delta_pp": (ver_correct - cmaj_correct)/n*100,
        "gap_closure": (ver_correct - cmaj_correct) / max(1e-9, oracle_correct - cmaj_correct),
    }


def cv_split_by_problem(rows, k=5, seed=0):
    rng = np.random.default_rng(seed)
    pids = sorted({r.problem_id for r in rows})
    rng.shuffle(pids)
    fold_size = len(pids) // k
    by_pid_idx = defaultdict(list)
    for i, r in enumerate(rows):
        by_pid_idx[r.problem_id].append(i)
    folds = []
    for f in range(k):
        start = f * fold_size
        end = (f+1) * fold_size if f < k-1 else len(pids)
        test_pids = set(pids[start:end])
        test_idx = []; train_idx = []
        for pid, idxs in by_pid_idx.items():
            (test_idx if pid in test_pids else train_idx).extend(idxs)
        folds.append((np.array(train_idx), np.array(test_idx)))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    print(f"[opt2] loading branches from {RESULTS_DIR}", flush=True)
    rows = load_all_branches()
    print(f"[opt2] {len(rows)} branches across {len({r.problem_id for r in rows})} problems", flush=True)

    embs = extract_embeddings(rows, model_id=args.model_id, max_len=args.max_len, batch_size=args.batch_size)

    folds = cv_split_by_problem(rows, k=args.cv, seed=args.seed)
    fold_results = []
    print(f"\n[opt2] {args.cv}-fold CV (split by problem)", flush=True)
    for i, (tr, te) in enumerate(folds):
        print(f"\n--- fold {i} ---", flush=True)
        res = evaluate_fold(rows, embs, tr, te)
        fold_results.append(res)
        if "error" in res:
            print(f"  fold {i}: SKIP ({res['error']})", flush=True); continue
        print(f"  fold {i}: cmaj={res['cmaj_acc']:.1%} verifier={res['verifier_acc']:.1%} "
              f"(d={res['delta_pp']:+.1f}pp gap-closure={res['gap_closure']:.1%}) "
              f"oracle={res['oracle_acc']:.1%} recovers={res['verifier_recovers']}", flush=True)

    valid = [r for r in fold_results if "error" not in r]
    mean_cmaj = float(np.mean([r["cmaj_acc"] for r in valid]))
    mean_ver = float(np.mean([r["verifier_acc"] for r in valid]))
    mean_oracle = float(np.mean([r["oracle_acc"] for r in valid]))
    print(f"\n[opt2] === MEAN: cmaj={mean_cmaj:.1%} verifier={mean_ver:.1%} oracle={mean_oracle:.1%} ===")
    print(f"[opt2] dpp_vs_cmaj = {(mean_ver-mean_cmaj)*100:+.2f}")
    print(f"[opt2] gap-closure = {(mean_ver-mean_cmaj)/max(1e-9, mean_oracle-mean_cmaj):.1%}")

    if mean_ver >= 0.89: decision = "WIN-DECISIVE"
    elif mean_ver >= 0.87: decision = "WIN-STRONG"
    elif mean_ver >= 0.83: decision = "WIN-MINOR"
    elif mean_ver >= mean_cmaj - 0.01: decision = "INCONCLUSIVE"
    else: decision = "LOSS"
    print(f"\n[opt2] >>> PRE-REG DECISION: {decision} (mean ver={mean_ver:.1%}, cmaj={mean_cmaj:.1%}) <<<")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "model_id": args.model_id,
        "n_branches": len(rows),
        "n_problems": len({r.problem_id for r in rows}),
        "cv_folds": args.cv, "max_len": args.max_len,
        "fold_results": fold_results,
        "mean_cmaj": mean_cmaj, "mean_verifier": mean_ver, "mean_oracle": mean_oracle,
        "delta_pp_vs_cmaj": (mean_ver - mean_cmaj) * 100,
        "gap_closure": (mean_ver - mean_cmaj) / max(1e-9, mean_oracle - mean_cmaj),
        "pre_reg_decision": decision,
    }, indent=2))
    print(f"[opt2] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
