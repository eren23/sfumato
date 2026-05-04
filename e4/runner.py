"""E4 runner: AR vs diffusion vs hybrid CoT on GSM8K.

Crucible-friendly. Reads config from env vars; emits stdout in the
LM-training-contract format so Crucible's OutputParser can parse it.

Conditions (env CONDITION):
    c1: pure AR (Qwen)
    c2: pure diffusion (LLaDA)
    c3: AR plan -> diffusion CoT -> AR answer (hybrid)
    c4: c3 + one extra (AR-extend, diffuse-again) round

Usage (local):
    MOCK_MODELS=1 CONDITION=c3 N_PROBLEMS=5 python e4/runner.py

Crucible parses lines like:
    step:{i}/{N} train_loss:{1-acc} val_loss:{1-acc} val_bpb:{flops}
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from e4 import ar_qwen, diff_llada, flops as flops_mod, grade  # noqa: E402


# Module-level state for env-driven commit-block count. Set in main() and
# read by run_condition() via _get_commit_n_blocks().
_COMMIT_N_BLOCKS: int = 1


def _get_commit_n_blocks() -> int:
    return _COMMIT_N_BLOCKS


def _make_esc_callback(diff_model, n_branches: int, esc_state: dict):
    """Build an ESC quorum=ceil(B/2)+1 batched step_callback.

    Fires only at sub-blocks >= ESC_MIN_BLOCK (default 2 of 4) so the answer
    span has time to land. Decodes each row's committed prefix, extracts the
    last numeric span, votes. When a quorum agrees, marks remaining rows
    `should_stop` and stashes the trigger block + winner in `esc_state`.

    Returns None when ESC is disabled (env ESC != "1") or the underlying
    tokenizer is unavailable (mock mode) — caller can skip passing a callback.
    """
    if os.environ.get("ESC", "0") != "1":
        return None
    tokenizer = getattr(diff_model, "_tokenizer", None)
    if tokenizer is None:
        return None
    from collections import Counter as _Counter

    quorum = (n_branches // 2) + 1
    esc_min_block = int(os.environ.get("ESC_MIN_BLOCK", "2"))

    def cb(state):
        # Default: keep all branches running.
        if state.sub_block < esc_min_block:
            return diff_llada.BatchStepDirective.continue_all(state.B)
        x = state.x_handle
        L = state.prompt_len
        end = state.block_end
        partial: list[str | None] = []
        for bi in range(state.B):
            if not state.active[bi]:
                partial.append(None)
                continue
            try:
                txt = tokenizer.decode(
                    x[bi, L:end].tolist(), skip_special_tokens=True
                )
                # Use the strict final-answer pattern. Mid-reasoning numbers
                # like "16 - 3 = 13" return "" → ESC waits for the answer
                # span ("#### N" / "Answer: N") to actually land.
                a = grade.extract_final_answer(txt)
                partial.append(a or None)
            except Exception:
                partial.append(None)
        counts = _Counter(a for a in partial if a)
        if not counts:
            return diff_llada.BatchStepDirective.continue_all(state.B)
        top_a, top_c = counts.most_common(1)[0]
        if top_c >= quorum:
            should_stop = [
                bool(state.active[bi]) and partial[bi] != top_a
                for bi in range(state.B)
            ]
            if esc_state.get("trigger_block") is None:
                esc_state["trigger_block"] = state.sub_block
                esc_state["winner"] = top_a
                esc_state["branches_pruned"] = sum(should_stop)
                esc_state["partial_answers"] = list(partial)
            return diff_llada.BatchStepDirective(should_stop=should_stop)
        return diff_llada.BatchStepDirective.continue_all(state.B)

    return cb


def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, str(int(default)))
    return raw.lower() in {"1", "true", "yes"}


def load_problems(n: int, dev_indices_path: Path) -> list[dict]:
    """Load the first n problems from the frozen dev set.

    Tries the local pinned indices first; falls back to streaming GSM8K from HF
    if the dataset is reachable. The mock path returns synthetic problems.
    """
    if env_bool("MOCK_MODELS", False):
        return [
            {
                "id": f"mock-{i}",
                "question": f"Mock problem {i}: 2 + {i} = ?",
                "answer": f"{2 + i}",
            }
            for i in range(n)
        ]

    if not dev_indices_path.exists():
        raise FileNotFoundError(
            f"Frozen dev indices missing at {dev_indices_path}. "
            "Run scripts/freeze_gsm8k.py first."
        )

    with dev_indices_path.open() as f:
        spec = json.load(f)

    from datasets import load_dataset  # type: ignore

    ds = load_dataset(spec["dataset"], spec.get("config", "main"), split=spec["split"])
    return [
        {
            "id": str(i),
            "question": ds[i]["question"],
            "answer": ds[i]["answer"].split("####")[-1].strip(),
        }
        for i in spec["indices"][:n]
    ]


def run_condition(
    problem: dict,
    condition: str,
    k_steps: int,
    ar_model,
    diff_model,
    seed: int,
) -> tuple[str, int, dict]:
    """Returns (predicted_answer, flops_used, trace_dict).

    trace_dict has the per-stage text outputs for diagnosis.
    """
    q = problem["question"]
    trace: dict[str, str] = {}
    if condition == "c1":
        cot, used = ar_model.generate_cot_and_answer(q, seed=seed)
        trace["cot"] = cot
        return grade.extract_answer(cot), used, trace
    if condition == "c2":
        text, used = diff_model.denoise_block(prompt=q, k_steps=k_steps, seed=seed)
        trace["diffusion_cot"] = text
        return grade.extract_answer(text), used, trace
    if condition == "c2c":
        # C2 + commit adapter on the final sub-block. Tests "is the commit
        # adapter alone enough to lift single-shot accuracy?" The diff wrapper
        # handles enabling/disabling the named PEFT adapter exactly once per
        # call (not per diffusion step). COMMIT_N_BLOCKS env var controls how
        # many trailing sub-blocks fire commit (default 1).
        text, used = diff_model.denoise_block(
            prompt=q,
            k_steps=k_steps,
            seed=seed,
            apply_commit=True,
            commit_n_blocks=_get_commit_n_blocks(),
        )
        trace["diffusion_cot"] = text
        return grade.extract_answer(text), used, trace
    if condition == "c3":
        plan, f1 = ar_model.generate_plan(q, max_tokens=32, seed=seed)
        cot, f2 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan, k_steps=k_steps, seed=seed
        )
        ans, f3 = ar_model.finalize_answer(question=q, plan=plan, cot=cot, seed=seed)
        trace.update(plan=plan, diffusion_cot=cot, finalize=ans)
        return grade.extract_answer(ans), f1 + f2 + f3, trace
    if condition == "c3p":
        # C3 minus the AR finalize step: AR plan -> diffusion CoT -> answer
        # extracted directly from LLaDA's output. Decomposes C3<C2 into
        # "does the plan help LLaDA?" vs "does the finalizer hurt?"
        plan, f1 = ar_model.generate_plan(q, max_tokens=32, seed=seed)
        cot, f2 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan, k_steps=k_steps, seed=seed
        )
        trace.update(plan=plan, diffusion_cot=cot)
        return grade.extract_answer(cot), f1 + f2, trace
    if condition == "c2hint":
        # C2 with a generic CoT hint prefix (no model planner). Tests
        # whether *any* prefix degrades, or specifically *generated* ones.
        text, used = diff_model.denoise_block(
            prompt=q + "\n\nLet's think step by step.",
            k_steps=k_steps,
            seed=seed,
        )
        trace["diffusion_cot"] = text
        return grade.extract_answer(text), used, trace
    if condition == "c2empty":
        # C2 with empty "Plan: " prefix (structure but no content).
        # Decomposes "is it the plan content or just the structural prefix?"
        text, used = diff_model.denoise_block(
            prompt=q + "\n\nPlan: ", k_steps=k_steps, seed=seed
        )
        trace["diffusion_cot"] = text
        return grade.extract_answer(text), used, trace
    if condition == "crev":
        # Reverse hybrid: LLaDA generates a short scaffold first (low budget),
        # then Qwen reads question + LLaDA's scaffold and finalizes.
        # Tests "diffuse-then-AR" asymmetry vs C3's "AR-then-diffuse-then-AR".
        scaffold_k = max(k_steps // 2, 4)
        scaffold, f1 = diff_model.denoise_block(
            prompt=q + "\n\nSketch the key calculations first.",
            k_steps=scaffold_k,
            seed=seed,
        )
        ans, f2 = ar_model.finalize_answer(
            question=q, plan="", cot=scaffold, seed=seed
        )
        trace.update(diffusion_scaffold=scaffold, finalize=ans)
        return grade.extract_answer(ans), f1 + f2, trace
    if condition == "c4":
        plan, f1 = ar_model.generate_plan(q, max_tokens=32, seed=seed)
        cot1, f2 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan, k_steps=k_steps, seed=seed
        )
        extension, f3 = ar_model.extend_cot(question=q, plan=plan, cot=cot1, seed=seed)
        cot2, f4 = diff_model.denoise_block(
            prompt=q + "\n\nPlan: " + plan + "\n\nDraft: " + cot1 + extension,
            k_steps=k_steps,
            seed=seed,
        )
        ans, f5 = ar_model.finalize_answer(question=q, plan=plan, cot=cot2, seed=seed)
        trace.update(
            plan=plan,
            diffusion_cot1=cot1,
            extension=extension,
            diffusion_cot2=cot2,
            finalize=ans,
        )
        return grade.extract_answer(ans), f1 + f2 + f3 + f4 + f5, trace
    if condition == "cmaj":
        # Self-consistency at diffusion level: N parallel LLaDA branches
        # (different seeds, temperature>0) -> majority vote on the extracted
        # numeric answer. Captures "germinate in parallel, then converge"
        # without any AR step. Temperature>0 is REQUIRED for branches to
        # diverge — at temp=0 LLaDA's gumbel sampling is deterministic and
        # all branches produce identical output (verified by sweep accident).
        # S0: branches run in a single batched (B, L+gen) forward pass.
        # S1: ESC quorum exit prunes branches once a majority agree at a
        # late sub-block boundary; winner is taken from the ESC quorum.
        # BATCHED=0 forces the legacy sequential loop for paired wallclock comparison.
        from collections import Counter

        n_branches = int(os.environ.get("BRANCHES", "5"))
        temperature = float(os.environ.get("TEMP", "0.7"))
        batched = os.environ.get("BATCHED", "1") == "1"
        seeds_b = [seed * 100 + b for b in range(n_branches)]
        esc_state: dict = {}
        if batched:
            cb = _make_esc_callback(diff_model, n_branches, esc_state)
            results = diff_model.denoise_block_batched(
                prompt=q,
                k_steps=k_steps,
                seeds=seeds_b,
                temperature=temperature,
                step_callback=cb,
            )
            branches = [r[0] for r in results]
            total = sum(r[1] for r in results)
        else:
            # BATCHED=0 — legacy per-seed sequential loop. Reproduces the
            # pre-S0 implementation for paired wallclock baselines.
            branches = []
            total = 0
            for s in seeds_b:
                cot, used = diff_model.denoise_block(
                    prompt=q,
                    k_steps=k_steps,
                    seed=s,
                    temperature=temperature,
                )
                branches.append(cot)
                total += used
        answers = [grade.extract_answer(b) for b in branches]
        counts = Counter(a for a in answers if a)
        if esc_state.get("winner"):
            winner = esc_state["winner"]
        else:
            winner = counts.most_common(1)[0][0] if counts else (answers[0] or "")
        for i, b in enumerate(branches):
            trace[f"branch_{i}"] = b
        trace["votes"] = " | ".join(answers)
        trace["winner"] = winner
        if esc_state.get("trigger_block") is not None:
            trace["esc_trigger_block"] = esc_state["trigger_block"]
            trace["esc_branches_pruned"] = esc_state["branches_pruned"]
        return winner, total, trace
    if condition == "cmajc":
        # cmaj + commit adapter on each branch's final sub-block, THEN
        # majority-vote on the extracted numeric answers. Tests whether
        # commit + branching double-dips (additive vs subsumed).
        # S0: branches run in a single batched call; commit-LoRA toggles
        # exactly twice (on at first_commit_block, off after last block).
        # S1: ESC quorum exit; same semantics as cmaj.
        # BATCHED=0 forces legacy sequential for paired baselines.
        from collections import Counter

        n_branches = int(os.environ.get("BRANCHES", "5"))
        temperature = float(os.environ.get("TEMP", "0.7"))
        batched = os.environ.get("BATCHED", "1") == "1"
        seeds_b = [seed * 100 + b for b in range(n_branches)]
        esc_state: dict = {}
        if batched:
            cb = _make_esc_callback(diff_model, n_branches, esc_state)
            results = diff_model.denoise_block_batched(
                prompt=q,
                k_steps=k_steps,
                seeds=seeds_b,
                temperature=temperature,
                apply_commit=True,
                commit_n_blocks=_get_commit_n_blocks(),
                step_callback=cb,
            )
            branches = [r[0] for r in results]
            total = sum(r[1] for r in results)
        else:
            branches = []
            total = 0
            for s in seeds_b:
                cot, used = diff_model.denoise_block(
                    prompt=q,
                    k_steps=k_steps,
                    seed=s,
                    temperature=temperature,
                    apply_commit=True,
                    commit_n_blocks=_get_commit_n_blocks(),
                )
                branches.append(cot)
                total += used
        answers = [grade.extract_answer(b) for b in branches]
        counts = Counter(a for a in answers if a)
        if esc_state.get("winner"):
            winner = esc_state["winner"]
        else:
            winner = counts.most_common(1)[0][0] if counts else (answers[0] or "")
        for i, b in enumerate(branches):
            trace[f"branch_{i}"] = b
        trace["votes"] = " | ".join(answers)
        trace["winner"] = winner
        if esc_state.get("trigger_block") is not None:
            trace["esc_trigger_block"] = esc_state["trigger_block"]
            trace["esc_branches_pruned"] = esc_state["branches_pruned"]
        return winner, total, trace
    if condition == "cmerge":
        # Parallel diffusion branches -> AR merger. Inverse of C3:
        # diffusion-germinate-multiple -> AR-converge-into-one. Same temp>0
        # requirement as cmaj for branches to actually diverge.
        # S0: branches run batched. ESC not applied — final AR merger needs
        # all branches as candidates regardless of mid-flight agreement.
        n_branches = int(os.environ.get("BRANCHES", "3"))
        temperature = float(os.environ.get("TEMP", "0.7"))
        seeds_b = [seed * 100 + b for b in range(n_branches)]
        results = diff_model.denoise_block_batched(
            prompt=q,
            k_steps=k_steps,
            seeds=seeds_b,
            temperature=temperature,
        )
        branches = [r[0] for r in results]
        total = sum(r[1] for r in results)
        joined = "\n\n---\n\n".join(
            f"Candidate {i+1}:\n{b}" for i, b in enumerate(branches)
        )
        ans, f_ar = ar_model.finalize_answer(
            question=q, plan="", cot=joined, seed=seed
        )
        for i, b in enumerate(branches):
            trace[f"branch_{i}"] = b
        trace["finalize"] = ans
        return grade.extract_answer(ans), total + f_ar, trace
    raise ValueError(f"Unknown condition: {condition}")


def _maybe_init_wandb(cfg: dict) -> object | None:
    """Init wandb if WANDB_API_KEY is set and WANDB_DISABLED is not '1'."""
    if os.environ.get("WANDB_DISABLED") == "1":
        return None
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[E4] wandb not installed; skipping logging.", flush=True)
        return None
    project = os.environ.get("WANDB_PROJECT", "sfumato-e4")
    name = (
        f"{cfg['condition']}-k{cfg['k_steps']}-seed{cfg['seed']}"
        f"-{cfg['ar_model'].split('/')[-1]}"
    )
    run = wandb.init(
        project=project,
        name=name,
        config=cfg,
        reinit=True,
        tags=[cfg["condition"], f"k={cfg['k_steps']}"],
    )
    return run


def _setup_fast_dllm() -> int:
    """One-shot pod-side setup for the S4 Fast-dLLM spike.

    Clones NVlabs/Fast-dLLM into /workspace/Fast-dLLM if missing, attempts to
    import the upstream module, and prints `dir(fast_dllm)` so we can pin the
    correct entry-point symbols in `e4/fast_dllm_adapter.py`.

    Triggered by CONDITION=fast_dllm_setup. Pure side-effect; no W&B / training.
    """
    import subprocess
    target = Path(env_str("FAST_DLLM_PATH", "/workspace/Fast-dLLM"))
    if not target.exists():
        print(f"[fast-dllm] cloning NVlabs/Fast-dLLM into {target}", flush=True)
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/NVlabs/Fast-dLLM.git", str(target),
        ])
    else:
        print(f"[fast-dllm] {target} exists; pulling latest", flush=True)
        subprocess.check_call(["git", "-C", str(target), "pull", "--ff-only"])
    print(f"[fast-dllm] listing top-level: {sorted(p.name for p in target.iterdir() if not p.name.startswith('.'))}", flush=True)
    sys.path.insert(0, str(target))
    try:
        import fast_dllm  # type: ignore
        attrs = sorted(a for a in dir(fast_dllm) if not a.startswith("_"))
        print(f"[fast-dllm] import OK; dir(fast_dllm) = {attrs}", flush=True)
        # Look for likely entry-point symbols.
        candidates = [
            "LLaDAModelWithKVCache", "wrap_llada", "wrap_llada_with_kv_cache",
            "parallel_decode_step", "parallel_decode_with_kv_cache",
        ]
        present = [c for c in candidates if hasattr(fast_dllm, c)]
        print(f"[fast-dllm] candidates present: {present}", flush=True)
    except ImportError as e:
        print(f"[fast-dllm] import FAILED: {e}", flush=True)
        # Show entry-point structure to help pin symbols.
        for sub in ("__init__.py", "fast_dllm/__init__.py"):
            p = target / sub
            if p.exists():
                print(f"[fast-dllm] -- {sub} --", flush=True)
                print(p.read_text()[:2000], flush=True)
        return 1
    # Crucible expects a terminal "step:N/N" marker for compliance; emit a fake one.
    print("step:1/1 train_loss:0.0000 val_loss:0.0000 val_bpb:0.000e+00", flush=True)
    return 0


def main() -> int:
    condition = env_str("CONDITION", "c1")
    if condition == "fast_dllm_setup":
        return _setup_fast_dllm()
    k_steps = env_int("K_STEPS", 0)
    n_problems = env_int("N_PROBLEMS", 5)
    seed = env_int("SEED", 0)
    ar_model_name = env_str("AR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    diff_model_name = env_str("DIFF_MODEL", "GSAI-ML/LLaDA-8B-Instruct")
    mock = env_bool("MOCK_MODELS", False)
    branches = env_int("BRANCHES", 1)

    # Optional LoRA adapters (E2 tracks). Empty string → None.
    raw_lora = env_str("LORA_PATH", "")
    raw_commit = env_str("COMMIT_LORA_PATH", "")
    lora_path = raw_lora if raw_lora else None
    commit_lora_path = raw_commit if raw_commit else None
    # v3 commit-LoRA follow-up knob: how many trailing sub-blocks fire commit.
    # Default 1 (only the last block, original behavior). v3 follow-up runs
    # with COMMIT_N_BLOCKS=3 to commit on blocks 2-4 of 4.
    global _COMMIT_N_BLOCKS
    _COMMIT_N_BLOCKS = env_int("COMMIT_N_BLOCKS", 1)

    random.seed(seed)
    dev_indices = REPO_ROOT / "e4" / "data" / "gsm8k_dev_200.json"
    problems = load_problems(n_problems, dev_indices)

    ar_model = ar_qwen.load(ar_model_name, mock=mock)
    diff_model = diff_llada.load(
        diff_model_name,
        mock=mock,
        lora_path=lora_path,
        commit_lora_path=commit_lora_path,
    )

    wandb_run = _maybe_init_wandb(
        {
            "condition": condition,
            "k_steps": k_steps,
            "n_problems": n_problems,
            "seed": seed,
            "ar_model": ar_model_name,
            "diff_model": diff_model_name,
            "branches": branches,
            "mock": mock,
        }
    )

    n_correct = 0
    total_flops = 0
    rows: list[dict] = []
    t0 = time.time()
    for i, prob in enumerate(problems):
        t_prob = time.time()
        pred, used, trace = run_condition(
            prob, condition, k_steps, ar_model, diff_model, seed=seed
        )
        wallclock_ms = int((time.time() - t_prob) * 1000)
        correct = grade.is_correct(pred, prob["answer"])
        n_correct += int(correct)
        total_flops += used
        rows.append(
            {
                "idx": i,
                "id": prob["id"],
                "condition": condition,
                "k_steps": k_steps,
                "seed": seed,
                "ar_model": ar_model_name,
                "diff_model": diff_model_name,
                "lora_path": lora_path or "",
                "commit_lora_path": commit_lora_path or "",
                "pred": pred,
                "gold": prob["answer"],
                "correct": correct,
                "flops": used,
                "wallclock_ms": wallclock_ms,
                "trace": trace,
            }
        )
        running_acc = n_correct / (i + 1)
        running_loss = 1.0 - running_acc
        # Crucible LM-training-contract stdout. val_bpb piggybacks for FLOPs.
        print(
            f"step:{i+1}/{n_problems} "
            f"train_loss:{running_loss:.4f} "
            f"val_loss:{running_loss:.4f} "
            f"val_bpb:{total_flops:.3e}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "step": i + 1,
                    "running_acc": running_acc,
                    "running_loss": running_loss,
                    "flops_used": used,
                    "flops_cumulative": total_flops,
                    "correct": int(correct),
                }
            )

    elapsed = time.time() - t0
    accuracy = n_correct / max(n_problems, 1)
    print(
        f"\n[E4] condition={condition} k={k_steps} seed={seed} "
        f"n={n_problems} accuracy={accuracy:.4f} flops={total_flops:.3e} "
        f"wallclock={elapsed:.1f}s",
        flush=True,
    )

    out_path = (
        REPO_ROOT
        / "e4"
        / "results"
        / f"raw_{condition}_k{k_steps}_seed{seed}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[E4] rows -> {out_path.relative_to(REPO_ROOT)}", flush=True)

    if wandb_run is not None:
        import wandb  # type: ignore

        wandb_run.summary["accuracy"] = accuracy
        wandb_run.summary["total_flops"] = total_flops
        wandb_run.summary["wallclock_s"] = elapsed
        wandb_run.summary["mean_flops_per_problem"] = total_flops / max(n_problems, 1)

        # Log full per-problem table (text outputs included). Trace stages
        # become a single concatenated string so the schema is uniform across
        # conditions.
        table = wandb.Table(
            columns=[
                "idx",
                "id",
                "gold",
                "pred",
                "correct",
                "flops",
                "trace",
            ]
        )
        for r in rows:
            trace_str = "\n\n".join(
                f"### {k}\n{v}" for k, v in r.get("trace", {}).items() if v
            )
            table.add_data(
                r["idx"],
                r["id"],
                r["gold"],
                r["pred"],
                bool(r["correct"]),
                r["flops"],
                trace_str[:6000],  # cap to avoid bloat
            )
        wandb_run.log({"problems": table})

        # Attach the raw jsonl as a versioned artifact (useful for replay).
        try:
            artifact = wandb.Artifact(
                f"{condition}-k{k_steps}-seed{seed}-rows", type="rows"
            )
            artifact.add_file(str(out_path))
            wandb_run.log_artifact(artifact)
        except Exception as exc:
            print(f"[E4] artifact upload skipped: {exc}", flush=True)

        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
