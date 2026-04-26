"""Build the Track 1 prefix-robustness training dataset for E2.

For each GSM8K-train example, emit 8 rows — one per prefix tier — so a
LoRA on LLaDA-8B-Instruct can learn to ignore upstream plan prefixes
regardless of quality. Tiers span: no prefix, minimal "Plan: ", a hint,
an XML opener with truncated gold, three Qwen2.5-Instruct planners
(0.5B / 1.5B / 7B at greedy ≤32 tokens), and the gold rationale itself.

Usage (mock locally, no GPU needed):
    python scripts/build_prefix_robust_dataset.py --n_train 8 --push False

Real run (on pod, GPU):
    HF_TOKEN=... python scripts/build_prefix_robust_dataset.py --push True

If --push is False, the resulting Dataset is saved to
e2/data/prefix_robust_dataset.parquet for inspection.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "e2" / "data"

PLAN_SYSTEM = (
    "You are a planning assistant. In <=32 tokens, sketch the calculation "
    "steps."
)


def _split_gsm8k_answer(answer_field: str) -> tuple[str, str]:
    """Return (rationale, numeric_answer). GSM8K answers end with '#### N'."""
    if "####" in answer_field:
        rationale, numeric = answer_field.rsplit("####", 1)
        return rationale.strip(), numeric.strip()
    return answer_field.strip(), ""


def _build_static_tiers(question: str, rationale: str) -> list[tuple[str, str]]:
    """Return list of (tier_name, prefix) for the deterministic tiers."""
    xml_content = rationale[:80]
    oracle = rationale[:200]
    return [
        ("none", ""),
        ("minimal", "Plan: "),
        ("hint", "Let's think step by step.\n"),
        ("xml", f"<plan>\n{xml_content}"),
        ("oracle", oracle),
    ]


def _generate_qwen_plans(
    model_name: str,
    questions: list[str],
    seed: int,
) -> list[str]:
    """Load `model_name`, generate one greedy plan per question, unload."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from tqdm import tqdm  # type: ignore

    print(f"[qwen] loading {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.requires_grad_(False)

    plans: list[str] = []
    torch.manual_seed(seed)
    for q in tqdm(questions, desc=f"plans:{model_name.split('/')[-1]}"):
        messages = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": q},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        plans.append(text)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return plans


def _mock_qwen_plans(model_name: str, questions: list[str]) -> list[str]:
    """Deterministic stand-in for --push False local dry-runs (no GPU)."""
    short = model_name.split("/")[-1]
    return [f"[mock plan from {short} for] {q[:40]}" for q in questions]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=7473)
    parser.add_argument("--push", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument(
        "--repo_id", type=str, default="eren23/sfumato-prefix-robust-gsm8k"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--qwen_models",
        type=str,
        default=(
            "Qwen/Qwen2.5-0.5B-Instruct,"
            "Qwen/Qwen2.5-1.5B-Instruct,"
            "Qwen/Qwen2.5-7B-Instruct"
        ),
    )
    parser.add_argument(
        "--mock_qwen",
        action="store_true",
        help="Skip real Qwen forward and emit deterministic placeholder plans. "
        "Auto-on when --push is False and no GPU is detected.",
    )
    args = parser.parse_args()

    from datasets import ClassLabel, Dataset, DatasetDict, load_dataset  # type: ignore

    print(f"[build] loading gsm8k/main:train", flush=True)
    ds = load_dataset("gsm8k", "main", split="train")
    n = min(args.n_train, len(ds))
    print(f"[build] using {n}/{len(ds)} train examples", flush=True)

    questions: list[str] = []
    rationales: list[str] = []
    gold_answers: list[str] = []
    for i in range(n):
        ex = ds[i]
        rationale, numeric = _split_gsm8k_answer(ex["answer"])
        questions.append(ex["question"])
        rationales.append(rationale)
        gold_answers.append(numeric)

    qwen_models = [m.strip() for m in args.qwen_models.split(",") if m.strip()]
    if len(qwen_models) != 3:
        print(
            f"[build] expected 3 qwen models, got {len(qwen_models)}: {qwen_models}",
            file=sys.stderr,
        )
        return 1

    auto_mock = False
    if args.mock_qwen:
        auto_mock = True
    else:
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available() and not args.push:
                print(
                    "[build] no GPU detected and --push False; "
                    "auto-enabling --mock_qwen",
                    flush=True,
                )
                auto_mock = True
        except ImportError:
            auto_mock = True

    qwen_tier_names = ["weak", "medium", "strong"]
    qwen_plans: dict[str, list[str]] = {}
    for tier, model_name in zip(qwen_tier_names, qwen_models):
        if auto_mock:
            qwen_plans[tier] = _mock_qwen_plans(model_name, questions)
        else:
            qwen_plans[tier] = _generate_qwen_plans(
                model_name, questions, seed=args.seed
            )

    rows: list[dict] = []
    for i, (q, rat, ga) in enumerate(zip(questions, rationales, gold_answers)):
        static_tiers = _build_static_tiers(q, rat)
        all_tiers: list[tuple[str, str]] = list(static_tiers)
        for tier in qwen_tier_names:
            all_tiers.append((tier, qwen_plans[tier][i]))
        for tier_name, prefix in all_tiers:
            full_prompt = f"{q}\n\n{prefix}" if prefix else q
            full_target = f"{rat}\n\n#### {ga}"
            rows.append(
                {
                    "question": q,
                    "prefix": prefix,
                    "prefix_tier": tier_name,
                    "gold_rationale": rat,
                    "gold_answer": ga,
                    "full_prompt": full_prompt,
                    "full_target": full_target,
                    "source_idx": i,
                }
            )

    print(f"[build] built {len(rows)} rows ({len(rows) // 8} problems × 8 tiers)")

    tier_names = ["none", "minimal", "hint", "xml", "weak", "medium", "strong", "oracle"]

    # Optional: log a wandb run for visibility into dataset gen.
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_DISABLED") != "1":
        try:
            import wandb  # type: ignore

            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "sfumato-e2"),
                name=f"track1-data-N{args.n_train}-seed{args.seed}",
                tags=["track1", "dataset-build"],
                reinit=True,
                config={
                    "n_train": args.n_train,
                    "qwen_models": qwen_models,
                    "seed": args.seed,
                    "n_rows": len(rows),
                    "tiers": tier_names,
                },
            )
            run.summary["n_rows"] = len(rows)
            run.summary["n_problems"] = len(rows) // 8
            run.summary["n_tiers"] = 8
            run.finish()
        except Exception as exc:
            print(f"[build] wandb log skipped: {exc}", flush=True)

    dataset = Dataset.from_list(rows)
    # Cast prefix_tier to ClassLabel so stratify_by_column works.
    dataset = dataset.cast_column("prefix_tier", ClassLabel(names=tier_names))
    # Stratified split needs test_size >= num_classes (8). Fall back to a
    # plain shuffled split for tiny smoke runs where 5% < 8 rows.
    test_n = max(int(round(len(dataset) * 0.05)), 1)
    if test_n >= len(tier_names):
        split = dataset.train_test_split(
            test_size=0.05, seed=args.seed, stratify_by_column="prefix_tier"
        )
    else:
        split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    if args.push:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            print(
                "[build] no HF token in env (HF_TOKEN / HUGGINGFACE_HUB_TOKEN); "
                "cannot push",
                file=sys.stderr,
            )
            return 1
        print(f"[build] pushing to hub: {args.repo_id} (private)", flush=True)
        dd.push_to_hub(args.repo_id, private=True, token=token)
        print(f"[build] pushed {args.repo_id}", flush=True)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / "prefix_robust_dataset.parquet"
        dataset.to_parquet(str(out_path))
        print(f"[build] wrote {out_path} ({len(dataset)} rows)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
