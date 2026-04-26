"""Qwen2.5-Instruct AR wrapper. Mock + real paths share one interface.

Mock path: deterministic synthetic outputs for local CI.
Real path: HF transformers chat-template generation, lazy-loaded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from e4 import flops as flops_mod

_COT_SYS = (
    "You are a careful math tutor. Think step by step, then give the final "
    "numeric answer on a line starting with 'Answer:'."
)
_PLAN_SYS = (
    "You are a planning assistant. In <=32 tokens, sketch the calculation "
    "steps needed to solve the problem. Do not solve it."
)
_FINAL_SYS = (
    "You are a careful math tutor. The CoT below was drafted by another "
    "system; check it, fix any errors, and output the final numeric answer "
    "on a line starting with 'Answer:'."
)
_EXTEND_SYS = (
    "Continue the partial chain-of-thought below by 1-2 short sentences "
    "that move toward the answer. Do not output the final answer yet."
)


@dataclass
class _Mock:
    name: str

    def generate_cot_and_answer(self, question: str, seed: int = 0) -> tuple[str, int]:
        random.seed(seed)
        text = f"Mock CoT for: {question}\nAnswer: {random.randint(0, 100)}"
        return text, flops_mod.qwen_forward_flops(name=self.name, n_tokens=128)

    def generate_plan(
        self, question: str, max_tokens: int = 32, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed)
        return (
            f"plan: think step by step about {question[:30]}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=max_tokens),
        )

    def finalize_answer(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed + 1)
        return (
            f"Answer: {random.randint(0, 100)}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=16),
        )

    def extend_cot(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        random.seed(seed + 2)
        return (
            f"\nMore reasoning {random.randint(0, 9)}",
            flops_mod.qwen_forward_flops(name=self.name, n_tokens=24),
        )


@dataclass
class _Real:
    name: str
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=dtype, device_map="auto"
        )
        model.requires_grad_(False)
        self._model = model

    def _chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.0,
    ) -> tuple[str, int]:
        import torch  # type: ignore

        self._ensure_loaded()
        torch.manual_seed(seed)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
            self._model.device  # type: ignore[attr-defined]
        )
        do_sample = temperature > 0.0
        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[attr-defined]
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        n_decoded = int(new_ids.shape[0])
        return text.strip(), flops_mod.qwen_forward_flops(self.name, n_decoded)

    def generate_cot_and_answer(self, question: str, seed: int = 0) -> tuple[str, int]:
        return self._chat(_COT_SYS, question, max_new_tokens=384, seed=seed)

    def generate_plan(
        self, question: str, max_tokens: int = 32, seed: int = 0
    ) -> tuple[str, int]:
        return self._chat(_PLAN_SYS, question, max_new_tokens=max_tokens, seed=seed)

    def finalize_answer(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        prompt = (
            f"Problem:\n{question}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Draft chain-of-thought:\n{cot}\n\n"
            "Now output only the final numeric answer."
        )
        return self._chat(_FINAL_SYS, prompt, max_new_tokens=64, seed=seed)

    def extend_cot(
        self, question: str, plan: str, cot: str, seed: int = 0
    ) -> tuple[str, int]:
        prompt = (
            f"Problem:\n{question}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Draft so far:\n{cot}"
        )
        return self._chat(_EXTEND_SYS, prompt, max_new_tokens=48, seed=seed)


def load(name: str, mock: bool = False) -> _Mock | _Real:
    return _Mock(name=name) if mock else _Real(name=name)
