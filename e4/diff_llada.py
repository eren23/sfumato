"""LLaDA-Instruct diffusion-LM wrapper.

Real path ports the reference sampler from ML-GSAI/LLaDA (Apache-2.0):
  https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

Mock path returns deterministic strings for local CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from e4 import flops as flops_mod

_LLADA_MASK_ID = 126336  # LLaDA's [MASK] token id
_LLADA_EOT_ID = 126081   # LLaDA's <|endoftext|> id (for optional masking, see LLaDA App. B.4)

_DENOISE_SYS = (
    "You are a careful math tutor. Think step by step about the problem "
    "below; show numeric work; end with 'Answer: <number>'."
)


@dataclass
class _Mock:
    name: str
    block_len: int = 256

    def denoise_block(
        self, prompt: str, k_steps: int, seed: int = 0
    ) -> tuple[str, int]:
        h = hashlib.sha256(f"{prompt}|{k_steps}|{seed}".encode()).hexdigest()[:8]
        text = f"Mock diffusion CoT (k={k_steps}) hash={h}"
        used = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.block_len, n_steps=k_steps
        )
        return text, used


def _add_gumbel_noise(logits, temperature: float):
    import torch  # type: ignore

    if temperature <= 0.0:
        return logits.to(torch.float64)
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_(min=1e-20, max=1.0)
    gumbel = -torch.log(-torch.log(noise))
    return logits / temperature + gumbel


def _num_transfer_tokens(mask_index, steps: int):
    """How many masked positions to commit at each diffusion step.

    Splits the total number of masks across `steps` rounds as evenly as
    possible. Returns a (B, steps) int tensor.
    """
    import torch  # type: ignore

    n_mask = mask_index.sum(dim=1, keepdim=True)
    base = n_mask // steps
    remainder = n_mask % steps
    out = base.repeat(1, steps)
    for b in range(out.shape[0]):
        out[b, : int(remainder[b].item())] += 1
    return out.to(torch.long)


@dataclass
class _Real:
    """LLaDA-Instruct semi-AR sampler.

    Inference flow (matches reference generate.py):
      1. apply chat template -> token ids
      2. append `gen_length` mask tokens
      3. for each block of `block_length`, do `steps_per_block = steps // num_blocks`
         denoising rounds, each committing `num_transfer_tokens[i]` of the
         lowest-confidence masks
    """

    name: str
    block_len: int = 256
    gen_length: int = 256
    sub_block_length: int = 32  # internal block_length within gen_length
    _model: object | None = field(default=None, repr=False)
    _tokenizer: object | None = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # NOTE: LLaDA's custom modeling code (trust_remote_code) targets
        # transformers ≤ 4.x. Pin transformers==4.46.3 on the pod.
        model = AutoModel.from_pretrained(
            self.name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.requires_grad_(False)
        self._model = model

    def _generate(
        self,
        prompt_ids,  # torch.LongTensor (1, L)
        steps: int,
        temperature: float,
    ):
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        device = self._model.device  # type: ignore[attr-defined]
        gen_length = self.gen_length
        block_length = self.sub_block_length
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        # Steps must be divisible by num_blocks; round up to nearest multiple.
        if steps % num_blocks != 0:
            steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks
        steps_per_block = steps // num_blocks

        x = torch.full(
            (1, prompt_ids.shape[1] + gen_length),
            _LLADA_MASK_ID,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids

        for b_idx in range(num_blocks):
            blk_start = prompt_ids.shape[1] + b_idx * block_length
            blk_end = prompt_ids.shape[1] + (b_idx + 1) * block_length
            block_mask = (x[:, blk_start:blk_end] == _LLADA_MASK_ID)
            n_transfer = _num_transfer_tokens(block_mask, steps_per_block)

            for s in range(steps_per_block):
                mask_index = x == _LLADA_MASK_ID
                logits = self._model(x).logits  # type: ignore[attr-defined]

                logits_n = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_n, dim=-1)

                p = F.softmax(logits.to(torch.float64), dim=-1)
                conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                # Don't commit positions past the current block.
                conf[:, blk_end:] = float("-inf")

                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, conf, torch.full_like(conf, float("-inf")))

                transfer = torch.zeros_like(x0, dtype=torch.bool)
                k = int(n_transfer[0, s].item())
                if k > 0:
                    _, sel = torch.topk(conf[0], k=k)
                    transfer[0, sel] = True
                x[transfer] = x0[transfer]

        return x[:, prompt_ids.shape[1] :]

    def denoise_block(
        self, prompt: str, k_steps: int, seed: int = 0
    ) -> tuple[str, int]:
        import torch  # type: ignore

        self._ensure_loaded()
        torch.manual_seed(seed)
        messages = [
            {"role": "system", "content": _DENOISE_SYS},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self._tokenizer(
            prompt_text, return_tensors="pt"
        )["input_ids"].to(self._model.device)  # type: ignore[attr-defined]

        gen_ids = self._generate(
            prompt_ids, steps=max(k_steps, 4), temperature=0.0
        )
        text = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]  # type: ignore[attr-defined]
        used = flops_mod.llada_forward_flops(
            name=self.name, n_tokens=self.gen_length, n_steps=max(k_steps, 4)
        )
        return text.strip(), used


def load(name: str, mock: bool = False, block_len: int = 256) -> _Mock | _Real:
    return _Mock(name=name, block_len=block_len) if mock else _Real(
        name=name, block_len=block_len
    )
