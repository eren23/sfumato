# Fast-dLLM v1 — pod setup recipe

One-shot recipe for wiring NVlabs/Fast-dLLM v1 into sfumato on an RTX
4090 spot pod. Run once per fresh pod.

```bash
# 1. Clone upstream into the pod's persistent volume.
cd /workspace
git clone https://github.com/NVlabs/Fast-dLLM
export FAST_DLLM_PATH=/workspace/Fast-dLLM

# 2. Install upstream's Python deps (whatever its requirements.txt lists
#    — likely transformers, accelerate, torch, possibly triton).
cd /workspace/Fast-dLLM
pip install -r requirements.txt   # if present
# Otherwise inspect README + setup.py and install pinned versions.

# 3. Sanity-check that the upstream module imports.
python3 -c "import sys; sys.path.insert(0, '$FAST_DLLM_PATH'); import fast_dllm; print(dir(fast_dllm))"
# Expected output: a dir() listing that includes either
#   `LLaDAModelWithKVCache` or `wrap_llada` (or similar). If neither
#   appears, inspect `fast_dllm/__init__.py` and pin the correct symbol
#   in `e4/fast_dllm_adapter.py:wrap_for_fast_dllm`.

# 4. Sfumato adapter dry-run (still in mock mode — should hard-fail
#    cleanly because FAST_DLLM=1 + MOCK_MODELS=1 is incoherent).
cd /workspace/sfumato
FAST_DLLM=1 FAST_DLLM_PATH=$FAST_DLLM_PATH python3 -c "
from e4 import fast_dllm_adapter as f
print('enabled:', f.is_enabled())
f._ensure_upstream_on_path()
print('upstream loaded OK')
"

# 5. Real-mode τ sweep (per PRE_REG.md):
#    Pick max τ such that N=20 c2c same-extracted-answer rate vs legacy ≥ 18/20.
for TAU in 0.85 0.90 0.95; do
  FAST_DLLM=1 FAST_DLLM_PATH=$FAST_DLLM_PATH FAST_DLLM_TAU=$TAU \
  CONDITION=c2c K_STEPS=64 N_PROBLEMS=20 SEED=0 \
  LORA_PATH=eren23/sfumato-llada-prefix-robust-v3 \
  COMMIT_LORA_PATH=eren23/sfumato-llada-commit-v3 \
  COMMIT_N_BLOCKS=3 \
  python3 e4/runner.py
done

# 6. WIN-confirm at chosen τ on N=200:
FAST_DLLM=1 FAST_DLLM_PATH=$FAST_DLLM_PATH FAST_DLLM_TAU=<chosen> \
CONDITION=c2c K_STEPS=64 N_PROBLEMS=200 SEED=0 \
LORA_PATH=eren23/sfumato-llada-prefix-robust-v3 \
COMMIT_LORA_PATH=eren23/sfumato-llada-commit-v3 \
COMMIT_N_BLOCKS=3 \
python3 e4/runner.py

# 7. Promote to cmajc only if c2c WINs (per PRE_REG.md):
FAST_DLLM=1 FAST_DLLM_PATH=$FAST_DLLM_PATH FAST_DLLM_TAU=<chosen> \
CONDITION=cmajc K_STEPS=64 BRANCHES=5 TEMP=0.7 N_PROBLEMS=200 SEED=0 \
LORA_PATH=eren23/sfumato-llada-prefix-robust-v3 \
COMMIT_LORA_PATH=eren23/sfumato-llada-commit-v3 \
COMMIT_N_BLOCKS=3 \
python3 e4/runner.py
```

## What's pinned in code, what isn't

- The **env-gate** (`FAST_DLLM` + `FAST_DLLM_PATH`) is committed.
- The **upstream symbol pinning** in
  `e4/fast_dllm_adapter.py:wrap_for_fast_dllm` is *speculative*. The
  shim tries `LLaDAModelWithKVCache` then `wrap_llada` then raises
  `ImportError`. Pin the actual symbol once observed on the pod.
- `parallel_decode_step` similarly tries both a method-on-wrapped-model
  and a module-level function. Pin the actual call once observed.

## Cost / time estimate

- Setup steps 1–4: ~10 min on a fresh pod.
- τ sweep (step 5): ~10 min × 3 = 30 min.
- N=200 c2c WIN-confirm (step 6): ~30 min pre-Fast-dLLM, **~5 min**
  post (the headline win).
- N=200 cmajc-v3 confirm (step 7): ~15 min post (multiplicative with S0).
- Total: **≤$1** GPU spot at $0.20/hr.
