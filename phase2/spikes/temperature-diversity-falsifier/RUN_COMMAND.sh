#!/usr/bin/env bash
# Spike: temperature-diversity-falsifier
# Pre-registered: phase2/spikes/temperature-diversity-falsifier/PRE_REG.md
# Dispatched via Crucible MCP run_project on sfumato_e4 (RTX 4090, on-demand).
#
# Equivalent local invocation (for reproduction without Crucible):
#
# for TAU in 0.5 0.7 1.0 1.3; do
#   CONDITION=cmaj \
#     K_STEPS=64 \
#     N_PROBLEMS=20 \
#     BRANCHES=5 \
#     TEMP=$TAU \
#     SEED=0 \
#     HF_LORA_REPO=eren23/sfumato-prefix-robust-gsm8k-v3 \
#     AR_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
#     DIFF_MODEL=GSAI-ML/LLaDA-8B-Instruct \
#     MOCK_MODELS=0 \
#     WANDB_RUN_NAME=spike-tau-$TAU \
#     python e4/runner.py
# done
#
# Per-tau wall-clock estimate: ~10 min on RTX 4090 (k=64, N=20, b=5).
# Total: ~40 min wallclock × $0.69/h on-demand (interruptible was unavailable
# due to volume-disk config) = $0.46 estimated.
#
# Crucible dispatch (one call per tau):
# mcp__crucible-fleet__run_project(
#   project_name="sfumato_e4",
#   overrides={
#     "CONDITION": "cmaj", "K_STEPS": "64", "N_PROBLEMS": "20",
#     "BRANCHES": "5", "TEMP": "<TAU>", "SEED": "0",
#     "HF_LORA_REPO": "eren23/sfumato-prefix-robust-gsm8k-v3",
#     "AR_MODEL": "Qwen/Qwen2.5-0.5B-Instruct",
#     "DIFF_MODEL": "GSAI-ML/LLaDA-8B-Instruct",
#     "MOCK_MODELS": "0",
#     "WANDB_RUN_NAME": "spike-tau-<TAU>",
#   })
echo "see header — dispatch via Crucible MCP"
