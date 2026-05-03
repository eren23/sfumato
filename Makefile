.PHONY: figures viz phase2-status help

PYTHON ?= .venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
PYTHON := $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
endif
ifeq ($(strip $(PYTHON)),)
$(error No Python interpreter found. Set PYTHON=... or create .venv/bin/python)
endif

help:
	@echo "Phase 2 targets:"
	@echo "  make figures       Regenerate all phase2/figures/*.{pdf,png,html,svg}"
	@echo "  make viz           Launch step-by-step inference visualizer (Gradio)"
	@echo "  make phase2-status Print phase2/STATUS.md"

figures:
	@$(PYTHON) phase2/figures/build_all.py

viz:
	@$(PYTHON) phase2/inference_viz/launch.py

phase2-status:
	@cat phase2/STATUS.md
