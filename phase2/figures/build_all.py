"""Build harness for Workstream A — regenerates all 5 Phase 2 figures
from source, asserts every output file exists, and runs the colourblind
audit on the unified palette.

Wired to ``make figures`` via the repo-root Makefile.

Usage:
  python phase2/figures/build_all.py               # build + audit
  python phase2/figures/build_all.py --no-audit    # skip CB audit
  python phase2/figures/build_all.py --audit-only  # just CB audit

Exits non-zero if any expected output file is missing or the colourblind
audit fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# Per-figure module + expected outputs
# ---------------------------------------------------------------------------
TARGETS = [
    ("fig1_prefix_hierarchy",   ["fig1_prefix_hierarchy.pdf",
                                 "fig1_prefix_hierarchy.png"]),
    ("fig2_branch_agreement",   ["fig2_branch_agreement.pdf",
                                 "fig2_branch_agreement.png"]),
    ("fig3_c2c_disentangling",  ["fig3_c2c_disentangling.pdf",
                                 "fig3_c2c_disentangling.png",
                                 "fig3_c2c_disentangling.html"]),
    ("fig4_compositionality",   ["fig4_compositionality.pdf",
                                 "fig4_compositionality.png"]),
    ("fig5_block_diagram",      ["fig5_block_diagram.excalidraw",
                                 "fig5_block_diagram.svg",
                                 "fig5_block_diagram.png"]),
]


def _import_module(name: str):
    """Import ``phase2/figures/<name>.py`` as a fresh module (so each call
    to main() in build_all gets a clean state)."""
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"phase2_figures_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to spec module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build() -> list[str]:
    """Run every figure script. Returns the list of relative paths
    written; raises on first failure."""
    written: list[str] = []
    for name, outputs in TARGETS:
        print(f"==> {name}")
        t0 = time.time()
        mod = _import_module(name)
        rc = mod.main()
        dt = time.time() - t0
        if rc != 0:
            raise RuntimeError(f"{name}.main() returned {rc}")
        # Verify outputs exist + are non-empty
        for out in outputs:
            p = HERE / out
            if not p.exists():
                raise FileNotFoundError(f"expected {p} after running {name} but it is missing")
            if p.stat().st_size == 0:
                raise RuntimeError(f"{p} exists but is empty")
            written.append(str(p.relative_to(REPO_ROOT)))
        print(f"    [{dt:.2f}s] ok ({len(outputs)} files)")
    return written


def run_audit() -> int:
    """Invoke the palette colourblind audit; return 0 on pass, 1 on fail."""
    from palette import audit_palette, PALETTE  # type: ignore

    print("==> palette colourblind audit")
    res = audit_palette()
    print(f"    palette: base={PALETTE.base}  v2={PALETTE.v2}  v3={PALETTE.v3}  "
          f"warn={PALETTE.warn}  ok={PALETTE.ok}")
    print(f"    ΔE2000 threshold = {res['threshold']}")
    print(f"    normal-vision min ΔE = {res['min_dE']:.2f}  "
          f"(pair {res['min_pair'][0]} vs {res['min_pair'][1]})")
    for mode, info in res["cvd"].items():
        flag = "ok " if info["min_dE"] >= res["threshold"] else "FAIL"
        print(f"    {flag}  {mode:>6} min ΔE = {info['min_dE']:.2f}  "
              f"(pair {info['pair'][0]} vs {info['pair'][1]})")
    if not res["ok"]:
        print("    AUDIT FAILED — palette is not colourblind-safe at threshold.")
        return 1
    print("    AUDIT PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-audit", action="store_true",
                        help="Skip the colourblind audit step.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Only run the colourblind audit; skip rebuilds.")
    args = parser.parse_args()

    t0 = time.time()
    if not args.audit_only:
        try:
            written = build()
        except Exception as e:
            print(f"BUILD FAILED: {e}", file=sys.stderr)
            return 2
        print()
        print(f"Wrote {len(written)} files in {time.time() - t0:.2f}s:")
        for p in written:
            print(f"  {p}")
        print()

    if args.no_audit:
        print("(audit skipped via --no-audit)")
        return 0

    rc = run_audit()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
