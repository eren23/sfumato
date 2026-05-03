"""Sfumato Phase 2 unified palette — imported by every figure.

Locked design (see Workstream A plan paragraph in phase2/STATUS.md):

  - Cool ramp encodes the base → v2 → v3 progression. Inherited from the
    paper draft for visual continuity.
  - Warm accent (amber-700) highlights failures / inversions.
  - Emerald accent marks targets / ceilings.

Every colour is a Tailwind v3 hex so the palette also reads natively in
the writeup CSS / blog markdown.

Colour-pair separation is enforced via `coloraide` at audit time
(``audit_palette()`` returns the worst Delta-E2000 across all pairs). The
audit also simulates protan / deutan / tritan colour-vision deficiencies
and asserts the simulated palette stays separable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    # Cool ramp (cardinal channel: base → v2 → v3 progression)
    base: str = "#9ca3af"   # gray-400
    v2:   str = "#60a5fa"   # blue-400
    v3:   str = "#1d4ed8"   # blue-700

    # Warm accent (failures / inversions / regressions)
    warn: str = "#b45309"   # amber-700
    warn_soft: str = "#fde68a"  # amber-200 (background fills only)

    # Emerald accent (targets / ceilings / "good" reference lines)
    ok: str = "#065f46"   # emerald-800

    # Neutrals
    ink: str = "#111827"  # gray-900 — primary text
    sub: str = "#374151"  # gray-700 — secondary text
    rule: str = "#e5e7eb"  # gray-200 — light rules / fills

    @property
    def cool_ramp(self) -> tuple[str, str, str]:
        return (self.base, self.v2, self.v3)

    @property
    def all(self) -> tuple[str, ...]:
        return (self.base, self.v2, self.v3, self.warn, self.ok)


PALETTE = Palette()


def audit_palette(min_delta_e: float = 9.0) -> dict:
    """Run a Delta-E2000 separation check + colour-vision-deficiency
    simulation over the locked palette.

    Threshold 9.0 is chosen to be ~4× the JND (≈2.3) and clears every pair
    under normal, protan, and deutan vision. Tritan (≈0.01% prevalence)
    floors at ~9.5 between v3 (deep blue) and ok (deep emerald); these two
    colours are never adjacent fills in the figures — ``ok`` is reserved
    for thin reference lines (cmaj ceiling, pre-reg target), ``v3`` for
    bar fills, so the in-figure adjacency that would matter for tritan
    doesn't occur.

    Returns a dict with:
      ``ok``           bool — every pair clears ``min_delta_e`` (and CVD
                       simulations pass too).
      ``min_pair``     (a, b) — the closest colour pair under normal vision.
      ``min_dE``       float — its Delta-E2000.
      ``cvd``          dict — per-CVD-mode min Delta-E.
      ``details``      list[(a, b, dE_normal, dE_protan, dE_deutan, dE_tritan)]

    Raises ImportError if ``coloraide`` is missing — install with::

        pip install coloraide
    """
    try:
        from coloraide import Color
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "coloraide is required for the colourblind audit. "
            "Run: pip install coloraide"
        ) from e

    # CVD simulation method names supported by coloraide ≥1.x.
    cvd_modes = ("protan", "deutan", "tritan")

    cols = list(PALETTE.all)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    details = []
    cvd_min: dict[str, float] = {m: float("inf") for m in cvd_modes}
    cvd_min_pair: dict[str, tuple[str, str]] = {m: ("", "") for m in cvd_modes}
    overall_min = float("inf")
    overall_min_pair = ("", "")

    for a, b in pairs:
        ca = Color(a)
        cb = Color(b)
        dE = ca.delta_e(cb, method="2000")
        if dE < overall_min:
            overall_min = dE
            overall_min_pair = (a, b)

        per_cvd = {}
        for mode in cvd_modes:
            ca_s = Color(a).filter("brightness", 1).filter(mode)
            cb_s = Color(b).filter("brightness", 1).filter(mode)
            dE_s = ca_s.delta_e(cb_s, method="2000")
            per_cvd[mode] = dE_s
            if dE_s < cvd_min[mode]:
                cvd_min[mode] = dE_s
                cvd_min_pair[mode] = (a, b)
        details.append((a, b, dE, per_cvd["protan"], per_cvd["deutan"], per_cvd["tritan"]))

    ok_normal = overall_min >= min_delta_e
    ok_cvd = all(v >= min_delta_e for v in cvd_min.values())

    return {
        "ok": ok_normal and ok_cvd,
        "min_pair": overall_min_pair,
        "min_dE": overall_min,
        "cvd": {
            mode: {"min_dE": cvd_min[mode], "pair": cvd_min_pair[mode]}
            for mode in cvd_modes
        },
        "details": details,
        "threshold": min_delta_e,
    }


if __name__ == "__main__":
    import json
    res = audit_palette()
    print(json.dumps(res, indent=2, default=str))
