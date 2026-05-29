# P.2 — Cross-head SAE feature overlap (HEADLINE result)

## Setup

- Model: F10 (305M composite, d_model=1024, n_layers=20)
- Two TopK SAEs (k=64, d_features=16384, 8k training steps each):
  - **AR head SAE** trained on activations at `ln_f` with `mode=ar`
  - **diff head SAE** trained on `ln_f` with `mode=diff`
- Both SAEs see the **same residual-stream position** (post-ln_f) on
  **the same 200M-token FineWeb-Edu corpus**. The only difference is
  which forward path (causal vs bidirectional attention) produced
  the activations.

## Headline number

**Mean best-cosine between AR features and diff features = 0.169.**

Per the pre-committed brief hypotheses:
- >0.8 → "one shared representation, two readouts"
- 0.4-0.8 → "partial sharing"
- **<0.4 → "modes specialise" ← we land here, strongly**

98.2% of AR features have NO close (cos > 0.5) sibling in the diff
SAE; only 0.42% have a sibling above 0.7.

## Detailed numbers

| Cosine threshold | % of AR features with diff sibling | % of diff features with AR sibling |
|---|---|---|
| ≥ 0.5 | **1.84%** | 1.59% |
| ≥ 0.7 | 0.42% | 0.41% |
| ≥ 0.9 | 0.01% (2 features) | 0.01% |

| Best-cosine distribution | AR→diff | diff→AR |
|---|---|---|
| mean | 0.169 | 0.140 |
| median | 0.129 | — |
| max | 0.904 | — |
| min | 0.102 | — |

## Interpretation

The composite's two heads use **substantially specialised (largely
distinct) feature directions inside the shared backbone**. (See
`sae_null_baseline.md` for the control: cross-mode best-cosine 0.169
vs same-mode different-seed 0.286 vs chance floor 0.124 — cross-mode
retains only ~28% of the above-chance alignment. "Disjoint"/
"orthogonal" overstates it; cross-mode is above chance, so a small
shared "bridge" subspace exists.) Even though both heads operate on
the same backbone weights, the residual-stream representations at the
final pre-head layer factorise into largely distinct sparse-feature
bases under each attention regime.

Reconciling with the Phase P.0 raw-activation cosine (0.54 at the
final layer):
- The 0.54 was on dense 1024-dim activation vectors, dominated by
  shared token-and-position embedding contributions.
- The 0.169 here is on *learned sparse feature directions* —
  the decomposed, interpretable structure. At that level the modes
  almost don't share.

## Implications for paper C

The paper's central claim is "AR axis tax, diff axis gain — composite
is a measurable trade-off, not a no-op." P.2 strengthens this:
**the modes don't even use the same internal feature vocabulary**.
This is mechanistically why both axes get measured costs/benefits
that don't cancel.

A weaker version of the claim could be "composite is just a single
representation with two readouts" — P.2 falsifies this directly at
the feature-direction level.

## What this opens

- **Distillation candidate**: if AR and diff feature bases are
  substantially specialised, F10 backbone is probably effectively *two
  half-rank subnetworks*. A distil into AR-only + diff-only models
  with shared embeddings might recover ~all the capability at ~half
  the params. Future paper D.
- **Phase K mode-switching circuit**: the K.2 recipe routes between
  AR and diff at the token level. The cross-head matched pairs
  (top-20, all with cosine 0.85-0.90) are the candidate "bridge
  features" that allow routing without representation breakage.
  Worth investigating in Phase P.3 attribution graphs.

## Artefacts

- JSON: `e5/interp/results/cross_head_overlap.json`
- HF Hub: `eren23/sfumato-composite-ckpts/interp/results/cross_head_overlap.json`
- Source SAEs: HF Hub `interp/saes/ln_f_ar/sae.pt`, `interp/saes/ln_f_diff/sae.pt`
