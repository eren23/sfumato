# Phase P.0 visualisations

Source: `/Users/eren/Documents/AI/sfumato/e5/interp/cache/p0_dump_f10.pt`

- Layers: 20
- Tokens: 67
- d_model: 1024

## Key numbers

- Final-layer mean cosine(AR, diff) = **0.543**
- Layer-0 cosine = 0.987, monotone decrease through depth
- Norm at last layer: AR=59.0, diff=58.0

Cosine drop from layer 0 → 19 is the pre-SAE preview of the P.2
cross-head overlap question. If modes shared one representation,
we'd see cosine ≈ 1.0 throughout. We observe substantial divergence.
