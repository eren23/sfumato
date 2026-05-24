# P.3 — per-prompt cross-head feature analysis

12 prompts × 2 modes × ln_f SAEs (k=64, d_features=16384). Output: e5/interp/results/p3_per_prompt/per_prompt/*

## Summary table

| Prompt | T | mean cos(AR,diff) | min cos | bridge AR top-N | bridge diff top-N | top-N shared indices |
|---|---|---|---|---|---|---|
| G1_janet_ducks | 43 | 0.568 | 0.060 | 1 | 1 | 0 |
| G2_baker_cookies | 27 | 0.573 | 0.117 | 1 | 1 | 0 |
| G3_apples_division | 24 | 0.602 | 0.184 | 2 | 2 | 0 |
| G4_age_problem | 30 | 0.611 | 0.303 | 1 | 1 | 0 |
| F1_history | 30 | 0.491 | 0.229 | 2 | 2 | 0 |
| F2_science | 27 | 0.705 | 0.404 | 1 | 1 | 0 |
| F3_geography | 31 | 0.466 | 0.165 | 1 | 2 | 0 |
| F4_narrative | 33 | 0.603 | 0.415 | 1 | 1 | 0 |
| M1_simple_arith | 11 | 0.439 | 0.175 | 1 | 1 | 0 |
| M2_two_step | 13 | 0.817 | 0.516 | 1 | 1 | 0 |
| M3_word_to_math | 23 | 0.753 | 0.459 | 1 | 1 | 0 |
| M4_equation_chain | 25 | 0.779 | 0.084 | 1 | 1 | 0 |

## Reading
- **bridge AR/diff top-N**: how many of P.2's 20 high-cosine matched pairs appear in the top-N firing features for this prompt. Bridge features are candidate "mode-switch routing tokens".
- **top-N shared indices**: how many feature INDICES coincide in AR's top-N and diff's top-N. Low values reinforce P.2's "modes specialise" finding; high values would say "same features fire just at different magnitudes".
- **min cos**: token position where AR and diff residual diverge the most. Often a math operator or quantity token.