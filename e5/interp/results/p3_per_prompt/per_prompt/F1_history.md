# F1_history

**Prompt:** `The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD, stretching from Britain in the north to Mesopotamia in the east.`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.491
- Min cos at token #0 = 0.229 (token: `'The'`)

## Top-8 AR-mode features by total activation

- AR feature **#8435**  total_act=488.09  peaks at pos 28 (`' east'`) ⛓️ BRIDGE
- AR feature **#14886**  total_act=464.88  peaks at pos 9 (`' Emperor'`)
- AR feature **#993**  total_act=243.25  peaks at pos 27 (`' the'`)
- AR feature **#4690**  total_act=191.51  peaks at pos 17 (`' from'`)
- AR feature **#15894**  total_act=171.85  peaks at pos 22 (`' to'`) ⛓️ BRIDGE
- AR feature **#10109**  total_act=148.11  peaks at pos 12 (`' in'`)
- AR feature **#4443**  total_act=119.79  peaks at pos 9 (`' Emperor'`)
- AR feature **#10502**  total_act=112.95  peaks at pos 19 (`' in'`)

## Top-8 diff-mode features by total activation

- diff feature **#14802**  total_act=836.84  peaks at pos 9 (`' Emperor'`)
- diff feature **#7380**  total_act=738.04  peaks at pos 7 (`' extent'`)
- diff feature **#13827**  total_act=623.08  peaks at pos 15 (`','`)
- diff feature **#10449**  total_act=563.98  peaks at pos 17 (`' from'`) ⛓️ BRIDGE
- diff feature **#827**  total_act=563.63  peaks at pos 0 (`'The'`)
- diff feature **#3372**  total_act=427.32  peaks at pos 3 (`' reached'`)
- diff feature **#4018**  total_act=406.52  peaks at pos 16 (`' stretching'`)
- diff feature **#1691**  total_act=376.21  peaks at pos 11 (`'jan'`) ⛓️ BRIDGE

## Mode-specific features (top-8 membership)

- AR-only top features: [993, 4443, 4690, 8435, 10109, 10502, 14886, 15894]
- diff-only top features: [827, 1691, 3372, 4018, 7380, 10449, 13827, 14802]
- shared top features (same index in both top-8): []