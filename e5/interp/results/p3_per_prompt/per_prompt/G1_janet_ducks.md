# G1_janet_ducks

**Prompt:** `Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How much does she make daily?
Answer:`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.568
- Min cos at token #1 = 0.060 (token: `':'`)

## Top-8 AR-mode features by total activation

- AR feature **#8435**  total_act=515.47  peaks at pos 9 (`' day'`) ⛓️ BRIDGE
- AR feature **#7298**  total_act=442.76  peaks at pos 35 (`' does'`)
- AR feature **#10109**  total_act=424.70  peaks at pos 12 (`' eats'`)
- AR feature **#12187**  total_act=252.59  peaks at pos 33 (`' How'`)
- AR feature **#9304**  total_act=249.93  peaks at pos 6 (`' 16'`)
- AR feature **#13452**  total_act=232.77  peaks at pos 14 (`' for'`)
- AR feature **#6191**  total_act=207.76  peaks at pos 25 (`' sells'`)
- AR feature **#7451**  total_act=196.70  peaks at pos 8 (`' per'`)

## Top-8 diff-mode features by total activation

- diff feature **#6082**  total_act=615.02  peaks at pos 37 (`' make'`)
- diff feature **#14787**  total_act=482.55  peaks at pos 34 (`' much'`)
- diff feature **#14810**  total_act=445.28  peaks at pos 7 (`' eggs'`) ⛓️ BRIDGE
- diff feature **#9667**  total_act=438.04  peaks at pos 39 (`'?'`)
- diff feature **#14386**  total_act=434.84  peaks at pos 21 (`' with'`)
- diff feature **#1755**  total_act=426.74  peaks at pos 26 (`' the'`)
- diff feature **#4335**  total_act=326.89  peaks at pos 16 (`' and'`)
- diff feature **#15540**  total_act=287.71  peaks at pos 9 (`' day'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [6191, 7298, 7451, 8435, 9304, 10109, 12187, 13452]
- diff-only top features: [1755, 4335, 6082, 9667, 14386, 14787, 14810, 15540]
- shared top features (same index in both top-8): []