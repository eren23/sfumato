# G2_baker_cookies

**Prompt:** `Question: A baker makes 240 cookies. He packs them in boxes of 12 each. How many boxes does he need?
Answer:`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.573
- Min cos at token #1 = 0.117 (token: `':'`)

## Top-8 AR-mode features by total activation

- AR feature **#10109**  total_act=360.47  peaks at pos 9 (`' packs'`)
- AR feature **#6191**  total_act=322.00  peaks at pos 13 (`' of'`)
- AR feature **#8435**  total_act=272.97  peaks at pos 22 (`' need'`) ⛓️ BRIDGE
- AR feature **#4978**  total_act=229.56  peaks at pos 20 (`' does'`)
- AR feature **#9465**  total_act=161.03  peaks at pos 16 (`'.'`)
- AR feature **#4467**  total_act=160.31  peaks at pos 18 (`' many'`)
- AR feature **#10879**  total_act=154.43  peaks at pos 1 (`':'`)
- AR feature **#7459**  total_act=150.92  peaks at pos 21 (`' he'`)

## Top-8 diff-mode features by total activation

- diff feature **#6082**  total_act=395.70  peaks at pos 20 (`' does'`)
- diff feature **#5502**  total_act=378.04  peaks at pos 11 (`' in'`)
- diff feature **#6383**  total_act=255.48  peaks at pos 2 (`' A'`)
- diff feature **#14810**  total_act=236.99  peaks at pos 12 (`' boxes'`) ⛓️ BRIDGE
- diff feature **#14195**  total_act=220.39  peaks at pos 3 (`' baker'`)
- diff feature **#11164**  total_act=198.29  peaks at pos 22 (`' need'`)
- diff feature **#4335**  total_act=182.90  peaks at pos 8 (`' He'`)
- diff feature **#5374**  total_act=181.77  peaks at pos 15 (`' each'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [4467, 4978, 6191, 7459, 8435, 9465, 10109, 10879]
- diff-only top features: [4335, 5374, 5502, 6082, 6383, 11164, 14195, 14810]
- shared top features (same index in both top-8): []