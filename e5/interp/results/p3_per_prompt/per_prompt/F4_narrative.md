# F4_narrative

**Prompt:** `She closed the book slowly, as if the story were a guest she did not want to send away, and looked out at the rain blurring the garden lights.`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.603
- Min cos at token #15 = 0.415 (token: `' not'`)

## Top-8 AR-mode features by total activation

- AR feature **#8435**  total_act=571.88  peaks at pos 4 (`' slowly'`) ⛓️ BRIDGE
- AR feature **#7298**  total_act=338.34  peaks at pos 24 (`' at'`)
- AR feature **#5419**  total_act=262.05  peaks at pos 26 (`' rain'`)
- AR feature **#12768**  total_act=195.55  peaks at pos 29 (`' the'`)
- AR feature **#9026**  total_act=195.00  peaks at pos 5 (`','`)
- AR feature **#11680**  total_act=176.46  peaks at pos 24 (`' at'`)
- AR feature **#3402**  total_act=119.57  peaks at pos 11 (`' a'`)
- AR feature **#9610**  total_act=113.34  peaks at pos 4 (`' slowly'`)

## Top-8 diff-mode features by total activation

- diff feature **#14810**  total_act=624.44  peaks at pos 19 (`' away'`) ⛓️ BRIDGE
- diff feature **#12987**  total_act=537.07  peaks at pos 15 (`' not'`)
- diff feature **#10877**  total_act=437.70  peaks at pos 18 (`' send'`)
- diff feature **#14177**  total_act=409.35  peaks at pos 13 (`' she'`)
- diff feature **#2417**  total_act=361.37  peaks at pos 30 (`' garden'`)
- diff feature **#10390**  total_act=345.39  peaks at pos 2 (`' the'`)
- diff feature **#2073**  total_act=325.62  peaks at pos 16 (`' want'`)
- diff feature **#10193**  total_act=314.28  peaks at pos 1 (`' closed'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [3402, 5419, 7298, 8435, 9026, 9610, 11680, 12768]
- diff-only top features: [2073, 2417, 10193, 10390, 10877, 12987, 14177, 14810]
- shared top features (same index in both top-8): []