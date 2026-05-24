# G3_apples_division

**Prompt:** `Question: There are 96 apples to share among 8 children equally. How many apples does each child get?
Answer:`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.602
- Min cos at token #1 = 0.184 (token: `':'`)

## Top-8 AR-mode features by total activation

- AR feature **#10109**  total_act=292.74  peaks at pos 3 (`' are'`)
- AR feature **#8435**  total_act=282.96  peaks at pos 10 (`' children'`) ⛓️ BRIDGE
- AR feature **#10879**  total_act=205.39  peaks at pos 1 (`':'`)
- AR feature **#15969**  total_act=154.15  peaks at pos 20 (`'?'`)
- AR feature **#2641**  total_act=138.07  peaks at pos 1 (`':'`)
- AR feature **#9465**  total_act=125.43  peaks at pos 12 (`'.'`)
- AR feature **#8967**  total_act=119.06  peaks at pos 17 (`' each'`) ⛓️ BRIDGE
- AR feature **#5419**  total_act=106.69  peaks at pos 11 (`' equally'`)

## Top-8 diff-mode features by total activation

- diff feature **#6082**  total_act=375.44  peaks at pos 16 (`' does'`)
- diff feature **#10390**  total_act=319.88  peaks at pos 8 (`' among'`)
- diff feature **#3233**  total_act=268.58  peaks at pos 7 (`' share'`)
- diff feature **#761**  total_act=261.84  peaks at pos 9 (`' 8'`) ⛓️ BRIDGE
- diff feature **#14810**  total_act=253.82  peaks at pos 22 (`'Answer'`) ⛓️ BRIDGE
- diff feature **#11164**  total_act=235.81  peaks at pos 23 (`':'`)
- diff feature **#15540**  total_act=223.36  peaks at pos 5 (`' apples'`)
- diff feature **#5335**  total_act=204.01  peaks at pos 22 (`'Answer'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [2641, 5419, 8435, 8967, 9465, 10109, 10879, 15969]
- diff-only top features: [761, 3233, 5335, 6082, 10390, 11164, 14810, 15540]
- shared top features (same index in both top-8): []