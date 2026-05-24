# G4_age_problem

**Prompt:** `Question: Alice is 12 years old. Her brother is 4 years younger. In 5 years, how old will Alice's brother be?
Answer:`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.611
- Min cos at token #3 = 0.303 (token: `' is'`)

## Top-8 AR-mode features by total activation

- AR feature **#10879**  total_act=419.33  peaks at pos 9 (`' brother'`)
- AR feature **#7298**  total_act=333.28  peaks at pos 21 (`' will'`)
- AR feature **#8435**  total_act=316.12  peaks at pos 6 (`' old'`) ⛓️ BRIDGE
- AR feature **#10109**  total_act=235.61  peaks at pos 10 (`' is'`)
- AR feature **#2641**  total_act=183.56  peaks at pos 1 (`':'`)
- AR feature **#15969**  total_act=175.40  peaks at pos 27 (`'\n'`)
- AR feature **#5529**  total_act=172.46  peaks at pos 23 (`"'s"`)
- AR feature **#9465**  total_act=135.14  peaks at pos 14 (`'.'`)

## Top-8 diff-mode features by total activation

- diff feature **#14810**  total_act=576.06  peaks at pos 6 (`' old'`) ⛓️ BRIDGE
- diff feature **#6082**  total_act=424.98  peaks at pos 3 (`' is'`)
- diff feature **#10390**  total_act=374.84  peaks at pos 23 (`"'s"`)
- diff feature **#9667**  total_act=366.48  peaks at pos 29 (`':'`)
- diff feature **#6836**  total_act=279.31  peaks at pos 9 (`' brother'`)
- diff feature **#2321**  total_act=247.23  peaks at pos 18 (`','`)
- diff feature **#7341**  total_act=198.57  peaks at pos 29 (`':'`)
- diff feature **#12128**  total_act=188.56  peaks at pos 5 (`' years'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [2641, 5529, 7298, 8435, 9465, 10109, 10879, 15969]
- diff-only top features: [2321, 6082, 6836, 7341, 9667, 10390, 12128, 14810]
- shared top features (same index in both top-8): []