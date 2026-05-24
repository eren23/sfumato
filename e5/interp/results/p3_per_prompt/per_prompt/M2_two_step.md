# M2_two_step

**Prompt:** `Let x = 3 + 4 * 2. Then x = `


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.817
- Min cos at token #0 = 0.516 (token: `'Let'`)

## Top-8 AR-mode features by total activation

- AR feature **#10109**  total_act=180.22  peaks at pos 11 (`' ='`)
- AR feature **#8865**  total_act=175.45  peaks at pos 7 (`' 2'`)
- AR feature **#1523**  total_act=171.95  peaks at pos 7 (`' 2'`)
- AR feature **#8435**  total_act=161.51  peaks at pos 5 (`' 4'`) ⛓️ BRIDGE
- AR feature **#13235**  total_act=141.00  peaks at pos 10 (`' x'`)
- AR feature **#15969**  total_act=140.02  peaks at pos 9 (`' Then'`)
- AR feature **#2987**  total_act=121.49  peaks at pos 9 (`' Then'`)
- AR feature **#14998**  total_act=119.59  peaks at pos 4 (`' +'`)

## Top-8 diff-mode features by total activation

- diff feature **#14810**  total_act=423.09  peaks at pos 4 (`' +'`) ⛓️ BRIDGE
- diff feature **#15540**  total_act=258.73  peaks at pos 10 (`' x'`)
- diff feature **#6082**  total_act=154.56  peaks at pos 11 (`' ='`)
- diff feature **#16052**  total_act=125.77  peaks at pos 0 (`'Let'`)
- diff feature **#14953**  total_act=123.42  peaks at pos 9 (`' Then'`)
- diff feature **#3342**  total_act=116.38  peaks at pos 9 (`' Then'`)
- diff feature **#13666**  total_act=103.57  peaks at pos 8 (`'.'`)
- diff feature **#9667**  total_act=96.68  peaks at pos 0 (`'Let'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [1523, 2987, 8435, 8865, 10109, 13235, 14998, 15969]
- diff-only top features: [3342, 6082, 9667, 13666, 14810, 14953, 15540, 16052]
- shared top features (same index in both top-8): []