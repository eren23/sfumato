# F2_science

**Prompt:** `Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.705
- Min cos at token #21 = 0.404 (token: `' help'`)

## Top-8 AR-mode features by total activation

- AR feature **#15733**  total_act=524.55  peaks at pos 3 (`' the'`)
- AR feature **#8435**  total_act=310.30  peaks at pos 18 (`' foods'`) ⛓️ BRIDGE
- AR feature **#9331**  total_act=283.49  peaks at pos 10 (`' some'`)
- AR feature **#14322**  total_act=218.92  peaks at pos 15 (`' to'`)
- AR feature **#6190**  total_act=217.71  peaks at pos 13 (`' use'`)
- AR feature **#6081**  total_act=205.55  peaks at pos 9 (`' and'`)
- AR feature **#11701**  total_act=191.29  peaks at pos 19 (`' with'`)
- AR feature **#9904**  total_act=153.37  peaks at pos 17 (`'ize'`)

## Top-8 diff-mode features by total activation

- diff feature **#10682**  total_act=734.88  peaks at pos 5 (`' by'`)
- diff feature **#12089**  total_act=428.13  peaks at pos 22 (`' of'`)
- diff feature **#14810**  total_act=387.65  peaks at pos 18 (`' foods'`) ⛓️ BRIDGE
- diff feature **#4000**  total_act=373.76  peaks at pos 19 (`' with'`)
- diff feature **#2417**  total_act=370.47  peaks at pos 20 (`' the'`)
- diff feature **#4031**  total_act=337.74  peaks at pos 10 (`' some'`)
- diff feature **#781**  total_act=303.51  peaks at pos 7 (`' green'`)
- diff feature **#2321**  total_act=284.15  peaks at pos 19 (`' with'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [6081, 6190, 8435, 9331, 9904, 11701, 14322, 15733]
- diff-only top features: [781, 2321, 2417, 4000, 4031, 10682, 12089, 14810]
- shared top features (same index in both top-8): []