# F3_geography

**Prompt:** `The Amazon rainforest, often referred to as the lungs of the planet, produces more than 20 percent of the world's oxygen supply through its dense vegetation.`


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.466
- Min cos at token #0 = 0.165 (token: `'The'`)

## Top-8 AR-mode features by total activation

- AR feature **#3267**  total_act=475.56  peaks at pos 8 (`' as'`)
- AR feature **#8435**  total_act=455.75  peaks at pos 25 (`' supply'`) ⛓️ BRIDGE
- AR feature **#13892**  total_act=256.34  peaks at pos 28 (`' dense'`)
- AR feature **#11383**  total_act=217.79  peaks at pos 9 (`' the'`)
- AR feature **#5492**  total_act=153.19  peaks at pos 14 (`','`)
- AR feature **#3330**  total_act=143.86  peaks at pos 16 (`' more'`)
- AR feature **#16322**  total_act=130.81  peaks at pos 9 (`' the'`)
- AR feature **#16292**  total_act=116.73  peaks at pos 23 (`"'s"`)

## Top-8 diff-mode features by total activation

- diff feature **#7508**  total_act=761.68  peaks at pos 28 (`' dense'`)
- diff feature **#14810**  total_act=461.02  peaks at pos 10 (`' lungs'`) ⛓️ BRIDGE
- diff feature **#15890**  total_act=415.81  peaks at pos 15 (`' produces'`)
- diff feature **#4335**  total_act=397.88  peaks at pos 5 (`' often'`)
- diff feature **#3068**  total_act=397.87  peaks at pos 18 (`' 20'`) ⛓️ BRIDGE
- diff feature **#10082**  total_act=359.56  peaks at pos 2 (`' rain'`)
- diff feature **#10682**  total_act=352.27  peaks at pos 26 (`' through'`)
- diff feature **#3447**  total_act=289.95  peaks at pos 7 (`' to'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [3267, 3330, 5492, 8435, 11383, 13892, 16292, 16322]
- diff-only top features: [3068, 3447, 4335, 7508, 10082, 10682, 14810, 15890]
- shared top features (same index in both top-8): []