# M1_simple_arith

**Prompt:** `The total cost is 50 - 12 + 7 = `


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.439
- Min cos at token #0 = 0.175 (token: `'The'`)

## Top-8 AR-mode features by total activation

- AR feature **#10109**  total_act=201.00  peaks at pos 5 (`' -'`)
- AR feature **#12187**  total_act=187.25  peaks at pos 9 (`' ='`)
- AR feature **#14370**  total_act=162.72  peaks at pos 9 (`' ='`)
- AR feature **#3215**  total_act=158.70  peaks at pos 8 (`' 7'`)
- AR feature **#2462**  total_act=93.32  peaks at pos 3 (`' is'`)
- AR feature **#8865**  total_act=83.04  peaks at pos 4 (`' 50'`)
- AR feature **#8435**  total_act=79.54  peaks at pos 2 (`' cost'`) ⛓️ BRIDGE
- AR feature **#12159**  total_act=58.65  peaks at pos 1 (`' total'`)

## Top-8 diff-mode features by total activation

- diff feature **#7133**  total_act=202.36  peaks at pos 9 (`' ='`)
- diff feature **#1755**  total_act=165.05  peaks at pos 3 (`' is'`)
- diff feature **#6082**  total_act=148.14  peaks at pos 7 (`' +'`)
- diff feature **#1656**  total_act=127.76  peaks at pos 0 (`'The'`)
- diff feature **#16052**  total_act=123.48  peaks at pos 1 (`' total'`)
- diff feature **#14810**  total_act=121.82  peaks at pos 2 (`' cost'`) ⛓️ BRIDGE
- diff feature **#11164**  total_act=104.90  peaks at pos 7 (`' +'`)
- diff feature **#7232**  total_act=98.03  peaks at pos 3 (`' is'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [2462, 3215, 8435, 8865, 10109, 12159, 12187, 14370]
- diff-only top features: [1656, 1755, 6082, 7133, 7232, 11164, 14810, 16052]
- shared top features (same index in both top-8): []