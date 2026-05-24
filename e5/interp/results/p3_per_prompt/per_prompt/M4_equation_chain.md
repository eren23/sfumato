# M4_equation_chain

**Prompt:** `If a = 5 and b = a + 2 then b = 7. So a + b = 5 + 7 = `


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.779
- Min cos at token #1 = 0.084 (token: `' a'`)

## Top-8 AR-mode features by total activation

- AR feature **#8435**  total_act=514.42  peaks at pos 22 (`' 7'`) ⛓️ BRIDGE
- AR feature **#13235**  total_act=353.95  peaks at pos 18 (`' b'`)
- AR feature **#14998**  total_act=327.61  peaks at pos 8 (`' +'`)
- AR feature **#11950**  total_act=286.76  peaks at pos 9 (`' 2'`)
- AR feature **#10109**  total_act=275.24  peaks at pos 23 (`' ='`)
- AR feature **#2987**  total_act=274.79  peaks at pos 10 (`' then'`)
- AR feature **#261**  total_act=273.20  peaks at pos 17 (`' +'`)
- AR feature **#1523**  total_act=246.37  peaks at pos 20 (`' 5'`)

## Top-8 diff-mode features by total activation

- diff feature **#14810**  total_act=935.76  peaks at pos 18 (`' b'`) ⛓️ BRIDGE
- diff feature **#15540**  total_act=440.66  peaks at pos 20 (`' 5'`)
- diff feature **#16052**  total_act=288.78  peaks at pos 0 (`'If'`)
- diff feature **#6082**  total_act=268.93  peaks at pos 19 (`' ='`)
- diff feature **#1537**  total_act=211.50  peaks at pos 10 (`' then'`)
- diff feature **#5335**  total_act=203.44  peaks at pos 21 (`' +'`)
- diff feature **#5934**  total_act=202.37  peaks at pos 5 (`' b'`)
- diff feature **#12093**  total_act=172.92  peaks at pos 10 (`' then'`)

## Mode-specific features (top-8 membership)

- AR-only top features: [261, 1523, 2987, 8435, 10109, 11950, 13235, 14998]
- diff-only top features: [1537, 5335, 5934, 6082, 12093, 14810, 15540, 16052]
- shared top features (same index in both top-8): []