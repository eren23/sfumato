# M3_word_to_math

**Prompt:** `John has 15 marbles. He gives Mary 4 and Tom 3. He has 15 - 4 - 3 = `


## Mode-divergence (cosine of residual at each token)

- Mean cos(AR, diff) = 0.753
- Min cos at token #1 = 0.459 (token: `' has'`)

## Top-8 AR-mode features by total activation

- AR feature **#10109**  total_act=344.87  peaks at pos 15 (`' has'`)
- AR feature **#10879**  total_act=234.58  peaks at pos 10 (`' and'`)
- AR feature **#8435**  total_act=215.28  peaks at pos 4 (`'bles'`) ⛓️ BRIDGE
- AR feature **#7298**  total_act=154.61  peaks at pos 10 (`' and'`)
- AR feature **#10020**  total_act=129.13  peaks at pos 6 (`' He'`)
- AR feature **#5492**  total_act=127.00  peaks at pos 14 (`' He'`)
- AR feature **#9465**  total_act=122.24  peaks at pos 5 (`'.'`)
- AR feature **#7355**  total_act=122.08  peaks at pos 15 (`' has'`)

## Top-8 diff-mode features by total activation

- diff feature **#6082**  total_act=317.99  peaks at pos 15 (`' has'`)
- diff feature **#6836**  total_act=305.90  peaks at pos 21 (`' ='`)
- diff feature **#4335**  total_act=276.00  peaks at pos 14 (`' He'`)
- diff feature **#14810**  total_act=219.22  peaks at pos 4 (`'bles'`) ⛓️ BRIDGE
- diff feature **#9667**  total_act=213.40  peaks at pos 21 (`' ='`)
- diff feature **#14467**  total_act=204.05  peaks at pos 8 (`' Mary'`)
- diff feature **#6629**  total_act=203.77  peaks at pos 16 (`' 15'`)
- diff feature **#9571**  total_act=160.07  peaks at pos 21 (`' ='`)

## Mode-specific features (top-8 membership)

- AR-only top features: [5492, 7298, 7355, 8435, 9465, 10020, 10109, 10879]
- diff-only top features: [4335, 6082, 6629, 6836, 9571, 9667, 14467, 14810]
- shared top features (same index in both top-8): []