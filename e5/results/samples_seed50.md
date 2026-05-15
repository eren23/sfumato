# Generation samples — 200M seed=50 (overnight checkpoints)

Modes compared per problem:

- `composite[ar_only]` — composite ckpt, greedy AR decode 128 tokens
- `composite[mode_switch]` — AR 96 + diff-revise last 32 (16 steps)
- `composite[paired]` — AR 64 + diff-fill next 64 (16 steps)
- `composite[diff_only]` — pure diff from full-mask suffix (16 steps)
- `ar_only ckpt` — pure-AR-trained model, greedy 128 tokens
- `diff_only ckpt` — pure-diff-trained model, mask-fill 128 tokens


---
## Problem 0

**Question:**
```
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```

**Gold answer:** `18`

**composite[ar_only]** ❌ pred=`30`
```
He spent $2 x $2 = $<<2*2=6>>6 for the first week.
He spent $6 x $2 = $<<6*2=6>>6 for the first two weeks.
He spent $6 x $6 = $<<6*6=24>>24 for the first week.
He spent $15 x $6 = $<<15*6=60>>60 for the first two weeks.
He spent $60 - $15 = $<<60-15=30>>30 on the first two weeks.
#### 30
```

**composite[mode_switch]** ❌ pred=`24`
```
He spent $2 x $2 = $<<2*2=6>>6 for the first week.
He spent $6 x $2 = $<<6*2=6>>6 for the first two weeks.
He spent $6 x $6 = $<<6*6=24>>24 $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
```

**composite[paired]** ❌ pred=`24`
```
He spent $2 x $2 = $<<2*2=6>>6 for the first week.
He spent $6 x $2 = $<<6*2=6>>6 for the first two weeks.
He spent $6 x $6 = $<<6*6=24>>24 $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
```

**composite[diff_only]** ❌ pred=`None`
```
$ $. $ $ $ $ $. $ $ $ $ $ $ $ $. $. $ $ $ $ $ $ $ $ $ $. $ $. $ $... $ $ $ $ $ $ $ $ $ $.. $. $ $. $ $ $ $. $ $ $ $ $
```


---
## Problem 1

**Question:**
```
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
```

**Gold answer:** `3`

**composite[ar_only]** ❌ pred=`None`
```
The total number of p be the first two times the first and the first and the second and the second and the second and the second equation, the first two times as the second and the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, th
```

**composite[mode_switch]** ❌ pred=`None`
```
The total number of p be the first two times the first and the first and the second and the second and the second and the second equation, the first two times as the second and the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation the the the the the the the the the the the the the the the the the the the
```

**composite[paired]** ❌ pred=`None`
```
The total number of p be the first two times the first and the first and the second and the second and the second and the second equation, the first two times as the second and the second equation, the second equation, the second equation, the second equation, the second equation, the second equation, the second equation the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

**composite[diff_only]** ❌ pred=`None`
```
..................................................................
```


---
## Problem 2

**Question:**
```
Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
```

**Gold answer:** `70000`

**composite[ar_only]** ❌ pred=`100`
```
He has $100,000/100*$<<400*100=400>>400
He has $200,000/100*$200=$<<200/100*100=100>>100,000
So he pays $100,000*$100=$<<100000*100=400>>400
#### 100
```

**composite[mode_switch]** ❌ pred=`100000000000000000000000000000000000000000000000000000000000`
```
He has $100,000/100*$<<400*100=400>>400
He has $200,000/100*$200=$<<200/100000000000000000000000000000000000000000000000000000000000
```

**composite[paired]** ❌ pred=`400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000`
```
He has $100,000/100*$<<400*100=400>>400
He has $200,000/100*$200=$<<200/100*100=100>>100,000
So he pays $100,000*$100=$<<100000*100=400>>400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

**composite[diff_only]** ❌ pred=`None`
```
$ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
```


---
## Problem 0

**Question:**
```
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```

**Gold answer:** `18`

**ar_only[ar_ckpt_greedy]** ❌ pred=`2`
```
She spent $2.5 x 2 = $<<2.5*2=4>>4.
She spent $2 x 2 = $<<2*2=4>>4.
She spent $2 x 2 = $<<2*2=4>>4.
She spent $4 x 2 = $<<4*2=8>>8 for the week.
She spent $4 x 2 = $<<4*2=8>>8 for the week.
She spent $4 x 2 = $<<4*2=8>>8 for the week.
She spent $4 x 2 =
```

**ar_only[ar_ckpt_greedy]** ❌ pred=`None`
```
The total number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number of blue and the number
```

**ar_only[ar_ckpt_greedy]** ❌ pred=`1000`
```
He pays $200,000*200=$<<500000*200=500000>>500,000
So he pays $500,000+$500=$<<1000+500=3000>>1000
#### 1000
```


---
## Problem 0

**Question:**
```
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```

**Gold answer:** `18`

**diff_only[diff_ckpt]** ❌ pred=`None`
```
..................................................................
```

**diff_only[diff_ckpt]** ❌ pred=`None`
```
..................................................................
```

**diff_only[diff_ckpt]** ❌ pred=`None`
```
..................................................................
```
