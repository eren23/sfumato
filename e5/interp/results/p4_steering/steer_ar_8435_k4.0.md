# P.4 steering — feature ar/#8435  k=4.0


## Prompt 1
```
Question: Janet's ducks lay 16 eggs per day. How much does she make daily?
Answer:
```


### ar baseline (no steer)
```
 First find the total number of eggs she lays: 16 eggs/day * 16 eggs/day = <<16*16=96>>96 eggs
Then find the total number of eggs she lays: 96 eggs/day * 96 eggs/
```


### ar STEERED (+4.0 × dir)
```
 First find the total number of eggs she lays: 16 eggs/day * 16 eggs/day = <<16*16=96>>96 eggs
Then find the total number of eggs she lays: 96 eggs/day * 96 eggs/
```

_token-equality with baseline: 100%_


### diff baseline (no steer)
```
 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per
```


### diff STEERED (+4.0 × dir)
```
 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks = 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per day 16 ducks per
```

_token-equality with baseline: 100%_


## Prompt 2
```
The Roman Empire under Trajan extended from
```


### ar baseline (no steer)
```
 the Mediterranean to the Mediterranean, and the Roman Empire extended from the Mediterranean to the Mediterranean.
The Roman Empire was the largest empire in the Roman Empire, and the Roman Empire was the largest empire in the Roman Empire. The Roman Empire was
```


### ar STEERED (+4.0 × dir)
```
 the Mediterranean to the Mediterranean, and the Roman Empire extended from the Mediterranean to the Mediterranean.
The Roman Empire was the first of the Roman Empire to be conquered by the Romans. The Romans conquered the Roman Empire in the 4th century BC
```

_token-equality with baseline: 57%_


### diff baseline (no steer)
```
 Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended
```


### diff STEERED (+4.0 × dir)
```
 Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended Roman Empire extended
```

_token-equality with baseline: 100%_


## Prompt 3
```
Let x = 3 + 4 * 2. Then x = 
```


### ar baseline (no steer)
```
 x + 3 * 2.
Now, we have x = 3 * 2 + 4 * 2 = 3.
Now, we have x = 3 * 2 + 4 * 2 = 3.
Now, we have x = 3
```


### ar STEERED (+4.0 × dir)
```
 x + 3 * 2.
Now, we have x = 3 * 2.
Now, we have x = 3 * 2.
Now, we have x = 3 * 2.
Now, we have x = 3 *
```

_token-equality with baseline: 48%_


### diff baseline (no steer)
```
 = 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4 * 4
```


### diff STEERED (+4.0 × dir)
```
 =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
```

_token-equality with baseline: 23%_
