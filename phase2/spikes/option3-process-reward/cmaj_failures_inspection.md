# cmaj-failures inspection (recoverable losses only)

Total problems in N=500 substrate: **211**
cmaj fails AND oracle could recover: **15** problems

These are the only problems where any verifier could matter.
Read 10 of them below — is the right branch distinguishable from the wrong ones?

---

## problem_id=1071  gold=`251`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'371': 3, '251': 2}` | cmaj_pick=`371` (3/5) | correct branches: **2/5**

### branch_0 ❌  extracted=`371`

```
Kayla invited 25 + 7 + 13 = 45 guests.
Of these, 45 - 4 = 41 guests will come.
The fee to rent the theater is $125 for a party of 20, plus $6 for each additional guest.
So, the cost of the party is $125 + (41 * $6) = $125 + $246 = $371.
Answer: 371
```

### branch_1 ✅  extracted=`251`

```
Kayla invited 25 + 7 + 13 = 45 guests.
There, 45 - 4 = 41 guests will attend the party.

The fee to rent the theater is $125 for a party of 20 guests plus $6 for each additional guest.
So, the total cost for the additional guests is (41 - 20) * $6 = $126.
The total cost for the party is $125 + $126 = $251.

Answer: 251
```

### branch_2 ✅  extracted=`251`

```
Kayla invited 25 + 7 + 13 = 45 guests.
She paid 45 guests - 4-show = 41 guests.
The fee to rent the theater is $125 for a party of 20, so there are 41 - 20 = 21 additional guests.
Each additional guest costs $6, so 21 guests cost 21 * $6 = $126.
The total cost is $125 + $126 = $251.
Answer: 251
```

### branch_3 ❌  extracted=`371`

```
Kayla invited 25 + 7 + 13 = 45 guests.
That means paid guests 45  - 4 = 41 guests.
The party will cost $225 + 41 * $6 = $125 + $246 = $371.
Answer: 371
```

### branch_4 ❌  extracted=`371`

```
Kayla invited 25 + 7 + 13 = 45 guests.
She invited 45 - 4 = 41 guests.
The fee to rent the theater is $125 for a party of 20, plus $6 for each additional guest.
Since therela invited 41 guests, she will need to pay an additional 41 * $6 = $246 for theTherefore, the total cost of the party is $125 + $246 = $371.
Answer: 371
```

---

## problem_id=1129  gold=`8`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'7': 2, '0': 1, '8': 1, '12': 1}` | cmaj_pick=`7` (2/5) | correct branches: **1/5**

### branch_0 ❌  extracted=`7`

```
Let A = 28 - 2 = 26 eggs =A + Madeline
A = 28 / 2 = 14 eggs
C + S  = 56 - 26 = 30 eggs
C = S + 12
C + S = 30
S + 12 + S = 30
2S + 12 = 30
2S = 18
S = 9
C = 21
Then
21 - 14 = 7 eggs
Answer: 7
```

### branch_1 ❌  extracted=`0`

```
A = Madeline
B = Arnold
C = Camden
D = Sarah
A + B = 56
C + D = D + 12 = 30
C = D + 12
A + B = 56
C = D + 12 = 12
 2A + 12 = 56
 2A = 44
 A = 22
C = 22
Camden painted C - B = 22 - 22 = 0 more eggs than Arnold.
Answer: 0
```

### branch_2 ❌  extracted=`7`

```
Let A be the number of eggs painted by Arnold and Madeline.
Let J be the number of eggs painted by Sarah.
C + 12 = 30
C = 18
J = 18 - 12 = 6
A + A + C + J = 56
2A + 18 + 6 = 56
2A = 22
A = 11
C - A = 18 - 11 = 7
Camden painted 7 more eggs than Arnold.
Answer: 7
```

### branch_3 ✅  extracted=`8`

```
Let both Madeline and Arnold paint x eggs.
Then Sarah painted 30 - 12 = 18 eggs.
Thus, x + x + 30 + 18 = 56.
So, 2x + 48 = 56.
Then, 2x = 8.
So, x = 4.
So, Arnold painted 4 eggs and Camden painted 30 - 18 = 12 eggs.
Therefore, Camden painted 12 - 4 = 8 more eggs than Arnold.
Answer: 8
```

### branch_4 ❌  extracted=`12`

```
Let A and Madeline the number of eggs painted by Arnold and Madeline, respectively.
Let C and S be the number of eggs painted by Camden and Sarah, respectively.
We know that A + Madeline + C + S = 56, and we know that C + S = 30.
We also know that C = S + 12.
Therefore, C = A + 12.
To, we painted C 12 more eggs than Arnold.
Answer: 12
```

---

## problem_id=1134  gold=`7`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'15': 4, '7': 1}` | cmaj_pick=`15` (4/5) | correct branches: **1/5**

### branch_0 ❌  extracted=`15`

```
Jane has saved $4 every week for 8 weeks, so she has saved a total of $4 * 8 = $32.
She wants to save a total of $60, so she needs to save an additional $60 - $32 = $28.
If she continues to save $4 every week, she will need to save $28 / $4 = 7 more weeks.
Therefore, it will take her a total of 8 weeks + 7 weeks = 15 weeks to have saved a total of $60.
Answer: 15
```

### branch_1 ❌  extracted=`15`

```
Jane has saved $4 every week for 8 weeks, so she has saved a total of $4 * 8 = $32.
She wants to save a total of $60, so she needs to save an additional amount of $60 - $32 = $28.
If she continues to save $4 every week, she will need to save an additional $28 / $4 = 7 weeks.
Therefore, it will take her an total 8 + 7 = 15 weeks to have saved a total of $60.
Answer: 15
```

### branch_2 ❌  extracted=`15`

```
Jane has saved $4 every week for 8 weeks, so she has saved a total of $4 * 8 = $32.
She wants to save a total of $60, so she still needs to save $60 - $32 = $28.
If she continues to save $4 every week, she will need an additional $28 / $4 = 7 more weeks to save the remaining $28.
Therefore, it will take her 8 + 7 = 15 weeks to save a total of $60.
Answer: 15
```

### branch_3 ✅  extracted=`7`

```
Jane has saved $4 every week for 8 weeks, so she has saved a total of 4 * 8 = $32.
She wants to save a total of $60, so she needs to save an additional 60 - 32 = $28.
If she continues to save $4 every week, she will need to save 28 / 4 = 7 more weeks to reach her goal of $60.
Answer: 7
```

### branch_4 ❌  extracted=`15`

```
Jane has saved $4 every week for 8 weeks, so she has saved a total of 4 * 8 = $32.
She wants to save a total of $60, so she needs to save an additional $60-2 = $28.
If she continues to save $4 each week, she will save her remaining $28 / $4 = 7 weeks.
Therefore, it will take her a total of 8 weeks already saved + 7 additional weeks = 15 weeks to have saved a total of $60.
Answer: 15
```

---

## problem_id=1182  gold=`1800`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'18': 1, '180': 2, '1800': 1, '54': 1}` | cmaj_pick=`180` (2/5) | correct branches: **1/5**

### branch_0 ❌  extracted=`18`

```
Bob spends $27000 distributed between 3 schools to buy books. He can buy 100 books for $500.
First, let's find out how much money Bob spends per school:
Bob spends $27000 / 3 = $9000 per school.
Next, let's find out how many books Bob can buy per school:
Bob can buy 100 books for $500, so he can buy $9000 / $500 = 18 books per school.
Answer: 18.
```

### branch_1 ❌  extracted=`180`

```
Bob spends $27000 distributed between 3 schools to buy books.
He can buy 100 books for $500.
So, he can buy $27000 / $500 = 540 books in total.
Since he distributes the books between 3 schools, he can buy 540 books / 3 = 180 books per school.
Answer: 180
```

### branch_2 ✅  extracted=`1800`

```
Bob spends $27000 distributed between 3 schools to buy books.
He can buy 100 books for $500.
So, he can buy $27000 / $500 = 5400 books total.
To find out how many books he can buy per school, we divide the total number of books by the number of schools:
5400 books / 3 schools = 1800 books per school
Therefore, Bob can buy 1800 books per school.
Answer: 1800
```

### branch_3 ❌  extracted=`54`

```
Bob spends $27000 distributed between 3 schools to buy books.
He can buy 100 books for $500.
So, he can buy 27000 / 500 = 54 books per school.
Answer: 54 books per school.
```

### branch_4 ❌  extracted=`180`

```
Bob spends $27000 distributed between 3 schools to buy books.
He can buy 100 books for $500.
So, he can buy 27000 / 500 = 540 books in total.
Since he distributes this amount between 3 schools, each school will receive 540 / 3 = 180 books.
Answer: 180
```

---

## problem_id=1186  gold=`19`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'37': 2, '19': 2, '22': 1}` | cmaj_pick=`37` (2/5) | correct branches: **2/5**

### branch_0 ❌  extracted=`37`

```
On Monday the total 3 miles + 3 miles + 3 miles = 9 miles
On Tuesday and Thursday 5 miles + 5 miles + 5 miles = 10 miles
On Wednesday the total 3 miles + 3 miles + 3 miles = 9 miles
On Friday the total 3 miles + 3 miles + 3 miles = 9 miles
So 9 miles + 9 miles + 9 miles + 10 miles = 37 miles
Answer: 37
```

### branch_1 ✅  extracted=`19`

```
On Monday in total, Sam ran 3 + 3 + 3 = 9 miles.
On Tuesday and Thursday, Sam ran 5 + 5 = 10 miles.

 total, in total Sam ran 9 + 10 = 19 miles this week.
Answer: 19
```

### branch_2 ❌  extracted=`37`

```
On Monday the total of 3 miles + 3+3+3=<<3+3+3=9>>9 miles
On Tuesday and Thursday the total of 5 miles + 5+5=<<5+5=10>>10 miles
On Wednesday and Friday the total of 3 miles + 3+3+3=<<3+3+3=9>>9 miles
On Friday the total of 9 miles + 10 + 9+9= <<9+10+9+9=37>>37 miles
Answer: 37
```

### branch_3 ❌  extracted=`22`

```
On Monday the total is 3 miles
On Tuesday the total is 5 miles
On Wednesday the total is 3 miles
On Thursday the total is 5 miles
On Friday the total is 3 miles
Thisogether the total is 3+5+3+3+5+3=22 miles
Answer: 22
```

### branch_4 ✅  extracted=`19`

```
On Monday the total 3 miles + 3 miles + 3 miles = 9 miles
On Tuesday and Thursday 5 miles + 5 miles = 10 miles
Alt week 9 miles + 10 miles = 19 miles
Answer: 19
```

---

## problem_id=1231  gold=`70`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'80': 2, '70': 2, '81.11': 1}` | cmaj_pick=`80` (2/5) | correct branches: **2/5**

### branch_0 ❌  extracted=`80`

```
To find the mean, we first need to add up all the scores and then divide by the number of quizzes.



Add up of her scores: 50 + 80 + 80 + 60 + 40 + 90 + 100 + 70 + 60 = 720



Number of quizzes: 9



Mean score = Add up of scores / Number of quizzes
Mean score = 720 / 9
Mean score = 80



Answer: 80
```

### branch_1 ✅  extracted=`70`

```
To find the mean score, we need to add all the scores and then divide by the number of quizzes. 

First, let's add the the scores:
50 + 80 + 80 + 60 + 40 + 90 + 100 + 70 + 60 = 630

Now, we divide this total by the number of quizzes, which is 9:
630 / 9 = 70

So, Sally's mean score is 70.

Answer: 70
```

### branch_2 ❌  extracted=`81.11`

```
To find the mean score, we need to add up all the scores and then divide by the number of quizzes.


Adding up all the scores:
50 + 80 + 80 + 60 + 40 + 90 + 100 + 70 + 60 = 730

There have been 9 quizzes, so we divide the total score by 9:
730 / 9 = 81.11

Answer: 81.11
```

### branch_3 ✅  extracted=`70`

```
To find the mean score, we need to add up all the scores and then divide by the number of quizzes. 

Sum of the scores: 50 + 80 + 80 + 60 + 40 + 90 + 100 + 70 + 60 = 630

Number of quizzes: 9

Mean score = Sum of the scores / Number of quizzes
Mean = 630 / 9
Mean = 70

Answer: 70
```

### branch_4 ❌  extracted=`80`

```
To find the mean score, we need to add up all the scores and then divide by the number of quizzes.

First, let's add up all the scores:
50 + 80 + 80 + 60 + 40 + 90 + 100 + 70 + 60 = 720

Next, we divide the sum by the number of quizzes, which is 9:
720 / 9 = 80

So, Sally's mean score is 80.

Answer: 80
```

---

## problem_id=1317  gold=`5`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'15': 3, '5': 2}` | cmaj_pick=`15` (3/5) | correct branches: **2/5**

### branch_0 ❌  extracted=`15`

```
Let c be the number of chickens and k be the number of cows.
We know that c + k = 20 and 2c + 4k = 70.
We can simplify the second equation to dividing everything by2, so c + 2k = 35.
We we have equations c + k = 20 and c + 2k = 35.
Subtracting the first equation from the second equation, we get c = 15.
Therefore, there are 15 chickens on Farmer Brown's farm.
Answer: 15
```

### branch_1 ✅  extracted=`5`

```
Let c be the number of chickens and k be the number of cows.
We know that c + k = 20 and 2c + 4k = 70.
We can simplify the second equation by dividing both sides by 2 to giving c + 2k = 35.
Now subtract the first equation from the second to eliminate k = 15.
Substitute k = 15 back into the first equation to find c = 5.
Therefore, the number of chickens Farmer Brown has on his farm is c = 5.
Answer: 5
```

### branch_2 ❌  extracted=`15`

```
Let c be the number of chickens and x be the number of cows.
We know that c + x = 20 and 2c + 4x = 70.
We can simplify the second equation by dividing everything by by 2 to get c + 2x = 35.
Now, we can subtract the first equation from this to to get x = 20 -
5 = 5.
Finally, we know that c + 5 = 20, so c = 20 - 5 = 15.
Answer: 15
```

### branch_3 ✅  extracted=`5`

```
Let c be the number of chickens and let k be the number of cows.
We know that c + k = 20 and 2c + 4k = 70.
We can simplify the second equation by dividing everything by 2 to giving c + 2k = 35.
Now, we subtract the first equation from this equation to get k = 15.
Substituting this value into the first equation, we get c = 5.
Therefore, there are 5 chickens on the farm.
Answer: 5
```

### branch_4 ❌  extracted=`15`

```
Let c be the number of chickens and let x be the number of cows.
We know that c + k = 20 and 2c + 4k = 70.
We can simplify the second equation by dividing both sides by 2, get c + 2k = 35.
Now we can subtract the first equation from this equation: c + 2k - (c + k) = 35 - 20.
This simplifies to: k = 15.
So, Farmer Brown has 15 chickens.
Answer: 15
```

---

## problem_id=299  gold=`31800`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'30000': 3, '31800': 1, '60000': 1}` | cmaj_pick=`30000` (3/5) | correct branches: **1/5**

### branch_0 ❌  extracted=`30000`

```
He drives 30*2=<<60=60>>60 miles per day
So he drives 6*5=<<60*5=300>>300 miles per week
So he drives 300*50=<<300*50=15000>>15000 miles per year
So he pays 15000*2=$<<15000*2=30000>>30,000 in miles per year
Answer: 30000
```

### branch_1 ✅  extracted=`31800`

```
He drives 30*2=60 miles each day
So he drives 60*5=300 miles a week
That he drives 300*50=15000 miles a year
So he pays 15000*2=$30000 for the miles per year
He also pays 150*12=$1800 in bonuses
So in total he pays 30000+1800=$31800 a year for driving
Answer: 31800
```

### branch_2 ❌  extracted=`30000`

```
He drives 30*2=<<60=60>>60 miles per day
He drives 60*5=<<<<300=300>>300 miles per week
He drives 300*50=<<15000=15000>>15000 miles per year
So he pays 15000*2=$<<15000*2=30000>>30,000 in driving per year
Answer: 30000
```

### branch_3 ❌  extracted=`30000`

```
He drives 30*2=<<60=60>>60 miles a day
So he drives 60*5=<<300=300>>300 miles a week
So he drives 300*50=<<15000=15000>>15000 miles a year
So he pays 15000*2=$<<30000=30000>>30,000 a year
Answer: 30000
```

### branch_4 ❌  extracted=`60000`

```
He drives 60*2=<<60*2=120>>120 miles a day
So he drives 120*5=<<120*5=600>>600 miles a week
So he pays 600*2=$<<600*2=1200>>1200 a week
So he pays 1200*50=$<<1200*50=60000>>60,000 a year
Answer: 60000
```

---

## problem_id=340  gold=`4`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'3': 2, '5': 2, '4': 1}` | cmaj_pick=`3` (2/5) | correct branches: **1/5**

### branch_0 ❌  extracted=`3`

```
Josh starts off practicing juggling 3 balls.
After each week he adds 1 ball to his juggling act, so after the first week he j juggling 4 balls.
After the second week he j juggling 5 balls.
After the third week he j juggling 6 balls.
After the fourth week he j juggling 7 balls.
But he slips and drops 3 of his balls, so he is left juggling 7 - 3 = 4 balls.
But one ball gets lost completely, so he is left juggling 4 - 1 = 3 balls.
Answer: 3
```

### branch_1 ❌  extracted=`5`

```
Josh starts off practicing juggling 3 balls and slowly gets better by adding 1 ball to his juggling act each week.
After the end of the fourth week, he is juggling 7 balls.
However, when he slips and drops three of his balls, he is left with 7 - 3 = 4 balls.
Two of the dropped balls are caught by people in the crowd, so he is left with 4 + 2 = 6 balls.
However, one dropped ball gets lost completely, so he is left with 6 - 1 = 5 balls.
Answer: 5
```

### branch_2 ❌  extracted=`3`

```
Josh starts off practicing juggling 3 balls.
He gets better adding 1 ball to his juggling act each week.
After the end of the fourth week, he should be 3 + 1 + 1 + 1 = 6 balls.
However, he slips and drops 3 of his balls,
2 of them are caught by people in the crowd, and one gets lost completely.
So, he has 6 - 2 - 1 = 3 balls left.
Therefore, with a sigh, Josh starts to juggle on stage with 3 balls.
Answer: 3
```

### branch_3 ✅  extracted=`4`

```
1. Josh starts juggling 3 balls. After 4 weeks, he adds 1 ball each week, so he is juggling 3 + 4 = 7 balls.
2. During the talent show, he drops 3 balls.
3. 2 of the dropped balls are caught by people, so he now has 7 - 2 = 5 balls left.
4. 1 of the dropped balls gets lost, so he now has 5 - 1 = 4 balls left.

Therefore, Josh starts to juggle with 4 balls.

Answer: 4
```

### branch_4 ❌  extracted=`5`

```
Josh starts with juggling 3 balls and adds 1 ball each week for 4 weeks, so he is juggling 3 + 4 = 7 balls at the end of the fourth week.
He then slips and drops 3 balls, so he is left with 7 - 3 = 4 balls.
2 of the dropped balls are caught by people in the crowd, so he is left with 4 + 2 = 6 balls.
However, one of the dropped balls gets lost completely, so he is left with 6 - 1 = 5 balls.
Answer: 5
```

---

## problem_id=419  gold=`3000`

_(question text not stored in substrate — see e4/data/gsm8k_dev_500.json)_

votes: `{'1000': 2, '3000': 2, '7000': 1}` | cmaj_pick=`1000` (2/5) | correct branches: **2/5**

### branch_0 ❌  extracted=`1000`

```
Stanley's brother, Oliver, needs to reach the elevation of Stanley's comb, which is at 4,000 feet. Oliver is currently at an elevation of 3,000 feet.

To find out how many feet Oliver needs to climb, we subtract his current elevation from the elevation of the comb:

\[ 4000 \text{ feet} - 3000 \text{ feet} = 1000 \text{ feet} \]

Oliver must climb 1,000 feet to reach the elevation of Stanley's comb.
```

### branch_1 ✅  extracted=`3000`

```
Toley is at an elevation of 3,000 feet, and Stanley is at an elevation of 10,000 feet, so the difference in elevation is 10,000 - 3,000 = 7,000 feet.
The comb is 4,000 feet below Stanley's elevation, so Oliver needs to climb 7,000 - 4,000 = 3,000 feet to reach the comb.
Answer: 3,000 feet.
Therefore: 3000
```

### branch_2 ✅  extracted=`3000`

```
Oliver's current elevation is 3,000 feet.
Stanley's current elevation is 10,000 feet.
Stanley's comb fell 4,000 feet below Stanley, so the comb's elevation is 10,000 feet - 4,000 feet = 6,000 feet.
To reach the elevation of Stanley's comb, Oliver needs to climb 6,000 feet - 3,000 feet = 3,000 feet.
The answer is: 3000
```

### branch_3 ❌  extracted=`7000`

```
To determine the elevation difference Oliver must climb to reach Stanley's comb, we need to subtract the elevation of Oliver's location from the elevation of the's location.

The elevation of the's location is 10,000 feet.
The elevation of Oliver's location is 3,000 feet.

The elevation difference is:
10,000 feet - 3,000 feet = 7000.

Therefore, Oliver must climb 7000 feet to reach the elevation of Stanley's comb.
Answer: 7000
```

### branch_4 ❌  extracted=`1000`

```
To determine how many feet Oliver must climb to reach the elevation of Stanley's comb, we need to calculate the difference in elevation between Stanley's comb and Oliver's current elevation.

Stanley's comb is at 4,000 feet, and Oliver is at 3,000 feet.

The difference in elevation is:
\[ 4000 - 3000 = 1000 \]

Therefore, Oliver must climb 1,000 feet to reach the elevation of Stanley's comb.

Answer: 1000 feet
```

---
