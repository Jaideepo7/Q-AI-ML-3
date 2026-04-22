# This document goes through three different machine learning Algorithms(ideal): 
- SM-2       : which is when to review each topic (spaced repetition scheduling)
- BKT        : what the user knows (knowledge state estimation)
- Thompson Sampling  : Which quiz strategy to use (exploration-exploitation)

## SM-2 Algorithm (SuperMemo 2) : 
### What it does : 
- this is an algorithm which determines optimal time to review certain material. This makes sure
to make sure same questions dont repeat too often, which ensures long term retention.
### How it works : 
- items you already know well will get reviewed less frequently while items you struggle with will get
  reviewed more often.

## BKT (Bayesian Knowledge Tracing) :
### What it does : 
- this is what estimates and keeps a value for if the user has mastered a topic/question. This value of how good they are with the question will be updated after every question.
### How it works: 
- The idea is to use bayesian inference to estimate hidden knowledge state based on if each answer is answered right or wrong.
- Would have four parameter:
* 1. L0  : initial knowledge state which could be like 30 percent meaning chance the learner already knows this
* 2. T   : transidtion state which is chance after a question information is retained
* 3. S   : slip probability, where the user knows it but still could get it wrong
*  4. G   : Guess probability, when they didnt know it but still got it right
 

### Example : 
Topic: "Photosynthesis"
Initial: P(L) = 0.3 (30% mastery)

Question 1: User answers CORRECTLY
→ P(L) increases to 0.55 (they probably know it)

Question 2: User answers CORRECTLY again
→ P(L) increases to 0.72 (even more confident)

Question 3: User answers INCORRECTLY
→ P(L) drops to 0.58 (maybe they guessed earlier, or slipped)

After 5 correct answers in a row:
→ P(L) reaches 0.95 (mastery threshold!)


### mastery_probability is where it will be stored per(user,topic)pair

## Thompson Sampling :
### What is it: 
- this is a decision-making algorithm that tries new things by balanacing exploration and exploitation when means it will use what works more for a learner.
### How it works: 
- The core idea is to maintain a probability distribution of each options expected reward. Then from these pick the best sample.
#### The Process
-  Model uncertainty with beta distribution :   Beat(alpha, beta) shows our belief about success rate : alpha = # of successes and beta = # of failures so basically higher alpha/beta raio means higher expected reward.
-  Now sample a randome value from each beta distribution
-  select the 'arm' with the highest sampled value
-  Update the selected arms distribution based on good or bad ouutcome. If good increase alpha (shift distribution right) if bad increase beta (shift distribution left)


## Example Scenario: 
4 Quiz Strategies:
1. Focus on weak topics    → Beta(1, 1) [unknown]
2. Mixed difficulty        → Beta(1, 1) [unknown]
3. Spaced repetition focus → Beta(1, 1) [unknown]
4. Random exploration      → Beta(1, 1) [unknown]

Round 1:
- Sample from each: [0.7, 0.3, 0.5, 0.2]
- Pick Strategy 1 (highest sample)
- User scores 0.8 → Update: Beta(1.8, 1.2)

Round 2:
- Sample: [0.6, 0.9, 0.4, 0.7]
- Pick Strategy 2
- User scores 0.5 → Update: Beta(1.5, 1.5)

After 50 rounds:
- Strategy 1: Beta(35, 15) → Expected reward ≈ 0.70
- Strategy 2: Beta(18, 22) → Expected reward ≈ 0.45
- Strategy 3: Beta(28, 17) → Expected reward ≈ 0.62
- Strategy 4: Beta(10, 30) → Expected reward ≈ 0.25

Now we mostly pick Strategy 1, but occasionally try others (exploration)



# How they would ideally work together : 
- Thompson Sampling : Choose Quiz Strategy ("focus on weak topics")
- BKT  : Identify which topics need practice  (filter the questions with low mastery)
- SM-2 : Checking is it time for certain topics to be reviewed again or for the first time
- GENERATE the quiz from all the selected topics
- User Answers Quiz
## Updata all three systems: 
- BKT : update the level of mastery
- SM-2 : schedule next review
- Thompson : Update strategy performance

# Proposed Architecture 
- Thompson Sampling (bandit layer)   - choose overall quiz strategy per session / adapt to user
- BKT (knowledge tracking)   - maintian mastery estimates for each topic and identifiy weak topics
- SM-2 (scheduling layer)    - optimize long term retention 
### Questions
- Should we use a mastery threshold (e.g., BKT P(L) > 0.9) to "graduate" topics, or keep reviewing everything?
- For Thompson Sampling: Should we have per-user bandits (personalized strategies) or global bandit (one-size-fits-all)?
- Do we need topic prerequisites? (e.g., can't learn "cellular respiration" until "glycolysis" is mastered)
- What's our review cadence? Daily check-ins, or only when user opens app?
