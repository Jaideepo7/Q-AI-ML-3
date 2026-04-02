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


### Questions
- Should we use a mastery threshold (e.g., BKT P(L) > 0.9) to "graduate" topics, or keep reviewing everything?
- For Thompson Sampling: Should we have per-user bandits (personalized strategies) or global bandit (one-size-fits-all)?
- Do we need topic prerequisites? (e.g., can't learn "cellular respiration" until "glycolysis" is mastered)
- What's our review cadence? Daily check-ins, or only when user opens app?
