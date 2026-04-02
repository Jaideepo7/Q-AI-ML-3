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

## Thompson Sampling :


### Questions
- Should we use a mastery threshold (e.g., BKT P(L) > 0.9) to "graduate" topics, or keep reviewing everything?
- For Thompson Sampling: Should we have per-user bandits (personalized strategies) or global bandit (one-size-fits-all)?
- Do we need topic prerequisites? (e.g., can't learn "cellular respiration" until "glycolysis" is mastered)
- What's our review cadence? Daily check-ins, or only when user opens app?
