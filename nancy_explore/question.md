## Research Brief: The Non-Binary Dynamics of Polychromic Diversity in Set RL

### Background Context
We are implementing Polychromic Exploratory Policy Optimization (Poly-EPO) for a CS224R project. The algorithm samples $N$ responses, clusters them by reasoning strategy, and builds subsets of size $n$. The objective function for a set $G$ is the product of its average reward and its diversity density:
$$f_{poly}(x, G) = \left( \frac{1}{n} \sum_{y_i \in G} r(x, y_i) \right) \cdot \left( \frac{|\{C(y_i)\}_{y_i \in G}|}{n} \right)$$

### The Core Problem (The "Perfect Subset" Fallacy)
Previous theoretical intuition suggested that as $n$ approaches $N$, the algorithm aggressively punishes duplicate answers because the probability of forming a "perfectly unique" subset (no duplicates) drops to zero. 

However, this assumes diversity is a binary threshold. In reality, the diversity score $d$ scales smoothly. A subset with a duplicate simply has a lower fractional score (e.g., $3/4$ instead of $4/4$), not a zero score. Therefore, common answers may still accumulate significant positive marginal advantage because they participate in sets that still have high expected rewards and non-zero diversity.

### Objective for the Agent
Write a Python simulation to calculate the exact, empirical marginal set advantage for different response profiles, proving how the smooth scaling of the diversity metric dampens the "aggressive filter" effect. 

### Research Questions to Code & Answer

* **RQ1: The Fractional Diversity Dampener.** For a fixed batch $N=8$, simulate a scenario with one common correct answer (appears 4 times), one rare correct answer (appears 1 time), and three incorrect unique answers. Calculate the exact marginal set advantage for the common vs. rare correct answer across all valid values of $n$ (from 2 to 7). Does the penalty for the common answer scale linearly, exponentially, or asymptotically as $n$ increases?
* **RQ2: The Reward/Diversity Covariance Override.** The Poly-EPO advantage relies on the covariance between reward and diversity. If a common answer is highly rewarded, at what threshold of $c$ (cluster size) does the loss in diversity credit completely outweigh the individual reward contribution, causing the advantage to become negative? Plot this crossing point for $N=16$ and $n \in \{2, 4, 8, 12\}$.
* **RQ3: Expected Unique Clusters vs Subset Size.** Mathematically define and plot the expected value of the diversity multiplier $\mathbb{E}[d(x, G)]$ for a subset of size $n$ drawn from a batch $N$ with a known frequency distribution of clusters. Show how this expected value shifts as $n$ scales, proving why tuning $n$ acts as a "soft" rather than "hard" filter.

### Expected Output
Please provide the raw Python code used for the simulation, the resulting data tables, and a brief analytical summary answering the three RQs based on the generated data.