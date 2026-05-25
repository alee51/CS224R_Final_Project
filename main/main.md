::: center
**Project Title: Minority Voting in RL Training for Response
Diversity**\
**Team Members:** Anastasia Lee, Emma Gao, Nancy Bao\
**Emails:** alee72@stanford.edu, emmagao@stanford.edu,
nbao0@stanford.edu\
:::

# Experiments

**Setup:** We used `Qwen-3-1.7B` and a random subset of 500 problems
from `DAPO-Math-17k-Processed` to sample $N = 8$ model responses (=\<
1024 tokens) per problem. Using the same prompt as the Poly-EPO
paper [@polyepo], we asked an LLM judge (`gemini-3.1-flash-lite`) to
assign each of the 8 responses into a cluster based on the response's
reasoning. Each problem's responses were assessed independently. For
each set of 8 responses, we took all $\binom{8}{4} = 70$ possible
subsets of $n =4$ responses and calculated the minority set score,
baseline, set advantage, and per-response marginal set advantage.

First, does it matter how we tiebreak when there are multiple groups
tied for minority? Do we average the rewards of the tied groups, or
randomly pick a group? Second, how does Chain of Thought-based
clustering compare to final answer-based clustering? To investigate, we
calculated advantage scores 4 different ways, either tiebreaking between
minorities by random choice (`-rand`) or averaging (`-avg`), and
clustering by answer (`ans`) or Chain of Thought (`cot`).

**Initial Results:** Our preliminary experiments inform two
implementation decisions: random tiebreaking is sufficient (Pearson
$r = 0.994$ between random and average tiebreaking), and answer-based
and CoT-based minority are distinct enough to warrant separate
evaluation (Pearson $r=0.519$) between the two. These results are
expected, as the variance from random tiebreaking is minimized from $70$
sets, and the same approach can easily lead to different answers due to
small intermediate arithmetic errors.

# Changes to Research Hypothesis or Objective

We chose to pivot from our proposal idea of training with a
($Cover@\tau$) objective, as we believe it'd be expensive with minimal
improvements to training.

Standard Poly-EPO [@polyepo] uses a **joint polychromic objective**
maximizing the multiplicative product of cluster diversity ($d$) and
mean cohort correctness:
$f_\textrm{poly}(x,y_{1:n}) = \frac{1}{n}\sum_{i=1}^n r(x,y_i)\cdot d(x,y_{1:n}).$

We introduce a **Set-Based Minority Voting objective** that only uses
the reward of \"minority\" responses for gradient updates:
$$f_\textrm{minority}(x,y_{1:n})=r(x,\textrm{minority}(y_{1:n})),$$
where $\text{minority}(y_{1:n})$ isolates the final answer string that
appeared with the lowest frequency within the subset. Set scores are
averaged to determine baseline values, which are subtracted to compute
set advantages.

**Justification:** Both GRPO and Poly-EPO allocate strong rewards to
majority modes. Minority voting will help prevent the common issue of
the model \"collapsing\" to one output during training. Forcing the
model to preserve its ability to maintain several answer outputs or
reasoning methods should allow it to generalize better to OOD prompts,
as a different reasoning method may be required there.

# Next Steps

1.  **Code Improvement:** Our pilot took 6.5 hours on an A100 GPU on
    Modal (\$16.33). Since our budget is $\sim$\$1,400, we will set up a
    lightweight, parallelized training framework inspired by VeRL to
    optimize GPU usage.

2.  **Further Experimentation:** We will switch to the higher-quality
    Polaris dataset [@polaris], sub-selecting a training block of
    lower-difficulty questions (size TBD). We'll train a GRPO model
    (baseline) and answer-based minority voting model; we'll add
    CoT-based minority and Poly-EPO-answer (paper `f_poly` with answer-hash diversity, not the paper's in-loop CoT judge) for comparison if compute permits.

3.  **Algorithm Refinement:** If our revised objective excessively
    degrades majority correctness modes, we may consider a hybrid
    objective that partially considers average total reward.

4.  **Evaluation & Analysis:** After training, we will evaluate model
    variants using $pass@k$ ($k \in \{1, 4, 16, 64\}$). Model
    performance will be tested across complex out-of-domain evaluation
    targets including high-difficulty Polaris splits, AIME 2025/2026,
    HMMT, and Beyond-AIME.

# Team Contributions

So far, Nancy has focused on setting up & running the initial
experiment, Emma on researching the minority objective, and Anastasia on
experimenting with cover @ $\tau$. For the remaining work, Nancy will
focus on coding the training framework, Anastasia will focus on training
execution, and Emma will focus on monitoring training statistics.

# Pilot Experiment Baseline Data Validation

::: {#tab:pass_k_empirical}
  **Metric**            **Value**      
  ----------------- ------------------ --
  Pass@1 Accuracy    9.03% (361/4000)  
  Pass@8 Accuracy         34.40%       

  : Baseline `Qwen-3-1.7B` Pilot Pass@k Performance ($n=500$ prompts,
  $4000$ rollouts)
:::

::: {#tab:empirical_diversity}
  **Prompt Cohort Stratum**                 **Substrate Axis**    **Median**   **Mean**   **Range**
  ----------------------------------------- -------------------- ------------ ---------- -----------
  All Prompts ($n=500$)                     Answer-hash               6          5.70       1--8
  All Prompts ($n=500$)                     LLM Reasoning             5          5.28       1--8
  $\ge 1$ Correct ($n=172$; correct only)   Answer-hash               1          1.00       1--1
  $\ge 1$ Correct ($n=172$; correct only)   LLM Reasoning             1          1.31       1--4
  All-Incorrect Prompts ($n=328$)           Answer-hash               6          5.77       1--8
  All-Incorrect Prompts ($n=328$)           LLM Reasoning             5          5.36       1--8

  : Distinct Response Trajectory Clusters Discovered Per Prompt Scope
:::
