Here is a full transcription of the whiteboard and your chat logs, followed by a breakdown of the overall project "vibes" to help you easily slot this information into your CS224R milestone report template.

### **Part 1: Whiteboard Transcription**

**Top Left: Conceptual Framework**

* **Set-based RL:** Optimizing for set of trajs [trajectories], not 1 traj.
* Fractions written: $\frac{20}{100}, \frac{10}{100} \rightarrow \frac{1}{5}, \frac{1}{10}$ (with an arrow pointing to "Use Set-RL Framework")
* **Graph:** Shows a curve dropping from 100% to 0% on the y-axis, over a flat x-axis.
* **Caption under graph:** 16 diff ans. Rank by popularity of answers (count).

**Left/Bottom Left: Minority Voting Logic**

* By doing minority voting, forces model to:
* $\rightarrow$ Upweight less popular ans
* correct $\rightarrow$ Then correct ans more likely to be gen
* incorrect $\rightarrow$ Make less popular ans correct
* $\rightarrow$ Also explore incorrect ans
* $\rightarrow$ If $0.5$ ans is marked as $0.1$, it'll be re-explored (arrow pointing to a note: *not in the RL sense, but in diversity of solns*)


* Try to prevent collapsing onto 1 wrong answer (Points to the table below)
* **Table:**

| Value | ML | Set-based |
| --- | --- | --- |
| -1 | .99 | .6 |
| -5 | .05 | .3 |
| -100 | .05 | .1 |

* *Note under table:* Avoids collapsing proba all to 1 (mass around 1 ans won't...)

**Top Middle: Evaluation & Metrics**

* **(Pass@k) $\rightarrow$ Test-Metrics**
* Beyond-AIME
* AIME-26, -25
* Minerva


* **Boxed Section (Minority Voting):**
* Produce 64 answers; see if model trained w/ minority voting's answers have more diverse chains of thought/ways of getting to solution / approaching the problem.
* (eval only) Look @ chains, see if useful



**Right Side: Compute, Models, and Objective Functions**

* **Datasets & Models:**
* DaPO - 17k $\rightarrow$ Qwen 1.7B | 1 epoch (400 steps)
* Polaris - 53k $\rightarrow$ Qwen 4B | 2 epochs (with an arrow labeled *v. expensive!*)


* **Polychromic Objective:** $\sum (\text{Reward} \times \text{Diversity (of CoT)})$, gen size = $8$, $\binom{8}{4}$ subsets (subset size = $4$)
* **Modify:** $\sum (\text{Reward} \times \text{Diversity (of CoT)})$
* $\rightarrow$ right (1)
* $\rightarrow$ wrong (0)



---

### **Part 2: The Project "Vibes" (Synthesis for your Report)**

Based on the whiteboard, your chat messages, and your TA (Ifdita) notes, you have a very solid, well-scoped systems-and-algorithms project. Here is the synthesized direction of your research:

**1. The Core Hypothesis (Algorithm / Objective)**
Standard RLHF training often causes models to "collapse" onto a single, standard way of solving a problem. Your project hypothesis is that by using a **Set-based RL framework** and implementing a **Minority Voting objective** (a variation of the Poly-EPO/Polychromic objective from your TA's paper), you can force the model to upweight rare but correct reasoning paths. This encourages highly diverse Chain of Thought (CoT) generation. As your notes point out, this is especially useful for scientific discovery where novel approaches are more valuable than textbook answers.

**2. The Engineering Strategy (Compute Constraint)**
Ifdita’s original paper used a heavy stack (Qwen 4B, 53k Polaris dataset, full VeRL framework on 4 H200s). Because you are constrained by Modal compute, a major part of your project's engineering contribution is **building a lightweight, custom version of the VeRL training framework**. You are scaling down the experiment to make it feasible: using a smaller model (Qwen 1.7B), a smaller dataset (DaPO 17k), and running fewer epochs (1 epoch / 400 steps).

**3. Training vs. Evaluation (Metrics)**

* **Training:** You need to optimize for the set test during training (Minority Voting). You are weighing the compute costs of your metrics. You noted that **Pass@k** is widely used because it's cheap to compute during training.
* **Evaluation:** You are looking at **Cover@tau** for evaluation. While too expensive for training (it changes the numerator to measure the actual distribution of correct answers), it provides fine-grained insight into how often the model outputs diverse, correct answers.
* **Test Sets:** AIME-25, AIME-26, Beyond-AIME, HMMT.

---

### **Part 3: Mapping to Your LaTeX Milestone Report**

To get that one-pager done quickly, here is how you can map these vibes into the provided LaTeX template:

* **Section 1 (Experiments):** Under **Setup**, describe your hardware (Modal), your lightweight VeRL implementation, the Qwen 1.7B model, and the DaPO 17k dataset. Under **Initial Results**, discuss any preliminary compute profiling you've done or the successful setup of your custom training loop.
* **Section 2 (Changes to Hypothesis):** If your original proposal was just standard Set-based RL, your **Revised Hypothesis** is now specifically focused on the *Minority Voting* mechanism and its impact on CoT diversity, heavily driven by the need to operate within your compute budget (moving away from Cover@tau during training to Pass@k).
* **Section 3 (Next Steps):** 1. Finalize the lightweight VeRL framework.
2. Run the 1-epoch training on Qwen 1.7B using the Minority Voting objective.
3. Evaluate using Pass@k and Cover@tau on the AIME/HMMT test sets, specifically qualitative evaluation of CoT diversity (looking at 64 generated answers to see if the paths to the solution actually differ).

*Quick Note:* Your text messages mention "Qwen 3b" as a suggestion, but your TA notes and the whiteboard both explicitly mention "Qwen 1.7B". You might want to clarify with your team which one you are actually pulling from HuggingFace before writing the final report!