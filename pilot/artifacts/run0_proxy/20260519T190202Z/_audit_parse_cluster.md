# Run0 parse/cluster audit
- Rollouts: 4000
## Mismatch rates (recompute vs stored)
| Check | Mismatches | Rate |
|---|---|---|
| extract_answer(completion) vs parsed_answer | 126 | 3.1500% |
| is_correct(completion,gold) vs correct | 58 | 1.4500% |
| cluster_id(parsed) vs cluster_id | 4000 | 100.0000% |
| canon(parsed)==canon(gold) vs stored correct | 2 | 0.0500% |

- Empty parsed_answer: 0
- Canon→multiple cluster_ids: 6
- Cluster_id→multiple canons: 12
- Prompts w/ >1 correct cluster: 0
- Nested \boxed regex truncation candidates: 128

## Extract mismatches

### a8c414e5-c522-49d6-a1af-2afcb37e3ddc (line ~24)
- gold: `3`
- stored parsed: `\text{No such`
- re-extract: `\]`
- correct stored/recomp: False / False
- cluster stored/recomp: 1879310132 / 1773168844
- canon(parsed): `\text{no such`
- canon(gold): `3`
- completion snippet: To solve the equation \(a^2 + 4 = (k^2 - 4)b^2\) for positive integers \(a\), \(b\), and \(k\), we start by rewriting the equation in a more convenient form: ↵ \[ ↵ a^2 + 4 = (k^2 - 4)b^2. ↵ \] ↵ This can be rearranged to: ↵ \[ ↵ a^2 + 4 = (k^2 - 4)b^2 \implies (k^2 - 4)b^2 - a^2 = 4 \implies (kb)^2 - a^2 = 4. ↵ ...

### 56a368fe-51a8-4879-9b96-053ea9485fea (line ~42)
- gold: `110`
- stored parsed: `\frac{1190`
- re-extract: `Therefore, the area of triangle $ABC$ is \(\boxed{\frac{1190}{29}}\).`
- correct stored/recomp: False / False
- cluster stored/recomp: 2103744368 / 548328424
- canon(parsed): `\frac{1190`
- canon(gold): `110`
- completion snippet: To solve this problem, we need to use some trigonometry and properties of right triangles. We can set up a coordinate system where point $A$ is at the origin, point $B$ is at $(x, 0)$, and point $C$ is at $(0, y)$. The altitude from $A$ to $BC$ splits $BC$ into two segments: 3 and 17. ↵  ↵ 1. Since the ...

### 56a368fe-51a8-4879-9b96-053ea9485fea (line ~48)
- gold: `110`
- stored parsed: `\frac{660`
- re-extract: `````
- correct stored/recomp: False / False
- cluster stored/recomp: 903155327 / 1530228938
- canon(parsed): `\frac{660`
- canon(gold): `110`
- completion snippet: Let's denote triangle \(ABC\) with \(\angle BAC = \theta\). We know that \(\tan \theta = \frac{22}{7}\). This means that if we drop the altitude from \(A\) to \(BC\) at point \(D\), we form two right triangles, \(ABD\) and \(ACD\), with \(\angle ABD = \alpha\) and \(\angle ACD = \alpha\) (since the ...

### a6bce30d-9781-402b-95ae-882c43e72b79 (line ~71)
- gold: `296`
- stored parsed: `\frac{19448`
- re-extract: `The answer is: \frac{19448}{9}`
- correct stored/recomp: False / False
- cluster stored/recomp: 1073187402 / 812947456
- canon(parsed): `\frac{19448`
- canon(gold): `296`
- completion snippet: From the first equation, we can express $y$ in terms of $x$: $y = 10 - 3x$.  Substituting this into the second equation, we find $10 - 3x + 3(10 - 3x) = 14$, which simplifies to $3x = 28$, so $x = \frac{28}{3}$.  Then $y = \frac{2}{3}$.  It follows that $10x^2 + 12xy + 10y^2 = \left(10\left(\frac{28...

### ce3091eb-d231-4776-8e7c-1765ff579257 (line ~111)
- gold: `160`
- stored parsed: `40\sqrt{1045`
- re-extract: `Thus, the distance between the centers of the circles is \(\boxed{40\sqrt{1045}}\).`
- correct stored/recomp: False / False
- cluster stored/recomp: 699998713 / 888136614
- canon(parsed): `40\sqrt{1045`
- canon(gold): `160`
- completion snippet: To solve the problem, we need to determine the distance between the centers of two circles with radii 15 and 95, given that the external tangents intersect at a 60-degree angle. Let's proceed step by step. ↵  ↵ 1. **Identify the geometry of the problem:** ↵    - Let the centers of the two circles be \(O_1...

### a470c9d1-e204-41f4-b6ff-2f5aa51645c3 (line ~150)
- gold: `60`
- stored parsed: `49`
- re-extract: `\]`
- correct stored/recomp: False / False
- cluster stored/recomp: 1418454050 / 1733643087
- canon(parsed): `49`
- canon(gold): `60`
- completion snippet: To find the depth of the rainwater in the bottom of the square-based pyramid after the rain, we can follow these steps: ↵  ↵ 1. **Understand the Geometry of the Pyramid:** ↵    - The pyramid has a square base with side length \(23\) cm. ↵    - The height of the pyramid is \(120\) cm. ↵  ↵ 2. **Volume of the Pyr...

### cfecb90b-3f7d-4493-af64-ff306ba84d0f (line ~269)
- gold: `2`
- stored parsed: `\frac{3`
- re-extract: `\[ \boxed{\frac{3}{2}}. \]`
- correct stored/recomp: False / False
- cluster stored/recomp: 755884582 / 580029781
- canon(parsed): `\frac{3`
- canon(gold): `2`
- completion snippet: First, let's determine the side lengths of the golden ticket. Let's denote the side lengths as \( s \) and \( s\varphi \), where \( \varphi = \frac{1 + \sqrt{5}}{2} \). The area of the original ticket is: ↵  ↵ \[ \text{Area} = s \cdot s\varphi = s^2 \varphi. \] ↵  ↵ When Charlie folds the ticket so that two...

### cfecb90b-3f7d-4493-af64-ff306ba84d0f (line ~271)
- gold: `2`
- stored parsed: `\frac{2`
- re-extract: `(II) Since $x_0 \in [0, \frac{\pi}{2}]$ and $f(x_0) = 2\`
- correct stored/recomp: False / False
- cluster stored/recomp: 1515881928 / 1719906980
- canon(parsed): `\frac{2`
- canon(gold): `2`
- completion snippet: To solve the problem, we first need to understand the properties of the golden ratio \(\varphi = \frac{1+\sqrt{5}}{2}\) and how it applies to the rectangle's side lengths. ↵  ↵ Given that the rectangle's side lengths are in the golden ratio, we can denote the side lengths as \(a\) and \(a \varphi\), whe...

## Correct mismatches

### 6137f3cc-cd8e-43dd-9213-2e4c3784c96e (line ~59)
- gold: `1000`
- stored parsed: `1000`
- re-extract: `1000`
- correct stored/recomp: True / False
- cluster stored/recomp: 1012351779 / 1544436214
- canon(parsed): `1000`
- canon(gold): `1000`
- completion snippet: Step 1:  We notice that the terms alternate in sign and we can pair them up to get (1990 - 1980) + (1970 - 1960) + ... + (10 - 0). This simplifies to 10 for each of the 100 pairs. The expression can be simplified further since there are 100 pairs of terms. We can calculate the sum of the pairs by mu...

### 6137f3cc-cd8e-43dd-9213-2e4c3784c96e (line ~64)
- gold: `1000`
- stored parsed: `1000`
- re-extract: `1000`
- correct stored/recomp: True / False
- cluster stored/recomp: 1012351779 / 1544436214
- canon(parsed): `1000`
- canon(gold): `1000`
- completion snippet: To solve the problem, let's break it down step by step. ↵  ↵ The given expression is: ↵  ↵ \[1990 - 1980 + 1970 - 1960 + \cdots - 20 + 10\] ↵  ↵ We can pair the terms as follows: ↵  ↵ \[(1990 - 1980) + (1970 - 1960) + \cdots + (10 - 0)\] ↵  ↵ Each pair simplifies to: ↵  ↵ \[10\] ↵  ↵ So, we have: ↵  ↵ \[10 + 10 + 10 + \cdots + 10\...

### a6bce30d-9781-402b-95ae-882c43e72b79 (line ~70)
- gold: `296`
- stored parsed: `$296$`
- re-extract: `$296$`
- correct stored/recomp: True / False
- cluster stored/recomp: 2024695799 / 1719866562
- canon(parsed): `$296$`
- canon(gold): `296`
- completion snippet: First, let's solve the system of equations: ↵  ↵ 1. $3x + y = 10$ can be rewritten as $y = 10 - 3x$. ↵  ↵ 2. Substitute $y = 10 - 3x$ into $x + 3y = 14$: ↵  ↵ $$x + 3(10 - 3x) = 14$$ ↵ $$x + 30 - 9x = 14$$ ↵ $$-8x = -16$$ ↵ $$x = 2$$ ↵  ↵ 3. Substitute $x = 2$ into $y = 10 - 3x$: ↵  ↵ $$y = 10 - 3(2)$$ ↵ $$y = 4$$ ↵  ↵ So, $x = 2$...

### 0b9df9e2-1b81-4e57-8bcb-0cbceabf89a7 (line ~84)
- gold: `71`
- stored parsed: `71`
- re-extract: `71`
- correct stored/recomp: True / False
- cluster stored/recomp: 546479777 / 1298459408
- canon(parsed): `71`
- canon(gold): `71`
- completion snippet: First, let's simplify the factorial expressions in the numerator and denominator. ↵  ↵ For 7!: ↵ 7! = 7 × 6 × 5 × 4 × 3 × 2 × 1 = 5040 ↵  ↵ For 8!: ↵ 8! = 8 × 7 × 6 × 5 × 4 × 3 × 2 × 1 = 40320 ↵  ↵ Now, let's substitute these values back into the original expression: ↵  ↵ (7! + 8!) / (6! + 7!) = (5040 + 40320) / (720 +...

### 0b9df9e2-1b81-4e57-8bcb-0cbceabf89a7 (line ~85)
- gold: `71`
- stored parsed: `71`
- re-extract: `71`
- correct stored/recomp: True / False
- cluster stored/recomp: 546479777 / 1298459408
- canon(parsed): `71`
- canon(gold): `71`
- completion snippet: Step 1:  I want to simplify the expression $\frac{7!+8!}{6!+7!}$. Both the numerator and the denominator involve factorials of consecutive integers. ↵ Step 2:  We know that $n!$ represents the product of all positive integers up to $n$, so $7!$ and $8!$ are the products of all numbers from 1 to 7 and ...

### 0b9df9e2-1b81-4e57-8bcb-0cbceabf89a7 (line ~87)
- gold: `71`
- stored parsed: `71`
- re-extract: `71`
- correct stored/recomp: True / False
- cluster stored/recomp: 546479777 / 1298459408
- canon(parsed): `71`
- canon(gold): `71`
- completion snippet: Let's solve the problem step by step. ↵  ↵ Given the expression: ↵  ↵ \[ ↵ \frac{7! + 8!}{6! + 7!} ↵ \] ↵  ↵ First, let's break down the factorials: ↵  ↵ \[ ↵ 7! = 5040 \\ ↵ 8! = 40320 \\ ↵ 6! = 720 ↵ \] ↵  ↵ Substitute these values into the expression: ↵  ↵ \[ ↵ \frac{7! + 8!}{6! + 7!} = \frac{5040 + 40320}{720 + 5040} ↵ \] ↵  ↵ Simplify the...

### 1dfe696e-f25b-434d-a3de-6361bd0f6901 (line ~119)
- gold: `75`
- stored parsed: `75`
- re-extract: `75`
- correct stored/recomp: True / False
- cluster stored/recomp: 491602499 / 310743884
- canon(parsed): `75`
- canon(gold): `75`
- completion snippet: Let's solve this step by step. ↵  ↵ 1. Each person has 10 friends. ↵ 2. Each person hugs all their friends. ↵  ↵ So, each of the 15 people hugs 10 friends.  ↵  ↵ However, if we simply multiply 15 (the number of people) by 10 (the number of friends), we would count each hug twice (once for each person involved).  ↵ ...

### 1dfe696e-f25b-434d-a3de-6361bd0f6901 (line ~120)
- gold: `75`
- stored parsed: `75`
- re-extract: `75`
- correct stored/recomp: True / False
- cluster stored/recomp: 491602499 / 310743884
- canon(parsed): `75`
- canon(gold): `75`
- completion snippet: To solve the problem of determining the total number of hugs exchanged at the party, we can approach the problem using combinatorial methods. Here's the step-by-step breakdown: ↵  ↵ 1. **Understand the Problem:** ↵    - There are 15 people at the party. ↵    - Each person has 10 friends. ↵    - Each person hu...

## Cluster mismatches

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~1)
- gold: `201`
- stored parsed: `0`
- re-extract: `0`
- correct stored/recomp: False / False
- cluster stored/recomp: 1715923189 / 791771871
- canon(parsed): `0`
- canon(gold): `201`
- completion snippet: Let $x=2^{51}$.  Then we seek the remainder of $x^4+2+202$ divided by $x^2+x+1$.  We compute $x^2+x+1\equiv 2x+1$ modulo $x^2+x+1$, so $(x^2+x+1)(x^2-x+1)\equiv (2x+1)(x^2-x+1)$ modulo $x^2+x+1$.  Expanding, $x^4-2x^3+x^2+1-2x^2+x\equiv -4x+1$ modulo $x^2+x+1$, so $-4x+1$ is the remainder when $x^4+...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~2)
- gold: `201`
- stored parsed: `-1`
- re-extract: `-1`
- correct stored/recomp: False / False
- cluster stored/recomp: 1239885873 / 1520823893
- canon(parsed): `-1`
- canon(gold): `201`
- completion snippet: To solve this problem, we can use the method of modular arithmetic. We want to find the remainder when $2^{202} + 202$ is divided by $2^{101} + 2^{51} + 1$. ↵  ↵ First, we note that $2^{202} = (2^{101})^2 \cdot 2$. So, we can rewrite the expression as: ↵  ↵ $2^{202} + 202 = (2^{101})^2 \cdot 2 + 202$ ↵  ↵ Now, ...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~3)
- gold: `201`
- stored parsed: `202`
- re-extract: `202`
- correct stored/recomp: False / False
- cluster stored/recomp: 1377652160 / 1603407851
- canon(parsed): `202`
- canon(gold): `201`
- completion snippet: Step 1:  To solve the problem, we need to understand the properties of modular arithmetic. In modular arithmetic, we can find the remainder of a division by subtracting multiples of the divisor to keep the dividend in a smaller range. ↵ Step 2:  Observe that \(2^{202} + 202 \equiv 2^{202} \pmod{2^{101...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~4)
- gold: `201`
- stored parsed: `205`
- re-extract: `205`
- correct stored/recomp: False / False
- cluster stored/recomp: 449302458 / 1968824085
- canon(parsed): `205`
- canon(gold): `201`
- completion snippet: To solve the problem, we need to determine the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\). Let's denote \(N = 2^{101} + 2^{51} + 1\). We need to find the remainder of \(2^{202} + 202\) when divided by \(N\). ↵  ↵ First, observe that \(2^{202} = (2^{101})^2\). We need to find...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~5)
- gold: `201`
- stored parsed: `202`
- re-extract: `202`
- correct stored/recomp: False / False
- cluster stored/recomp: 1377652160 / 1603407851
- canon(parsed): `202`
- canon(gold): `201`
- completion snippet: To solve this problem, we need to find the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\). ↵  ↵ Let's start by analyzing the given expressions and simplifying them using modular arithmetic. ↵  ↵ First, let's denote: ↵ \[ N = 2^{101} + 2^{51} + 1 \] ↵  ↵ We want to find the remainder when ...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~6)
- gold: `201`
- stored parsed: `Therefore, the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\) is Answer: \(202\).`
- re-extract: `Therefore, the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\) is Answer: \(202\).`
- correct stored/recomp: False / False
- cluster stored/recomp: 1324932181 / 1844207439
- canon(parsed): `therefore the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} `
- canon(gold): `201`
- completion snippet: Step 1: Let's begin by understanding the problem. We need to find the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\). ↵  ↵ Step 2: We can observe that \(2^{202} + 202 = 2 \cdot 2^{201} + 202 = 2 \cdot 2^{101} \cdot 2^{100} + 202\). Similarly, we can write \(2^{202} + 202\) in th...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~7)
- gold: `201`
- stored parsed: `202`
- re-extract: `202`
- correct stored/recomp: False / False
- cluster stored/recomp: 1377652160 / 1603407851
- canon(parsed): `202`
- canon(gold): `201`
- completion snippet: To solve the problem, we need to find the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\). Let's start by rewriting the divisor in a more convenient form. ↵  ↵ Notice that \(2^{202} + 202\) can be expressed in terms of the divisor \(2^{101} + 2^{51} + 1\). Let's denote \(N = 2^{1...

### 1653ee27-05d2-49ea-b9fb-3cdd58a05730 (line ~8)
- gold: `201`
- stored parsed: `404`
- re-extract: `404`
- correct stored/recomp: False / False
- cluster stored/recomp: 600023466 / 690507946
- canon(parsed): `404`
- canon(gold): `201`
- completion snippet: To solve the problem, we need to find the remainder when \(2^{202} + 202\) is divided by \(2^{101} + 2^{51} + 1\). Let's break down the problem and solve it step-by-step using Python and SymPy. ↵  ↵ 1. **Define the polynomials and the division:** ↵    - Let \(P(x) = 2^{202} + 202\). ↵    - Let \(D(x) = 2^{1...

## Multi-cluster correct answers (format splits)

## Nested boxed truncation samples
- **a8c414e5-c522-49d6-a1af-2afcb37e3ddc**: full-depth=`\text{No such } k \text{ exists.}` vs regex=`\text{No such `
- **56a368fe-51a8-4879-9b96-053ea9485fea**: full-depth=`\frac{1190}{29}` vs regex=`\frac{1190`
- **56a368fe-51a8-4879-9b96-053ea9485fea**: full-depth=`\frac{660}{7}` vs regex=`\frac{660`
- **a6bce30d-9781-402b-95ae-882c43e72b79**: full-depth=`\frac{19448}{9}` vs regex=`\frac{19448`
- **ce3091eb-d231-4776-8e7c-1765ff579257**: full-depth=`40\sqrt{1045}` vs regex=`40\sqrt{1045`
- **a470c9d1-e204-41f4-b6ff-2f5aa51645c3**: full-depth=`31.09` vs regex=`49`
- **a470c9d1-e204-41f4-b6ff-2f5aa51645c3**: full-depth=`31` vs regex=`49`
- **a470c9d1-e204-41f4-b6ff-2f5aa51645c3**: full-depth=`31.09` vs regex=`49`

## Brace-heavy parsed answers: 397
Correct rollouts: same canon, multiple cluster_ids: 0
