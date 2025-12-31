---
title: "Weight Decay and Learning Rate from a Moving Average Perspective"
date: "2025-12-05"
author: "Vuk Rosić"
excerpt: "Exploring the relationship between Weight Decay and Learning Rate by reinterpreting the model parameter update rule as an Exponential Moving Average (EMA)."
tags: ["Mathematics", "Deep Learning", "Optimization", "LLM"]
credit: "Translated and adapted from original work by Jianlin Su (kexue.fm)"
---

> **Note**: This article is translated and adapted from the original Chinese blog post by **Jianlin Su** at [kexue.fm](https://kexue.fm/archives/11459). We have expanded on the original derivations and added further technical context, making it more accessibe to wider audience.

### Introduction

Defining the right **Weight Decay** and **Learning Rate** is fundamental to training Large Language Models. These two numbers decide whether your model becomes smart or fails completely. While modern optimizers like **AdamW** have improved things, we still lack a clear scientific theory for how to set these values perfectly.

This article introduces a new perspective: viewing the training process as a **"Sliding Average Memory."**

Instead of just "updating parameters," imagine your model is building a long-term memory of the data it sees. Weight Decay and Learning Rate control how fast this memory updates and how quickly it forgets.

**Here is what we will learn, step by step:**

1.  **The Sliding Average View**: First, we will demystify the optimization process. We'll show that updating a model's weights isn't magic—it's mathematically identical to calculating a **"weighted moving average"** of the gradients. This simple shift in perspective is the key that unlocks the rest of our analysis.
2.  **Unrolling the History**: Does a model remember the very first batch of data it saw? We will "unroll" the recursive training steps to trace exactly how past data contributes to the current model weights, revealing how older information slowly fades over time.
3.  **The Memory Period**: Just like humans, models forget. We will derive a precise formula for the **"Memory Period"**—calculating exactly how many steps a specific piece of training data stays relevant in the model's "mind" before it vanishes completely.
4.  **The Optimal Schedule**: Finally, we will answer the practical questions. We'll use our new formulas to mathematically prove that the Learning Rate shouldn't be chosen randomly. Instead, we'll derive an **"Inverse Square Root"** schedule that perfectly balances acquiring new information while retaining the old.

Whether you are a student trying to understand the basics or a researcher optimizing a massive run, this guide will give you a deeper, intuitive understanding of the mechanics behind LLM training.

### Sliding Average #

The general form of Weight Decay is:
$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t + \lambda_t \boldsymbol{\theta}_{t-1}) \tag{1}
$$

**Breaking this down:**
-   $\boldsymbol{\theta}_{t-1}$ is the parameter matrix from the previous step.
-   $\eta_t$ is the Learning Rate at step $t$, which controls how big of a step we take.
-   $\boldsymbol{u}_t$ is the update direction (gradients) provided by the optimizer (like the gradient in SGD).
-   $\lambda_t \boldsymbol{\theta}_{t-1}$ is the **Weight Decay** term. It acts as a regularization force that shrinks weights towards zero by a fraction $\lambda_t$ at each step.
    -   **Why is this needed?** Without this constraint, weights can grow indefinitely large, allowing the model to learn extremely complex, "jagged" boundaries that weave around every single data point.
    -   **Fitting Noise:** Real data is messy. If a model has the freedom to use massive weights, it can over-interpret random noise in the training set as meaningful signal (overfitting). Weight decay forces the model to be "conservative," preferring simpler, smoother solutions that capture the broad trends rather than the noisy specifics.


Introducing the notation:

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t, & \hat{\boldsymbol{m}}_t &= \boldsymbol{m}_t / (1 - \beta_1^t) \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t^2, & \hat{\boldsymbol{v}}_t &= \boldsymbol{v}_t / (1 - \beta_2^t)
\end{aligned} \tag{2}
$$

**1. Deconstructing the Variables:**
-   **$\boldsymbol{g}_t$**: The gradient at the current step. It points in the direction of steepest ascent.
-   **$\boldsymbol{m}_t$ (First Moment/Momentum)**: An exponential moving average of past gradients. It helps smooth out noisy gradients and allows the optimizer to build up velocity in consistent directions, acting like a heavy ball rolling down a hill.
-   **$\boldsymbol{v}_t$ (Second Moment)**: An exponential moving average of *squared* gradients. This estimates the "magnitude" or variance of the gradients. Large values mean the landscape is steep or unstable in that dimension.
-   **$\hat{\boldsymbol{m}}_t, \hat{\boldsymbol{v}}_t$ (Bias Correction)**: We typically initialize $m_0 = 0$. Since the decay rate $\beta$ is close to 1 (e.g., 0.9 or 0.999), the early estimates of $m_t$ are heavily biased towards zero.
    -   The term $(1 - \beta^t)$ acts as a correction factor. At step $t=1$, if $\beta=0.9$, then $(1 - 0.9^1) = 0.1$, so we divide by 0.1 (multiply by 10) to scale up the initial small value.
    -   As $t \to \infty$, $\beta^t \to 0$, so the divisor becomes 1, and the bias correction disappears once the moving average has warmed up.

**2. How different optimizers define the update $\boldsymbol{u}_t$:**

-   **SGDM (Stochastic Gradient Descent with Momentum):**
    $$ \boldsymbol{u}_t = \boldsymbol{m}_t $$
    Uses simple momentum. It accumulates velocity but doesn't adapt the learning rate per parameter.

-   **RMSProp:**
    $$ \boldsymbol{u}_t = \boldsymbol{g}_t / (\sqrt{\boldsymbol{v}_t} + \epsilon) $$
    Divides the gradient by the root mean square of recent gradients. If gradients are consistently large, it reduces the step size to prevent instability.

-   **Adam (Adaptive Moment Estimation):**
    $$ \boldsymbol{u}_t = \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon) $$
    The gold standard. It combines Momentum (smoothing) and RMSProp (scaling) with bias correction.
    -   **How it adapts:** The magic lies in the division by $\sqrt{\hat{\boldsymbol{v}}_t}$. Since these operations form a vector, they happen element-wise for every single parameter.
    -   **Mechanism:** If a specific parameter has frequently large gradients (high variance $\hat{\boldsymbol{v}}_t$), the denominator becomes large, effectively **lowering** its learning rate to prevent overshooting. Conversely, for parameters with rare or small gradients, the denominator is small, **boosting** their effective learning rate so they don't get left behind.

-   **SignSGDM:**
    $$ \boldsymbol{u}_t = \text{sign}(\boldsymbol{m}_t) $$
    Ignores magnitude entirely, stepping only based on the direction of momentum. This is theoretically interesting for stability and communication efficiency.

-   **Muon:**
    $$ \boldsymbol{u}_t = \text{msign}(\boldsymbol{m}_t) $$
    A newer optimizer designed for large-scale training.
    -   **Matrix Sign:** Unlike standard optimizers that treat parameters as a flat vector of numbers, Muon respects the 2D structure of weight matrices.
    -   **How it works:** It effectively computes a Singular Value Decomposition (SVD) of the momentum matrix ($U \Sigma V^T$), replaces all singular values in $\Sigma$ with 1s, and reconstructs the matrix. This "orthogonalizes" the update, ensuring that the optimizer takes steps of unit magnitude in every principal direction. This is fundamentally different from element-wise sign, which only looks at individual values in isolation.

With the exception of SGDM, the examples listed here are all considered forms of **adaptive learning rate optimizers**.

Our starting point is the **Exponential Moving Average (EMA)** perspective.

**What is an EMA?**
In statistics, an EMA is used to smooth out noisy data. The standard formula for updating an average $A_t$ with a new observation $x_t$ is:
$$ A_t = (1 - \alpha) A_{t-1} + \alpha x_t $$
Here, $\alpha$ is a small number (like 0.01). This says: "Keep 99% of the old average, and mix in 1% of the new observation."

**Applying this to Weight Decay:**
We can rewrite the standard weight update rule from Equation (1) to look exactly like this EMA formula:

$$
\boldsymbol{\theta}_t = \underbrace{(1 - \lambda_t \eta_t)}_{\text{Keep majority of old weights}} \boldsymbol{\theta}_{t-1} + \underbrace{\lambda_t \eta_t}_{\text{Mix in small fraction}} \underbrace{\left(-\frac{\boldsymbol{u}_t}{\lambda_t} \right)}_{\text{"New Observation"}} \tag{3}
$$

**Why does this matter?**
This algebraic trick changes how we interpret training.
-   **Standard View:** "Take the weights and subtract the gradient."
-   **EMA View:** "The weights are a **moving average** of the 'target' $-\boldsymbol{u}_t / \lambda_t$."
    -   **What is this target?** Imagine a tug-of-war. The optimizer wants to push weights in direction $\boldsymbol{u}_t$, while weight decay pulls them back to $0$. The point $-\boldsymbol{u}_t / \lambda_t$ is the **equilibrium point** where these two forces would perfectly balance out.
    -   So, instead of taking a single step, the model is constantly *smoothing* its trajectory towards this shifting equilibrium point.
-   The term $\lambda_t \eta_t$ acts exactly like the coefficient $\alpha$ in EMA. It controls the "memory length" of the system. If $\lambda_t \eta_t$ is very small, the model remembers the past for a long time (high inertia). If it is large, the model updates quickly but forgets the past sooner.


At this point, Weight Decay appears as a weighted average form of model parameters and $-\boldsymbol{u}_t/\lambda_t$. 

> At first the expression (3) may seem like a weighted sum, not a weighted average. But the defining property that turns a "weighted sum" into a "weighted average" is precisely that the coefficients add up to 1.


The sliding average perspective is not new; articles such as *«How to set AdamW's weight decay as you scale model and dataset size»* and *«Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training»* have already discussed it. This article, however, calculates various aspects more carefully under this perspective.

In the following sections, we mainly take Adam as an example, and finally discuss the applicability of other optimizers. The calculation process will overlap significantly with *«Asymptotic Estimation of AdamW's Weight RMS (Part 1)»* and *«(Part 2)»*, and readers can cross-reference them.

### Iterative Expansion #

For simplicity, let us first consider constant $\lambda, \eta$. Let $\beta_3 = 1 - \lambda \eta$. Think of $\beta_3$ as a "retention rate" - it tells us how much of the old weight matrix is kept at each step (e.g., 0.9999).

Then the update rule becomes:
$$ \boldsymbol{\theta}_t = \beta_3 \boldsymbol{\theta}_{t-1} + (1 - \beta_3)(-\boldsymbol{u}_t / \lambda) $$
This form looks familiar—it's exactly the same structure as the momentum term $m_t$ or variance term $v_t$ in Equation (2).

**1. Unrolling the Recursion (Equation 4):**
Let's unroll the update rule $\boldsymbol{\theta}_t = \beta_3 \boldsymbol{\theta}_{t-1} + (1 - \beta_3)(-\boldsymbol{u}_t / \lambda)$ step by step.

**Step $t=1$:**

$$ \boldsymbol{\theta}_1 = \beta_3 \boldsymbol{\theta}_0 + (1-\beta_3)(-\boldsymbol{u}_1/\lambda) $$

**Step $t=2$:**
Substitute $\boldsymbol{\theta}_1$ into the rule for $\boldsymbol{\theta}_2$:

$$ \boldsymbol{\theta}_2 = \beta_3 \underbrace{[\beta_3 \boldsymbol{\theta}_0 + (1-\beta_3)(-\boldsymbol{u}_1/\lambda)]}_{\boldsymbol{\theta}_1} + (1-\beta_3)(-\boldsymbol{u}_2/\lambda) $$

$$ \boldsymbol{\theta}_2 = \beta_3^2 \boldsymbol{\theta}_0 + \beta_3(1-\beta_3)(-\boldsymbol{u}_1/\lambda) + (1-\beta_3)(-\boldsymbol{u}_2/\lambda) $$

**Step $t=3$:**
Multiply the previous result by $\beta_3$ again and add the new term. This creates a pattern where older updates get multiplied by more $\beta_3$'s.

Generalizing this to time $t$, we get the sum of all past updates, weighted by powers of $\beta_3$:
If we expand this formula recursively back to the start ($t=0$), we get a sum of all past updates, weighted by how "old" they are:

$$
\boldsymbol{\theta}_t = \underbrace{\beta_3^t \boldsymbol{\theta}_0}_{\text{Decayed Initial Weight}} + \underbrace{(1 - \beta_3) \sum_{i=1}^t \beta_3^{t-i} (-\boldsymbol{u}_i / \lambda)}_{\text{Weighted Sum of All Past Updates}} \tag{4}
$$

-   The first term shows that the initial weights $\boldsymbol{\theta}_0$ decay exponentially and eventually vanish.
-   The second term accumulates every past update $\boldsymbol{u}_i$, but recent updates (where $t-i$ is small) have higher weights ($\approx 1$), while old updates (where $t-i$ is large) have tiny weights ($\approx 0$).

**2. The Mean Field Approximation (Equation 5):**
For Adam, $\boldsymbol{u}_t = \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)$. As training progresses ($t \to \infty$), the bias correction terms $\beta^t$ vanish, so we can ignore the "hats".
The expression for $\boldsymbol{u}_i$ is nonlinear (it involves division). We need to simplify the summation term in Equation (4): $\sum \beta_3^{t-i} (\boldsymbol{m}_i / \sqrt{\boldsymbol{v}_i})$.

Sums of ratios are notoriously difficult to analyze because $\sum (a/b) \neq (\sum a) / (\sum b)$. To proceed, we apply a **Mean Field Approximation**, treating the numerator (momentum) and denominator (variance) as separate fields.

*   **Step 1: Define Normalized Weights**
    Let's define a probability distribution of weights $\alpha_{t,i}$ so they sum to 1:
    $$ \alpha_{t,i} = \frac{\beta_3^{t-i}}{\sum_{j=1}^t \beta_3^{t-j}} = \frac{\beta_3^{t-i}(1-\beta_3)}{1-\beta_3^t} $$
    
*   **Step 2: Rewrite the Sum as an Expectation**
    The term we want to approximate is the weighted average of the ratio:
    $$ \text{Target} = \sum_{i=1}^t \alpha_{t,i} \left( \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}} \right) = \mathbb{E}_{\alpha} \left[ \frac{m}{\sqrt{v}} \right] $$

*   **Step 3: Apply The Approximation**
    We approximate the "Average of Ratios" as the "Ratio of Averages":
    $$ \mathbb{E}_{\alpha} \left[ \frac{m}{\sqrt{v}} \right] \approx \frac{\mathbb{E}_{\alpha}[m]}{\sqrt{\mathbb{E}_{\alpha}[v]}} $$
    
*   **Step 4: Substitute Back**
    Plugging this back into our variables gives us Equation (5):

$$
\frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{u}_i \underbrace{\approx}_{\text{Mean Field}} \frac{\sum \alpha_{t,i} \boldsymbol{m}_i}{\sqrt{\sum \alpha_{t,i} \boldsymbol{v}_i}} = \frac{\bar{\boldsymbol{m}}_t}{\sqrt{\bar{\boldsymbol{v}}_t}} \tag{5}
$$

Where $\bar{\boldsymbol{m}}_t$ and $\bar{\boldsymbol{v}}_t$ are simply the sliding averages of the momentum and variance terms:
$$
\bar{\boldsymbol{m}}_t \triangleq \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{m}_i, \quad \bar{\boldsymbol{v}}_t \triangleq \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \boldsymbol{v}_i
$$

Now we need to calculate $\bar{\boldsymbol{m}}_t$ and $\bar{\boldsymbol{v}}_t$.
To do this, we first need to express $\boldsymbol{m}_t$ and $\boldsymbol{v}_t$ as sums of past gradients, similar to how we expanded $\boldsymbol{\theta}_t$ in Eq (4):
$$ \boldsymbol{m}_t = (1-\beta_1)\sum_{j=1}^t \beta_1^{t-j} \boldsymbol{g}_j $$
$$ \boldsymbol{v}_t = (1-\beta_2)\sum_{j=1}^t \beta_2^{t-j} \boldsymbol{g}_j^2 $$

Now, substitute this expanded form of $\boldsymbol{m}_t$ back into the definition of $\bar{\boldsymbol{m}}_t$:
$$
\bar{\boldsymbol{m}}_t = \frac{1-\beta_3}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \underbrace{\left( (1-\beta_1)\sum_{j=1}^i \beta_1^{i-j} \boldsymbol{g}_j \right)}_{\boldsymbol{m}_i}
$$
This gives us a double summation. By rearranging the terms, we arrive at Equation (6):

$$
\bar{\boldsymbol{m}}_t = \frac{(1-\beta_3)(1-\beta_1)}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_1^{i-j} \boldsymbol{g}_j = \frac{(1-\beta_3)(1-\beta_1)}{(1-\beta_3^t)(\beta_3 - \beta_1)} \sum_{j=1}^t (\beta_3^{t-j+1} - \beta_1^{t-j+1}) \boldsymbol{g}_j \tag{6}
$$

**Understanding the Summation Logic:**
1.  **The Inner Sum**: The term $\sum_{j=1}^i \beta_1^{i-j}$ represents the accumulation of momentum up to step $i$. This is a **geometric series**, where past gradients decay by a factor of $\beta_1$ at each step.
2.  **Swapping the Order**: The derivation swaps the double summation from $\sum_{i=1}^t \sum_{j=1}^i$ to $\sum_{j=1}^t \sum_{i=j}^t$. Intuitively, instead of asking "what gradients are included at step $i$?", we ask "how many times does a specific gradient $\boldsymbol{g}_j$ appear in all future steps?".
3.  **The Result**: The final coefficient $\frac{\beta_3^{t-j+1} - \beta_1^{t-j+1}}{\beta_3 - \beta_1}$ tells us the exact weight of the gradient from step $j$ in the final model. It shows that older gradients (small $j$, large $t-j$) have exponentially less influence.

 $$
\bar{\boldsymbol{v}}_t = \frac{(1-\beta_3)(1-\beta_2)}{1-\beta_3^t} \sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_2^{i-j} \boldsymbol{g}_j^2 = \frac{(1-\beta_3)(1-\beta_2)}{(1-\beta_3^t)(\beta_3 - \beta_2)} \sum_{j=1}^t (\beta_3^{t-j+1} - \beta_2^{t-j+1}) \boldsymbol{g}_j^2 \tag{7}
$$

The exchange of summation symbols is based on the identity $\sum_{i=1}^t \sum_{j=1}^i a_i b_j = \sum_{j=1}^t \sum_{i=j}^t a_i b_j$. In summary, we have:

$$
\boldsymbol{\theta}_t = \beta_3^t \boldsymbol{\theta}_0 + (1 - \beta_3^t) (-\bar{\boldsymbol{u}}_t / \lambda) \tag{8}
$$

The weight $\boldsymbol{\theta}_t$ is the training result we want, expressed as a weighted average of $\boldsymbol{\theta}_0$ and $-\bar{\boldsymbol{u}}_t / \lambda$. Here, $\boldsymbol{\theta}_0$ is the initial weight, and $\bar{\boldsymbol{u}}_t$ is data-dependent. Under the mean field approximation, it is approximately equal to $\bar{\boldsymbol{m}}_t / \sqrt{\bar{\boldsymbol{v}}_t}$, and $\bar{\boldsymbol{m}}_t$ and $\bar{\boldsymbol{v}}_t$ can be expressed as weighted sums of the gradients at each step. Taking $\bar{\boldsymbol{m}}_t$ as an example, the weight of the gradient at step $j$ is proportional to $\beta_3^{t-j+1} - \beta_1^{t-j+1}$.

### Memory Period #

We are mainly concerned with pre-training, which is characterized by being **Single-Epoch** (most data is seen only once). Therefore, one of the keys to training good results is not to forget early data. Assuming the training data has been globally shuffled, it is reasonable to assume that the data in every Batch is equally important.

Data is linearly superimposed into $\bar{\boldsymbol{m}}_t$ in the form of gradients. Assuming that the gradient at each step contains only the information of the current Batch, then for a certain Batch not to be forgotten, the coefficient $\beta_3^{t-j+1} - \beta_1^{t-j+1}$ must not be too small. Consider the function $f(s) = \beta_3^s - \beta_1^s$. It is a function that increases first and then decreases, but because $\beta_3$ is closer to 1 than $\beta_1$, the number of increasing steps is small, and in the long range, it is mostly an exponential decay.

Overall, the trend is that the coefficient gets smaller as the distance increases. To ensure the model does not forget every Batch, the coefficient at the farthest point must not be too small. Assuming the coefficient must be no less than $c \in (0, 1)$ to be remembered, when $s$ is large enough, $\beta_1^s$ tends to 0 first, so $\beta_3^s - \beta_1^s \approx \beta_3^s$.

From $\beta_3^s \ge c$, we can solve for $s$:
1.  Take the natural logarithm ($\ln$) of both sides: $s \cdot \ln(\beta_3) = \ln(c)$.
2.  Use the **Taylor Expansion approximation** $\ln(1-x) \approx -x$ for small $x$. Since $\beta_3 = 1 - \lambda \eta$, we have $\ln(\beta_3) \approx -\lambda \eta$.
3.  Substitute this back: $s \cdot (-\lambda \eta) \approx \ln(c)$.
4.  Solve for $s$: $s \approx \frac{\ln c}{-\lambda \eta} = \frac{-\ln c}{\lambda \eta}$.

Since $-\ln c$ is just a positive constant, we conclude that $s$ is proportional to $1/\lambda \eta$. This result, $s \le O(1/\lambda \eta)$, is the model's **Memory Period**. It tells us that higher weight decay or learning rate leads to a shorter memory span.


So, can we simply set $\lambda=0$ blindly to make the memory period infinite, so we don't have to worry about forgetting? Theoretically yes, but this is not a good choice. Another function of Weight Decay is to help the model forget the initialization. From equation (8), the weight of the initialization $\boldsymbol{\theta}_0$ is $\beta_3^t$. If $\beta_3$ is too large or the training steps $t$ are too small, the proportion of initialization is still high, and the model may still be in an underfitting stage.

In addition, Weight Decay is beneficial for stabilizing the "internal health" of the model. In *«Asymptotic Estimation of AdamW's Weight RMS (Part 1)»*, we derived that the asymptotic result of AdamW's Weight RMS is $\sqrt{\eta / 2\lambda}$. If $\lambda=0$, the Weight RMS will expand at a rate of $\sqrt{\eta t}$. This means that setting $\lambda=0$ directly may bring about internal abnormalities such as weight explosion.

Therefore, $\beta_3$ cannot be too small (to avoid forgetting early data) and cannot be too large (to avoid underfitting or weight explosion). A more suitable setting is to make $1/\lambda \eta$ proportional to the training steps. If it is a Multi-Epoch training scenario, consider making $1/\lambda \eta$ proportional to the training steps of a single Epoch.

### Dynamic Version #

In actual training, we apply dynamically changing LR Schedules, such as Cosine Decay, Linear Decay, WSD (Warmup-Stable-Decay), etc. Therefore, the above results for static Weight Decay and Learning Rate do not completely match practice, and we need to generalize them to the dynamic version.

Starting from equation (3), using the approximation $1 - \lambda_t \eta_t \approx e^{-\lambda_t \eta_t}$, and expanding iteratively, we get:

$$
\boldsymbol{\theta}_t = (1 - \lambda_t \eta_t)\boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t \approx e^{-\lambda_t \eta_t} \boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t = e^{-\kappa_t} \left( \boldsymbol{\theta}_0 - \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{u}_i \right) \tag{9}
$$

Here, we used an approximation to convert the product of decay terms into an exponential sum.
Instead of multiplying $(1 - \lambda_i \eta_i)$ at every step, we sum the exponents:
$$ \prod_{i=1}^t (1 - \lambda_i \eta_i) \approx \exp\left( \sum_{i=1}^t (-\lambda_i \eta_i) \right) = e^{-\kappa_t} $$
This allows us to treat the discrete schedule as a continuous integral-like term $\kappa_t$, greatly simplifying the math for dynamic schedules.


Where $\kappa_t = \sum_{i=1}^t \eta_i \lambda_i$. Continuing to set $z_t = \sum_{i=1}^t e^{\kappa_i} \eta_i$, we can obtain the same mean field approximation:

$$
\bar{\boldsymbol{u}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{u}_i = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}} \approx \frac{\bar{\boldsymbol{m}}_t}{\sqrt{\bar{\boldsymbol{v}}_t}} \tag{10}
$$

Where $\bar{\boldsymbol{m}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{m}_i$ and $\bar{\boldsymbol{v}}_t \triangleq \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{v}_i$. Substituting the expansions of $\boldsymbol{m}_t, \boldsymbol{v}_t$:

$$
\bar{\boldsymbol{m}}_t = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{m}_i = \frac{1-\beta_1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \sum_{j=1}^i \beta_1^{i-j} \boldsymbol{g}_j = \sum_{j=1}^t \boldsymbol{g}_j \underbrace{\frac{1-\beta_1}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_1^{i-j} \eta_i}_{\text{denoted as } \bar{\beta}_1(j,t)} \tag{11}
$$

$$
\bar{\boldsymbol{v}}_t = \frac{1}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \boldsymbol{v}_i = \frac{1-\beta_2}{z_t} \sum_{i=1}^t e^{\kappa_i} \eta_i \sum_{j=1}^i \beta_2^{i-j} \boldsymbol{g}_j^2 = \sum_{j=1}^t \boldsymbol{g}_j^2 \underbrace{\frac{1-\beta_2}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_2^{i-j} \eta_i}_{\text{denoted as } \bar{\beta}_2(j,t)} \tag{12}
$$

It can be seen that compared to the static Weight Decay and Learning Rate, the dynamic version does not change much in form, except that the weighting coefficients of the gradient become slightly more complex $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$. Specifically, when $\beta_1, \beta_2 \to 0$, $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$ simplify to:

$$
\bar{\beta}_1(j,t) = \bar{\beta}_2(j,t) = \frac{e^{\kappa_j} \eta_j}{z_t} \tag{13}
$$

### Optimal Schedule #

There are many things we can do next. The most basic one is to calculate $\bar{\beta}_1(j,t)$ and $\bar{\beta}_2(j,t)$ and estimate the memory period according to specific WD Schedules and LR Schedules. However, here we choose to do something more extreme—directly reverse-engineer an **Optimal WD Schedule and LR Schedule**.

Specifically, we assumed earlier that the data has been globally shuffled, so the data in each Batch is equally important. However, the coefficient obtained in the static version $\bar{\beta}_1(j,t) \propto \beta_3^{t-j+1} - \beta_1^{t-j+1}$ is not constant, but changes with distance, which does not fully match "each Batch is equally important." If conditions permit, we expect it to be equal to a constant. Based on this expectation, we can solve for the corresponding $\lambda_j, \eta_j$.

For simplicity, we start with $\beta_1, \beta_2 \to 0$. In this case, the expectation condition can be written as $\forall 0 \le i, j \le t, e^{\kappa_i} \eta_i / z_t = e^{\kappa_j} \eta_j / z_t$.
We want every batch to have the **exact same weight**.
Looking at Equation (10), the weight for step $i$ is $e^{\kappa_i} \eta_i$. To make this constant, we equate the terms for step $j$ and $j-1$:
$$ e^{\kappa_{j-1}} \eta_{j-1} = e^{\kappa_j} \eta_j $$
Rearranging gives $\frac{\eta_{j-1}}{\eta_j} = e^{\kappa_j - \kappa_{j-1}}$.
Since $\kappa_t$ is the sum of $\lambda \eta$, the difference $\kappa_j - \kappa_{j-1}$ is simply $\lambda_j \eta_j$.
Substituting this back, we get:
$$
\frac{e^{\lambda_j \eta_j}}{\eta_j} = \frac{1}{\eta_{j-1}} \implies \eta_{j-1} = \eta_j e^{\lambda_j \eta_j} \tag{14}
$$


This provides a numerical method for solving $\lambda_j, \eta_j$: after obtaining $\eta_{j-1}$ at each step, $\lambda_j, \eta_j$ can be obtained by solving this nonlinear equation, so the entire sequence can be obtained recursively starting from $\eta_1$. If a more analytical result is desired, derivatives can be used to approximate the difference: taking the logarithm of both sides gives $\lambda_j \eta_j + \log \eta_j - \log \eta_{j-1} = 0$. Treating $\lambda_j, \eta_j$ as continuous functions $\lambda_s, \eta_s$, and $\log \eta_j - \log \eta_{j-1}$ as the derivative approximation of $\log \eta_s$, we have:

$$
\lambda_s \eta_s + \frac{\dot{\eta}_s}{\eta_s} \approx 0 \tag{15}
$$

If $\lambda_s$ is taken as a constant $\lambda$, then we can solve the differential equation:
1.  Start with $\lambda \eta + \frac{\dot{\eta}}{\eta} = 0$.
2.  Multiply by $\eta$ to separate variables: $\dot{\eta} = -\lambda \eta^2$.
3.  Integrate both sides: $\int \eta^{-2} d\eta = \int -\lambda dt \implies -\frac{1}{\eta} = -\lambda t + C$.
4.  Solve for $\eta$: $\eta(t) = \frac{1}{\lambda t - C}$. Adding boundary conditions gives the final form:

$$
\eta_s \approx \frac{\eta_{\max}}{\lambda \eta_{\max} s + 1} \tag{16}
$$


This is the **Best LR Schedule under constant Weight Decay**. It does not require a preset endpoint $t$ and minimum learning rate $\eta_{\min}$, which means it can be trained infinitely, similar to the Stable stage of WSD, but it automatically balances the coefficients of the gradient at each step. However, it also has a drawback: when $s \to \infty$, it tends to 0. From *«Asymptotic Estimation of AdamW's Weight RMS (Part 2)»*, we know that Weight RMS will tend to $\lim_{s \to \infty} \eta_s / 2\lambda_s$, so this drawback may bring the risk of weight collapse.

To solve this problem, we can consider letting $\lambda_s = \alpha \eta_s$, where $\alpha = \lambda_{\max} / \eta_{\max}$ is a constant. In this case, we perform a similar derivation:
1.  Substitute $\lambda = \alpha \eta$ into the ODE: $\alpha \eta^2 + \frac{\dot{\eta}}{\eta} = 0 \implies \dot{\eta} = -\alpha \eta^3$.
2.  Integrate $\eta^{-3} d\eta = -\alpha dt$: the left side becomes $-\frac{1}{2}\eta^{-2}$.
3.  Solving for $\eta$ yields an **inverse square root** relationship.

$$
\eta_s \approx \frac{\eta_{\max}}{\sqrt{2 \lambda_{\max} \eta_{\max} s + 1}}, \quad \lambda_s \approx \frac{\lambda_{\max}}{\sqrt{2 \lambda_{\max} \eta_{\max} s + 1}} \tag{17}
$$


Correspondingly, $e^{\kappa_s} \approx \sqrt{2 \lambda_{\max} \eta_{\max} s + 1}$, $e^{\kappa_s} \eta_s \approx \eta_{\max}$, $z_t \approx \eta_{\max} t$, and $\bar{\beta}_1(j,t) = \bar{\beta}_2(j,t) \approx 1/t$.

### General Results #

The current results, such as Eq (16) and Eq (17), are based on $\beta_1, \beta_2 = 0$. Do the results need to change when they are not equal to 0? More generally, the above results are based on the Adam optimizer; to what extent can they be generalized to other optimizers?

First, let's look at the problem when $\beta_1, \beta_2 \neq 0$. The answer is that when $t$ is large enough, the conclusion does not need to be changed much. Taking $\bar{\beta}_1(j,t)$ as an example, under the above optimal schedule, $e^{\kappa_i} \eta_i$ is equal to a constant (related to $t$). Then according to the definition:

$$
\bar{\beta}_1(j,t) = \frac{1-\beta_1}{z_t} \sum_{i=j}^t e^{\kappa_i} \beta_1^{i-j} \eta_i \propto \sum_{i=j}^t \beta_1^{i-j} = \frac{1 - \beta_1^{t-j+1}}{1-\beta_1} \tag{18}
$$

When $t$ is large enough, $\beta_1^{t-j+1} \to 0$, so this can also be seen as a constant independent of $j$. As mentioned earlier, for $\beta_1, \beta_2$, the fact that "$t$ is large enough" is almost certain, so the results for $\beta_1, \beta_2 = 0$ can be used directly.

As for the optimizers, the ones we mentioned earlier are SGDM, RMSProp, Adam, SignSGDM, and Muon, which can be divided into two categories. Among them, SGDM is one category; its $\bar{\boldsymbol{u}}_t$ is directly $\bar{\boldsymbol{m}}_t$, and does not even need to use the mean field approximation, so the results up to Eq (15) are applicable. However, Eq (16) and Eq (17) are probably not the most suitable, because the asymptotic Weight RMS of SGDM also depends on the gradient norm [Reference], so the gradient norm needs to be taken into account, which is relatively more complicated.

The remaining RMSProp, Adam, SignSGDM, and Muon are classified into another category, all belonging to adaptive learning rate optimizers. Their update rules all have the homogeneous form of $\text{gradient} / \sqrt{\text{gradient}^2}$. In this case, if we still believe in the mean field approximation, we can get the same $\bar{\boldsymbol{m}}_t$, same $\beta_1(j,t)$, so the results up to Eq (15) are also applicable; and for this type of homogeneous optimizer, it can be proven that Weight RMS is also asymptotically proportional to $\sqrt{\eta / \lambda}$, so Eq (16) and Eq (17) can also be reused.

### Hypothesis Discussion #

Our derivation has come to a temporary halt. In this section, we will discuss the assumptions relied upon in the derivation.

Throughout the text, there are two main large assumptions used in the derivation worth discussing. The first assumption is the **Mean Field Approximation**, first introduced in *«Rethinking Learning Rate and Batch Size (Part 2): Mean Field»*. The Mean Field itself is certainly not new; it is a classic approximation in physics. However, using it to analyze the dynamics of optimizers appears to be a first by the author. Currently, it has been used to estimate the Batch Size, Update RMS, Weight RMS, etc., of optimizers, and the results seem reasonable.

Regarding the validity of the Mean Field Approximation, we cannot comment too much; it reflects a kind of belief. On the one hand, based on the reasonableness of existing estimation results, we believe it will continue to be reasonable, at least providing valid asymptotic estimates for some scalar indicators. On the other hand, for adaptive learning rate optimizers, due to the nonlinearity of their update rules, the difficulty of analysis increases greatly. Apart from the Mean Field Approximation, we actually have no other calculation tools to use.

The most typical example is Muon. Because it is a non-Element-wise operation, previous calculation methods like those for SignSGD that work component-wise lose their effect, while the Mean Field Approximation still works (refer to *«Rethinking Learning Rate and Batch Size (Part 3): Muon»*). Therefore, the Mean Field Approximation actually provides a unified, effective, and concise calculation method for the analysis and estimation of a large class of adaptive learning rate optimizers. Currently, there seems to be no other method with the same effect, so we can only continue to believe in it.

The second major assumption is **"the gradient at each step only contains the information of the current Batch."** This assumption is inherently wrong because the gradient depends not only on the data of the current Batch but also on the parameters of the previous step, and the parameters of the previous step naturally contain historical information. However, we can try to remedy this. Theoretically speaking, every Batch brings new information; otherwise, this Batch would have no meaning. So the remedy is to change it to "the gradient at each step contains roughly the same incremental information."

Of course, upon careful reflection, this statement is also controversial, because the more you learn and the wider the coverage, the less unique information later Batches have. However, we can struggle a bit more, that is, divide knowledge into two major categories: "laws" and "facts." Factual knowledge—for example, that a certain theorem was discovered by a certain mathematician—can only be relied upon by memory. So we can consider changing it to "the gradient at each step contains roughly the same factual knowledge." In short, from practice, the LR Schedule obtained by "treating every gradient step equally" seems to really have benefits, so we can always try to construct an explanation for it.

The recent paper *«How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining»* provides indirect evidence. It considered curriculum learning where data quality goes from low to high and found that aggressive LR Decay would negate the advantages of curriculum learning. Our result is that the weight of each Batch is Eq (13), proportional to the Learning Rate. If LR Decay is too aggressive, the weight of the high-quality data later on will be too small, resulting in poor performance. Being able to reasonably explain this phenomenon conversely shows the rationality of our assumption.

### Experimental Verification #

To verify the theoretical derivation of the "Optimal Schedule," we conducted a benchmark sweep on the **Blueberry-Nano** model (88M parameters) using a **8M token** training subset. We compared the standard constant baseline against both the Inverse Time (Eq 16) and Inverse Sqrt (Eq 17) schedules, testing the impact of coupling weight decay decay.

| Configuration | Schedule | Coupled WD | WD ($\lambda$) | Val Loss |
|:---|:---|:---:|:---:|:---:|
| **Baseline** | Constant | No | 0.2 | 4.84 |
| **Optimal (Eq 17)** | Inverse Sqrt | **Yes** | 0.4 | **4.75** |
| **Eq 17 Variation** | Inverse Sqrt | No | 0.2 | 4.76 |
| **Eq 16 Variation** | Inverse Time | No | 0.2 | 4.85 |

**Full Sweep Visualization (Steps, Time, and Zoomed View):**

![Full Sweep Comparison](./assets/full_sweep_comparison.png)

**Theoretical Insights and Empirical Learnings:**

The benchmark results provide conclusive empirical support for the article's core mathematical premises:

1.  **Validation of Equal Batch Weighting**: The superiority of the Inverse Sqrt schedule ($1/\sqrt{s}$) over the Inverse Time schedule ($1/s$) confirms that for adaptive optimizers like Muon, treating every gradient step as contributing equal incremental information is the "Optimal" strategy. The $1/s$ schedule decays too aggressively, essentially "wasting" the high-quality data seen in later stages of training.
2.  **Coupling as an Information Stabilizer**: Decaying Weight Decay alongside the Learning Rate (coupling) acts as a critical mechanism to maintain a consistent "Memory Period" relative to the training progress. The zoomed-in plot reveals that Experiment 6 (Coupled, WD=0.4) achieves the smoothest and deepest convergence, whereas constant WD configurations eventually hit a plateau where the regularization becomes misaligned with the effective step size.
3.  **Enhanced Sample and Temporal Efficiency**: The "Optimal" schedule isn't just better in terms of final loss; it is significantly more efficient. By maintaining a theoretically calibrated training trajectory, we extract more value from every 1M tokens, achieving lower loss in less wall-clock time than standard constant-parameter baselines.

### Summary #

This article understands Weight Decay (WD) and Learning Rate (LR) from the perspective of sliding average, and explores the optimal WD Schedule and LR Schedule under this perspective.