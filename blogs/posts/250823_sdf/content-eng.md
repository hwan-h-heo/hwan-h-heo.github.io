title: SDF and Eikonal Equation
date: August 23, 2025
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

<button id="copyButton">
<i class="bi bi-share-fill"></i>
</button>

<div id="myshare_modal" class="share_modal">
<div class="share_modal-content">
<span class="share_modal_close">×</span>
<p><strong>Link Copied!</strong></p>
<div class="copy_indicator-container">
<div class="copy_indicator" id="share_modalIndicator"></div>
</div>
</div>
</div>

---

#### TL;DR

Let's explore the derivation and proof of the relationship between a Signed Distance Field (SDF) and the Eikonal Equation.
While I've written a similar post on velog before, this article aims to provide a more rigorous treatment of the subject.

## Introduction

A **SDF** (Signed Distance Function) is a representation that, for each point in space, provides the shortest distance to a surface. The set of points where this distance value is zero—the zero-level set or isosurface—defines the actual surface.

![](https://ar5iv.labs.arxiv.org/html/1901.05103/assets/x2.png)

SDFs are critical not only in modern fields like 3D generative AI and neural surface reconstruction but also in classical computer graphics for tasks such as physical simulation, animation, and mesh repair for watertightness.

When reading papers related to SDFs, one inevitably encounters the [Eikonal equation](https://en.wikipedia.org/wiki/Eikonal_equation):
$$ \| \nabla f_\text{SDF} \| = 1 \\
$$
and the Eikonal regularization term (or Eikonal loss) derived from it:

$$ \arg \min_\theta \mathbb E_x \left ( \| \nabla f_{\text{SDF}}(x) \| -1   \right )^2
$$

where $\theta$ represents the MLP's parameters. In this post, we will explore why an SDF can be defined by this Eikonal equation.

---

## From SDF to the Eikonal Equation

First, let's establish a rigorous definition of an SDF.

For a set $\Omega \subset \mathbb{R}^n$ and its boundary $\partial\Omega$, the Signed Distance Function (SDF) $f: \mathbb{R}^n \to \mathbb{R}$ is defined as follows:

<p id="p-2">$$ f(x) = \begin{cases} d(x, \partial\Omega) & \text{if } x \in \Omega^c \\ 0 & \text{if } x \in \partial\Omega \\ -d(x, \partial\Omega) & \text{if } x \in \Omega \end{cases}
$$</p>

Here, $$d(x, \partial\Omega) = \inf_{y \in \partial\Omega} \|x - y\|$$ is the Euclidean Distance between a point $x$ and the set $\partial\Omega$.

What we aim to prove is the **Eikonal Equation**, which states that for every point $x$ where $f$ is differentiable:

$$ \|\nabla f(x)\| = 1
$$

---

## Proof 1 

> *Proof using Lipschitz Continuity*

This proof will rigorously use the triangle inequality to show that $f$ is a function with a Lipschitz constant of 1. From this, we will first derive the supremum of the gradient ($\|\nabla f\| \le 1$). The infimum ($\|\nabla f\| \ge 1$) will then be proven by considering the direction in which the SDF value changes most rapidly.

### 1.1. Supremum of the SDF Gradient

<!--$$\|\nabla f(x)\| \le 1-->
<!--$$-->

For any two points $x_1, x_2 \in \mathbb{R}^n$, let $p_1 \in \partial\Omega$ be the closest point on the boundary to $x_1$. Then,
$$ d(x_1, \partial\Omega) = \|x_1-p_1\|
$$
This relationship follows from the definition of the SDF.

Furthermore, by the definition of $f(x_1)$, the following holds for any point $p_2 \in \partial\Omega$:
$$ f(x_1) \le d(x_1, \partial\Omega) \le \|x_1-p_2\|
$$
Thus, for $f(x_1)$ and $f(x_2)$, we can establish the following relationship:
<p>
$$
\begin{aligned}
f(x_2) - f(x_1) & \ge d(x_2, \partial\Omega) - \|x_1-p_2\| \\ &= \|x_2-p_2\| - \|x_1-p_2\| .
\end{aligned}
$$
</p>
Using the reverse triangle inequality, this can be simplified as follows:
<p>
$$
\begin{aligned}
f(x_2) - f(x_1) &\ge -\|(x_2-p_2) - (x_1-p_2)\| \\ & = -\|x_2-x_1\| \\  \therefore |f(x_1) - f(x_2)| & \le \|x_1-x_2\|
\end{aligned}
$$
</p>

*cf.* $ \|a\|-\|b\| \ge -\|a-b\| $ 

This shows that the function $f$ is a **[1-Lipschitz Continuous Function](https://en.wikipedia.org/wiki/Lipschitz_continuity)**, meaning the distance between the function's values does not increase more than the distance between the points themselves.

If $f$ is differentiable at a point $x$, then by the definition of the gradient, we have:

<p>
$$
\begin{aligned}
|\nabla f(x) \cdot v| & = \lim_{t \to 0} \frac{|f(x+tv) - f(x)|}{t} \\ & \le \lim_{t \to 0} \frac{\|tv\|}{t} = \|v\|
\end{aligned}
$$
</p>

Now, let's choose the unit vector $v = \frac{\nabla f(x)}{\|\nabla f(x)\|}$ (assuming $\nabla f(x) \neq 0$):
$$\left | \nabla f(x) \cdot \frac{\nabla f(x)}{\|\nabla f(x)\|} \right | = \frac{\|\nabla f(x)\|^2}{\|\nabla f(x)\|} = \|\nabla f(x)\|$$
Therefore,
$$\|\nabla f(x)\| \le \left \|\frac{\nabla f(x)}{\|\nabla f(x)\|} \right \| = 1$$ 
which implies $\|\nabla f(x)\| \le 1$.

***

### 1.2. Infimum of the SDF Gradient

Next, we will prove that $\|\nabla f(x)\| \ge 1$ to establish the lower bound (infimum) of the gradient's magnitude.

Let's consider a point $x \in \mathbb{R}^n \setminus \partial\Omega$ where $f$ is differentiable. We also assume there exists a **unique** closest point $p \in \partial\Omega$ to $x$. (Such points constitute most of the space, excluding the 'medial axis'.)

Now, consider the unit vector $$ n = \frac{x-p}{\|x-p\|} $$, which points from $p$ to $x$, representing the direction of fastest movement away from the boundary.

To analyze the directional derivative in this direction, consider the point $x+tn$ for a sufficiently small positive number $t > 0$. Under the assumption that $p$ is the unique closest point to $x$, for a small enough $t$, the closest point to $x+tn$ will also be $p$. Therefore, the distance function value is calculated as:
<p>
$$
\begin{aligned}
d(x+tn, \partial\Omega) &= \|(x+tn) - p\| \\ &= \|(x-p) + tn\| \\ & = \|\|x-p\|n + tn\| \\ & = (\|x-p\|+t) = d(x, \partial\Omega)+t
\end{aligned}
$$
</p>

Applying the sign from the SDF definition, we get the following:
- If $x \in \Omega^c$ $(f > 0)$, then $f(x) = d(x, \partial\Omega)$, so $$ f(x+tn) = d(x, \partial\Omega)+t = f(x)+t$$ 
- If $x \in \Omega$ $(f < 0)$, then $f(x) = -d(x, \partial\Omega)$, so $$f(x+tn) = -(d(x, \partial\Omega)+t) = f(x)-t$$

Now, we can compute the directional derivative along $n$:
- For $x \in \Omega^c$:
    $$
    \nabla f(x) \cdot n = \lim_{t\to 0^+} \frac{f(x+tn)-f(x)}{t} = \lim_{t\to 0^+} \frac{t}{t} = 1
    $$
- For $x \in \Omega$ (where the direction of fastest increase is $-n$):
    $$
    \nabla f(x) \cdot (-n) = \lim_{t\to 0^+} \frac{f(x-tn)-f(x)}{t} = \lim_{t\to 0^+} \frac{t}{t} = 1
    $$
    Therefore, $\nabla f(x) \cdot n = -1$.

In both cases, $|\nabla f(x) \cdot n| = 1$.
From the Cauchy-Schwarz inequality, $|\nabla f(x) \cdot n| \le \|\nabla f(x)\| \|n\|$, and the fact that $\|n\|=1$, we can conclude:

$$ 1 \le \|\nabla f(x)\| $$

### 1.3. Conclusion

From $\|\nabla f(x)\| \le 1$ proven in section 1.1 and $\|\nabla f(x)\| \ge 1$ proven in section 1.2, it follows that for all points $x$ where $f$ is differentiable, the Eikonal equation holds:

$$
\|\nabla f(x)\| = 1
$$

---

## Proof 2 

> *Proof using Geometric Properties of Distance Function*

The second proof offers a more intuitive approach using the geometric properties of the distance function. For rigor, we first state the following well-known theorem regarding the gradient of a distance function.

> **Theorem.** If the distance function $d(x, S)$ is differentiable at a point $x$, and $p$ is the unique closest point to $x$ on $S$, then the gradient of $d$ is given by:
> $$ \nabla d(x, S) = \frac{x-p}{\|x-p\|} $$
>
> This theorem is derived from the geometric fact that the level sets of the distance function are sets of points equidistant from $S$. The gradient is perpendicular to these level sets and points in the direction of the fastest increase in value. This direction is precisely the unit vector from $p$ to $x$.

### 2.1. Gradient Calculation

As in the first proof, we assume a point $x \in \mathbb{R}^n \setminus \partial\Omega$ where $f$ is differentiable, and a unique closest point $p$ on $\partial\Omega$. We can calculate $\nabla f(x)$ by considering two cases.

**Case 1: $x \in \Omega^c$**

In this case, by the definition of an SDF, $f(x) = d(x, \partial\Omega)$. Applying the theorem directly gives:
$$ \nabla f(x) = \nabla d(x, \partial\Omega) = \frac{x-p}{\|x-p\|} $$

**Case 2: $x \in \Omega$**

In this case, $f(x) = -d(x, \partial\Omega)$, so the gradient is:
$$ \nabla f(x) = -\nabla d(x, \partial\Omega) = -\frac{x-p}{\|x-p\|} $$

### 2.2. Gradient's Norm

In both cases, the gradient $\nabla f(x)$ is either a unit vector or a unit vector multiplied by -1. Therefore, calculating the Euclidean norm of the gradient yields:

<p>
$$ 
\begin{aligned}
\|\nabla f(x)\| &= \left\| \pm \frac{x-p}{\|x-p\|} \right\| \\ & = \frac{\|x-p\|}{\|x-p\|} = 1 
\end{aligned}
$$
</p>

This shows that $f$ satisfies the Eikonal equation at all points where it is differentiable.

---

## Discussion


**Rigor and Assumptions**: Both proofs rely on the assumptions of **differentiability** of $f$ and the **uniqueness of the closest point**. In reality, an SDF is not differentiable at all points in space. The set of non-differentiable points is called the ['medial axis'](https://en.wikipedia.org/wiki/Medial_axis) or 'skeleton', which corresponds to the set of points that are equidistant to two or more points on $\partial\Omega$. However, since this set has a Lebesgue measure of 0, it is most accurate to state that an SDF satisfies the Eikonal equation ***almost everywhere***.

The Eikonal loss plays a critical role by forcing the scalar field output by the network to be more than just an implicit function; it compels the field to represent a true physical 'distance' in space. This ensures that the level sets remain stable and allows for the direct and accurate calculation of the surface normal vectors via the gradient, which is essential for high-quality rendering and geometric analysis.

In 3D reconstruction using Implicit Neural Networks (INNs), such as in NeRF, SDFs are often learned to reconstruct surfaces. Calculating the Eikonal loss term in this context requires the second derivative of the MLP. However, the `tcnn` library, which is essential for training INNs, does not officially support second derivatives. Consequently, SDF methods that use a hash-grid (or feature-grid) representation compute the gradient not analytically but numerically, using ***finite differences*** to obtain a ***numerical gradient***.

<video id="video-2"  controls style='width: 100%' autoplay playsinline loop >
    <source src='https://research.nvidia.com/labs/dir/neuralangelo/assets/numerical_gradient.mp4' type='video/mp4'>
</video>

A prime example is [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/), which calculates gradients in this manner.
This is partly to circumvent the technical limitations of the tcnn library, but more importantly, the numerical gradient provides a smoothing effect on the gradient field, which helps stabilize optimization in the early stages of training. Neuralangelo actively leverages this by employing a coarse-to-fine strategy that progressively recovers finer details as training advances, leading to the generation of incredible geometric detail as the hash grid resolution increases.

<video id="video-1"  controls style='width: 100%' autoplay playsinline loop >
    <source src='https://research.nvidia.com/labs/dir/neuralangelo/assets/coarse_to_fine.mp4' type='video/mp4'>
</video>

*Ref*: [Boundary regularity for the distance functions, and the eikonal equation](https://arxiv.org/pdf/2409.01774)