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


#### TL; DR

SDF (Signed Distance Field) 와 Eikonal Equation 의 관계에 대한 유도와 증명을 알아보자. 
velog 에서 비슷한 글을 작성한 적 있지만, 엄밀성을 더하여 글을 작성한다. 

## Introduction

**SDF** (Signed Distance Function) 는 공간 내의 각 점에 대해 해당 점에서 표면까지의 최단 거리를 나타내며, 그 거리 값이 0인 점들의 집합, 즉 zero-level set (isosurface) 이 바로 실제 surface 를 정의하는 representation 이다. 

![](https://ar5iv.labs.arxiv.org/html/1901.05103/assets/x2.png)

이는 Surface reconstruction, 3D generation 등 최신 3D AI 분야뿐만 아니라, watertight mesh repair, 충돌 처리, 애니메이션 등 물리학적 시뮬레이션에서도 중요한 역할을 한다.

SDF 과 관련된 논문을 읽다 보면 필연적으로 마주하게 되는 공식이 바로 [eikonal equation](https://en.wikipedia.org/wiki/Eikonal_equation):
$$ \| \nabla f_\text{SDF} \| = 1 \\
$$
와 이를 통해서 유도된 eikonal regularization term (or eikonal loss)

$$ \arg \min_\theta \mathbb E_x \left ( \| \nabla f_{\text{SDF}}(x) \| -1   \right )^2
$$

인데, ($\theta$: MLP's parameter) 이 글을 통해 SDF 가 왜 이러한 eikonal equation 을 통해 정의될 수 있는지 알아보자.

---

## SDF to Eikonal Equation

우선 SDF 에 대한 정의부터 엄밀하게 짚고 넘어가도록 하겠다. 

어떤 집합 $\Omega \subset \mathbb{R}^n$ 와 그 경계 $\partial\Omega$ 에 대해, Signed Distance Function (SDF) $f: \mathbb{R}^n \to \mathbb{R}$ 는 다음과 같이 정의된다.

<p id="p-2">$$ f(x) = \begin{cases} d(x, \partial\Omega) & \text{if } x \in \Omega^c \\ 0 & \text{if } x \in \partial\Omega \\ -d(x, \partial\Omega) & \text{if } x \in \Omega \end{cases}
$$</p>


여기서 $$d(x, \partial\Omega) = \inf_{y \in \partial\Omega} \|x - y\|$$ 는 점 $x$ 와 집합 $\partial\Omega$ 사이의 Euclidean Distance 이다.

우리가 증명하고자 하는 **Eikonal Equation** 은 $f$가 미분 가능한 모든 점 $x$에서 다음을 만족한다는 것이다.

$$ \|\nabla f(x)\| = 1
$$


---



## Proof 1 

> *Proof using Lipschitz Continuity*

이 증명은 원래의 아이디어인 삼각부등식을 더 엄밀하게 사용하여 $f$가 Lipschitz constant 1 을 갖는 함수임을 보이고, 이를 통해 gradient 의 supremum ($\|\nabla f\| \le 1$) 을 유도하도록 하겠다. infimum ($\|\nabla f\| \ge 1$) 은 SDF 의 값이 가장 빠르게 변하는 방향을 고려하여 증명한다.

### 1.1. Supremum of SDF Gradient

<!--$$\|\nabla f(x)\| \le 1-->
<!--$$-->

임의의 두 점 $x_1, x_2 \in \mathbb{R}^n$ 에 대하여, boundary 위의 최근접점을 $p_1 \in \partial\Omega$ 이라 하면, 
$$ d(x_1, \partial\Omega) = \|x_1-p_1\|
$$
SDF 의 정의에 따라 위의 관계로 나타낼 수 있다. 

또한 $f(x_1)$ 의 정의에 따라, 다음이 성립하므로,
$$ f(x_1) \le d(x_1, \partial\Omega) \le \|x_1-p_2\|
$$ 
$f(x_1)$ 과 $f(x_2)$ 에 대해 다음의 관계가 성립한다. 
<p>
$$
\begin{aligned}
f(x_2) - f(x_1) & \ge d(x_2, \partial\Omega) - \|x_1-p_2\| \\ &= \|x_2-p_2\| - \|x_1-p_2\| .
\end{aligned}
$$
</p>
위의 관계는 삼각부등식을 이용하여 다음과 같은 형태로 정리된다. 
<p>
$$
\begin{aligned}
f(x_2) - f(x_1) &\ge -\|(x_2-p_2) - (x_1-p_2)\| \\ & = -\|x_2-x_1\| \\  \therefore |f(x_1) - f(x_2)| & \le \|x_1-x_2\|
\end{aligned}
$$
</p>

*cf.* $ \|a\|-\|b\| \ge -\|a-b\| $ 

    
이는 함수 $f$가 Lipschitz constant 1을 갖는 **[1-Lipschitz Continuous Function](https://en.wikipedia.org/wiki/Lipschitz_continuity)** 임을 의미한다. (즉, 두 점 사이의 거리가 1 이상으로 증가하지 않는 함수이다)

만약 $f$가 점 $x$에서 미분 가능하다면, gradient 의 정의에 의해 다음이 성립한다.

<p>
$$
\begin{aligned}
|\nabla f(x) \cdot v| & = \lim_{t \to 0} \frac{|f(x+tv) - f(x)|}{t} \\ & \le \lim_{t \to 0} \frac{\|tv\|}{t} = \|v\|
\end{aligned}
$$
</p>

이제 unit vector $v = \frac{\nabla f(x)}{\|\nabla f(x)\|}$ 에 대해 ($\nabla f(x) \neq 0$ 라면),
$$\left | \nabla f(x) \cdot \frac{\nabla f(x)}{\|\nabla f(x)\|} \right | = \frac{\|\nabla f(x)\|^2}{\|\nabla f(x)\|} = \|\nabla f(x)\|$$
따라서 
$$\|\nabla f(x)\| \le \left \|\frac{\nabla f(x)}{\|\nabla f(x)\|} \right \| = 1$$ 
이므로, $\|\nabla f(x)\| \le 1$ 이다.


***

### 1.2. Infimum of SDF Gradient

이번에는 $\|\nabla f(x)\| \ge 1$ 임을 증명하여 gradient 크기의 하한 (infimum) 을 보이도록 하겠다.

우선, $f$ 가 미분 가능한 점 $$x \in \mathbb{R}^n \setminus \partial\Omega$$ 를 생각하자. 또한 $x$에서 경계 $\partial\Omega$ 까지의 거리를 최소화하는 **유일한** 최근접점 $p \in \partial\Omega$ 가 존재한다고 가정한다. (이러한 점들은 'medial axis'를 제외한 대부분의 공간에 해당한다.)

이제 $p$ 에서 $x$ 를 향하는, 즉 경계로부터 가장 빠르게 멀어지는 방향을 나타내는 단위 벡터 $$ n = \frac{x-p}{\|x-p\|} $$ 을 생각할 수 있다.

이 방향으로의 directional derivative 를 고려하기 위해, 충분히 작은 양수 $t > 0$ 에 대한 점 $x+tn$ 을 생각할 수 있다. $x$ 의 최근접점 $p$ 가 유일하다는 가정 하에, 충분히 작은 $t$ 에 대해서 $x+tn$의 최근접점 역시 $p$가 된다. 따라서 거리 함수 값은 다음과 같이 계산된다.
<p>
$$
\begin{aligned}
d(x+tn, \partial\Omega) &= \|(x+tn) - p\| \\ &= \|(x-p) + tn\| \\ & = \|\|x-p\|n + tn\| \\ & = (\|x-p\|+t) = d(x, \partial\Omega)+t
\end{aligned}
$$
</p>


이를 SDF의 정의에 따라 부호를 고려하여 정리하면 다음과 같다.
- 만약 $x \in \Omega^c$ $(f > 0)$ 이면, $f(x) = d(x, \partial\Omega)$ 이므로, $$ f(x+tn) = d(x, \partial\Omega)+t = f(x)+t$$ 
- 만약 $x \in \Omega$ $(f < 0)$ 이면, $f(x) = -d(x, \partial\Omega)$ 이므로 $$f(x+tn) = -(d(x, \partial\Omega)+t) = f(x)-t$$

이제 $n$ 방향으로의 directional derivative 를 계산하면 다음과 같다.
- $x \in \Omega^c$ 의 경우:
    $$
    \nabla f(x) \cdot n = \lim_{t\to 0^+} \frac{f(x+tn)-f(x)}{t} = \lim_{t\to 0^+} \frac{t}{t} = 1
    $$
- $x \in \Omega$ 의 경우 (값이 가장 빠르게 증가하는 방향은 $-n$ 이다):
    $$
    \nabla f(x) \cdot (-n) = \lim_{t\to 0^+} \frac{f(x-tn)-f(x)}{t} = \lim_{t\to 0^+} \frac{t}{t} = 1
    $$
    따라서 $\nabla f(x) \cdot n = -1$ 이다.

두 경우 모두 $|\nabla f(x) \cdot n| = 1$ 이 성립한다.
Cauchy-Schwarz 부등식 $|\nabla f(x) \cdot n| \le \|\nabla f(x)\| \|n\|$ 과 $\|n\|=1$ 이라는 사실로부터,

$$ 1 \le \|\nabla f(x)\| $$

이 성립함을 알 수 있다.

### 1.3. Conclusion

1.1절에서 증명한 $\|\nabla f(x)\| \le 1$ 과 1.2절에서 증명한 $\|\nabla f(x)\| \ge 1$ 로부터, $f$ 가 미분 가능한 모든 점 $x$ 에 대하여 Eikonal equation 이 성립함을 알 수 있다.

$$
\|\nabla f(x)\| = 1
$$

---

## Proof 2 

> *Proof using Geometric Properties of Distance Function*

두 번째 증명은 거리 함수의 기하학적 성질을 이용하는 보다 직관적인 접근법이다. 증명의 엄밀성을 위해, 거리 함수의 gradient 에 대한 잘 알려진 다음 정리를 먼저 명시한다.

> **Theorem.** 거리 함수 $d(x, S)$가 점 $x$에서 미분 가능하고, $p$가 $S$ 상의 $x$에 대한 유일한 최근접점이라면, $d$의 gradient 는 다음과 같다.
> $$ \nabla d(x, S) = \frac{x-p}{\|x-p\|} $$
>
> 이 정리는 거리 함수의 level set 이 $S$ 로부터 같은 거리에 있는 점들의 집합이며, gradient 는 level set 에 수직이고 값이 가장 빠르게 증가하는 방향을 가리킨다는 기하학적 사실로부터 유도된다. 이 방향은 바로 $p$ 에서 $x$ 로 향하는 단위 벡터이다.

### 2.1. Gradient Calculation

앞선 증명과 마찬가지로 $f$ 가 미분 가능한 점 $x \in \mathbb{R}^n \setminus \partial\Omega$ 와, $\partial\Omega$ 상의 유일한 최근접점 $p$ 를 가정하자. 두 가지 경우로 나누어 $\nabla f(x)$ 를 계산할 수 있다.

**Case 1: $x \in \Omega^c$**

이 경우 SDF 의 정의에 의해 $f(x) = d(x, \partial\Omega)$ 이다. 따라서 위 정리를 직접 적용하면 다음과 같다.
$$ \nabla f(x) = \nabla d(x, \partial\Omega) = \frac{x-p}{\|x-p\|} $$

**Case 2: $x \in \Omega$**

이 경우 $f(x) = -d(x, \partial\Omega)$ 이므로, gradient 는 다음과 같다.
$$ \nabla f(x) = -\nabla d(x, \partial\Omega) = -\frac{x-p}{\|x-p\|} $$

### 2.2. Gradient's Norm

두 경우 모두, gradient $\nabla f(x)$ 는 단위 벡터이거나 단위 벡터에 -1을 곱한 형태이다. 따라서 gradient 의 Euclidean norm 을 계산하면 다음과 같다.

<p>
$$ 
\begin{aligned}
\|\nabla f(x)\| &= \left\| \pm \frac{x-p}{\|x-p\|} \right\| \\ & = \frac{\|x-p\|}{\|x-p\|} = 1 
\end{aligned}
$$
</p>

이는 $f$가 미분 가능한 모든 점에서 Eikonal equation 을 만족함을 보인다.

---

## Discussion


**Rigor and Assumptions**: 두 증명 모두 $f$의 **미분 가능성** 과 **최근접점의 유일성** 을 가정했다. 실제 SDF 는 공간 내 모든 점에서 미분 가능하지는 않다. 미분 불가능한 점들의 집합을 ['medial axis'](https://en.wikipedia.org/wiki/Medial_axis) 또는 'skeleton' 이라 부르며, 이는 $\partial\Omega$ 상의 두 개 이상의 점과 동일한 최단 거리를 갖는 점들의 집합에 해당한다. 하지만 이러한 점들의 집합은 Lebesgue measure 0을 가지므로, SDF는 ***거의 모든 곳에서 Eikonal equation 을 만족한다*** 고 표현하는 것이 가장 엄밀하다.

NeRF 등 Implicit Neural Network (INN) 를 이용한 3D reconstruction 에서 surface reconstruction 을 위해 SDF 를 학습하는 경우가 있다. 이 경우에 eikonal loss term 을 계산하려면 MLP 에 대한 second derivative 가 필요한데, INN 학습에 필수적인 tcnn library 에는 공식적으로 second derivative 기능을 제공하지 않는다. 따라서 Hash-Grid (feature-grid) representation 을 사용하는 SDF method 는 gradient 를 analytic 하게 계산하는 대신 ***finite difference*** 를 이용하여 ***numerical gradient*** 를 계산한다. 

<video id="video-2"  controls style='width: 100%' autoplay playsinline loop >
    <source src='https://research.nvidia.com/labs/dir/neuralangelo/assets/numerical_gradient.mp4' type='video/mp4'>
</video>


대표적으로 [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/) 등이 해당 방식을 통해 gradient 를 계산한다.
이는 부분적으로 tcnn 라이브러리의 기술적 한계를 우회하기 위한 것이지만, 더 중요한 것은 numerial gradient 가 gradient field 에 smoothing 효과를 제공하여 학습 초기 단계에서 최적화를 안정화하는 데 도움이 된다는 것이다. Neuralangelo는 학습이 진행됨에 따라 점진적으로 미세한 디테일을 복구하는 'Coarse-to-fine' strategy 을 사용하여 이를 적극적으로 활용한다. 이는 neuralangelo 의 original Instant-NGP 대비 학습 시간 증가의 원인 중 하나이지만, Hash Grid Resolution 이 높아질수록 엄청난 geometric detail 이 생기는 것을 볼 수 있다.

<video id="video-1"  controls style='width: 100%' autoplay playsinline loop >
    <source src='https://research.nvidia.com/labs/dir/neuralangelo/assets/coarse_to_fine.mp4' type='video/mp4'>
</video>

*Ref*: [Boundary regularity for the distance functions, and the eikonal equation](https://arxiv.org/pdf/2409.01774)


