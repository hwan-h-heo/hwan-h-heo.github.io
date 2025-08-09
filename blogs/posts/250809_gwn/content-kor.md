title: Generalized Winding Number
date: July 31, 2025
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

### TL; DR
 
Winding Number 의 정의와, 이를 3D 로 확장한 Generalized Winding Number 가 왜 SDF 계산에서 쓰일 수 있는지 알아보자! 
<br/>

## Introduction 

Graphics 에서 Signed Distance Field (SDF) 를 계산하는 것은 애니메이션, 충돌 처리, 물리학적 시뮬레이션 등을 위해 필수적이다. 
SDF 는 그 이름과 같이, '부호를 가진 거리 함수' 인데, *어떤 점으로부터 물체까지의 가장 가까운 거리* 에, 물체의 외부라면 positive, 내부이면 negative 의 값을 내뱉는 함수이다. 

<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-3.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Signed Distance Field</figcaption>
</figure>

이는 'Eikonal Equation' 이라는 non-trival PDE problem 을 풀어야 정확히 구할 수 있지만, 모든 공간에 대해서 해당 방정식의 해를 정확하게 구하는 것은 계산적으로 매우 어려운 일이기 때문에, 보통은 구하기 쉬운 '거리' 를 먼저 계산한 뒤 '내부 vs. 외부' 만을 판별하여 부호를 붙인다. 

하지만 이때 '외부 vs. 내부' 를 구분하는 것 또한 non-trivial problem 으로, 거리를 계산하는 것보다 훨씬 어려운 문제이다. 
`flood fill` 알고리즘을 이용하여 내외부를 결정짓는 방법도 있지만, 이는 mesh 에 있는 hole 이나 self intersction 등을 고치지는 못한다. 

이 글에서는 SDF 를 우아하게 정의하고, broken mesh 의 hole 이나 self intersction 등의 문제를 repair 하는 능력을 갖춘 ***Generalized Winding Number*** 를 알아볼 것이다.  

## What is Winding Number?
<br/>

<figure id="figure-1" >
  <img src='./250809_gwn/assets/image.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Winding Number</figcaption>
</figure>

Winding Number 는 특정 점을 기준으로 ***닫힌 곡선이 얼마나 많이 감겨 있는지*** 를 나타내는 정수 값이다. 이 개념은 복소해석학, 대수적 위상수학 등 다양한 수학 분야 뿐 아니라 컴퓨터 그래픽스에서도 중요하게 다루어진다. 

이 글에서는 Cauchy integral 로 정의되는 winding number 의 직관적 이해부터, polygonal mesh 의 SDF 계산을 위해 winding number 가 어떻게 이용되는지 알아보자.

### 1. Intuition

***Winding Number*** 는 말 그대로 '감긴 횟수' 를 의미한다. 

평면 위의 한 점 $p$ 와 그 점을 지나지 않는 닫힌 곡선 $C$ 를 상상해보자. 이제 *곡선 $C$ 를 따라 한 바퀴 걸어간다고 생각했을 때, 점 $p$를 중심으로 몇 바퀴 회전했는지를 나타내는 값* 이 바로 Winding Number 이다.

- **시계 반대 방향**으로 회전하면 **positive (+)** 로,
- **시계 방향**으로 회전하면 **negative (-)** 로 센다
- 만약 곡선이 점 $p$를 감싸지 않는다면 winding number 는 **0** 이 된다.

이 직관을 토대로, 이를 좀 더 수학적인 용어로 확장해보겠다.

### 2. Definition w/ Cauchy Integral

복소평면에서 닫힌 경로 $\gamma$ 와 이 경로 위에 있지 않은 점 $a$ 에 대한 와인딩 넘버 $n(\gamma, a)$ 는 다음과 같은 코시 적분으로 정의된다.

<p>
$$
n(\gamma, a) = \frac{1}{2\pi i} \oint_\gamma \frac{1}{z-a} dz
$$
</p>

이 적분 값은 항상 정수가 되며, 이는 $\gamma$ 가 점 $a$ 를 몇 번 감았는지를 나타낸다.

---

#### 왜 코시 적분으로 와인딩 넘버를 정의할 수 있을까?

1. **$ \frac{1}{z-a} $의 의미:** 이 함수는 점 $a$ 에서 극(pole)을 갖는 함수이며, 복소로그함수의 도함수이기도 하다. 즉, $ \log(z-a) $를 미분하면 $ \frac{1}{z-a} $가 된다.

2. **적분의 의미 = 각도의 변화량:** 복소로그함수 $ \log(w) $는 다음과 같이 표현할 수 있다:
$$
\log(w) = \ln|w| + i \arg(w)
$$
    여기서 $ \arg(w) $는 복소수 $w$의 편각 ([argument](https://en.wikipedia.org/wiki/Argument_(complex_analysis))), 즉 원점으로부터의 각도를 의미한다. 
    ![image.png](./250809_gwn/assets/image-2.png)

    따라서 식 $$ \oint_\gamma \frac{1}{z-a} dz $$ 는, 경로 $\gamma$ 를 따라 $ \log(z-a) $ 의 변화량을 구하는 것과 같다. 경로가 닫혀 있으므로 $ \ln|z-a| $ 부분의 시작점과 끝점의 값은 같아져 적분 값에 영향을 주지 않는다 ([Cauchy’s Integral Theorem](https://en.wikipedia.org/wiki/Cauchy%27s_integral_theorem)). 
    결국 이 적분은 경로 $\gamma$ 를 따라 점 $a$를 기준으로 한 각도, 즉 $ \arg(z-a) $ 의 총 변화량을 나타낸다.
    
3. **$ 2\pi i $로 나누기:** 경로 $\gamma$ 를 따라 한 바퀴 완전히 돌면 각도 $ \arg(z-a) $ 는 $ 2\pi $ 의 ***정수 배*** 만큼 변하게 된다. 즉, $k$ 바퀴 돌았다면 총 각도 변화량은 $ 2\pi k $ 가 될 것이다. 따라서 적분 값을 $ 2\pi i $로 나누어주면 총 회전수인 정수 $k$ 만을 얻게 된다.

<aside>
💡 <strong id="strong-2" > 직관적 요약</strong> : 코시 적분 공식은 본질적으로 경로를 따라 움직이면서 기준점을 중심으로 한 각도의 총 변화량을 계산하는 방법이다. $ \frac{1}{z-a} $는 각도 변화를 측정하고, 적분을 통해 그 변화를 모두 더한 후 $ 2\pi i $로 나누어 회전 횟수라는 정수 값을 깔끔하게 추출해내는 것.
</aside>

---

## Generalized Winding Number

2차원 평면에서의 winding number 개념을 3차원 공간과 일반적인 surface mesh 로 확장한 것이 **Generalized Winding Number (GWN)** 이다. 

---

### 1. Discreatization of Winding Numver

위에서 살펴본 winding number 의 정의에 대하여, 일반성을 잃지 않고 기준점 $p$ 를 원점으로 가정할 때, winding number 는 *spherical coordinates 의 각도 변화를 적분하여* 정의할 수 있을 것이다.

<p>
$$
w(p) = \frac{1}{2\pi} \int_C d\theta
$$
</p>

직관적으로 이는 곡선 $C$ 를 $p$ 주위의 단위 원 (unit circle)에 투영했을 때, 그 투영된 경로의 부호를 고려한 길이를 $2\pi$ 로 나눈 값과 같다.

만약 곡선 $C$ 가 여러 개의 선분으로 이루어진 polyline, piecewise linear 라면, 이 적분은 각 선분이 만드는 각도의 합으로 정확하게 이산화될 수 있다.

<p>
$$
w(p) = \frac{1}{2\pi} \sum_{i=1}^n \theta_i
$$
</p>

여기서 $\theta_i$는 곡선의 연속된 두 정점 $c_i, c_{i+1}$ 과 점 $p$가 이루는 signed angle 이다.

### 2. Winding Number in 3D: Solid Angle

Winding Number 는 3차원으로 즉시 일반화될 수 있다. 이때 2D space 에서의 **각도(angle)** 는 3D의 **입체각 (solid angle, $\Omega$)** 으로 대체된다. 
입체각은 한 점에서 표면을 바라볼 때 그 표면이 얼마나 큰 공간을 차지하는지를 나타내는 척도이다.

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Angle_solide_coordonnees.svg/500px-Angle_solide_coordonnees.svg.png' width=40%>

점 $p$ 에 대한 닫힌 서피스 $S$ 의 winding number 는 $p$ 에서 $S$ 가 만드는 총 solid angle 을 구 전체의 solid angle 인 $4\pi$로 나눈 값으로 정의된다.

<p>
$$
\omega(p) = \frac{1}{4\pi} \int_{S} \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|^3} dA
$$
</p>

- 이 수식에서 $p$ 는 winding number 를 계산하고자 하는 3차원 공간상의 한 점, $S$ : closed surface (mesh), $x$ : surface $S$ 위의 한 점, $\mathbf{n}_x$ : 점 $x$ 에서의 normal vector 를 나타낸다. 

- ${(x-p) \cdot \mathbf{n}_x} / {\|x-p\|^3} dA$ : 이 항은 점 $p$ 에서 바라본 미소 면적 $dA$의 differential solid angle 을 나타낸다. 

    Surface 위의 어떤 면적 $dA$ 에 대해, 관측점 $p$ 에서 실제로 '보이는' 면적은 $dA$ 의 실제 크기가 아니라, 방향에 수직인 tangent space 에 projected 된 면적 $dA_{\perp}$ 이다. 

    <p>
    $$
    dA_{\perp} = dA \cdot \cos(\theta)
    $$
    </p>
    
    이제 이를 $p$ 를 중심으로 하는 반지름이 1인 sphere 에 투영한다고 생각하면, solid angle 은 다음과 같이 나타낼 수 있을 것이다 $(r = \||x-p\||)$. 
    
    <p>
    $$
    d\Omega = \frac{dA_{\perp}}{r^2} = \frac{dA \cdot \cos(\theta)}{r^2}
    $$
    </p>
    
    이 때, $\cos(\theta)$ 는 dot product 에 의해 다음과 같으므로,
    
    <p>
    $$
    \cos(\theta) = \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|} \\ \because \|\mathbf{n}_x\| = 1
    $$
    
    </p>
    
    이를 이용해 solid angle 에 대한 수식을 다음과 같이 정리할 수 있다. 
    
    <p>
    $$
    \begin{aligned}
    d\Omega &= \frac{dA}{r^2} \cdot \left( \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|} \right) \\
    &= \frac{(x-p) \cdot \mathbf{n}_x}{\|x-p\|^3} dA
    \end{aligned}
    $$
    </p>

- ${1}/{4\pi}$ : normalization term 으로, 구 전체의 입체각인 $4\pi$ 와 같다. 



이 역시 서피스가 삼각형 메시로 구성되어 있다면, 각 삼각형이 만드는 입체각 $\Omega_f$의 합으로 정확하게 discreatize 될 수 있을 것이다.

<p>
$$
w(p) = \sum_{f=1}^m \frac{1}{4\pi} \Omega_f(p)
$$
</p>

여기서 $\Omega(p, T)$는 점 $p$에서 삼각형 $T$가 이루는 입체각을 의미한다.

<aside>
💡 <strong id="strong-2" > Summary </strong> : 코시 적분 공식으로 어렵게 정의되어 있지만, 점 $p$ 를 중심으로하는 단위원에 mesh 의 모든 face 를 사영하여 나오는 겉넓이가 전체에서 어느 비율을 차지하는지를 계산한다고 생각하면 된다. 이 때 2D 에서의 winding number 가 <em id="em-1" >반시계방향</em> 으로 감긴 값을 positive 로 계산했듯이, GWN 에서는 표면의 겉면(Normal 방향)이 보이면 양수, 뒷면이 보이면 음수로 계산되어 합산되는 원리이다. 즉 mesh surface 위라면 이 값은 0.5 에, mesh 내부라면 1에 가깝게, mesh 외부라면 0 에 가까운 값을 갖게될 것이다. 
</aside>

---

## GWN Field for SDF

<br/>

### SDF from GWN

GWN의 가장 큰 장점은 **Robustness** 이다. 전통적인 방법들은 메시가 "깔끔할 것"(e.g., no holes, no Self-intersections, normal-consistent, watertight manifold) 을 요구하지만, GWN 은 이러한 가정이 깨진 **Broken Mesh** 에서도 의미 있는 값을 계산할 수 있다.

<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-5.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> GWN + UDF from Polygonal Mesh </figcaption>
</figure>

앞서 살펴봤듯이 GWN $w(p)$ 는 mesh surface 에서 0.5 가 되므로, 
$$ f(p) = \omega(p) - 0.5$$ 
로 정의되는 implicit function $f$ 의 isosurface 가 곧 GWN 을 통해 정의되는 mesh surface 임을 알 수 있다. 이를 응용하면, mesh 의 내/외부를 판별하기 어려운 *broken mesh* 에서도 SDF 가 잘 정의되는 *repaired mesh* 를 계산할 수 있다. 


1. **부호 판별**

    - $\omega(p) \approx 1$ 이면 점 $p$는 **내부**에 있다. ($\text{sign}(p) = -1$)
    - $\omega(p) \approx 0$ 이면 점 $p$는 **외부**에 있다. ($\text{sign}(p) = +1$)
    - Threshold 0.5를 사용하여 `if ω(p) > 0.5 then inside else outside` 와 같이 부호를 결정할 수 있다. 이 방법은 closed mesh 가 아니어도 매우 잘 작동힌다. GWN 은 적분 기반의 정의 덕분에 입력 mesh 의 위상적 연결성이나 기하학적 완벽성에 의존하지 않는다. 
2. **거리 계산**: 부호와 별개로, 점 $p$에서 메시 $S$ 까지의 기하학적인 최단 거리를 계산한다. (이는 BVH 를 사용하여 효율적으로 계산할 수 있다.)
3. **SDF 결합**: 위 두 단계에서 얻은 부호와 거리를 결합하여 최종 SDF 값을 얻는다.

<p>
$$
\text{SDF}(p) = \text{sign}(0.5-w(p)) \times \text{distance}(p, S)
$$
</p>

---

### Why GWN?

GWN의 가장 큰 힘은 **Robustness** 에 있다. 입력 메시가 **열려 있거나(open), 비다양체(non-manifold)이거나, 그 외의 문제가 있어도** 의미 있는 값을 계산해낸다.

- **Harmonic Function**: GWN은 입력 메시 자체를 제외한 모든 공간에서 Harmonic Function 이다. 이는 어떤 점에서의 함수값이 그 점 주변의 평균값과 같다는 의미이며, 이로 인해 불필요한 oscillation 이 최소화된다.

    - **왜 중요한가?**: 이 성질 덕분에 mesh 가 불완전하더라도 그 주변의 Winding Number Filed 는 매우 매끄럽고 예측 가능하게 변화힌다. 이는 GWN이 각 facet 에 의해 생성된 개별적인 조화 함수들의 합으로 표현되기 때문.
    
- 경계에서의 **Jump Discontinuity**: GWN은 입력 메시의 면을 가로지를 때 값이 $ \pm 1 $ 만큼 급격하게 변하는 점프 불연속성을 가진다.

    - **직관적 의미**: SDF 함수는 surface 에 가까워질수록 값이 0에 수렴하여 내부와 외부의 구분이 모호해진다 (eikonal equation). 반면 GWN 은 표면에 아무리 가까이 다가가도 내부와 외부의 값 차이가 뚜렷하게 유지된다.
    
    - **왜 중요한가?**: 이 성질 덕분에 GWN은 점이 내부에 있는지 외부에 있는지에 대한 매우 확실한 "confidence measure" 역할을 한다.


<figure id="figure-2" >
  <img src='./250809_gwn/assets/image-4.png' alt='img alt' width='70%'>
  <figcaption style='text-align: center; font-size: 15px;'><strong>Figure: </strong> Mesh Repair with GWN</figcaption>
</figure>
    
    
이러한 특징들 때문에 GWN 은 불완전한 mesh 의 내부/외부를 판별하고, 이를 기반으로 **Graphcut**과 같은 energy minimization 을 사용하여 메시를 복구하는 데 매우 효과적으로 사용된다. 

--- 

## Conclusion 

이 글을 통해 우리는 winding number 의 정의와, 이를 일반화한 Generalized Winding Number 에 대해서 알아보고 이 개념이 어떻게 Signed Distance Field 계산에 쓰일 수 있는지 알아보았다. 

3D Gen 분야의 mesh pre-processing 에서 watertight conversion 이 필수적이기 때문에 (cf: [Building Large 3D Generative Model (1)](/blogs/posts/?id=250702_build_large_3d_1)), 해당 분야 종사자라면 graphics 에서 watertight mesh 를 구축하기 위해 어떤 방법들이 쓰이는지 자세하게 알 필요가 있는 것 같다. 

물론 GWN 또한 계산 비용이 크고 broken mesh 를 완벽하게 고치는 방법은 아니기 때문에 heat method 처럼 normal vector 에 대한 diffusion simulation 으로 SDF 를 계산하는 방법 ([GSD](https://nzfeng.github.io/research/SignedHeatMethod/index.html)) 등이 제시되고 있다. 다음 글에서는 heat diffusion 을 통해 unsinged distance field 를 계산하는 방법과, GSD 등에 대해서 알아보도록 하겠다. 

- **Reference**: [Robust Inside-Outside Segmentation using Generalized Winding Numbers" (Jacobson et al., 2013)](https://dl.acm.org/doi/10.1145/2461912.2461916)

***Stay Tuned!***
