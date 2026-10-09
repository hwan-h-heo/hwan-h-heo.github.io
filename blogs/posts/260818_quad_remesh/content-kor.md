<link rel="stylesheet" href="/blogs/posts/260818_quad_remesh/assets/quad-remesh.css">
<script defer src="/blogs/posts/260818_quad_remesh/assets/quad-remesh.js"></script>

# Quad Remeshing: Cross Field & Integer-Grid Map

잘 만들어진 quad mesh를 보면 사각형 하나하나보다 먼저 눈에 들어오는 것이 있다. 눈 둘레를 감싸는 edge loop, 볼을 가로지르는 긴 strip, 관절의 형태를 따라 휘어지는 격자다. 각각의 면은 조금씩 찌그러져 있어도 이웃한 면들이 같은 흐름을 이어 간다. 이런 구조를 삼각형 메시에서 어떻게 얻을 수 있을까?

인접한 삼각형 두 개를 묶으면 사각형 하나는 쉽게 만들 수 있다. 하지만 이 선택을 메시 전체에서 반복한다고 해서 눈 둘레를 도는 loop까지 저절로 생기지는 않는다. 멀리 떨어진 곳의 방향과 간격도 서로 맞아야 하고, 격자의 줄 수가 달라지는 곳에는 그 변화를 받아 줄 연결 구조가 필요하다.

이 글에서는 **표면에 방향장을 구하고, 그 방향을 따르는 좌표계에 정수 격자를 놓는 방식**으로 quad remeshing을 살펴본다. 모든 remeshing 알고리즘이 같은 단계를 거치는 것은 아니지만, 이 접근을 이해하면 cross field, singularity, parameterization, mixed-integer optimization이 왜 한 문제 안에 함께 등장하는지 알 수 있다.

![방향장, 좌표, 정수 격자에서 quad mesh로 이어지는 과정](./assets/armadillo_pipeline_10k.webp)

Armadillo 예시는 Varco3D에서 사용하는 자체 구현 [CUDA 가속 CoMISo-like integer PDE solver](https://github.com/hwanhuh/CUDA-Lattice-Quadratice-Solver)로 만든 quad remesh 결과다. 계산 과정은 간단한 표면 위의 Manim 영상으로 살펴본다.

## 1. Triangle Pairing & Lattice

입력 삼각형 메시를 $M=(V,E,F)$라고 하자. 여기서는 면의 방향이 일관되고, 내부 edge에는 두 면이, 경계 edge에는 한 면이 붙는 manifold 표면을 가정한다.

인접한 삼각형을 국소적인 모양만 보고 묶으면 각 사각형의 품질은 어느 정도 좋아질 수 있다. 그러나 어떤 면에서 고른 방향이 다음 면에서도 이어진다는 보장은 없다. 마주 보는 edge를 따라가다 보면 흐름이 지그재그로 꺾이거나 짧은 loop로 닫히고, 짝을 찾지 못한 삼각형이 남기도 한다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/pairing-vs-lattice.webp" aria-label="삼각형의 내부 edge를 지우는 local pairing과, 표면 전체에 방향을 맞춘 격자의 비교.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/pairing-vs-lattice.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/pairing-vs-lattice.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>삼각형의 내부 edge를 지우는 local pairing과, 표면 전체에 방향을 맞춘 격자의 비교.</figcaption>
</figure>

삼각형 메시는 방향과 좌표를 계산하는 바탕, 즉 *calculation carrier*다. 최종 quad의 꼭짓점과 edge는 그 위에 새로 생기므로 입력 삼각형의 연결을 그대로 따를 필요가 없다. 이렇게 연결을 새로 정하려면 표면 전체에서 격자가 어떻게 흘러갈지부터 정해야 한다.

표면 위 격자의 성질은 다음 네 가지로 나누어 생각하면 편하다.

| 요소 | 정해야 하는 것 |
|---|---|
| Orientation | quad의 두 축이 향하는 방향 |
| Scale | 격자의 간격과 밀도 |
| Phase | 격자선이 실제로 지나가는 위치 |
| Topology | strip과 loop가 이어지는 방식, 줄 수가 바뀌는 위치 |

가령 평면에 같은 방향과 간격의 격자 두 개를 그려도, 하나를 반 칸 옮기면 꼭짓점은 서로 다른 곳에 놓인다. Orientation과 scale은 같지만 phase가 다른 경우다. 이 격자를 곡면에 감으면 한 바퀴에 몇 칸이 들어가는지까지 맞춰야 한다. 앞의 세 가지를 국소적으로 잘 정했더라도 마지막 연결이 어긋날 수 있다.

Cross field는 방향을, sizing field는 간격을 정하는 데 쓰인다. 실제 선의 위치는 좌표를 구하면서 결정하고, seam과 cycle에 놓이는 정수 조건으로 전체 연결을 맞춘다. 어떤 방향과 밀도가 좋은지는 용도에 따라 달라진다. 변형을 위한 메시인지, 날카로운 형상을 보존할 메시인지에 따라 선호하는 격자도 달라진다.

### Integer Lattice

원하는 결과를 먼저 좌표로 표현해 보자. 평면에 두 좌표 $u,v$가 있으면 정수선

$$
u=k,\qquad v=l,\qquad k,l\in\mathbb Z
$$

을 그릴 수 있다. 두 선의 교차점은 격자 꼭짓점이 되고, 인접한 정수선으로 둘러싸인 $[k,k+1]\times[l,l+1]$은 한 칸이 된다.

표면의 한 조각 $C_i$에도 좌표 사상

$$
\Phi_i=(u_i,v_i):C_i\rightarrow\mathbb R^2
$$

를 주면 같은 일을 할 수 있다. Parameter plane에 그린 정수 격자를 표면으로 되가져오는 것이다. 사상이 적절하다면 다음 대응을 얻는다.

$$
\begin{aligned}
\text{integer lattice point}&\longleftrightarrow\text{quad vertex},\\
\text{integer grid segment}&\longleftrightarrow\text{quad edge},\\
\text{unit square}&\longleftrightarrow\text{quad face}.
\end{aligned}
$$

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/grid-extraction.webp" aria-label="UV의 정수 cell을 삼각형 안에서 보간해 원래 표면의 quad로 되가져온다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/grid-extraction.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/grid-extraction.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>UV의 정수 cell을 삼각형 안에서 보간해 원래 표면의 quad로 되가져온다.</figcaption>
</figure>

이때 격자선 하나는 입력 삼각형 여러 개를 가로질러도 된다. 입력 삼각형의 꼭짓점이 모두 정수 좌표를 가져야 하는 것도 아니다. 삼각형 안에서 보간한 $u,v$가 정수가 되는 위치에 새 꼭짓점과 선을 만들기 때문이다. 입력 메시의 해상도와 출력 quad의 해상도를 구분해야 하는 이유도 여기에 있다.

### Chart & Seam

닫힌 곡면 전체에 겹침이나 특이점 없이 하나의 평면 좌표를 줄 수는 없다. 실제 계산에서는 표면을 잘라 여러 chart로 다루고, 절단선인 seam 양쪽에 좌표의 복사본을 둔다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/cut-charts.webp" aria-label="Seam의 좌표를 두 chart에 복제해도 원본 표면점의 대응은 유지된다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/cut-charts.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/cut-charts.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>Seam의 좌표를 두 chart에 복제해도 원본 표면점의 대응은 유지된다.</figcaption>
</figure>

각 chart를 따로 잘 펴는 것만으로는 부족하다. 같은 표면점을 두 좌표로 표현하더라도 seam을 건너는 격자선은 이어져야 한다. 정사각형 격자는 $90^\circ$ 회전하거나 정수 칸만큼 이동해도 같은 격자이므로, 이 두 변환을 chart 사이의 연결 규칙으로 삼을 수 있다. 뒤에서 살펴볼 integer-grid map의 정수 조건이 여기서 나온다.

그런데 격자를 이어 붙이는 과정에는 좌표의 원점만으로 해결할 수 없는 문제가 하나 더 있다. 구면 같은 표면에서는 어디선가 격자 자체의 연결 모양이 달라져야 한다.

## 2. Singularity & Topology

평면의 정사각형 격자에서는 내부 꼭짓점마다 edge 네 개와 면 네 개가 만난다. 꼭짓점에 연결된 edge 수를 valence라고 하면 regular vertex의 valence는 $4$다. 정육면체의 꼭짓점에는 면 세 개만 모이므로 valence가 $3$이다. 이런 꼭짓점을 extraordinary vertex 또는 irregular vertex라고 부른다.

정육면체의 각 면을 촘촘하게 나누어도 원래 여덟 꼭짓점에는 여전히 세 방향의 격자가 모인다. 면 내부와 원래 edge 위에 생긴 새 꼭짓점은 valence $4$가 되지만, 여덟 corner의 연결 구조는 그대로 남는다. 정육면체를 둥글게 변형해 구면에 가깝게 만들어도 이 관계는 바뀌지 않는다.

### Valence & Index

Regular vertex 주변을 $90^\circ$짜리 sector 네 개로 보면 valence가 다른 꼭짓점도 같은 기준으로 셀 수 있다. $n_v$개의 sector가 만나는 꼭짓점의 index를

$$
I_v=\frac{4-n_v}{4}
$$

로 둔다. 여기서 세는 것은 실제 3D 각도가 아니라 정사각형 격자의 조합적인 sector 수다. 휘어진 quad의 corner angle이 정확히 $90^\circ$일 필요는 없다.

| Valence | Index | Regular vertex와의 차이 |
|---:|---:|---|
| $3$ | $+\frac14$ | sector 하나가 부족하다 |
| $4$ | $0$ | sector 네 개가 모인다 |
| $5$ | $-\frac14$ | sector 하나가 더 있다 |

이 차이는 꼭짓점 주위를 도는 격자 방향에서도 드러난다. 정육면체의 corner 주변에서 한 축을 $+u$라고 정하고, 인접한 면으로 옮기며 같은 가지를 따라가 보자. 세 sector를 거쳐 출발한 면으로 돌아오면 처음의 가지와 인접한 가지 사이에 quarter-turn 차이가 남는다. 순회 방향과 부호 규약에 따라 $+u\rightarrow+v$ 또는 $+u\rightarrow-v$로 표현할 수 있다.

네 가지를 한꺼번에 그린 십자 모양은 $90^\circ$ 돌아도 같아 보인다. 하지만 그중 어느 가지를 $+u$라고 불렀는지는 달라졌다. 이처럼 한 바퀴를 돈 뒤 방향의 이름을 원래대로 붙일 수 없는 지점이 cross field의 singularity와 연결된다. 구부러진 표면에서는 표면 자체의 회전도 함께 고려해야 하므로, 실제 계산 방법은 tangent frame을 도입한 뒤에 살펴보겠다.

### Euler Characteristic

이런 꼭짓점이 몇 개나 필요한지도 표면의 topology와 관련이 있다. 경계가 없는 all-quad mesh에서 모든 면은 edge 네 개를 갖고, 각 edge는 두 면에 공유되므로 $4F=2E$다. 모든 vertex valence의 합은 $2E$이므로

$$
\begin{aligned}
\sum_v\left(4-\operatorname{val}(v)\right)
&=4V-2E\\
&=4V-4F\\
&=4(V-E+F)\\
&=4\chi(M).
\end{aligned}
$$

따라서 index의 합은 Euler characteristic과 같다. 위 식에서 $V,E,F$는 각각 꼭짓점, edge, 면의 개수다.

$$
\boxed{\sum_v I_v=\chi(M)}
$$

정육면체의 표면은 구면과 위상동형이고 $\chi(S^2)=2$다. 모든 singularity를 valence-$3$으로만 구성한다면 $8\times\frac14=2$이므로 정확히 여덟 개가 필요하다. 정육면체의 여덟 corner가 그 예다. 다른 구면 메시에서는 valence-$5$나 더 높은 valence의 꼭짓점을 함께 쓸 수도 있지만, index의 합은 여전히 $2$여야 한다.

반면 torus는 $\chi(T^2)=0$이어서 모든 꼭짓점이 regular인 quad grid도 가능하다. Positive와 negative index를 함께 두어 합이 $0$이 되게 할 수도 있다. 경계가 있는 표면에서는 경계와 corner의 기여를 추가해야 하므로 위의 내부 꼭짓점 공식만 적용할 수는 없다.

좋은 remeshing은 irregular vertex를 무조건 없애는 문제가 될 수 없다. 표면이 요구하는 index를 만족시키면서, 필요한 singularity를 형상과 격자 흐름에 어울리는 위치에 두어야 한다. 이제 이 구조를 입력 삼각형 위의 방향장으로 표현해 보자.

## 3. Cross Field

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/carrier-patch.webp" aria-label="이웃한 세 면의 cross와, 강조한 면을 정규직교 기저로 다시 본 모습. A·B·C가 양쪽에서 대응한다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/carrier-patch.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/carrier-patch.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>이웃한 세 면의 cross와, 강조한 면을 정규직교 기저로 다시 본 모습. A·B·C가 양쪽에서 대응한다.</figcaption>
</figure>

Quad의 edge가 표면을 따라가려면 그 방향도 각 점의 접평면 안에 있어야 한다. 점 $p$의 단위 법선을 $\mathbf n_p$라고 하면

$$
T_pM=\left\{\mathbf v\in\mathbb R^3\;\middle|\;\mathbf v\cdot\mathbf n_p=0\right\}
$$

이 접평면이다. 삼각형 메시에서는 각 면이 평면이므로 face $f$마다 정규직교 기저 $B_f=(\mathbf b_{f,1},\mathbf b_{f,2})$를 두고 방향을 각도 하나로 표현할 수 있다.

$$
\mathbf d_f(\theta_f)
=\cos\theta_f\,\mathbf b_{f,1}+\sin\theta_f\,\mathbf b_{f,2}
$$

### Parallel Transport

문제는 이웃한 두 면이 서로 다른 접평면과 기저를 쓴다는 데 있다. 두 면에서 모두 $\theta=0$이라고 해도 3D 공간에서 같은 방향이라는 뜻은 아니다. 반대로 표면을 따라 자연스럽게 이어지는 방향도 기저가 다르면 서로 다른 각도로 기록된다.

공유 edge로 이어진 종이 두 장을 떠올리면 비교 방법이 분명해진다. 한 삼각형을 공유 edge 주위로 회전시켜 두 삼각형을 같은 평면에 펼친다. 그 위에 그린 화살표도 함께 회전시킨 뒤 이웃 면의 기저로 읽는다. 이 연산이 discrete parallel transport다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/parallel-transport.webp" aria-label="공유 edge를 축으로 면과 화살표를 함께 펼친 뒤 방향을 비교한다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/parallel-transport.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/parallel-transport.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>공유 edge를 축으로 면과 화살표를 함께 펼친 뒤 방향을 비교한다.</figcaption>
</figure>

Face $f$의 방향을 $g$로 옮겼을 때 각도에 더해지는 값을 $\alpha_{fg}$라고 정의하면

$$
\mathcal T_{f\rightarrow g}(\theta_f)=\theta_f+\alpha_{fg}
$$

로 쓸 수 있다. 앞으로 이웃한 방향을 비교하거나, 방향의 가지를 전파하거나, 곡선을 추적할 때는 이 보정을 먼저 적용한다.

### 4-RoSy Representation

정사각형 격자의 edge에는 앞뒤 구분이 없고, 두 축의 이름을 바꿔도 같은 격자 방향이다. 그래서 $\mathbf d\sim-\mathbf d$일 뿐 아니라 $\theta\sim\theta+\frac\pi2$다. 한 면에서 표현해야 할 방향은 다음 네 가지의 묶음이다.

$$
\mathcal C_f=\left\{\theta_f,\theta_f+\frac\pi2,\theta_f+\pi,\theta_f+\frac{3\pi}2\right\}
$$

이 묶음을 표면 전체에 배치한 것이 cross field, 또는 4-RoSy field다. 각 면에서 네 가지 중 하나를 임의로 고르고 각도를 비교하면 같은 십자끼리도 $90^\circ$ 차이가 나는 문제가 생긴다. 각도를 네 배 한 다음 단위원 위의 점으로 저장하면 이 중복을 없앨 수 있다.

$$
\mathbf q_f=\begin{pmatrix}\cos4\theta_f\\\sin4\theta_f\end{pmatrix},
\qquad z_f=e^{i4\theta_f}
$$

$e^{i4(\theta_f+\pi/2)}=e^{i4\theta_f}$이므로 네 가지가 모두 같은 값이 된다. Parallel transport도 이 표현에서는 $\mathcal R(4\alpha_{fg})$라는 2차원 회전으로 옮겨진다.

### Field Optimization

이제 이웃한 cross는 transport 후에 비슷하도록 하고, 날카로운 edge나 경계처럼 따라야 할 방향이 있는 곳에서는 그 방향에 맞춘다. 이를 나타내는 개념적인 에너지는

$$
\begin{aligned}
E_{\mathrm{cross}}
=&\sum_{(f,g)}w_{fg}
\left\|\mathbf q_g-\mathcal R(4\alpha_{fg})\mathbf q_f\right\|^2\\
&+\sum_f\lambda_f\left\|\mathbf q_f-\mathbf q_f^\star\right\|^2
\end{aligned}
$$

이다. 첫 항의 $w_{fg}$는 이웃 면 사이의 매끄러움을 얼마나 강하게 요구할지 정한다. 두 번째 항의 $\mathbf q_f^\star$는 형상에서 얻은 선호 방향이고, $\lambda_f$는 그 방향의 신뢰도와 중요도를 반영한다.

예를 들어 날카로운 edge에서는 cross의 한 축을 edge의 접선과 맞출 수 있다. Principal curvature direction도 유용하지만, 거의 평평한 곳이나 두 주곡률이 비슷한 곳에서는 방향이 불안정하다. 이런 곳까지 강하게 고정하면 노이즈를 따라가는 field가 되기 쉬우므로 guide의 가중치를 낮춘다.

계산한 $\mathbf q_f$에서 각도를 되찾을 때는

$$
\theta_f=\frac14\operatorname{atan2}(q_{f,y},q_{f,x})
$$

를 쓴다. 이 각도는 여전히 $\frac\pi2$ modulo로 정의된다. 또 위의 quadratic energy만으로 단위 길이가 보장되지는 않는다. 특히 guide 없이 smoothness 항만 최소화하면 모든 $\mathbf q_f$가 영벡터인 해도 가능하므로 정규화나 anchor 등 추가 처리가 필요하다.

![Armadillo 위에서 계산한 4-RoSy cross field](./assets/armadillo_cross_field.png)

이제 표면에서 격자가 향할 방향은 보인다. 다만 각 십자는 좌표축의 이름을 갖고 있지 않다. $+u$를 어느 가지에 붙여야 하는지, 이 선택을 이웃 면으로 계속 옮겨도 모순이 없는지는 아직 확인하지 않았다. 앞서 정육면체 corner에서 보았던 singularity가 바로 이 단계에서 다시 나타난다. Complex 표현으로 푼 field의 singularity는 결과의 winding에 담겨 있으며, 위치나 index를 직접 제어하려면 별도의 제약이나 후처리가 필요하다.

## 4. Winding & Branch Lifting

한 face의 cross만 보고 그 근처에 singularity가 있는지 알기는 어렵다. 이웃한 면들을 따라 한 바퀴 돌아야 한다. 정육면체에서 격자축을 추적했던 것처럼, 계산한 field에서도 하나의 가지를 고르고 계속 이어 가는 것이다.

### Winding & Singularity Index

Valence-$3$ 꼭짓점의 이상적인 quad sector 세 개를 따라 한 가지를 추적해 보자. 출발점으로 돌아오면 십자 모양은 같지만, 고른 가지는 처음보다 quarter-turn 돌아가 있다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/branch-holonomy.webp" aria-label="Valence-3의 세 sector를 한 바퀴 돌면 선택한 가지가 90° 달라진다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/branch-holonomy.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/branch-holonomy.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>Valence-3의 세 sector를 한 바퀴 돌면 선택한 가지가 90° 달라진다.</figcaption>
</figure>

이를 phase의 winding으로 표현할 수 있다. Singularity를 감싼 작은 원판에서 연속적인 기준 frame을 잡고 $z=e^{i4\theta}$의 phase를 따라간다고 하자. Loop $\gamma$를 한 바퀴 도는 동안 phase를 끊지 않고 펼쳐 센 변화량이 $2\pi m_\gamma$이면

$$
m_\gamma=\frac{1}{2\pi}\Delta_\gamma\arg z,
\qquad
\boxed{I_\gamma=\frac{m_\gamma}{4}}
$$

이다. 네 배 각도로 저장했으므로 cross의 index는 winding의 $\frac14$이다. Valence-$3$의 sector defect에 대응하는 경우 $m_\gamma=1$, $I_\gamma=+\frac14$이다.

실제 삼각형 메시에서는 face마다 독립적인 frame을 쓰므로 raw angle을 그대로 더해서는 이 값을 얻을 수 없다. 이웃 면 사이의 parallel transport와 loop를 돌 때 생기는 기하학적 holonomy를 함께 반영해야 한다. 특히 다면체 꼭짓점의 angle defect를 빠뜨리면 표면의 곡률과 field의 singularity를 혼동하게 된다.

<details class="quad-note" id="detail-holonomy">
<summary>수학 보충 · Holonomy, Index, Monodromy</summary>

**Parallel transport 자체의 holonomy.** 닫힌 경로 $\gamma$를 따라 접벡터를 평행이동한 뒤 출발 접평면에서 비교했을 때 남는 회전을 holonomy라고 한다. 양의 방향으로 도는 작은 경로가 원판 $D$를 둘러싼다면, 방향 규약을 고정했을 때 Levi-Civita connection의 회전은

$$
h_\gamma\equiv\int_D K\,dA\pmod{2\pi}
$$

로 주어진다. 삼각형 메시의 내부 꼭짓점 $v$에 곡률이 집중되어 있다고 보는 polyhedral metric에서는 이 값에 대응하는 것이 angle defect

$$
\Omega_v=2\pi-\sum_{f\ni v}\beta_{f,v}
$$

다. $\beta_{f,v}$는 입력 삼각형의 실제 corner angle이다. 이 회전은 **표면의 connection**에 속하므로, 특정 cross field를 고르기 전에도 정의된다. [Crane 등의 discrete connection 설명](https://www.geometry.caltech.edu/pubs/CDS10.pdf)은 이 구분을 삼각형 unfolding으로 보여 준다.

**Field의 index에는 상대적인 회전도 들어간다.** $\alpha_{fg}$를 앞에서 정의한 $f\to g$의 transport angle로 두고, transported cross와 이웃 cross 사이의 작은 차이를

$$
\delta_{fg}=\operatorname{wrap}_{(-\pi/4,\,\pi/4]}
\left(\theta_g-\theta_f-\alpha_{fg}\right)
$$

로 잡자. 여기서 wrap은 $\pi/2$의 배수를 더하거나 빼서 대표값을 고른다. 양의 방향으로 정렬한 vertex fan과 일관된 부호 규약에서는

$$
\boxed{I_v=\frac1{2\pi}\left(\Omega_v+\sum_{(f,g)\in\partial v}\delta_{fg}\right)}
$$

로 index를 읽는다. 면 사이의 field 회전과 기하학적 회전을 **합쳐야** $\frac14\mathbb Z$ 값이 나온다. 단순히 $\sum\delta_{fg}$만 세거나, 반대로 $\Omega_v$만 cross의 index라고 부르면 다른 양을 혼동하게 된다. 이 식은 선택한 discrete matching에 대한 식이며, 차이가 $\pm\pi/4$에 걸리거나 샘플 사이 회전이 너무 크면 matching의 모호성도 처리해야 한다. [MIQ의 singularity index formulation](https://graphics.rwth-aachen.de/media/papers/bommes_zimmer_2009_siggraph_011.pdf)도 angle defect와 frame transition을 함께 사용한다.

매끄러운 표면의 regular point에서는 작은 loop의 곡률 적분이 있을 수 있어도 field index는 $0$일 수 있다. 반대로 평평한 원판에서도 중심에서 정의되지 않는 cross를 만들면 nonzero index를 가질 수 있다. 그래서 곡률과 singularity는 관련이 있지만 같은 데이터는 아니다.

**Branch monodromy는 이 중 이산적인 정보를 본다.** 한 가지를 lift하여 한 바퀴 추적했을 때 가지 번호가 $q_\gamma\in\mathbb Z_4$만큼 바뀐다면, 격자의 회전 성분은 $R^{q_\gamma}$다. 축 변환을 어느 방향으로 기록하는지에 따라 $q_\gamma$와 $4I_\gamma$ 사이 부호가 바뀔 수 있지만, modulo $4$ 정보만으로는 서로 정수만큼 다른 index를 구별할 수 없다는 점은 같다. 좌표 chart의 affine monodromy에는 이 회전 외에 translation도 포함된다.

본문 영상은 이상적인 quad sector의 branch 회전을 보여 준다. 휘어진 carrier의 실제 angle defect가 정확히 $\pi/2$라는 뜻은 아니다. Cut을 넣으면 chart 내부에서 이 loop가 닫히지 않게 하고, 남는 branch 전환을 seam의 연결식에 기록할 수 있다.

</details>

### Branch Lifting & Cut

좌표를 구하려면 한 가지를 $+u$로, 그 옆의 가지를 $+v$로 골라야 한다. Singularity가 없는 seed face에서

$$
\mathbf e_u=\mathbf d(\theta),
\qquad
\mathbf e_v=\mathbf d\left(\theta+\frac\pi2\right)
$$

라고 정하자. 이웃 face로 옮길 때는 $\mathbf e_u$를 parallel transport한 뒤, 이웃 cross의 네 가지 중 가장 가까운 것을 새 $\mathbf e_u$로 선택한다. Face 사이의 연결을 나타내는 dual graph를 따라 이 선택을 전파하는 과정이 branch lifting이다.

그런데 같은 face에 서로 다른 경로로 도달할 수 있다. 두 경로 사이에 $+\frac14$ singularity가 있으면 한쪽은 오른쪽 가지를, 다른 쪽은 위쪽 가지를 $+u$라고 요구할 수 있다. 같은 영역 안에서 두 요구를 모두 만족시킬 수는 없다.

Cut은 이 충돌을 좌표계의 경계로 옮기는 방법이다. 두 경로가 하나의 chart 안에서 다시 만나지 않도록 자르고, seam 양쪽의 vertex와 edge를 좌표 계산에서 서로 다른 복사본으로 취급한다. 각 chart 안에서는 branch label을 일관되게 정하고, seam을 건널 때 축이 어떻게 바뀌는지 따로 기록한다. Singularity는 그대로 남지만 좌표를 정의하는 데 생겼던 모순은 chart 사이의 변환으로 표현된다.

인접 face 사이에서 가지 이름이 바뀌는 양은 $C_4$의 회전으로 기록할 수 있다. $C_4$는 $0,1,2,3$번의 quarter-turn으로 이루어진 회전군이다. 다음 그림에서는 이 회전을 $R_{ff}$로 표시한다.

![연결된 branch transition을 따라가는 모습](./assets/armadillo_rff_propagation.webp)

*굵은 선은 연결된 branch transition, 붉은 점은 singularity다. 선이 나타나는 순서는 연결을 보여 주기 위한 것이다.*

선의 위치는 각 면에서 대표 가지를 고르는 방식에 따라 달라진다. 중요한 것은 loop를 따라 누적했을 때 남는 불일치다. 이 선들을 모두 잘랐다고 유효한 chart가 보장되는 것도 아니다. 회전이 $0$인 연결도 handle이나 cycle을 열기 위해 자를 수 있고, distortion을 줄이려고 절단을 추가할 수도 있다.

<details class="quad-note" id="detail-cut">
<summary>구현 보충 · Cut Graph &amp; Seam</summary>

Cut은 3D 표면에 틈을 만드는 연산이 아니라, **좌표를 저장하는 연결 관계를 복제하는 연산**이다. 입력 꼭짓점 $p$를 chart 안에서는 $p^+,p^-$로 나누고, 두 좌표 사이에 $\Phi_-(p^-)=R^r\Phi_+(p^+)+\mathbf t$를 둔다. 조각을 화면에서 벌려 그려도 원본 표면점의 identity는 같다. Piecewise-linear 좌표에서는 seam edge 양 끝에 같은 affine transition을 적용하면 선형 보간에 의해 edge 내부에서도 그 관계가 성립한다.

Branch labeling의 충돌을 없애는 것과 parameter domain을 disk로 만드는 것도 구분해야 한다. Singularity를 boundary로 옮겼더라도 handle이나 non-contractible cycle이 남을 수 있다. 반대로 겉보기에 그럴듯한 선망을 그렸다고 해서 complement가 유효한 chart가 되었다고 결론낼 수 없다. 실제로 면의 adjacency를 끊고 경계 loop와 연결 성분을 검사해야 한다.

구현을 이해하는 한 가지 기준은 triangle dual graph의 spanning tree다. 처음에 모든 면을 분리한 뒤 tree의 edge를 따라 한 면씩 붙이면, 다른 vertex identification을 추가하지 않는 triangle-copy 구성에서는 disk인 영역을 얻는다. 하지만 cut이 매우 많아질 수 있으므로, 실제 알고리즘은 branch 제약과 chart 크기, 경계, distortion을 고려해 일부 연결을 복구하거나 추가로 자른다. 따라서 $R_{ff}\ne0$인 edge 집합만으로 충분한 cut graph를 얻었다고 볼 수는 없다. Branch를 일관되게 고르는 orientation chart와 실제 좌표를 푸는 parameterization chart, separatrix로 둘러싸인 coarse quad patch는 서로 다를 수 있다.

Seam 변환 $G_1=(R_1,\mathbf t_1)$과 $G_2=(R_2,\mathbf t_2)$를 차례로 건너면 합성은

$$
G_2\circ G_1=(R_2R_1,\ R_2\mathbf t_1+\mathbf t_2)
$$

다. Translation들을 그냥 더할 수 있는 것은 회전이 없는 특수한 경우다. 역방향 seam도 독립적으로 정하지 않고

$$
G^{-1}=(R^{-1},-R^{-1}\mathbf t)
$$

로 맞춘다. 이 규칙으로 loop의 holonomy를 누적해야 reciprocal·cycle 조건과 정수 격자의 phase를 함께 확인할 수 있다.

Chart 안에서 기준 가지나 좌표 원점을 바꾸면 개별 seam의 표기는 달라진다. 그래도 한 바퀴의 변환은 기준 chart 변환에 의한 conjugation만 받으므로, identity인지 아닌지와 같은 연결의 성질은 보존된다. Cut의 위치는 표현의 선택이지만, 그 뒤에 기록해야 할 monodromy를 생략할 수는 없다.

</details>

<details class="quad-note" id="detail-gauss">
<summary>수학 보충 · Theorema Egregium</summary>

가우스의 **Theorema Egregium**은 Gaussian curvature $K$가 first fundamental form, 즉 표면의 내재적인 metric으로 결정된다고 말한다. 국소 isometry $\Phi$는

$$
\langle d\Phi_p(\mathbf a),d\Phi_p(\mathbf b)\rangle
=\langle\mathbf a,\mathbf b\rangle
$$

를 모든 접벡터에 대해 만족하므로 곡률도 보존한다. 평면의 곡률은 $0$이다. 따라서 $K(p)\ne0$인 열린 영역을 평면에 옮길 때는 metric distortion을 피할 수 없다. 이는 smooth surface의 국소적인 주장이다. [정리와 isometry의 정의](https://www.silviofanzon.com/2024-Differential-Geometry-Notes/sections/chap_4.html#conclusion-fts-and-theorema-egregium)를 함께 보면 조건이 분명해진다.

삼각형 메시의 각 face는 평평해서 하나씩은 길이를 보존하며 펼칠 수 있다. 어려움은 그 삼각형들을 다시 붙일 때 생긴다. 내부 꼭짓점 주위 각도의 합이 $2\pi$가 아니면 fan을 겹침과 틈 없이 평면에서 닫을 수 없다. 꼭짓점까지 cut하여 열린 fan으로 만들 수는 있지만, 두 seam을 원래대로 붙일 때의 angle defect까지 사라진 것은 아니다. Smooth curvature가 분포한 영역을 자르는 경우에도 cut에서 떨어진 내부의 $K$는 그대로다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/gauss-and-cuts.webp" aria-label="구면의 왜곡과 원통의 등거리 펼침. Cut을 넣어도 곡률은 바뀌지 않는다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/gauss-and-cuts.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/gauss-and-cuts.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>구면의 왜곡과 원통의 등거리 펼침. Cut을 넣어도 곡률은 바뀌지 않는다.</figcaption>
</figure>

영상의 구면 삼각형은 세 변이 geodesic이고 세 내각이 모두 $\pi/2$다. Gauss–Bonnet으로도

$$
\alpha+\beta+\gamma-\pi=\int_D K\,dA
$$

이므로 구면에서는 내각의 합이 $\pi$보다 크다. Isometry라면 geodesic과 각도가 보존되어야 하는데, 평면의 직선 삼각형에서는 이를 동시에 만족시킬 수 없다. 이 비교는 단순히 3D silhouette를 펴는 애니메이션과 다르다.

원통의 옆면은 반대 사례다. 원점 이동을 제외하고

$$
X_k(s,z)=\left(\frac{\sin ks}{k},\frac{1-\cos ks}{k},z\right),
\qquad X_0(s,z)=(s,0,z)
$$

라는 변형을 생각하면 $k=1/r$에서 원통이고 $k\to0$에서 평면 strip이 된다. 모든 단계에서 $\|\partial_sX_k\|=\|\partial_zX_k\|=1$, $\partial_sX_k\cdot\partial_zX_k=0$이므로 metric은 $ds^2+dz^2$로 일정하다. 영상은 이 식으로 같은 material point들을 움직인다. 그래도 둘레 좌표의 주기성 때문에 한 줄의 seam이 필요하다.

이처럼 **왜곡 없이 펼칠 수 있는가**는 metric의 문제이고, **하나의 좌표와 branch를 전역적으로 정의할 수 있는가**는 topology와 monodromy의 문제다. Cut은 두 번째 문제를 chart boundary로 옮긴다. 작은 chart를 쓰면 왜곡을 줄이는 데 도움이 되지만, Theorema Egregium이 금지하는 isometry를 만들어 주지는 않는다.

</details>

### Streamline & Coordinate Grid

Branch가 정해진 chart에서는 방향장을 따라 곡선을 추적할 수도 있다.

$$
\gamma'(s)=\mathbf e_u(\gamma(s))
$$

를 만족하는 곡선이 streamline이다. 삼각형 경계를 건널 때마다 방향을 transport하고 다음 branch를 선택한다. Singularity에 연결되는 특별한 streamline인 separatrix들을 연결하면 표면을 큰 quad patch로 나누는 layout을 만들 수 있다.

그러나 임의의 시작점에서 streamline을 여러 개 그렸다고 해서 그 사이에 같은 수의 격자 칸이 들어가거나, 반대편 경계에서 선들이 맞아떨어지는 것은 아니다. 방향을 추적하는 것과 전체 격자의 간격·위치를 정하는 것은 별개의 요구다. Explicit layout을 먼저 만드는 방법도 있지만, 이 글에서는 chart의 좌표 $u,v$를 구하고 그 정수선을 추출하는 경로를 따른다. 이를 위해 다음으로 정할 값이 격자의 간격이다.

## 5. Coordinate Solve

Branch lifting을 마치면 각 face에 두 방향 $\mathbf e_{u,f},\mathbf e_{v,f}$가 생긴다. 여기에 원하는 간격을 주면 $u,v$가 어느 방향으로 얼마나 빨리 증가해야 하는지 정할 수 있다. 다만 face마다 정한 요구가 하나의 좌표 함수로 모두 실현될 수 있는지는 별도로 풀어야 한다.

### 1D Parameterization

길이가 $\ell_i$인 구간에서 좌표가 물리적 거리 $h_i$마다 $1$씩 증가하기를 원한다면 양 끝의 값은

$$
\frac{u_{i+1}-u_i}{\ell_i}\approx\frac1{h_i},
\qquad
u_{i+1}-u_i\approx\frac{\ell_i}{h_i}
$$

를 만족해야 한다. 양 끝에 별다른 제약이 없는 열린 선분에서는 한 점의 값을 정하고 이 차이를 순서대로 더하면 된다.

충돌은 끝점의 값이나 닫힌 경로에 추가 조건이 있을 때 드러난다. 각 구간이 요구하는 증가량의 합은 $10.4$인데 전체를 정확히 $10$칸으로 연결해야 한다면, 모든 구간의 요구를 그대로 만족시킬 수 없다. 어느 구간에서 얼마나 조정할지 정해야 한다. 최소제곱은 이런 차이를 전체 구간에 나누어 반영하는 한 방법이다.

표면에서는 이런 요구가 수많은 삼각형과 loop에서 동시에 얽힌다. 한 경로를 따라 좌표를 누적했을 때와 다른 경로로 도착했을 때 같은 값을 얻어야 하므로, face별로 좋은 방향을 고르는 것보다 조건이 강하다.

### Sizing Field

Face $f$에서 두 방향의 목표 간격을 $h_{u,f},h_{v,f}$라고 하자. 그 간격만큼 이동할 때 좌표가 약 $1$ 증가하도록 목표 gradient를

$$
\mathbf g_{u,f}=\frac1{h_{u,f}}\mathbf e_{u,f},
\qquad
\mathbf g_{v,f}=\frac1{h_{v,f}}\mathbf e_{v,f}
$$

로 둔다. 작은 $h$는 큰 gradient와 촘촘한 격자에 대응한다. 두 간격이 같으면 isotropic sizing이고, 다르면 두 축의 밀도가 다른 anisotropic sizing이다.

평평한 영역에는 큰 간격을, 곡률이 크거나 세부 형상을 보존해야 하는 영역에는 작은 간격을 줄 수 있다. 사용자 brush나 simulation 해상도처럼 외부에서 주어진 밀도도 반영할 수 있다. 다만 이웃 면 사이에서 간격이 갑자기 바뀌면 가늘고 긴 cell이 생기기 쉬우므로 sizing field를 매끄럽게 하고 변화율을 제한한다. Anisotropic sizing에서도 두 축의 간격 비율이 지나치게 커지지 않도록 조절한다.

목표 개수와의 관계도 여기서 짐작할 수 있다. 목표 방향들이 직교하고 실제 간격이 목표를 잘 따른다면 cell 하나의 면적은 대략 $h_u h_v$이므로

$$
N_{\mathrm{target}}\approx\int_M\frac1{h_u(p)h_v(p)}\,dA
$$

로 필요한 밀도를 추산할 수 있다. 균일한 간격 $h$라면 $N\approx A(M)/h^2$다. 이 식은 sizing을 정하는 근사이며, seam의 정수 조건이나 singularity, 후처리까지 반영한 정확한 출력 개수는 아니다.

이제 좌표에서 요구하는 것은 $\nabla_Mu_f\approx\mathbf g_{u,f}$와 $\nabla_Mv_f\approx\mathbf g_{v,f}$다. 여기서 방향을 읽을 때 한 가지를 구분해야 한다. $\nabla_Mu$는 $u$가 증가하는 방향이고, $u=k$인 등위선은 그 gradient에 수직이다. 목표를 정확히 따른다면 $u=k$인 격자선은 $\mathbf e_v$ 방향으로, $v=l$인 선은 $\mathbf e_u$ 방향으로 놓인다.

### Integrability & Least Squares

Cross field에는 방향만 있고 gradient의 크기나 좌표 원점은 없다. Sizing으로 크기까지 붙여도 그 field가 어떤 함수의 gradient가 된다는 보장은 없다. 예를 들어 단일값 함수의 gradient를 닫힌 경로를 따라 적분하면 $0$이어야 하지만, 독립적으로 정한 목표 field는 그렇지 않을 수 있다. Cross와 sizing은 좌표의 미분이 닮기를 바라는 목표이고, solver는 실제로 존재하는 좌표 함수 중 그 목표에 가까운 것을 찾는다.

가장 직접적인 에너지는 면적 가중 최소제곱이다.

$$
\begin{aligned}
E_{\mathrm{coord}}(u,v)
=&\sum_f A_f\left\|\nabla_Mu_f-\mathbf g_{u,f}\right\|^2\\
&+\sum_f A_f\left\|\nabla_Mv_f-\mathbf g_{v,f}\right\|^2.
\end{aligned}
$$

$A_f$는 삼각형의 면적이다. 방향을 더 잘 따르는 것과 간격을 더 잘 맞추는 것이 이 에너지 안에서 함께 평가된다. 여기서 최소화하는 것은 목표 gradient와의 차이이므로, 이 식 하나가 모든 종류의 distortion이나 뒤집힘을 막는 것은 아니다.

삼각형 메시에서는 chart vertex마다 $u,v$를 저장하고 면 내부에서 선형 보간한다. 삼각형의 선형 basis function을 $\phi_1,\phi_2,\phi_3$라고 하면

$$
\nabla_Mu_f
=u_1\nabla_M\phi_1+u_2\nabla_M\phi_2+u_3\nabla_M\phi_3
$$

이다. Gradient가 꼭짓점 값의 선형식이므로 전체 문제는 sparse quadratic least squares가 된다. 좌표에 상수를 더해도 gradient는 같기 때문에, 자유롭게 남는 원점은 기준점을 고정하는 등의 방식으로 정해야 한다.

Seam과 정수 변수를 고정해 두면 이 최소제곱 문제의 내부 평형 조건은

$$
\Delta_Mu=\operatorname{div}_M\mathbf g_u,
\qquad
\Delta_Mv=\operatorname{div}_M\mathbf g_v
$$

라는 Poisson-type equation으로 이어진다. 목표 field의 국소적인 유입·유출에 맞춰 좌표값의 변화를 조정하는 식이다. 실제 해는 이 식과 함께 주어진 경계, feature, seam 조건을 만족해야 한다.

국소적인 적분 가능성과 전역적인 적분 가능성도 다르다. 평면의 simply connected 영역에서는 curl-free 조건이 핵심이지만, torus처럼 줄일 수 없는 loop가 있는 표면에서는 국소 curl이 없어도 한 바퀴의 적분값인 period가 남을 수 있다. 또 Gaussian curvature가 $0$이 아닌 영역을 평면으로 펼치면서 길이와 각도를 모두 보존할 수는 없으므로 어느 정도의 distortion도 받아들여야 한다.

좌표 계산은 이런 요구를 조율하는 과정이다. 그런데 지금까지의 식에서는 좌표값이 모두 실수여도 괜찮았다. 실제 격자를 닫으려면 앞의 $10.4$칸 예제처럼 몇 칸으로 연결할지 정해야 한다. 이 선택이 다음 단계의 정수 변수다.

## 6. Integer-Grid Map

Chart를 자른 뒤에는 같은 3D 점 $p$가 seam 양쪽에서 서로 다른 좌표를 갖는다. Chart $i$의 좌표와 chart $j$의 좌표를 잇는 규칙을

$$
\Phi_j(p)=R^{r_{ij}}\Phi_i(p)+\mathbf t_{ij},
\qquad
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\quad r_{ij}\in\{0,1,2,3\}
$$

로 쓰자. $R^{r_{ij}}$는 두 chart의 축이 몇 번의 quarter-turn으로 대응되는지 나타내고, $\mathbf t_{ij}$는 좌표 원점의 차이다. Branch lifting과 cut을 정했다면 회전은 대체로 이미 정해져 있다. 이제 좌표와 translation을 함께 맞춰야 한다.

### Integer Transition

Translation이 실수여도 seam 양쪽의 좌표를 위 식으로 연결할 수는 있다. 하지만 정수 격자까지 같은 위치에 놓이려면

$$
\mathbf t_{ij}\in\mathbb Z^2
$$

여야 한다. Quarter-turn 회전과 정수 translation은 정수 격자를 그대로 보존하기 때문이다.

$$
R^{r_{ij}}\mathbb Z^2+\mathbf t_{ij}=\mathbb Z^2
$$

회전이 없는 간단한 경우에 $\mathbf t_{ij}=(0.4,0)^T$라고 해 보자. Chart $i$의 $u=k$인 선은 chart $j$에서 $u=k+0.4$인 선에 대응한다. 방향과 간격은 같아도 상대편의 정수선과는 만나지 않는다. 이 상태로 양쪽에서 각각 격자를 꺼내면 seam에서 edge가 끊기거나 T-junction이 생기고, cell 경계가 닫히지 않을 수 있다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/integer-seam.webp" aria-label="정수 translation에서는 격자가 맞지만, 0.4칸 옮기면 seam에서 어긋난다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/integer-seam.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/integer-seam.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>정수 translation에서는 격자가 맞지만, 0.4칸 옮기면 seam에서 어긋난다.</figcaption>
</figure>

### Cycle Period

Seam 하나를 맞췄다면 이제 여러 seam을 건너 한 바퀴 돌았을 때도 맞는지 봐야 한다. Loop $\gamma$를 따라 좌표 변환들을 합성하면

$$
G_\gamma(\mathbf x)=R_\gamma\mathbf x+\mathbf t_\gamma
$$

가 된다. Singularity를 감싸지 않고 한 점으로 줄일 수 있는 loop에서는 $R_\gamma=I$, $\mathbf t_\gamma=\mathbf0$이어야 한다. 같은 위치에 돌아왔을 때 좌표가 달라져서는 안 되기 때문이다.

Singularity를 감싸면 $R_\gamma\ne I$일 수 있다. 앞서 본 $+\frac14$ index는 축의 quarter-turn과 대응한다. Torus의 handle처럼 줄일 수 없는 loop에서는 회전이 $I$여도 $\mathbf t_\gamma\ne\mathbf0$인 translational period가 남을 수 있다.

예를 들어 직사각형의 양쪽을 붙여 원통을 만들 때 seam의 이동량을 $(n,0)$으로 잡으면 둘레에 $n$칸을 놓는 셈이다. $n$을 바꾸면 주변 좌표가 조금 움직이는 데 그치지 않고 둘레를 도는 cell의 수가 달라진다. 일반 표면에서도 정수 period는 loop를 따라 누적되는 격자 칸 수를 나타낸다. 정수 변수가 최종 connectivity에 영향을 주는 이유다.

### Singularity Coordinates

Singularity의 위치에도 호환 조건이 있다. 한 바퀴의 변환이 $G_\gamma(\mathbf x)=R_\gamma\mathbf x+\mathbf t_\gamma$이면 그 중심은

$$
(I-R_\gamma)\mathbf x_s=\mathbf t_\gamma
$$

를 만족해야 한다. 예를 들어 $90^\circ$ 회전과 $\mathbf t_\gamma=(1,0)^T$의 fixed point는 $(\frac12,\frac12)^T$다. 정수 translation만으로 singularity가 정수 격자 꼭짓점에 놓인다는 보장까지 얻지는 못한다.

Singularity $s$를 extraordinary vertex로 추출하려면 그 좌표도

$$
\mathbf x_s=\Phi(s)\in\mathbb Z^2
$$

로 함께 제약해야 한다. 그래야 앞서 본 valence-$3$/$5$ 꼭짓점이 cell 내부가 아니라 정수선의 교차점에 놓인다.

이 글의 자체 구현에서는 seam translation을 $\mathbf t_{ij}\in2\mathbb Z^2$로 제한한다. 이 조건은 seam 변환을 합성해도 유지되므로, valence-$3$/$5$의 quarter-turn monodromy에서 fixed point가 정수 좌표에 놓이게 된다. 일반적인 정수 translation보다 강한 조건으로 해당 singularity의 정수 좌표 조건을 만족시킨 것이다.

### Mixed-Integer Optimization

Chart vertex의 실수 좌표를 모두 모은 벡터를 $\mathbf x$, seam translation이나 period, singularity의 정수 좌표를 나타내는 정수 자유도를 $\mathbf z$라고 하자. Mixed-integer problem은 일반적으로

$$
\begin{aligned}
\min_{\mathbf x,\mathbf z}\quad&f(\mathbf x,\mathbf z)\\
\text{subject to}\quad&A\mathbf x+B\mathbf z=\mathbf b,\\
&C\mathbf x+D\mathbf z\le\mathbf d,\\
&\mathbf x\in\mathbb R^n,\quad\mathbf z\in\mathbb Z^m
\end{aligned}
$$

로 쓸 수 있다. 목적함수가 quadratic이고 제약이 linear이면 MIQP다. 일반적인 quadratic 목적함수는 $\mathbf y=(\mathbf x^T,\mathbf z^T)^T$에 대해 $\frac12\mathbf y^TQ\mathbf y+\mathbf q^T\mathbf y$이며, 연속 변수와 정수 변수 사이의 교차항도 포함할 수 있다.

앞 절의 coordinate energy를 쓰는 문제에서는 다음 특수형으로 구조를 읽는 편이 쉽다.

$$
\begin{aligned}
\min_{\mathbf x,\mathbf z}\quad&\frac12\mathbf x^TH\mathbf x+\mathbf c^T\mathbf x\\
\text{subject to}\quad&A\mathbf x+B\mathbf z=\mathbf b,\\
&C\mathbf x+D\mathbf z\le\mathbf d,\\
&\mathbf x\in\mathbb R^n,\quad\mathbf z\in\mathbb Z^m.
\end{aligned}
$$

$H,\mathbf c$는 원하는 방향과 간격에 맞추는 에너지를 담는다. 등식 제약은 seam과 cycle의 호환성을 연결하고, 필요하면 부등식으로 변수 범위나 추가 조건을 둔다. 정수 변수에 직접 에너지를 걸지 않아도, 어떤 정수를 선택했는지에 따라 가능한 좌표와 그 distortion이 달라진다. 다만 일반적인 determinant 양수 조건이 이 식의 linear inequality로 곧바로 표현되는 것은 아니다.

정수 변수들을 고정하면 남은 좌표 문제는 sparse quadratic solve로 풀 수 있다. 어려운 부분은 어떤 정수 조합을 고르느냐다. 실용적인 방법은 먼저 정수 조건을 풀어 continuous relaxation을 구하고, 일부 변수를 정수로 고정한 뒤 나머지를 다시 푸는 과정을 반복한다. CoMISo 계열의 rounding·fixing·re-solve 전략이 이런 접근에 해당한다.

<details class="quad-note" id="detail-comiso">
<summary>알고리즘 보충 · CoMISo Greedy Rounding</summary>

여기서는 CoMISo의 고전적인 **iterative greedy rounding** 경로를 살펴본다. 가능한 정수 조합을 모두 탐색하는 exact MIQP solver와는 다르며, 이후 추가된 backend 전체를 같은 알고리즘으로 묶어 부르는 것도 아니다. 기준은 [Bommes·Zimmer·Kobbelt의 알고리즘 설명](https://graphics.rwth-aachen.de/media/papers/bommes_2011_cas1_1.pdf)과 [CoMISo의 등식 제약 소거 방식](https://www.graphics.rwth-aachen.de/software/comiso/)이다.

**1. 등식 제약을 반영한 연속 문제를 푼다.** Seam과 feature 조건을 소거하여 가능한 좌표만 남기고, 정수 조건을 잠시 풀어 quadratic energy의 minimizer를 구한다. 모든 입력 UV를 반올림할 후보로 두는 것이 아니라, translation이나 지정된 corner처럼 정수여야 하는 자유도의 index 집합 $\mathcal I$를 따로 관리한다.

소거할 때도 정수 lattice를 보존해야 한다. 일반적인 실수 nullspace basis를 구한 뒤 임의의 축소 변수를 정수로 만든다고 원래 정수 조건이 보존되는 것은 아니다. 원 논문은 연속 변수를 우선 소거하고, 정수 변수만 남은 제약은 정규화 뒤 $\pm1$ 계수의 변수를 안전하게 소거할 수 있는 경우 등을 다룬다. 이 조건이 없는 임의의 정수 등식 시스템에 같은 보장을 적용할 수는 없다.

**2. 현재 해에서 정수에 가장 가까운 후보를 고른다.** 아직 고정하지 않은 집합을 $\mathcal I_k$라 하면 기본 greedy 규칙은

$$
j=\operatorname*{argmin}_{i\in\mathcal I_k}
\left|y_i^{(k)}-\operatorname{round}(y_i^{(k)})\right|,
\qquad m_j=\operatorname{round}(y_j^{(k)})
$$

이다. 그리고 $y_j=m_j$를 새로운 등식으로 추가한다. 이것은 integer-feasible set에 대한 전역적인 최적 선택이라는 증명이 아니라, 작은 perturbation을 먼저 시도하는 heuristic이다.

**3. 고정값을 나머지 방정식에 대입하고 다시 푼다.** 축소 에너지를 $E(\mathbf y)=\frac12\mathbf y^T H\mathbf y-\mathbf b^T\mathbf y$라고 쓰고, 누적된 고정 변수들을 $F$, 남은 자유 변수를 $U$로 나누면 다음 solve는

$$
H_{UU}\mathbf y_U=\mathbf b_U-H_{UF}\mathbf m_F
$$

가 된다. 행·열을 없애는 대신 고정 변수 위치에 identity 행·열을 남기는 구현도 가능하다. 어느 쪽이든 고정값의 영향은 우변으로 전달되고, 다음 정수 후보는 **갱신된 해**에서 고른다.

희소 시스템에서는 한 변수의 고정으로 처음 영향을 받는 residual도 국소적일 수 있다. 논문의 three-level solver는 이 부분에 local Gauss–Seidel을 적용하고, 설정한 반복 횟수 안에 수렴하지 않으면 conjugate gradient, 이어 sparse Cholesky를 사용한다. 각 단계는 설정에 따라 끌 수 있으므로 모든 실행이 세 단계를 항상 거치는 것은 아니다. 서로의 rounding 결정을 바꾸지 않는다는 influence bound가 있는 후보들을 묶는 simultaneous rounding도 포함하지만, 단순히 공간상 멀다는 이유만으로 독립이라고 가정하지 않는다.

이 과정을 $\mathcal I_k$가 빌 때까지 반복한다. 이미 고른 정수는 유지하므로 정확히 푼 연속 subproblem의 feasible set은 줄어들고, 최소 에너지는 내려갈 수 없다. 마지막에도 seam residual, integer residual, map validity는 별도로 검사해야 한다. 이는 global optimum을 찾았다는 인증과는 다른 검증이다.

<figure class="post-media quad-remesh-video">
  <video controls loop muted playsinline preload="none" width="1280" height="720" poster="/blogs/posts/260818_quad_remesh/assets/manim/lattice-fixing.webp" aria-label="정수 하나를 고정할 때마다 남은 좌표를 다시 풀고 다음 후보를 고른다.">
    <source src="/blogs/posts/260818_quad_remesh/assets/manim/lattice-fixing.mp4" type="video/mp4">
    <a href="/blogs/posts/260818_quad_remesh/assets/manim/lattice-fixing.mp4">MP4 영상 보기</a>
  </video>
  <figcaption>정수 하나를 고정할 때마다 남은 좌표를 다시 풀고 다음 후보를 고른다.</figcaption>
</figure>

영상은 이 절차를 작은 KKT 시스템으로 재현한 예시다. 정수 변수 $(t_u,t_v,c_u,c_v)$ 네 개에 대한 초기 relaxation은

$$
(t_u,t_v,c_u,c_v)
\approx(4.913835,\ 0.691573,\ 6.628090,\ 1.911269)
$$

이며 실제 계산 순서는 다음과 같다.

| 단계 | 새로 고정한 조건 | 다시 푼 에너지 |
|---|---|---:|
| Relaxation | 없음 | $4.904531$ |
| 1 | $t_u=5$ | $4.905699$ |
| 2 | $c_v=2$ | $4.906447$ |
| 3 | $t_v=1$ | $4.921409$ |
| 4 | $c_u=6$ | $5.003745$ |

처음의 $c_u=6.628090$을 즉시 반올림하면 $7$이다. 그러나 $t_v=1$을 고정한 뒤 다시 풀면 $c_u\approx6.471282$가 되어, 마지막 선택은 $6$이 된다. 값들을 처음에 한꺼번에 반올림하는 것과 결과가 달라지는 구체적인 이유다.

</details>

이때 seam별 translation을 독립적으로 반올림해서는 안 된다. 일반적으로

$$
\operatorname{round}(\mathbf a+\mathbf b)
\ne\operatorname{round}(\mathbf a)+\operatorname{round}(\mathbf b)
$$

이기 때문이다. 예를 들어 회전이 없는 세 seam의 한 좌표 이동량이 $0.4,0.4,-0.8$이면 합은 $0$이지만, 각각 반올림한 $0,0,-1$의 합은 $-1$이다. Seam 하나하나의 작은 수정이 loop 전체를 한 칸 어긋나게 만든다.

그래서 정수 결정은 seam과 cycle 제약을 만족하는 변수들에 대해 이루어져야 한다. 정수를 고른 뒤에는 그 값을 유지한 채 연속 좌표를 다시 푼다. 정수여야 하는 것은 격자의 연결을 정하는 자유도이며, 대부분의 triangle-corner UV는 끝까지 실수다.

이제 chart를 건너도 정수 격자가 맞고, loop를 돌아도 정해진 칸 수로 닫히며, singularity도 격자 꼭짓점에 놓이는 좌표를 얻었다. 하지만 아직 확인할 것이 남았다. 좌표가 삼각형을 뒤집거나 눌러 버렸다면 정수선이 잘 맞아도 정상적인 quad를 꺼내기 어렵다.

## 7. Quad Extraction

### Fold-over & Map Validity

방향이 정해진 입력 삼각형의 꼭짓점을 $A\rightarrow B\rightarrow C$ 순서로 읽자. UV에서도 같은 방향을 유지하는지 확인하려면 signed area를 계산하면 된다.

$$
A_f^{\mathrm{UV}}=\frac12\det\left(\Phi(B)-\Phi(A),\Phi(C)-\Phi(A)\right)
$$

기준 방향을 양수로 잡았을 때 면적이 음수면 fold-over, $0$이면 퇴화다. 같은 내용을 좌표 사상의 local Jacobian $J_f$로 쓰면

$$
\begin{aligned}
\det J_f>0&\quad\text{orientation preservation},\\
\det J_f=0&\quad\text{degeneracy},\\
\det J_f<0&\quad\text{fold-over}.
\end{aligned}
$$

로 구분한다.

![두 삼각형의 정상적인 펼침과 fold-over 비교](./assets/armadillo_foldover_geometry.png)

*가운데는 정상적인 펼침, 오른쪽은 한 면을 뒤집은 비교용 배치다. 붉은 영역이 겹친 부분이다.*

이런 영역에서는 하나의 unit cell을 표면으로 되가져왔을 때 경계가 꼬이거나, 내부가 정상적인 disk가 되지 않을 수 있다. 방향과 간격을 맞추는 에너지는 determinant의 부호를 보장하지 않는다. Seam의 정수 조건 역시 격자의 연결 규칙을 정할 뿐, 삼각형의 뒤집힘을 막아 주지는 않는다.

실제 parameterization에서는 distortion barrier나 orientation-preserving 제약을 쓰고, 문제가 있는 곳의 sizing을 완화하거나 chart를 나누어 다시 풀기도 한다. 모든 삼각형의 방향을 보존해도 chart 전체가 겹치지 않는다는 보장까지 자동으로 얻는 것은 아니다.

이 지점에서 [Mixed-Integer Quadrangulation](https://publications.rwth-aachen.de/record/133928) & [Integer-Grid Maps for Reliable Quad Meshing](https://ris.uni-paderborn.de/record/60452)의 문제의식도 연결된다. MIQ는 field에 정렬된 seamless parameterization & 정수 조건을 함께 다루고, IGM은 유효한 quad 추출을 위한 조건을 formulation에 포함하는 방법을 제안한다.

다만 직접 구현하는 과정에서는 IGM 논문이 제시한 reliability를 재현하지 못했고, 현재 구현은 MIQ 계열의 정수 조건 & 별도의 validity 검사에 기반한다.

### Cell Connectivity

이상적인 integer-grid map에서는 $\Phi^{-1}(\mathbb Z^2)$에서 꼭짓점을, $u=k$와 $v=l$의 등위선에서 edge를 얻는다. 그러나 선분의 위치만 알아서는 mesh connectivity가 완성되지 않는다. Seam 양쪽에서 발견한 두 교차점이 같은 꼭짓점인지 식별하고, 선분을 순서대로 연결해 닫힌 경계를 만든 뒤, 그 경계 안이 하나의 cell인지 확인해야 한다.

수치 오차는 이 과정을 더 어렵게 만든다. 정수선이 입력 꼭짓점을 거의 통과하면 이웃 삼각형들이 같은 교차를 조금 다르게 판단할 수 있다. 매우 얇은 UV 삼각형에서는 서로 다른 event가 거의 겹쳐 보인다. 단순히 가까운 좌표끼리 합치면 필요한 edge를 없애거나 잘못된 면을 만들 수도 있다.

![Cross field부터 추출한 quad까지의 단계별 결과](./assets/armadillo_patch_sequence.png)

*방향장에서 좌표와 정수선을 거쳐 quad의 연결이 만들어진다.*

이 때문에 추출에서는 교차점의 좌표뿐 아니라 어느 삼각형과 edge에서 나온 것인지도 기록한다. 공유 edge와 seam transition으로 같은 event를 식별하고, 꼭짓점 주위의 edge 순서에 따라 cell의 경계를 연결한다. 얻은 loop가 닫혀 있는지, 방향이 맞는지, 하나의 disk를 둘러싸는지를 확인해야 비로소 면으로 사용할 수 있다.

[QEx: Robust Quad Mesh Extraction](https://www.graphics.rwth-aachen.de/publication/03204/)은 수치적으로 불완전한 parameterization에서 이 연결을 일관되게 복원하고 local fold-over를 처리하는 문제를 다룬다.

### Geometry Cleanup

Connectivity를 얻은 뒤에도 꼭짓점 위치와 face 모양은 더 개선할 수 있다. 생성한 꼭짓점을 원본 표면에 projection하고 접선 방향으로 smoothing하며, feature와 경계에 놓여야 할 점은 해당 위치로 맞춘다. Quad의 각도와 aspect ratio, scaled Jacobian을 개선하고 매우 짧은 edge나 작은 면을 정리하는 작업도 여기에 포함된다.

이 과정은 앞서 만든 strip과 singularity 배치를 고려해야 한다. 강한 Laplacian smoothing으로 날카로운 형상을 무너뜨리거나, 불필요한 valence-$3$/$5$ 쌍을 정리하다 주변 흐름까지 바꾸면 좌표 단계에서 얻은 장점을 잃을 수 있다. 마지막에는 self-intersection과 non-manifold 여부, 원본 표면과의 거리 오차도 확인한다.

![최종 quad-dominant mesh](./assets/armadillo_final_quads.png)

## 8. Fundamentals & Implementation

Quad remeshing은 방향장, 좌표, 정수 조건, 추출이 함께 맞아야 하는 문제다. Cross field의 방향을 좌표로 옮기고, seam과 cycle의 정수 조건을 맞춘 뒤, 그 격자를 실제 메시로 복원한다.

이 구조를 수학과 graphics 관점에서 이해하고 나니, 구현을 시도하고 최적화하는 일이 훨씬 수월해졌다. 각 단계가 풀어야 할 문제와 만족해야 할 조건을 알고 있으면 agent에게 요청할 작업도, 결과를 검증할 기준도 분명해진다. 자체 CUDA solver를 활용한 이 구현은 Varco3D에 사용되고 있으며, 이 구현으로 전환하면서 상용 앱인 QuadRemesher의 계약을 더 이어가지 않게 됐다.

핵심을 이해하지 않은 채 agent에게 “QuadRemesher를 만들어줘”라고만 요청하면, 공개 알고리즘을 가져다 조합하는 수준을 넘어서기가 어렵다. 결과가 어긋났을 때 어느 단계의 어떤 조건을 다시 살펴봐야 하는지 판단할 수 있어야, 구현을 원하는 방향으로 개선할 수 있다.

바이브 코딩을 하더라도 최소한의 기본기는 확실히 다져 두어야 한다고 느꼈다. 문제의 구조를 이해한 상태에서 agent와 함께 구현하고 검증하는 과정이 있어야, 원하는 결과에 맞춰 알고리즘을 개선하고 최적화할 수 있다.
