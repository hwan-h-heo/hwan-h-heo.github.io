title: Visualizing Hierarchical Surface Decoding in Three.js
date: August 03, 2026
author: Hwan Heo
--- 여기부터 실제 콘텐츠 ---

<figure class="post-media">
  <iframe class="post-voxel-demo" src="./assets/hierarchical-surface-decoding.html?v=20260803-camera3" title="Live Three.js hierarchical surface decoding demo" loading="eager"></iframe>
  <figcaption>Precomputed hierarchy와 기존 voxel renderer를 불러와 반복 재생하는 live Three.js 결과. Pause와 reduced-motion을 지원한다.</figcaption>
</figure>

## Abstract

고해상도 3D field를 복원할 때 모든 좌표를 같은 비용으로 query할 필요는 없다. 먼저 낮은 해상도에서 surface가 존재할 영역을 찾고, 그 주변만 재귀적으로 세분화하면 된다. [FlashVDM](https://github.com/Tencent-Hunyuan/FlashVDM)의 hierarchical decoding을 읽으며 이 과정 자체를 시각화해보고 싶었다.

처음에는 Manim으로 개념을 정리했고, 이후 실제 GLB surface에서 deterministic voxel hierarchy를 생성해 Three.js animation으로 옮겼다. 이 글은 논문의 알고리즘을 재현하는 튜토리얼이 아니라, **coarse-to-fine decoding을 어떤 움직임으로 설명할 것인가**에 대한 짧은 구현 기록이다.

---

## 1. 무엇을 시각화할 것인가

VecSet 계열 VAE는 latent token으로부터 SDF field를 복원하기 위해 3D 공간의 point coordinates를 decoder에 query한다. 가장 단순한 방식은 고해상도 dense grid의 모든 점을 query하는 것이지만, 실제 surface는 전체 volume 중 일부만 지난다. 빈 공간까지 같은 해상도로 평가하는 셈이다.

FlashVDM은 이를 octree 형태의 hierarchical decoding으로 줄인다. coarse resolution에서 active region을 먼저 식별하고, active cell의 범위에서만 다음 해상도의 query를 만든다. 이 과정을 반복하면 계산은 전체 dense volume이 아니라 surface 주변으로 집중된다. 기존 [Varco3D 회고](/blogs/posts/varco3d-a-year-in-review-2025-retrospective/)에서 이를 “coarse resolution에서 active voxel을 찾은 뒤, 그 범위 안에서만 해상도를 높이는 과정”으로 설명했다.

여기서 한 가지를 분명히 구분할 필요가 있다. 아래의 animation은 FlashVDM decoder를 브라우저에서 실행한 결과가 아니다. VAE가 SDF를 예측하는 장면을, **mesh surface와 교차하는 voxel을 남기는 기하학적 분류**로 치환한 시각적 해석이다. FlashVDM이 보고한 point-query locality나 neural attention도 animation 안에서 계산하지 않는다. 논문에서 가져온 것은 “낮은 해상도에서 후보를 좁히고 필요한 곳만 세분화한다”는 decoding 문법이다.

---

## 2. Manim으로 만든 storyboard

처음 만든 Manim 장면은 `4 → 8 → 16 → 32`의 작은 hierarchy였다. 완전히 채워진 $4^3$ query volume에서 몇 개의 점을 VAE block으로 보내 active 여부를 분류하고, 남은 coarse cell의 여덟 child slot만 다음 해상도에 펼쳤다. 이전 level의 cell과 grid는 희미하게 남겨, 해상도가 교체되는 것이 아니라 탐색 범위가 안쪽으로 좁혀지는 구조를 보이게 했다. 전체 scene 코드는 [여기](/blogs/posts/260803_hierarchical_decoding_threejs/assets/flashvdm-volume-decoding-scene.py)에서 볼 수 있다.

첫 level의 dense query와 화면상의 위치는 다음 두 함수로 만든다.

```python
RESOLUTIONS = [4, 8, 16, 32]

def _dense_cells(self, resolution):
    return np.indices((resolution, resolution, resolution)).reshape(3, -1).T

def _to_point(self, coord, resolution):
    return ((coord + 0.5) / resolution - 0.5) * self.VOLUME_SIDE
```

`_dense_cells(4)`는 64개 cell을 빠짐없이 만들고, `_to_point`는 integer index를 voxel center로 옮긴다. active indices는 별도의 `sample_r{resolution}.npz`에서 읽어 중복 제거와 범위 검사를 거친다. 이때 source data의 축을 화면 convention에 맞추기 위해 `NPZ_AXIS_ORDER = (0, 2, 1)`도 명시했다.

active parent의 candidate는 octree 정의 그대로 여덟 개다.

```python
offsets = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
], dtype=int)
children = (parent_cells[:, None, :] * 2 + offsets[None, :, :]).reshape(-1, 3)
```

Manim scene의 `_validate_hierarchy`는 각 finer active cell이 이 candidate set 안에 있는지 먼저 확인한다. 따라서 영상에서도 새로운 voxel이 임의의 위치에서 나타나는 대신 반드시 이전 parent 안에서 출발한다.

coarse classification은 전체 decoder를 흉내 내기보다 16개 query를 골라 VAE block을 왕복시키는 짧은 연출로 압축했다. 생존 여부 자체는 이미 로드한 active set과 비교한다.

```python
active4_set = self._cell_tuple_set(active[4])
sample_keep = np.array(
    [tuple(int(v) for v in cell) in active4_set for cell in sample_cells],
    dtype=bool,
)

dot.animate.set_color(keep_colors[4] if keep else pruned_color)
```

카메라는 `frame.reorient(-62, 68, 0, ORIGIN, 8 / 1.2)`로 고정하고, 모든 `run_time`에 `SPEED_SCALE = 2.0`을 곱했다. 각 level의 query와 active cell은 `FadeIn`되고 이전 level은 낮은 opacity로 남는다. 즉 이 버전의 timing은 의도적으로 느리고 단계적이다. 이후 Three.js 버전에서 바꾸게 된 지점이 바로 이 부분이다.

<figure class="post-media">
  <video controls muted playsinline preload="metadata" aria-label="Manim hierarchical volume decoding storyboard">
    <source src="./assets/flashvdm-volume-decoding-flow.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>Manim 원본의 19–49초 핵심 구간을 1.5배속으로 편집한 storyboard. 실제 FlashVDM inference trace가 아니라 query, classification, refinement의 관계를 설명하기 위한 개념 영상이다.</figcaption>
</figure>

정지된 도식으로는 parent와 child의 포함 관계가 잘 보였지만, 단계마다 화면 전체가 한 번에 교체되니 “decoding이 진행된다”기보다 LoD 모델을 차례로 켜는 것처럼 보였다. 웹 버전에서는 각 parent가 독립적인 작은 transition을 갖도록 바꾸기로 했다.

---

## 3. GLB를 deterministic hierarchy로 바꾸기

브라우저에서 GLB를 voxelize하면 첫 화면의 CPU stall과 기기별 결과 차이를 피하기 어렵다. 따라서 source GLB는 offline Python pipeline에서만 읽고, runtime은 precomputed JSON manifest와 binary payload만 fetch한다.

전처리는 scene graph의 모든 triangle primitive를 순회해 world transform을 vertex에 bake한 뒤, 전체 AABB의 가장 긴 축을 기준으로 uniform normalization한다.

$$
\mathbf{v}'=(\mathbf{v}-\mathbf{c})\frac{0.96}{\max(\mathbf{b}_{max}-\mathbf{b}_{min})}
$$

정규화된 triangle은 $512^3$ grid에서 conservative triangle–AABB intersection으로 surface voxelize한다. 내부를 채우는 solid voxelization이 아니라, triangle surface와 cell이 교차할 때만 active로 기록한다. 이후 production hierarchy는 별도로 voxelize하지 않고 finest indices를 정수 나눗셈해 만든다.

```text
512 reference surface
→ 256
→ 128 → 64 → 32 → 16 → 8
```

실제 animation level은 `8 → 16 → 32 → 64 → 128`이다. $512^3$는 얇은 부위를 놓치지 않기 위한 offline reference일 뿐 runtime asset에 포함되지 않는다. 페이지의 고정된 구도에서는 finest $128^3$ surface를 software z-buffer로 투영해 보이는 cell과 주변 depth margin을 남기고, 다시 그 leaves에서 모든 coarse parent를 유도했다. 따라서 각 runtime level은 서로 다른 방식으로 sampling한 point cloud가 아니라 하나의 finest set에서 연결된 hierarchy다.

각 parent에는 여덟 child의 생존 여부를 한 byte로 저장한다.

$$
\text{bit}=4d_x+2d_y+d_z, \qquad (d_x,d_y,d_z)\in\{0,1\}^3
$$

즉 bit 0부터 7까지가 `(000), (001), (010), (011), (100), (101), (110), (111)`에 대응한다. 좌표는 `Uint16`, child mask는 `Uint8`로 기록하며, vertex나 face는 binary에 넣지 않는다. 같은 GLB, 옵션, dependency lock, seed에서는 JSON과 binary가 byte-identical하게 생성된다.

---

## 4. Parent 하나를 여덟 child로 보이기

Three.js 쪽에서는 voxel마다 `Mesh`나 JavaScript object를 만들지 않는다. 하나의 unit cube를 `InstancedBufferGeometry`로 공유하고, instance attribute에는 integer coordinate와 retained/rejected state만 둔다. parent coordinate와 child mask는 asset load 시 typed array로 한 번 펼치며, 매 frame에는 시간과 transition progress 같은 작은 uniform만 갱신한다.

child index를 $\mathbf{i}_c$, resolution을 $r_c$라 하면 parent와 각 center는 다음처럼 정해진다.

$$
\mathbf{i}_p=\left\lfloor\frac{\mathbf{i}_c}{2}\right\rfloor, \qquad
\text{center}(\mathbf{i},r)=\left(\frac{\mathbf{i}+0.5}{r}-0.5\right)S
$$

shader는 child를 parent center에서 시작해 정확한 slot으로 보낸다.

$$
\mathbf{p}(t)=\operatorname{mix}\left(
\text{center}(\mathbf{i}_p,r_p),
\text{center}(\mathbf{i}_c,r_c),
e(t)
\right)
$$

처음에는 여덟 slot이 모두 나타난다. mask가 켜진 child는 자신의 위치에 정착하고, 꺼진 child는 이동 도중 작아지며 사라진다. voxel 사이에는 resolution에 따라 간격을 두어, 고해상도에서도 surface가 하나의 불투명한 덩어리로 뭉치지 않게 했다. 이 연출 덕분에 “active voxel만 남는다”는 결과보다 **어떤 후보가 검사되고 버려졌는지**를 함께 보여줄 수 있었다.

---

## 5. 단계가 아니라 lineage를 움직이기

가장 오래 조정한 부분은 animation schedule이었다. 처음에는 `8→16`이 끝난 뒤 `16→32`를 시작했다. 각 단계만 보면 명확했지만 전체는 계속 깜빡였고, 왼쪽에서 오른쪽으로 resolution layer가 밀리는 평범한 wave처럼 보였다.

최종 schedule은 level이 아니라 voxel lineage를 기준으로 한다. 화면의 좌상단 coarse corner에서 Manhattan distance를 계산해 BFS-like front를 만들고, 한 coarse voxel이 surface로 확인되면 전체 coarse pass가 끝나기를 기다리지 않고 즉시 child refinement를 시작한다. 여기서 BFS-like는 실제 graph queue가 아니라 corner distance로 만든 결정적 순서다. 다음 level에서는 parent 내부의 octant rank가 local order를 정한다.

따라서 먼저 드러난 sub-volume은 `8 → 16 → 32 → 64 → 128`을 중간에 멈추지 않고 내려간다. 그 뒤에서는 아직 coarse classification이 진행 중이고, 앞에서는 이미 fine voxel이 만들어진다. parent가 완전히 사라지기 전 child가 움직이고, child의 rejected slot이 collapse되는 동안 retained child는 다시 자신의 child를 펼친다. 약 세 level이 겹치지만 모든 lineage가 같은 local duration을 사용해 속도는 일정하다.

---

## 6. 좋은 visualization과 좋은 Hero는 다르다

이 animation은 처음에 portfolio home의 hero로 만들었다. 구현 의도는 잘 전달됐지만, 그것이 오히려 문제였다. dense volume이 분류되고 surface가 드러나는 데는 분명한 시작과 끝이 있다. 방문자는 자연스럽게 결과를 기다렸고, 오른쪽의 3D object가 왼쪽의 소개 문장보다 먼저 읽혔다. 반복해도 배경이 되지 못하고 매번 작은 demo를 다시 재생하는 느낌이 강했다.

반면 기존 particle wave는 완성된 상태로 바로 나타나며 특정한 결말을 요구하지 않는다. 정보의 위계에서도 typography 뒤에 머문다. 같은 색과 motion language로 연결해도, 하나는 ambient field이고 다른 하나는 설명을 요구하는 sequence였다.

그래서 home에는 기존 wave만 남기고, voxel animation은 이 Note 안의 artifact로 옮겼다. 시각화가 실패한 것은 아니다. 오히려 hierarchical decoding을 충분히 잘 설명했기 때문에 상시 배경보다 독자가 선택해 재생하는 본문 안에서 더 자연스럽다. 연구 아이디어를 설명하는 visual과 사이트의 첫인상을 만드는 hero는 서로 다른 목적을 가진다는 것, 이 작업에서 얻은 가장 실용적인 결론이다.
