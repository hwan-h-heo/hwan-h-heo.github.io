
> Sparse 3D Generation은 최적화가 까다롭다. 입력마다 active token 수가 달라지고, 그 변화가 TensorRT, `torch.compile` 같은 범용 최적화 도구의 효과를 제한한다. 이 글에서는 VARCO3D 2.0의 **unconditional cross-attention을 고정 벡터 경로로 치환**하고, memory-bound tensor path를 custom CUDA kernel과 cuBLASLt epilogue로 fusion하여 A100 BF16 환경의 15-step denoise latency를 asset별 동일 가중 평균 기준 **`25.66%`** 줄인 과정을 설명한다. 또한 bitwise-exact null-context path와 tolerance-qualified fusion path를 분리하고, production에서 이를 검증하고 fallback하는 numerical contract를 정리한다.

## 들어가며

AI Researcher에게 inference optimization은 대개 연구의 본류보다 제품화 과정에서 뒤따라오는 engineering task에 가깝다.

나 역시 주된 관심은 model architecture와 representation을 설계하고 품질을 개선하는 일이었다. Viewer, 실험용 web application, profiling utility처럼 연구 loop에 직접 필요한 도구는 종종 만들었지만, custom CUDA extension과 runtime optimization은 고수준 CUDA programming이 가능한 ML Engineer와 함께 다루는 영역이라고 생각했다.

AI Agent는 이 영역의 진입 비용을 낮춘다.

Repository 탐색, profiler trace와 source code의 연결, CUDA/C++ implementation, Python binding, correctness test와 benchmark를 빠르게 반복할 수 있게 됐다. 다만 구현 비용이 낮아졌다고 해서 좋은 최적화가 자동으로 나오지는 않는다.

이번 작업의 첫 질문도 익숙한 형태였다.

> 모델 최적화를 위해 CUDA Graph, TensorRT, `torch.compile` 등을 VARCO3D 2.0 Sparse 3D Generation 모델에 적용할 수 있어?

Agent가 sparse transformer block을 Dynamo로 probe한 결과, block 하나가 `13`개 graph와 `12`개 graph break로 쪼개졌다. 주요 원인은 다음과 같았다.

- `SparseTensor`와 layout 처리에서 발생하는 .item()과 bincount
- Python slice, list, dict 기반 layout과 cache
- `FlashAttention` dispatcher의 Python control flow

기존 A100 측정에서 MLP만 compile한 경로도 안정적인 wall-time 이득을 만들지 못했고, BF16 결과 차이가 발생했다. Profiler상 전체 비용의 약 87.6%는 이미 FlashAttention과 cuBLAS/cuBLASLt GEMM에 쓰이고 있었다.

***Agent의 답변:***

> *전체 모델 적용이 불가능한 것은 아니지만, 현재 구조에서 바로 적용하는 것은 현실성이 낮습니다.
1. 전체 `torch.compile`은 보류합니다.
2. TensorRT 전체 변환은 비용 대비 기대 이득이 낮습니다.
3. 시도한다면 sparse metadata를 graph 밖에서 계산하고, 한 block의 tensor-only body만 `fullgraph=True, dynamic=True`로 재작성해야 합니다.
4. TensorRT도 기술적으로는 가능하지만, 입력을 `SparseTensor` 대신 `feats`, `coords`, `cu_seqlens` 등의 tensor tuple로 flattening하고, `varlen FlashAttention`과 sparse operation을 위한 TensorRT plugin도 작성해야 합니다. 그렇지 않으면 dense Linear와 MLP 일부만 TensorRT로 들어가고 나머지는 PyTorch fallback으로 분할될 가능성이 높습니다. 즉, 현재 모델에서는 단순한 변환보다는 runtime porting에 가깝습니다.*

따라서 full `torch.compile`과 TensorRT port를 우선순위에서 내리는 판단 자체는 합리적이었다.
그러나 다음 두 문장은 같은 뜻이 아니다.

"현재 구조는 범용 compiler에 잘 맞지 않는다."

vs.

"현재 forward path에는 최적화할 계산이 없다."

문제는 답이 아니라 질문의 범위였다. "어떤 범용 최적화 도구를 적용할 수 있는가?"라고 물었기 때문에 답도 그 도구들의 비용 대비 효과를 판정하는 데서 끝났다. 도구를 제한하지 않고 질문을 다시 바꿨다.

> **현재 forward path에는 애초에 실행할 필요가 없는 계산이 없는가? 필요한 계산도 더 짧은 경로로 실행할 수 없는가?**

이 글은 그 질문에서 시작해 다음 두 방향을 실제 production workload에서 검증한 기록이다.

1. Model semantics를 분석해 불필요한 workload 자체를 제거한다.
2. Compiler가 묶지 못한 국소 tensor path의 memory round trip을 직접 fusion한다.


---

## 1. Sparse 3D Generation은 왜 최적화하기 어려운가

### 1.1. Sparse representation은 workload를 동적으로 만든다

3D grid의 해상도를 두 배로 높이면 voxel 수는 여덟 배가 된다.

$$
(2N)^3 = 8N^3
$$

고해상도 3D generation이 전체 공간을 dense하게 처리하기 어려운 이유다. 최근 model들은 geometry가 존재할 가능성이 있는 영역만 sparse voxel 또는 latent token으로 표현한다.

일반적인 pipeline은 다음과 같다.

1. Coarse 3D structure를 생성한다.
2. 활성 영역을 sparse coordinate 또는 latent token으로 변환한다.
3. Active token만 Transformer로 denoise한다.
4. Sparse representation을 mesh로 복원한다.

![](./assets/sparse-pipeline.png)

Sparse representation은 비어 있는 공간의 계산을 피한다. 대신 active token 수가 입력마다 달라진다.

| 3.9K voxel tokens | 24K voxel tokens | 13.4K voxel tokens |
| --- | --- | --- |
| ![](./assets/sparse-tokens-3k.png) | ![](./assets/sparse-tokens-24k.png) | ![](./assets/sparse-tokens-13k.png) |

Token 수가 바뀌면 다음 값도 함께 변한다.

- Transformer feature matrix와 GEMM의 $M$ dimension
- Varlen attention metadata
- Sparse coordinate와 batch layout
- Intermediate activation과 peak memory
- CFG branch의 tensor shape
- Allocator가 reserve하는 memory 크기

즉 dynamic shape는 한 tensor의 크기만 바꾸지 않는다. Attention, GEMM, memory와 control flow가 함께 달라진다.

### 1.2. 범용 최적화 공식이 놓치는 지점

2D diffusion model은 한 sampling trajectory에서 latent shape가 대체로 고정된다. LLM도 sequence length는 dynamic하지만 KV cache, paged attention과 continuous batching을 중심으로 execution model이 정리돼 있다.

Sparse 3D에는 아직 비슷한 표준 runtime이 없다. 그렇다고 막연히 어렵다고 판단한 것은 아니다. 대표 block과 실제 profiler를 기준으로 각 경로를 검토했다.

| 방법 | 실제 관찰 | 판단 |
| --- | --- | --- |
| CUDA Graph | Token 수를 bucket으로 고정하려면 padding과 graph별 memory pool이 필요하다. Padding token도 attention과 GEMM에 들어가 sparse compute의 이점을 줄인다. | Production token 분포에서 관리 비용 대비 기대 이득이 작아 보류했다. |
| `torch.compile` | 대표 sparse transformer block이 `13`개 graph와 `12`개 graph break로 나뉘었다. 주요 원인은 `SparseTensor.shape/layout`의 `.item()`·`bincount`, Python slice/list/dict 기반 layout과 cache, FlashAttention dispatcher의 Python branch였다. MLP-only compile도 안정적인 wall-time 이득이 없었고 BF16 결과가 달라졌다. | 전체 model compile은 보류했다. 시도한다면 metadata를 graph 밖에서 계산하고 tensor-only body만 `fullgraph=True, dynamic=True`로 검증하며, block 단위 이득이 `3%` 미만이면 중단한다. |
| TensorRT | 바로 변환하면 dense Linear·MLP 일부만 engine에 들어가고 FlashAttention, custom kernel과 sparse metadata path는 PyTorch fallback으로 나뉠 가능성이 컸다. 전체 engine화를 위해서는 SparseTensor flattening, `torch.library` schema와 fake/meta kernel, varlen attention·sparse op plugin, token 범위별 optimization profile이 필요했다. | 기술적으로 가능하지만 사실상 별도 porting project였다. 현재 구조에서는 비용 대비 기대 이득이 낮아 보류했다. |

Profiler상 전체 비용의 약 `87.6%`는 이미 FlashAttention과 cuBLAS/cuBLASLt GEMM이었다. 이 수치는 최적화 여지가 `12.4%`뿐이라는 뜻은 아니다. 이미 빠른 kernel도 불필요하게 호출될 수 있고, kernel 사이의 memory 이동과 dispatch는 여전히 줄일 수 있다.

다만 방향은 분명해졌다.

1. FlashAttention과 GEMM 자체를 처음부터 다시 만들지 않는다.
2. Model semantics를 분석해 필요 없는 workload를 제거한다.
3. Compiler가 잡지 못한 국소 tensor path는 직접 fusion한다.
4. 각 candidate는 실제 denoise latency가 줄어들 때만 채택한다.

범용 도구를 검토한 결과는 막다른 길이 아니라, 문제의 크기를 줄여준 출발점이었다.

---

## 2. Null-context attention은 계산할 필요가 없다

### 2.1. Zero tensor 죽이기

현대적인 image-to-3D 생성 모델은 DINO 계열의 VFM에서 input image feature를 추출하고, 이를 cross-attention으로 3D denoiser에 주입하는 경우가 많다. 학습 중 image condition을 dropout하면 inference에서 2D 생성 모델과 유사한 CFG를 적용할 수 있다.

다만 차이점은 2D Gen 처럼 Text-Encoder에 빈 text 를 넣어 negative condition 을 만드는 것이 아니라, 다음과 같이 명시적인 `zero valued tensor` 를 unconditional branch 에서 사용한다는 것이다.

```python
uncond_context = torch.zeros_like(cond_context)
```

값은 모두 0이지만 일반 tensor와 같은 execution path를 통과한다.

- Zero tensor allocation과 initialization
- Positive·negative condition 결합
- K/V projection과 Q/K normalization
- Full-length cross-attention
- Output projection

첫 수정은 이 tensor를 명시적인 symbolic state로 바꾸는 것이었다.

```text
zero-valued tensor
→ symbolic null condition
```

처음에는 zero tensor allocation과 null 판정을 없애는 정도의 최적화라고 생각했다. 그런데 forward path를 더 따라가면서 질문이 바뀌었다.

> 값이 전부 0인 condition을 attention에 넣으면, 실제로 무엇이 계산되는가?

### 2.2. Null attention에서 query는 중요하지 않다

Tensor의 row를 token으로 두는 row-vector convention을 사용하자. Condition-side value projection을 다음처럼 쓸 수 있다.

$$
V = ZW_V + \mathbf{1}_L b_V^\top,
$$

여기서 $Z\in\mathbb{R}^{L\times D_c}$는 context token, $V\in\mathbb{R}^{L\times D_v}$는 value matrix다. 단순 affine projection에서 null context가 $Z=0$이면,

$$
V = \mathbf{1}_L b_V^\top.
$$

즉 모든 valid context token의 value row가 같은 vector다.

Query 수가 $N$일 때 attention probability matrix를 $A\in\mathbb{R}^{N\times L}$라고 하자. Mask가 적용된 valid position에 대해 각 row의 softmax weight 합은 1이므로,

$$
A\mathbf{1}_L = \mathbf{1}_N.
$$

따라서 null-context attention output은

$$
\begin{aligned}
AV
&= A\mathbf{1}_L b_V^\top \\
&= \mathbf{1}_N b_V^\top.
\end{aligned}
$$

이다. Query, key, attention logit과 softmax distribution이 어떻게 달라져도 각 output row는 같은 $b_V$가 된다.

Output projection까지 포함하면,

$$
\begin{aligned}
Y_-
&= (\mathbf{1}_N b_V^\top)W_O + \mathbf{1}_N b_O^\top \\
&= \mathbf{1}_N c^\top,
\end{aligned}
$$

여기서 고정 row vector $c$는

$$
c^\top = b_V^\top W_O + b_O^\top
$$

로 정의할 수 있다.

더 일반적으로 condition-side transform을 $\phi$라고 할 때 필요한 조건은 입력 tensor 자체가 0이라는 사실이 아니다. $\phi$ 이후의 value row가 모든 valid context position에서 같은 vector $v_0$가 되는지가 핵심이다. 이 경우 위 식의 $b_V$를 $v_0$로 바꾸면 같은 유도가 성립한다.

이 최적화에는 다음 조건이 필요하다.

- Inference에서 attention dropout이 비활성화돼 있음
- Valid context token이 하나 이상 존재함
- Condition-side transform 이후 모든 valid value row가 동일함
- Attention row의 valid-position weight 합이 1임
- Image token이 shape token과 softmax normalization을 공유하는 joint-attention이 아님

결론은 단순하다.

> ***Null-context attention을 더 빠르게 만들 필요가 없었다. Full unconditional attention workload 자체가 필요하지 않았다.***

<figure class="post-media">
  <video controls playsinline preload="metadata" aria-label="Null-context attention simplification animation">
    <source src="./assets/null-context-attention.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>Zero-context case를 보여주는 illustration. 이 최적화는 transformed value row들이 모두 동일한 경우라면 더 일반적으로 적용된다. Context가 수치적으로 0일 필요는 없다.</figcaption>
</figure>

### 2.3. Reference-compatible path에서 canonical cache까지

기존 batched CFG path는 conditional branch와 unconditional branch를 같은 token 길이로 구성했다.

```text
[conditional context, full zero-valued context]
→ Q/K/V projection
→ Q/K normalization
→ cross-attention
→ [2N, D] output projection
```

수학적으로는 다음처럼 완전히 분리할 수 있다.

```text
conditional branch:
    original image-condition attention
    → [N, D] output projection

unconditional branch:
    cached constant row c
    → expand([N, D])
```

초기 byte-exact implementation은 null attention 자체는 제거하되, output projection의 GEMM shape가 달라질 때 생길 수 있는 BF16 차이를 피하기 위해 reference-compatible shape 제약을 유지했다. 이 경로는 정확했지만, unconditional output projection의 계산과 일부 reconstruction overhead가 남았다.

다음 단계에서는 $c$를 한 번 계산해 cache하고 broadcast하려 했다. 하지만 direct $M=1$ projection으로

$$
c^\top = b_V^\top W_O + b_O^\top
$$

를 계산하면 기존 production GEMM의 unconditional row와 raw BF16 bit가 일치하지 않았다. 수학적으로 같은 식이어도 GEMM의 $M$ dimension이 달라지면 library dispatch, tiling과 reduction order가 달라질 수 있기 때문이다.

추가 실험에서 모든 row가 $b_V$인 `M=256` canonical input으로 output projection을 실행하고 첫 row만 cache하면, 검증한 A100·PyTorch·CUDA production 환경에서 기존 reference row와 bitwise하게 일치한다는 것을 확인했다. 이 결과로 최종 runtime path를 다음처럼 구성할 수 있었다.

```text
conditional:
    [N, D] projection

unconditional:
    canonical M=256에서 미리 계산한 [1, D] row cache
    → expand([N, D])
```

즉 batched CFG에서도 더 이상 unconditional branch를 위해 `[2N,D]` output projection을 유지할 필요가 없다. 다만 `M=256`은 수학적 상수가 아니라 환경 의존적인 empirical calibration point다. Fast path는 startup qualification을 통과한 경우에만 활성화하며, 자세한 numerical contract는 7절에서 설명한다.

이 구현 과정에서 Agent는 다음 반복 작업을 맡았다.

- Null condition의 생성과 전달 경로 추적
- Symbolic sentinel과 reference fallback 구현
- Batched CFG와 serial CFG call site 분리
- Canonical cache 생성과 invalidation 구현
- Feature flag와 startup self-test 추가
- Real-payload benchmark와 raw-bit equivalence test
- Public repository에 대한 최소 patch와 검증

Agent의 가장 큰 효과는 kernel을 대신 작성한 데만 있지 않았다. 한 가설이 틀리면 버리고, 다른 shape와 fallback을 빠르게 시험하면서 반증 비용을 낮춘 데 있었다.

### 2.4. Shortcut Evaluation

#### VARCO 3D 2.0

Fusion과 null-context를 같은 10개 payload에서 독립적으로 on/off했다.

| Path | Sum of 10 asset means | Reduction vs `pure_eager` | Output |
| --- | ---: | ---: | --- |
| `pure_eager`  | `239.829s` | - | Reference |
| `null_context [N, D], [1, D]` | `229.348s` | `4.37%` | Byte-exact |

Null context 최적화는 byte-exact 한 결과물을 만들면서 asset별로 `3.96~5.87%`의 latency를 줄일 수 있었다.

#### Public 3D Gen Models: Hunyuan3D, Direct3D, and Trellis

VARCO3D에서 결과가 나온 뒤에는 이 아이디어가 우리 implementation에만 해당하는지 확인하고 싶었다.
최근 image-to-3D model 다수는 우리 모델과 마찬가지로 Unconditional branch가 zero image token을 사용하고, 같은 최적화를 적용할 수 있다고 생각해 대표적인 3 public model: Hunyuan3D 2.1, Direct3D-s2, Trellis2 에 byte-exact 최적화를 적용해보았다.

| Model | 적용 범위 | Denoise latency reduction | Output |
| --- | --- | ---: | --- |
| Hunyuan3D 2.1 | Shape denoiser의 separate cross-attention | `2.5%` | Byte-exact |
| Direct3D-S2 | Dense stage의 cross-attention | `4.4%` | Byte-exact |
| TRELLIS.2 | Image-condition cross-attention | `5.2%` | Byte-exact |

각 수치는 같은 repository와 inference 설정에서 reference path와 optimized path를 비교한 paired ablation이다. Input, seed와 model setting은 각 model 안에서 동일하게 유지했다. Model 간 absolute latency는 비교하지 않았다.

물론 모든 image-conditioning block에 기계적으로 적용할 수 있는 것은 아니다.

- Direct3D-S2의 sparse stage는 condition coordinate positional embedding 때문에 zero feature가 위치별로 다른 value가 된다. 이 stage에는 적용하지 않았다.
- Joint-attention은 image token과 shape token이 하나의 softmax normalization을 공유할 수 있다. Image token을 제거하면 shape token의 normalization도 바뀐다. (Hunyuan3D 2.0)

따라서 적용 여부는 “negative condition이 0인가”가 아니라, **condition-side transform 이후의 value row가 실제로 동일한가**를 기준으로 판단해야 한다.

이 결과로 null-context 최적화가 VARCO3D의 우연한 구현 특성은 아니라는 점을 확인했다. 동시에 적용 조건이 수학적으로 분명하기 때문에, 맞지 않는 stage에는 넣지 않을 수 있었다.

---

## 3. Compiler가 놓친 틈을 직접 fusion하기

Null-context처럼 수학적으로 지울 수 있는 branch는 흔하지 않다. 대부분의 active path는 실제 latent와 timestep에 의존한다. 의미론적 제거 다음에는 profiler가 필요했다.

앞선 compiler probe에서 누적 CUDA kernel time의 약 `87.6%`가 FlashAttention과 cuBLAS/cuBLASLt GEMM이라는 사실은 이미 확인했다. 가장 큰 두 구간은 `flash_attn`과 `flex_gemm`을 사용하고 있었고, 새로운 attention kernel이나 GEMM을 처음부터 만드는 것은 이번 작업의 범위를 벗어났다.

하지만 “대부분의 시간이 이미 최적화된 kernel에 쓰인다”는 사실과 “더 줄일 수 있는 실행 경로가 없다”는 말은 같지 않다.

- Kernel 앞뒤의 normalization, indexing과 dtype conversion
- Intermediate tensor를 global memory에 썼다가 다시 읽는 경로
- Python dispatcher와 작은 CUDA launch가 반복되는 local chain

Full `torch.compile`이 안정적으로 동작했다면 Inductor가 노렸을 영역도 주로 이런 pointwise chain과 intermediate materialization이었을 것이다. Sparse metadata와 custom backend 때문에 전체 graph를 compile하기 어렵다면, 역할과 입출력이 분명한 국소 경로를 직접 fusion하면 된다.

### 3.1. Fusion은 정확히 무엇을 없애는가

GPU에서 PyTorch operation을 실행하면 각 operation은 하나 이상의 CUDA kernel launch로 이어진다.

```python
y = x * scale
z = y + shift
out = z * gate
```

Eager PyTorch에서는 일반적으로 `mul`, `add`, `mul`이 각각 별도의 GPU 작업으로 실행된다. 각 단계에서 다음 비용이 반복된다.

1. CUDA kernel launch
2. Global memory에서 input tensor load
3. Elementwise operation
4. Intermediate tensor store
5. 다음 kernel의 intermediate reload

연산 자체가 단순해도 kernel launch와 memory traffic이 반복된다. 특히 작은 operation이 수천 번 이상 호출되면 무시하기 어려운 비용이 된다.

Kernel fusion은 이 chain을 하나의 CUDA kernel 내부에서 처리한다.

```python
out = (x * scale + shift) * gate
```

`x`를 한 번 읽은 뒤 register 내부에서 scale, shift와 gate를 적용하고 최종 결과만 global memory에 쓴다. 이를 통해 다음 비용을 줄일 수 있다.

- CUDA kernel launch
- Intermediate tensor allocation
- Intermediate global-memory write
- 다음 kernel의 global-memory read
- Python과 ATen dispatch overhead

이번 모델에서는 단순한 contiguous elementwise operation보다 sparse batch indexing이 섞인 패턴이 더 중요한 대상이었다.

### 3.2. AI Agent와 한 candidate씩 검증한 Custom CUDA fusion

Agent를 사용하면 CUDA candidate를 빠르게 구현할 수 있다. 동시에 여러 optimization을 한꺼번에 넣으면 어떤 변경이 실제 latency를 줄였는지 알 수 없다. 따라서 한 cycle에서는 하나의 candidate만 다뤘다.

```text
Real workload profile
→ candidate 하나 선택
→ 없앨 memory·launch 경로를 먼저 정의
→ reference path와 feature flag 유지
→ Agent가 fused implementation과 binding 작성
→ correctness test
→ profiler에서 실제 replacement 확인
→ real-payload ablation
→ accept 또는 reject
```

채택 기준도 고정했다.

1. 기대한 eager operation이 실제 profiler에서 사라졌는가?
2. 해당 path의 numerical contract를 만족하는가?
3. Microbenchmark가 아니라 전체 denoise latency가 줄었는가?
4. 지원하지 않는 shape에서 reference path로 안전하게 돌아가는가?

이 기준이 세워진 이후엔 Agent는 implementation loop를 빠르게 돌렸다.

---

## 4. Production까지 살아남은 Fused Kernels

### 4.1. Q/K RMSNorm, RoPE, QKV

Transformer `self-attention` 앞에는 다음 연산이 반복됐다.

```text
Q/K 분리
→ RMSNorm
→ dtype conversion
→ complex view와 RoPE multiply
→ real tensor 복원
→ QKV stack/cat
```

하나씩 보면 큰 비용은 아니다. 하지만 수십 개의 Transformer block과 여러 denoise step이 반복되면서 normalized Q/K tensor, complex temporary와 restack tensor가 계속 materialize됐다.

이를 `qk_rms_norm_rope_qkv_inplace` kernel 하나로 합쳤다.

Kernel은 packed QKV tensor에서 직접 다음을 수행한다.

1. Q와 K에 RMSNorm을 적용
2. RoPE rotation을 계산
3. 결과를 원래 QKV 위치에 기록

한 denoise step의 representative profile에서 이 path는 다음 eager operation을 대체했다.

- Multi-head RMSNorm 8회
- `_to_copy`와 `copy_` 각각 16회
- `mul` 8회
- `stack`과 `cat` 각각 4회
- `view_as_complex`와 `view_as_real` 각각 8회

여기서 빨라진 것은 RMSNorm 하나가 아니다. Attention preprocessing 전체의 tensor lifecycle이 짧아졌다.

### 4.2. Sparse LayerNorm + AdaLN affine

기존 path는 다음과 같았다.

```python
normalized = layer_norm(x)
scale_token = scale[batch_ids]
shift_token = shift[batch_ids]
out = normalized * scale_token + shift_token
```

`scale[batch_ids]`와 `shift[batch_ids]`는 각각 `[N,D]` tensor를 materialize한다.

`sparse_layer_norm_affine` kernel은 token row를 normalize한 뒤, 같은 kernel에서 해당 `batch_id`의 scale과 shift를 읽어 modulation을 적용한다.

다음 intermediate가 사라졌다.

- 별도 LayerNorm output
- Token별 scale tensor
- Token별 shift tensor
- 별도 affine output

이 candidate는 Q/K fusion보다 절대 효과는 작았지만, token 범위 전반에서 일관되게 latency를 줄였다.

### 4.3. Leave-one-out ablation

아래 표는 다른 채택 fusion을 모두 켠 상태에서 candidate 하나만 비활성화한 leave-one-out 결과다. 양수는 해당 candidate를 끄면 느려졌다는 뜻이다.

| Candidate | Calls / iter | 10-asset avg | 판단 |
| --- | ---: | ---: | --- |
| `qk_rms_norm_rope_qkv_inplace` | 60 | `+93.3ms` (`+6.43%`) | 채택. 가장 큰 active-path 기여 |
| `sparse_layer_norm_affine` | 120 | `+25.4ms` (`+1.89%`) | 채택. 전 token 구간에서 일관된 개선 |
| `layer_norm` | 62 | `+15.6ms` (`+1.17%`) | 채택. 작지만 안정적인 개선 |
| `sparse_batch_mul_add` | 120 | `+11.4ms` (`+0.88%`) | 채택. 반복 residual path 단순화 |
| `qk_rms_norm_cross_inplace` | 60 | `-1.3ms` (`≈0%`) | 중립. 기본 성능 근거로 보지 않음 |
| `gelu_tanh` | 60 | `-14.0ms` (`-0.84%`) | **기각. Custom kernel이 오히려 느림** |

`layer_norm` candidate는 residual path의 독립 normalization을 전용 kernel로 치환했고, `sparse_batch_mul_add`는 batch-indexed modulation과 residual affine chain을 하나의 kernel로 단순화했다. `qk_rms_norm_cross_inplace`는 cross-attention preprocessing을 줄였지만 큰 payload에서 이득이 재현되지 않아 기본 성능 근거에서는 제외했다.

이 과정에서 profiler의 operation 수가 줄었다는 사실과 product latency가 줄었다는 사실은 분리해서 봐야 한다. 그 차이가 가장 선명하게 드러난 candidate가 GELU였다.

---

## 5. 더 빠른 GELU kernel이 틀린 fusion 이었던 이유

### 5.1. Kernel을 바꿨다고 Fusion이 된 것은 아니다

위 profiler 표에서 언급한 `gelu` 에 대해서 잠시 돌아가보자.

해당 최적화를 진행한 근거는 물론 Profiler였다.
Kernel call분석에는 standalone `aten::gelu`가 step당 48회 반복해서 나타났다.

처음 선택한 candidate는 명확해 보였다.

> Custom `gelu_tanh` CUDA kernel을 만들고 기존 call을 교체한다.

표면적으로는 성공했다.

- `aten::gelu`가 profiler에서 사라졌다.
- Custom kernel로 정상 교체됐다.
- Numerical tolerance를 통과했다.

그러나 실제 denoise latency는 오히려 늘었다.

실행 경로는 여전히 다음과 같았다.

```text
MLP GEMM output store
→ GELU input load
→ GELU output store
→ 다음 GEMM input load
```

PyTorch GELU는 해당 A100과 tensor shape에서 이미 효율적으로 구현돼 있었다. Kernel 하나를 다른 kernel로 바꿨지만 줄이려던 global-memory 왕복은 그대로였다.
즉, 단순히 GELU 수식을 빠른 CUDA 커널로 바꾸는 것은 메모리 I/O 횟수를 줄이지 못했다.

여기서 candidate를 폐기하고 넘어갈 수도 있었다. 대신 질문을 한 번 더 바꿨다.

> GELU가 느린 것이 아니라면, 어떤 boundary를 없애야 실제로 빨라지는가?

### 5.2. Fusion boundary를 cuBLASLt epilogue 안으로

기존 MLP path는 다음과 같았다.

```text
Linear(N → M)
→ BF16 activation 저장
→ standalone GELU
→ 두 번째 BF16 activation 저장
→ Linear(M → N)
```

즉, Standalone GELU의 경계만 건드려서는 첫 GEMM output의 store와 GELU의 load를 없앨 수 없다. 그래서 fusion boundary를 앞선 GEMM까지 넓혔다.

cuBLASLt의 bias+GELU epilogue를 사용하면 GEMM accumulator에 bias와 GELU를 적용한 뒤 최종 activation만 저장할 수 있다.

```text
GEMM + bias + GELU epilogue
→ Linear(M → N)
```

이 path는 다음 비용을 없앤다.

- Standalone GELU launch
- GELU input global-memory read
- GELU 이전 intermediate BF16 activation
- Intermediate write/read 한 차례

최종 profiler에서 standalone `aten::gelu`는 사라지고, 같은 수의 `_addmm_activation` cuBLASLt epilogue call로 바뀌었다.

이 경험 이후 fusion candidate를 operation 이름으로 고르지 않게 됐다.

> **Fusion boundary는 source-code operation이 아니라, 없애려는 memory round trip을 기준으로 정해야 한다.**

다만 cuBLASLt epilogue는 기존 eager path와 BF16 rounding boundary, GELU approximation과 internal evaluation order가 같다고 가정할 수 없다. 따라서 이 path는 bitwise-exact optimization으로 분류하지 않았고, 별도의 tolerance test와 model-level regression으로 관리했다.

---

## 6. Dynamic-token production benchmark

### 6.1. Real payload에서 잰 denoise latency

Active token 수가 `3,664–30,227`인 production asset 10개에 대해 같은 manifest를 재사용했다. 모든 latency는 warm-up 뒤 반복 측정한 CUDA-synchronized 15-step denoise wall time이다. GPU는 NVIDIA A100-SXM4-80GB, model dtype은 BF16이며, PyTorch 2.7.1+cu128과 CUDA 12.8을 사용했다.

비교 path는 다음과 같다.

- `pure_eager`: VARCO custom fusion과 cuBLASLt GELU를 끄고 null-context fast path도 끈 reference
- `kernel_fusion`: custom fusion과 cuBLASLt GELU를 켜고 null-context fast path는 끈 path
- `prior_exact_null`: null attention은 제거했지만 reference-compatible output-projection 제약을 유지한 이전 exact path
- `canonical_cache_null`: `M=256` canonical cache와 `[N,D] + cached [1,D]` broadcast를 사용하는 최종 path
- `canonical_cache_combined`: kernel fusion과 최종 null-context path를 함께 적용한 path

아래 reduction은 합산 시간의 비율이 아니다. 각 asset의 상대 latency reduction을 먼저 계산한 뒤 10개 asset에 동일한 가중치를 주어 평균한 **per-asset macro average**다. Speedup은 asset별 speedup의 geometric mean이다.


| Comparison | Equal-weight latency reduction |
| --- | ---: |
| `canonical_cache_null` vs `pure_eager` | **`4.37%`** |
| `kernel_fusion` vs `pure_eager` | **`18.734%`** |
| `canonical_cache_combined` vs `pure_eager` | **`25.66%`** |
| `canonical_cache_null` vs `prior_exact_null` | `0.53%` |
| `canonical_cache_combined` vs `prior_exact_combined` | `1.78%` |

최종 combined path는 pure eager 대비 asset별 동일 가중 평균 latency를 `25.66%` 줄였다. Canonical cache 자체도 기존 byte-exact null implementation보다 null-only path에서 `0.53%`, fusion과 결합한 path에서 `1.78%` 추가 개선을 만들었다.

### 6.2. Profiler가 드러낸 두 종류의 최적화

| Condition | CUDA launches | standalone `aten::gelu` | cuBLASLt `_addmm_activation` | FlashAttention kernel time |
| --- | ---: | ---: | ---: | ---: |
| `pure_eager` | 53,354 | 720 | 0 | `30.055 s` |
| `kernel_fusion` | 14,844 | 0 | 720 | `30.763 s` |
| `prior_exact_null` | 57,492 | 720 | 0 | `29.031 s` |
| `prior_exact_combined` | 18,982 | 0 | 720 | `29.720 s` |

Fusion group은 eager 대비 CUDA launch를 `72.18%` 줄였고, standalone GELU 720회를 cuBLASLt epilogue로 옮겼다.

반면 이전 exact null path는 compact unconditional workload를 구성하는 copy와 indexing launch 때문에 총 launch 수가 늘었지만, FlashAttention workload와 wall time은 줄었다. Launch count 하나만으로 성능을 판단할 수 없는 이유다. 이후 canonical cache는 남아 있던 output-projection·reconstruction 경로를 더 줄였고, 6.1절의 추가 latency 개선으로 확인됐다.


---

## 7. Fast path를 위한 Numerical Contract

수학적으로 같은 함수가 항상 같은 floating-point bit pattern을 만드는 것은 아니다. 특히 BF16 GEMM에서는 tensor shape, library dispatch, reduction order와 rounding boundary가 결과의 마지막 bit를 바꿀 수 있다.

이 글에서 `bitwise exact` 또는 `byte-exact`는 **동일 shape의 대상 tensor를 contiguous하게 본 raw BF16 bit pattern이 모두 같다**는 뜻이다. 단순한 실수값 equality나 최종 mesh file의 serialization byte equality를 의미하지 않는다.

검증은 BF16 tensor를 integer view로 reinterpret한 뒤 비교한다.

```python
import torch

def bf16_bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("expected BF16 tensors")
    if a.shape != b.shape:
        return False

    a_bits = a.contiguous().view(torch.int16)
    b_bits = b.contiguous().view(torch.int16)
    return torch.equal(a_bits, b_bits)
```

Null-context path에서는 대상 cross-attention block의 output projection 이후 tensor와 paired denoiser output을 검증 경계로 사용했다.

### 7.1. Tensor shape도 contract다

기존 batched CFG output projection을 다음처럼 두자.

$$
Y = XW_O + \mathbf{1}_{2N}b_O^\top,
$$

$$
X =
\begin{bmatrix}
X_+ \\
\mathbf{1}_N b_V^\top
\end{bmatrix}
\in\mathbb{R}^{2N\times D}.
$$

수학적으로는 이를 다음처럼 분리할 수 있다.

$$
Y_+ = X_+W_O + \mathbf{1}_N b_O^\top,
$$

$$
Y_- = \mathbf{1}_N c^\top,
\qquad
c^\top=b_V^\top W_O+b_O^\top.
$$

검증한 production shapes에서는 `[N,D]` conditional projection이 기존 `[2N,D]` projection의 conditional slice와 raw-bit로 일치했다. 초기 fully split implementation에서 mismatch가 발생한 지점은 direct `M=1` GEMM으로 만든 unconditional row였다.

즉 다음 두 계산은 실수 산술에서는 같지만 BF16 bit pattern은 달랐다.

$$
\operatorname{Linear}_{M=1}(b_V)
\overset{\mathrm{bitwise}}{\ne}
\operatorname{row}_{-}
\left(
\operatorname{Linear}_{M=2N}(X)
\right).
$$

`M=1`은 GEMV 또는 skinny-GEMM 계열의 다른 구현으로 dispatch될 수 있다. 일반적으로 cuBLAS/cuBLASLt는 $M$, $N$, $K$, dtype, layout과 alignment에 따라 algorithm과 tile configuration을 선택할 수 있으며, BF16에서는 accumulation과 rounding order의 차이가 최종 bit에 드러날 수 있다.

이 설명은 관찰한 mismatch에 대한 가능한 mechanism이다. 실제 채택 판단은 특정 kernel family나 algorithm ID가 같다는 추정에 의존하지 않고, 오직 raw-bit qualification 결과에 의존한다.

### 7.2. Direct M=1 cache가 탈락한 이유

가장 직접적인 구현은 startup에서 한 row만 projection하는 것이다.

```text
input:  [1, D] = b_V
linear: [1, D] W_O + b_O
cache:  [1, D]
```

하지만 이 path는 reference `[2N,D]` GEMM의 unconditional row와 bitwise하게 일치하지 않았다. 차이가 작더라도 denoiser step마다 residual path로 누적될 수 있고, 이 최적화의 목표는 “품질상 유사함”이 아니라 기존 exact null path를 더 짧게 실행하는 것이었다.

따라서 direct `M=1` 결과를 허용 오차로 승인하지 않고, production reference와 같은 raw bit를 만드는 calibration shape를 탐색했다.

### 7.3. Canonical cache: qualification 이후의 bitwise-exact

Startup에서 모든 row가 $b_V$인 canonical input을 만든다.

$$
X_{\mathrm{canonical}} =
\mathbf{1}_{256}b_V^\top
\in\mathbb{R}^{256\times D}.
$$

같은 dtype, layout과 output projection을 적용한다.

$$
C_{\mathrm{canonical}} =
X_{\mathrm{canonical}}W_O
+
\mathbf{1}_{256}b_O^{\top}.
$$

각 row는 수학적으로 동일하므로 첫 row를 cache한다.

$$
c_{\mathrm{cache}}^{\top} =
C_{\mathrm{canonical}}[0,:].
$$

Runtime unconditional output은 allocation 없이 broadcast 가능한 view로 구성한다.

$$
Y_{-} =
\operatorname{expand}
\left(
 c_{\mathrm{cache}}, [N,D]
\right).
$$

`M=256`은 테스트한 canonical shape들 중에서 경험적으로 선택했다. 관찰된 `M=1` behavior를 피하기에 충분히 컸고, qualification을 통과한 production environment에서 reference BF16 bit를 재현했다. 선택된 정확한 cuBLAS algorithm이나 tile configuration에 대해서는 어떤 가정도 하지 않는다.

#### Startup qualification과 fallback

Startup self-test는 model의 모든 대상 layer에 대해 다음 순서로 수행한다.

1. Canonical `M=256` input으로 cache를 생성한다.
2. Worker에 설정된 production-equivalent reference shape에서 기존 `[2N,D]` path를 실행한다.
3. `[N,D]` conditional slice와 reference conditional slice를 raw-bit 비교한다.
4. Cached unconditional row와 reference unconditional row를 raw-bit 비교한다.
5. 모든 대상 layer와 configured validation shape가 통과할 때만 canonical-cache path를 활성화한다.
6. 하나라도 실패하면 기존 exact reference-compatible path로 fallback한다.

이 qualification은 다음 변화에 대응한다.

- GPU architecture 변경
- CUDA·cuBLAS·PyTorch version 변경
- Dtype, layout, stride 또는 alignment 변경
- Weight reload, parameter pointer 또는 parameter version 변경
- 대상 layer configuration 변경

Self-test는 모든 가능한 shape와 미래 environment에 대한 수학적 증명이 아니다. 현재 worker의 configured validation scope에서 기대한 numerical behavior가 재현되는지 확인하고 fast path를 gate하는 장치다.

### 7.4. cuBLASLt GELU epilogue가 bitwise-exact를 약속할 수 없는 이유

기존 eager path는 GEMM output을 BF16으로 저장한 뒤 GELU를 적용한다.

$$
y =
\operatorname{GELU}
\left(
\operatorname{round}_{\mathrm{BF16}}(Wx+b)
\right).
$$

cuBLASLt epilogue는 더 높은 precision의 accumulator에서 bias와 GELU를 적용한 뒤 BF16으로 저장할 수 있다.

$$
y' =
\operatorname{round}_{\mathrm{BF16}}
\left(
\operatorname{GELU}(Wx+b)
\right).
$$

두 식은 rounding boundary가 다르다. 또한 vendor epilogue의 GELU approximation과 evaluation order가 eager implementation과 동일하다고 가정할 수 없다. 따라서 같은 nominal activation을 계산하더라도 bitwise equivalence를 요구하지 않았다.

두 performance qualification run에서 30개 MLP layer가 사전에 정의한 layer-level tolerance를 통과했고, 관찰된 최대 절대 차이는 `0.0625`였다. NaN과 Inf가 없는지 확인하고, 별도의 model-level regression도 통과한 경우에만 cuBLASLt fast path를 활성화했다.

Production에는 다음 세 경로를 구분해 유지했다.

1. cuBLASLt bias+GELU fast path
2. 기존 BF16 rounding boundary를 유지하는 optimized path
3. 완전한 sequential reference path

### 7.5. 하나의 optimization, 하나의 contract

“품질 regression 없음”과 “모든 optimization이 byte-exact함”을 같은 의미로 사용하지 않았다.

| Optimization | Numerical contract | Qualification |
| --- | --- | --- |
| Null-context elimination + canonical cache | 대상 block·denoiser output의 raw BF16 bitwise equality | 모든 대상 layer에서 reference row·output raw-bit 비교, 실패 시 prior exact path fallback |
| Q/K RMSNorm + RoPE fusion | Reference operation과 exact 또는 사전에 정의한 BF16 tolerance | Shape·dtype별 differential test와 model regression |
| Sparse LayerNorm·AdaLN·batch affine fusion | Reference operation과 exact 또는 사전에 정의한 BF16 tolerance | Per-op differential test와 real-payload regression |
| cuBLASLt bias+GELU | Non-bitwise-exact, bounded numerical error | Layer-level tolerance, NaN/Inf 검사, model-level regression |

빠른 path가 생길수록 observable fallback은 더 중요하다. Custom extension이 조용히 실패해 eager path로 돌아가면 서비스는 정상처럼 보이지만 latency regression을 놓칠 수 있다.

모든 optimization에 다음 장치를 유지했다.

- Reference implementation과 독립 feature flag
- Startup qualification과 cache invalidation
- Parameter pointer·version, dtype, device와 layout validation
- GPU·CUDA·PyTorch environment fingerprint 확인
- Unsupported shape와 runtime exception fallback
- Fallback 횟수와 reason logging
- Real-payload benchmark와 regression test

Custom extension은 serving worker에서 JIT compile하지 않고 production environment와 같은 조건의 prebuilt wheel로 배포했다.

---

## Conclusion

처음 받은 Agent의 답은 틀리지 않았다. 대표 block은 `13`개 graph와 `12`개 graph break로 쪼개졌고, representative profiler의 CUDA kernel time 대부분은 이미 FlashAttention과 GEMM에 있었다. Full `torch.compile`과 TensorRT port를 우선순위에서 내린 판단은 합리적이었다.

이번 작업이 이어진 이유는 그 답을 부정했기 때문이 아니라 질문의 단위를 바꿨기 때문이다.

> 범용 compiler가 전체 model을 최적화할 수 있는가?

에서

> 현재 forward path에서 실행할 필요가 없는 계산과 불필요하게 끊긴 memory path는 무엇인가?

로 질문을 바꿨다.

결과적으로 VARCO3D 2.0의 15-step denoise latency를 asset별 동일 가중 평균 기준, 출력 품질 하락 없이 `25.66%` 줄일 수 있었다.

CUDA 커널 전문 엔지니어의 관점에서 보면 `M=256` canonical cache를 찾은 과정이나 GELU candidate를 다시 설계한 과정은 다소 우회적으로 보일 수 있다. 나 역시 이번 작업을 통해 fusion의 핵심이 단순히 kernel 수를 줄이는 데 있는 것이 아니라, 실제 global-memory round trip과 intermediate materialization을 제거하는 데 있다는 점을 시행착오로 배웠다.

하지만 이 작업의 의미는 처음부터 CUDA 전문가처럼 문제를 풀었다는 데 있지 않다. 모델의 conditioning semantics와 forward equation을 이해하는 리서처가 불필요한 계산을 수식으로 특정하고, Agent를 repository 탐색, 구현, benchmark와 regression test의 실행자로 활용했다는 데 있다.

이 작업은 AI Agent가 CUDA 전문성을 대체하는 방식이라기보다, 도메인 지식을 가진 리서처가 자신이 직접 다룰 수 있는 engineering boundary를 넓히는 방식에 가까웠다.

---

![](./assets/varco3d2-result-a.png)

이 설계는 일회성 실험으로 끝나지 않았다. 최적화된 kernel은 VARCO3D 2.0 serving path에 들어갔고, 이 과정에서 정리한 profiling, candidate selection, numerical qualification과 fallback 원칙은 사내 `worker-forward-optimize` skill로 정리했다.

![](./assets/varco3d2-result-b.png)

*Generated by VARCO3D 2.0.*

AI Agent 시대에 개인의 작업 반경은 분명 넓어진다. AI Researcher가 webapp tool을 만들고, ML Engineering task를 직접 끌어오고, 때로는 custom CUDA extension까지 production path에 넣을 수 있다.

하지만 그 확장은 'AI의 판단에 모든 것을 맡긴다'는 뜻이 아니다. 무엇이 불필요한 계산인지, 어떤 numerical contract가 필요한지, 어떤 결과를 production에 채택할 수 있는지는 여전히 사람이 정해야 한다.

더 멀리 가기 위해 모든 것을 알아야 하는 시대는 조금씩 지나가고 있다. 대신 무엇을 모르는지 알고, 무엇을 검증해야 하는지 정하고, 어디까지 책임질 수 있는지 판단하는 능력이 더 중요해지고 있다.

그 기준을 세울 수 있다면, 익숙하지 않은 영역도 더 이상 완전히 남의 일이 아니다.
