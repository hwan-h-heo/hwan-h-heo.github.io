## Abstract

CUDA에서 GPU 연산은 kernel이라는 함수 단위로 실행된다. `x * scale`, `+ shift`, `ReLU`를 따로 계산하면 kernel을 세 번 호출해야 하고, 각 kernel이 만든 중간 tensor는 다음 kernel이 읽을 수 있도록 global memory에 저장된다.

Kernel fusion은 이어지는 연산을 하나의 kernel로 합쳐 이 중간 tensor를 저장하고 다시 읽는 memory traffic과 kernel launch 횟수를 줄이는 최적화다. Elementwise처럼 계산은 가벼운데 memory 접근과 launch 비용이 큰 연산에서 특히 효과적이다.

그렇다고 여러 연산을 무조건 하나로 합치는 것이 정답은 아니다. 중간값을 thread나 block 안에 오래 유지하면 synchronization, register, shared memory 사용량이 늘어나고, 이미 잘 튜닝된 library kernel을 포기하면서 오히려 본 계산이 느려질 수 있다.

이번 글에서는 값을 만드는 producer와 그 값을 사용하는 consumer를 하나의 kernel로 묶는 vertical fusion을 중심으로 elementwise chain, GEMM prologue·epilogue, reduction, indexed workload를 차례로 살펴본다. 각 경우에 줄어드는 byte 수와 새로 생기는 비용을 함께 계산하며 어디까지 합쳐야 실제 성능 개선으로 이어지는지 정리한다.

---

## 1. Kernel 사이의 Intermediate

### Materialization과 두 종류의 fusion

연산 graph에서 값을 만드는 쪽을 producer, 그 값을 사용하는 쪽을 consumer라고 한다. 둘이 별도 kernel에서 실행되면 중간값은 producer가 끝난 뒤에도 남아 있어야 하며 이 값을 global memory의 tensor로 저장하는 과정을 materialization이라고 한다.

![abs와 sum이 별도 kernel로 실행되는 unfused Nsight Systems trace](./assets/nvidia-unfused-trace.webp)

*`*abs`가 output-sized intermediate를 쓰고 `sum`이 읽는 trace. Kernel 실행 구간만 보이며 reduction 내부 단계는 나타나지 않는다. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 1](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

![abs와 sum이 하나의 kernel로 합쳐진 fused Nsight Systems trace](./assets/nvidia-fused-trace.webp)

*Reduction 안에서 `abs`를 계산해 per-element intermediate를 없앤 trace. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 2](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

```text
unfused: producer → global-memory materialization → consumer
fused:   producer → register / shared memory → consumer
```

위의 fused path가 vertical fusion이다. Producer가 만든 값을 consumer가 register나 shared memory에서 바로 사용하므로 중간 tensor가 사라진다.

Horizontal fusion은 서로 독립적인 작은 연산을 한 kernel에 묶는다. 여러 reduction을 함께 실행해 dispatch 비용이나 입력 읽기를 공유할 수 있지만 producer–consumer 중간값이 사라지는 것은 아니다.

별도 consumer가 중간값을 HBM이 아니라 L2에서 읽을 수도 있다. 하지만 cache residency는 capacity, eviction, traversal order에 따라 달라지며 kernel이 register나 shared memory에 중간값을 유지하는 것과는 다르다. Logical traffic은 식으로 계산하고 실제 traffic은 profiler에서 확인해야 한다.

### Byte Budget

`x`, `scale`, `shift`가 모두 $N$개 원소를 가진 tensor이고 각 표현식이 별도 kernel을 launch하는 인위적인 elementwise chain을 생각해 보자.

```python
u = x * scale
v = u + shift
out = relu(v)
```

원소당 크기가 $b$바이트일 때 $S=Nb$라 두자. Cache reuse를 제외한 unfused path에는 tensor 크기의 read 또는 write가 여덟 번 발생한다.

$$
T_{\mathrm{unfused}} \approx 8S.
$$

Fused kernel은 `u`와 `v`를 일시적인 값으로 유지한다.

```cpp
float value = x[i] * scale[i] + shift[i];
out[i] = fmaxf(value, 0.0f);
```

$$
T_{\mathrm{fused}} \approx 4S,
$$

세 번의 input read와 최종 store만 남는다는 계산이다. $8S\rightarrow4S$는 cache reuse를 제외한 값이며 `scale`이나 `shift`가 scalar, 짧은 broadcast vector, cache-resident data라면 실제 traffic은 더 작아진다. 반대로 stride와 write allocation 때문에 커질 수도 있다.

Byte budget은 없앨 수 있는 logical traffic을 찾는 데 사용한다. 실제 speedup은 예측하지 않는다.

### CUDA Graph와의 차이

CPU submission과 launch overhead가 병목이라면 workflow를 실행 가능한 graph로 미리 준비해 launch 설정 비용을 줄이는 CUDA Graph를 먼저 검토할 수 있다.

CUDA Graph에서도 각 연산은 별도의 kernel node로 남고 그 사이의 materialized intermediate도 사라지지 않는다. Fusion은 data boundary를 줄이고 CUDA Graph는 작업 제출 비용을 줄이므로 두 방법을 함께 적용할 수도 있다. 이 구분은 [CUDA 13.3 Programming Guide의 CUDA Graph 모델](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html)을 따른다.

[VARCO3D 2.0 최적화](/blogs/posts/optimizing-sparse-3d-generation-inference-kor/)에서도 두 방법의 차이를 확인했다. `aten::gelu`를 standalone custom `gelu_tanh` kernel로 교체했지만 eager path보다 느렸다. GEMM output을 저장한 뒤 GELU가 다시 읽고 쓰는 경계가 그대로 남았기 때문이다.

Bias와 GELU를 cuBLASLt epilogue로 옮기자 이 왕복이 사라지고 standalone call도 `_addmm_activation`으로 바뀌었다. Custom GELU가 consumer만 바꿨다면 epilogue fusion은 GEMM과 GELU 사이의 경계를 없앴으며 결과는 bitwise exact가 아닌 tolerance 기준으로 검증했다.

---

## 2. Intermediate Ownership

중간값을 없애려면 값을 누가 끝까지 소유하는지 알아야 한다. 한 thread에서 끝나는 계산과 여러 block이 같은 결과에 기여하는 계산은 필요한 synchronization이 다르다.


| Ownership 범위  | 대표적인 경계                      | 주로 추가되는 비용                            |
| ------------- | ---------------------------- | ------------------------------------- |
| Thread 하나     | Elementwise chain            | Live register와 instruction 수          |
| Warp 또는 block | Row reduction, normalization | Shuffle/shared memory와 barrier        |
| GEMM tile     | Prologue 또는 epilogue         | Layout 변환과 mainloop resource pressure |
| 여러 block      | 큰 reduction 또는 충돌하는 scatter  | Workspace, atomic 또는 추가 kernel        |


### Thread-local 연산

예를 들어

$$
out_i=\operatorname{clamp}(x_i s_i+t_i,\ell,u)
$$

에서는 한 thread가 index $i$를 처음부터 끝까지 소유하므로 shape, stride, aliasing, side effect에 문제가 없다면 graph compiler가 연산을 자동으로 fusion할 수 있다.

[PyTorch Inductor](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html)는 이런 graph에서 Triton code를 생성한다. 경계를 정하는 주체는 graph compiler이며 Triton은 kernel language와 compiler를 제공한다. [XLA](https://openxla.org/xla/architecture)와 [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)도 compilation stage에서 비슷한 결정을 내린다.

Python 표현식이 한 줄이라고 kernel도 하나라고 단정할 수는 없다. 생성된 graph나 kernel trace를 확인해야 한다.

### GEMM Prologue와 Epilogue

Linear layer는 GEMM 뒤에 bias와 activation을 적용하는 경우가 많다. 세 연산을 별도 kernel로 실행하면 stage 사이에 output 크기의 tensor가 생기지만 epilogue fusion은 GEMM 결과를 global memory에 저장하기 전에 bias와 activation을 적용해 이 중간값을 없앤다.

![CUTLASS GEMM hierarchy와 epilogue tile 및 epilogue functor](./assets/cutlass-gemm-epilogue.webp)

*전통적인 CUTLASS GEMM hierarchy. Thread별 accumulator fragment가 epilogue tile과 functor를 거쳐 global memory에 저장된다. 이 그림은 인용한 CUTLASS 모델의 register-accumulator path를 설명하며 Blackwell SM100 경로는 아래에서 따로 구분한다. Source: [NVIDIA CUTLASS — Efficient GEMM in CUDA, Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue)*

다음 계산에서

$$
Z=\alpha AB+\beta C,
\qquad
D=\operatorname{GELU}(Z+b),
$$

$S$를 $M\times N$ output의 byte 크기라 두자. 비교하는 두 path가 공통으로 읽는 A, B, optional C와 더 작은 broadcast bias는 제외한다. GEMM, bias, activation을 따로 실행하면 output과 intermediate에서 대략

$$
T_{\mathrm{unfused}}\approx5S
$$

의 traffic이 생긴다. Fused epilogue는 같은 범위에서 최종 store만 남겨 $T_{\mathrm{fused}}\approx S$까지 줄일 수 있다. Training은 다음처럼 activation과 pre-activation을 모두 반환하기도 한다.

$$
D=\operatorname{GELU}(Z),
\qquad
\mathrm{Aux}=Z.
$$

`Aux`는 backward에서 다시 사용한다. Auxiliary tensor가 남아도 GEMM과 activation 사이의 kernel 경계는 사라진다.

Accumulator의 위치는 architecture에 따라 다르다. Hopper까지의 전통적인 CUTLASS path는 MMA 결과를 register fragment 형태로 epilogue에 넘기는 경우가 많지만 Blackwell SM100의 TCGen05 path는 accumulator를 TMEM에 저장한다. Epilogue는 TMEM의 subtile을 register로 불러와 fusion을 적용한 뒤 shared memory와 TMA를 거쳐 저장한다.

두 path 모두 global output을 materialize하기 전에 변환을 적용하지만 accumulator가 항상 thread register에 있는 것은 아니다. 자세한 흐름은 공식 [CUTLASS SM100 epilogue 설명](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#background-sm100-epilogue-flow)에서 확인할 수 있다.

Prologue는 GEMM의 반대쪽 경계를 다룬다. Quantized weight에서

$$
W=\operatorname{dequant}(W_q;s,z)
$$

라면 dense temporary weight를 만들지 않고 operand tile을 불러올 때 unpack, scaling, layout transform을 수행할 수 있지만 instruction과 register 사용량, operand 공급 압력은 늘어난다.

Asynchronous copy나 pipeline이 이 비용 일부를 숨길 수 있다. 하지만 pipelining은 latency를 숨기는 방법일 뿐 fusion의 성능을 보장하지 않는다.

### Reduction

LayerNorm, RMSNorm, Softmax는 여러 원소를 하나의 statistic으로 줄이며 한 row가 warp나 block 안에 들어오면 같은 kernel에서 reduction과 normalization까지 처리할 수 있다.

한 block이 row 전체를 소유할 수 없다면 input을 다시 읽거나 작은 partial-result workspace를 사용해야 하며 atomic이나 추가 kernel이 남을 수도 있다.

평균과 분산에 필요한 최소 Welford state는 $(n,\mu,M_2)$다. 비어 있지 않은 두 partial state $A$, $B$에 대해 $\delta=\mu_B-\mu_A$라 두면 다음처럼 합친다.

$$
n=n_A+n_B,
\qquad
\mu=\mu_A+\delta\frac{n_B}{n},
$$

$$
M_2=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n}.
$$

이 상태는 warp와 block의 hierarchical reduction에서 합칠 수 있지만 floating-point 연산 순서가 달라지므로 bit 단위 결과도 달라질 수 있다. Combine rule은 [Chan, Golub, LeVeque의 parallel variance 분석](https://doi.org/10.1080/00031305.1983.10483115)을 따른다.

Consumer가 각 tile을 바로 사용할 수 있다면 큰 per-element intermediate를 없애고 한 block이 결과를 끝까지 소유할 수 없다면 작은 partial workspace만 남긴다.

FlashAttention이 대표적인 사례다. Score tile을 online softmax와 $V$ multiplication에 바로 전달해 전체 score와 probability matrix를 만들지 않는다.

### Indexed·sparse workload

Index가 들어오면 read locality와 write ownership이 data에 따라 달라진다. `batch_ids`로 parameter row를 고르는 token-wise affine transform을 보자.

```python
scale_token = scale[batch_ids]
shift_token = shift[batch_ids]
out = x * scale_token + shift_token
```

Eager mode에서는 affine operation 전에 `[T,C]` gather 결과 두 개가 생길 수 있다. Compiler, DSL kernel, fused custom operator는 token을 처리할 때 선택된 parameter 값을 바로 읽어 이 중간값을 없애며 구현 전에는 다음 항목을 확인한다.

- Channel 방향이 연속적이어서 인접 lane이 `x`, `out`, 선택된 parameter row를 coalesced access하는가?
- 가까운 token이 같은 batch나 spatial key를 충분히 재사용해 grouping 또는 shared-memory staging 비용을 회수하는가?
- Sorting이나 binning 비용을 여러 downstream stage에서 나눠 부담할 수 있는가?
- Scatter에서는 같은 key를 warp나 block 안에서 먼저 줄여 global atomic 수를 줄일 수 있는가?

다음과 같은 scatter에서

```cpp
atomicAdd(output + index[i], value[i]);
```

Update가 hot destination에 몰리면 atomic이 serialize되고 accumulation order도 data에 따라 달라진다. Warp aggregation, block-local partial, spatial binning은 global atomic을 줄이지만 추가 작업이나 새로운 materialization이 필요할 수 있다.

Irregular gather/scatter가 항상 direct CUDA를 요구하지는 않는다. Graph compiler, Triton 같은 GPU DSL, library primitive, custom CUDA operator를 모두 후보로 두고 representation과 conflict 분포를 기준으로 선택한다.

Layout도 함께 확인해야 한다. Transform 하나를 없애면서 contiguous access가 strided access로 바뀌면 오히려 느려질 수 있으므로 consumer-oriented packing이나 shared-memory transpose는 upstream conversion, tail, alignment를 포함한 전체 경로에서 측정한다.

---

## 3. Fusion의 비용

### Resource 사용량과 Library 성능

Fusion은 중간값의 live range를 늘린다. Register pressure가 높아지면 occupancy가 떨어지거나 local memory spill이 생기고 shared memory 사용량이 늘면 resident block 수가 줄어든다. Barrier가 warp나 block을 대기시키는 데다 producer와 consumer가 서로 다른 tile shape를 선호할 수도 있다.

Occupancy 자체가 목표는 아니다. 하지만 active warp가 너무 적으면 memory와 instruction latency를 숨기기 어렵다.

가장 놓치기 쉬운 비용은 튜닝된 library kernel의 성능이며 실제 shape에서 다음 두 경로를 비교해야 한다.

```text
fast library GEMM + small separate kernel
versus
custom fused GEMM with a slower mainloop
```

cuBLASLt나 CUTLASS가 지원하는 epilogue는 tuned mainloop를 유지할 수 있다. 더 넓은 custom fusion은 줄어든 traffic과 launch 비용이 mainloop 성능 하락, workspace, dispatch, backward recomputation 비용보다 클 때만 의미가 있다.

### Numerical Contract

Fusion은 intermediate의 rounding point를 없앨 수 있다. Unfused BF16 GEMM은 FP32 accumulator를 BF16으로 저장할 때 반올림하지만 fused epilogue는 최종 store까지 높은 precision을 유지한다. FMA contraction이나 reduction tree가 달라져도 finite-precision 결과가 바뀐다.

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\not\equiv
\operatorname{fl}(a+\operatorname{fl}(b+c)).
$$

Benchmark를 시작하기 전에 필요한 numerical contract를 정한다.


| Contract                    | 요구 사항                                |
| --------------------------- | ------------------------------------ |
| Mathematical equivalence    | 실수 산술에서 같은 연산                        |
| Tolerance-based equivalence | 명시한 absolute/relative error 범위 안의 결과 |
| Determinism                 | 명시한 조건에서 같은 구현을 반복 실행했을 때 같은 결과      |
| Byte-exact reference match  | 지정한 reference path와 같은 bit           |


네 contract는 서로 독립적이다. 결과가 deterministic해도 reference와 다를 수 있고 tolerance 안에 있으면서 낮은 bit는 실행마다 달라질 수 있다. 선택한 contract에 따라 intermediate rounding, Tensor Core mode, FMA, reduction reassociation, atomic order의 허용 범위가 달라진다.

---

## 4. 구현 선택과 검증

### Control Surface

Fusion을 구현하는 방법은 하나가 아니다. 없애려는 경계를 표현할 수 있는 가장 높은 수준부터 검토한다.


| Control surface                        | 적합한 경우                                                                                                           |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Framework/compiler fusion              | Graph와 규칙적인 layout이 전체 경계를 드러낼 때                                                                                 |
| CUDA Graph                             | Launch/submission overhead가 지배적이고 materialization은 남아도 될 때                                                       |
| CCCL algorithm: Thrust 또는 CUB          | Transform, reduce, scan, sort, reduce-by-key가 관리되는 primitive와 맞을 때. 조합 과정에 temporary storage나 여러 launch가 남을 수 있음 |
| cuBLASLt 또는 지원되는 vendor fused operator | GEMM post-op이 지원되는 epilogue와 layout에 맞을 때                                                                        |
| CUTLASS / CuTe DSL / GPU kernel DSL    | Tile dataflow나 prologue·epilogue를 더 세밀하게 제어해야 할 때                                                                |
| Custom CUDA operator                   | 위 수준에서 ownership, indexing, atomic, integration을 효과적으로 표현할 수 없을 때                                                |


CCCL은 Thrust와 CUB를 포함한다. Thrust는 high-level parallel algorithm을 제공하고 CUB는 `DeviceReduce` 같은 device·block·warp primitive를 제공한다. Temporary storage와 실행 방식도 candidate의 비용에 포함해야 하며 여러 primitive를 조합해도 single-kernel fusion이 보장되지는 않는다. 공식 [CCCL/CUB `DeviceReduce` 문서](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html)에서 그 범위를 확인할 수 있다.

### 검증과 탈락 기준

Baseline과 candidate는 같은 end-to-end 구간, shape 분포, dtype, warm-up 조건에서 측정한다. 측정 항목은 fusion으로 줄이려는 비용과 연결해야 한다.


| 주장                       | 확인할 항목                                                                                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 경계가 사라졌다                 | Kernel trace와 launch 수, 대상 allocation의 부재, 예상한 logical intermediate write/read byte                                                                             |
| Physical traffic이 줄었다    | Nsight Compute **Memory Workload Analysis**의 device-memory·L2 byte/throughput과 cache behavior                                                                   |
| Resource 비용이 이득을 지우지 않았다 | **Launch Statistics**와 **Occupancy**의 thread당 register, static/dynamic shared memory, theoretical·achieved occupancy, local-memory traffic 또는 spill instruction |
| 본 계산이 느려지지 않았다           | 해당하면 GEMM/mainloop throughput, 전체 kernel time, cast·reorder·workspace·allocation·dispatch를 포함한 end-to-end latency                                               |
| Correctness가 유지됐다        | 선택한 numerical contract 아래의 대표 shape, tail-heavy shape, conflict가 집중된 입력                                                                                         |


표의 section 이름은 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)를 따른다. Raw metric identifier는 architecture와 tool version에 따라 달라질 수 있으므로 section 단위로 표기했다.

아래 항목 중 하나에 해당하면 candidate를 버리거나 fusion 범위를 좁힌다.

- 목표 intermediate가 여전히 쓰이고 읽힌다.
- 중요한 shape에서 end-to-end 개선이 run-to-run noise 안에 있거나 오히려 느려진다.
- Spill, shared-memory 사용량, occupancy 하락, 느려진 library mainloop가 byte 절감 효과를 삼킨다.
- Preprocessing, workspace, backward recomputation이 비용을 측정 구간 밖으로 옮겼을 뿐이다.
- 선택한 numerical contract를 위반한다.

Fusion의 목표는 최대한 큰 kernel을 만드는 것이 아니다. 측정된 비용을 없애고 더 큰 비용을 만들지 않는 범위까지만 합친다.

---

## Closing

Kernel fusion은 intermediate의 수명을 바꾼다. Materialize되는 값을 찾고 그 값을 소유하는 thread, warp, block을 확인하면 fusion할 수 있는 범위가 정해진다.

Thread-local elementwise chain은 비교적 단순하지만 GEMM prologue와 epilogue는 빠른 mainloop를 유지해야 한다. Reduction은 소유 범위가 여러 thread와 block으로 넓어지고 indexed·sparse workload는 locality와 write conflict가 입력마다 달라진다.

병목이 launch submission에 있다면 CUDA Graph로 충분할 수 있다. 튜닝된 library와 작은 별도 kernel이 end-to-end에서 더 빠르면 그 경계를 유지한다. 좋은 fused kernel은 특정 materialization을 없애고 성능 측정과 numerical contract를 모두 통과한 kernel이다.

---

## References

Architecture별 설명은 2026년 8월 현재의 CUDA 13.3 Programming Guide와 CUTLASS 4.6.1을 기준으로 확인했다. 다른 toolkit이나 GPU generation을 대상으로 할 때는 instruction path와 profiler 세부 사항을 다시 확인해야 한다.

1. NVIDIA, *Kernel Fusion in NVIDIA CUDA: Optimizing Memory Traffic and Launch Overhead*: [https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)
2. NVIDIA, *CUDA 13.3 Programming Guide — CUDA Graphs*: [https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html)
3. NVIDIA, *CUTLASS — Efficient GEMM in CUDA*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
4. NVIDIA, *CUTLASS — SM100 Epilogue Flow*: [https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#background-sm100-epilogue-flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#background-sm100-epilogue-flow)
5. NVIDIA, *CCCL/CUB DeviceReduce*: [https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html)
6. NVIDIA, *Nsight Compute Profiling Guide*: [https://docs.nvidia.com/nsight-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
7. PyTorch, *Torch Compiler — Getting Started*: [https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html)
8. OpenXLA, *XLA Architecture*: [https://openxla.org/xla/architecture](https://openxla.org/xla/architecture)
9. NVIDIA, *TensorRT Performance Best Practices*: [https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
10. T. F. Chan, G. H. Golub, R. J. LeVeque, *Algorithms for Computing the Sample Variance: Analysis and Recommendations*: [https://doi.org/10.1080/00031305.1983.10483115](https://doi.org/10.1080/00031305.1983.10483115)
