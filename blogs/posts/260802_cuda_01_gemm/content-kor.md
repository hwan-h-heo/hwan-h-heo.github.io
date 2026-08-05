## Abstract

General Matrix Multiplication, 줄여서 GEMM은 두 행렬을 곱하고 필요하면 기존 행렬을 일정 비율로 더하는 연산이다.

딥러닝의 Linear layer와 attention projection, MLP의 주된 연산은 GEMM이고 일부 convolution도 입력을 실제 행렬로 펼치지 않는 implicit GEMM 형태로 계산된다. 그만큼 딥러닝과 HPC에서 GEMM은 GPU 시간을 많이 차지한다.

GEMM 최적화는 Tensor Core instruction을 직접 다루기 전에 어디까지 library에 맡길지 정하는 데서 시작한다. 표준 dense GEMM은 cuBLAS를 기준으로 잡는다. Layout, epilogue, workspace, algorithm 후보를 직접 제어해야 하면 cuBLASLt를 검토한다. 두 API로 필요한 dataflow를 표현할 수 없을 때 CUTLASS나 custom kernel까지 내려간다.

이 순서는 절대적인 성능 순위가 아니다. 처음부터 custom kernel을 작성하는 대신 구현 범위를 필요한 만큼만 넓히기 위한 기준이다.

이 글은 naive GEMM, 계층적 tiling, Tensor Core pipeline을 차례로 설명한다. 특정 GPU에서 가장 빠른 tile을 찾는 방법보다 library가 내부에서 처리하는 문제와 shape에 따라 달라지는 선택을 다룬다. 마지막에는 실제 workload를 위한 shape manifest와 검증 절차를 정리한다.

본문의 API와 아키텍처 설명은 2026년 8월, CUDA 13.3과 CUTLASS 4.6.1 공식 문서를 기준으로 확인했다. cuBLASLt epilogue, dtype, grouped GEMM, CUTLASS scheduler의 지원 범위는 toolkit과 GPU 세대에 따라 달라지므로, 실제 적용 전에는 배포 환경의 문서와 API query를 다시 확인해야 한다.

---

## 1. Library와 Custom Kernel 사이

GEMM, cuBLAS, cuBLASLt, CUTLASS, CuTe, Tensor Core는 서로 다른 추상화 계층에 있다.

```text
Mathematical operation
  GEMM
    ↓
Library / API
  cuBLAS / cuBLASLt
    ↓
Kernel construction
  CUTLASS device-wide operators and CuTe components
    ↓
Kernel policy
  tile, pipeline, epilogue, scheduler
    ↓
Hardware
  SM, Tensor Core, CUDA Core, register, shared memory, HBM
```

아래 계층으로 내려갈수록 데이터 흐름을 더 세밀하게 제어할 수 있다. 대신 tail, alignment, numerical behavior, architecture portability를 직접 책임져야 한다.

먼저 현재 API로 연산을 표현할 수 있는지 확인한다. 표현할 수 있다면 실제 workload에서도 충분히 빠른지 측정한다. Custom kernel은 두 조건 중 하나를 만족하지 못할 때 검토해도 늦지 않다.


| 필요한 것                                      | 첫 후보                                     | 다음 단계로 내려갈 조건                           |
| ------------------------------------------ | ---------------------------------------- | --------------------------------------- |
| 표준 dense GEMM                              | cuBLAS                                   | layout, epilogue, workspace, 후보 제어가 필요함 |
| 유연한 layout·compute type·epilogue           | cuBLASLt                                 | 필요한 조합이나 dataflow를 API가 표현하지 못함         |
| Custom mainloop·quantized decode·scheduler | CUTLASS device operator / CuTe component | 제공된 구성 요소로도 표현하거나 유지하기 어려움              |
| 완전히 특수한 데이터 흐름                             | Custom CUDA kernel                       | 유지·검증 비용보다 측정된 이득이 큼                    |


표의 `cuBLAS first`는 표준 GEMM의 비교 기준을 적은 코드로 확보한다는 뜻이다. Shape에 따라 cuBLASLt가 처음부터 더 잘 맞을 수 있다. CUTLASS로 작성한 kernel이 두 library보다 항상 빠른 것도 아니다.

cuBLASLt는 cuBLAS의 상위 버전이 아니다. Layout, algorithm, heuristic을 더 유연하게 기술하는 별도 API다. [NVIDIA cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)

일반적인 GEMM은 다음과 같다.

$$
D=\alpha\,\operatorname{op}(A)\operatorname{op}(B)+\beta C
$$

Transpose가 없을 때

$$
A\in\mathbb{R}^{M\times K},\qquad
B\in\mathbb{R}^{K\times N},\qquad
C,D\in\mathbb{R}^{M\times N}
$$

이고

$$
D_{ij}=\alpha\sum_{k=0}^{K-1}A_{ik}B_{kj}+\beta C_{ij}
$$

이다. Multiply와 add를 각각 1 FLOP로 세면 주된 matrix product의 연산량은 약 $2MNK$다. Alpha와 beta를 적용하는 연산은 이 근사에서 제외한다.

---

## 2. Data Reuse and Hierarchical Tiling

### Naive GEMM

아래 kernel은 $\alpha=1$, $\beta=0$, transpose 없음으로 제한한 $D=AB$를 계산한다. A, B, D는 row-major contiguous buffer라고 가정한다. 이는 tile reuse를 설명하기 위한 예제이며 column-major가 기본인 cuBLAS의 인자 해석과는 다르다.

```cpp
__global__ void naive_gemm(
    const float* A,
    const float* B,
    float* D,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    D[row * N + col] = acc;
}
```

같은 row의 output은 A의 한 row를 공유한다. 같은 column의 output은 B의 한 column을 공유한다. Cache와 broadcast가 traffic 일부를 줄여 주지만 working set이 커지면 이 재사용을 cache hit에만 맡길 수 없다.

고성능 GEMM은 output을 CTA, warp 또는 warp-group, instruction 수준의 tile로 나눈다. 각 계층이 A와 B의 일부를 명시적으로 재사용한다.

### Tile과 Arithmetic Intensity

CTA 하나가 $B_M\times B_N$ output tile을 계산하고 K dimension을 $B_K$ 단위로 순회한다고 하자. 한 K tile에서 수행하는 연산량은

$$
2B_MB_NB_K\quad\text{FLOPs}
$$

이고 A와 B를 HBM에서 한 번씩 읽는 데 필요한 byte는 element당 크기를 $b$라 할 때

$$
b(B_MB_K+B_KB_N)=bB_K(B_M+B_N)
$$

이다. Output load/store, CTA 사이의 cache reuse, alignment·padding traffic을 생략하고 A/B tile이 CTA 안에서 완전히 재사용된다고 가정하면 input 기준 arithmetic intensity는

$$
\operatorname{AI}_{\text{input}}
\approx
\frac{2B_MB_NB_K}{bB_K(B_M+B_N)}
=
\frac{2B_MB_N}{b(B_M+B_N)}
\quad\text{FLOP/byte}
$$

가 된다. 정사각형 tile $B_M=B_N=T$에서는 $\operatorname{AI}_{\text{input}}\approx T/b$다.


| CTA output tile | FP16/BF16 input ($b=2$) | FP32 input ($b=4$) |
| --------------- | -----------------------: | ------------------: |
| $64\times64$    | 약 32 FLOP/byte          | 약 16 FLOP/byte     |
| $128\times128$  | 약 64 FLOP/byte          | 약 32 FLOP/byte     |


같은 $B_K$와 dtype을 사용하면 `128 × 128` tile의 input reuse는 `64 × 64`의 두 배다. 이 수치는 성능 예측치가 아니다.

Tile이 커지면 shared memory, accumulator storage, CTA당 thread 수가 늘어난다. M이나 N이 작을 때는 동시에 실행할 CTA 수와 tail utilization도 줄어든다. Output traffic까지 포함하면 short-K의 실제 intensity는 위 근사보다 낮다.

```cpp
for (int k0 = 0; k0 < K; k0 += BK) {
    // Cooperative global-to-shared load:
    // A[BM, BK], B[BK, BN]
    __syncthreads();

    // Update the output accumulator from shared-memory tiles.
    __syncthreads();
}
```

첫 barrier는 input load가 끝난 뒤 연산을 시작하게 하고 두 번째 barrier는 다음 K tile이 buffer를 덮기 전에 현재 연산이 끝났음을 보장한다. 실제 Tensor Core kernel은 더 세밀한 asynchronous pipeline과 barrier를 사용한다.

![Global memory에서 thread-block, warp, thread tile과 epilogue로 이어지는 CUTLASS GEMM hierarchy](./assets/cutlass-gemm-hierarchy.webp)

*Block-, warp-, thread-level tile reuse와 epilogue data movement를 보여 주는 warp-level CUTLASS schematic이다. Hopper WGMMA나 Blackwell의 operand path를 그대로 나타낸 그림은 아니다. Source: [NVIDIA CUTLASS — Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)*

```text
Global GEMM
  ↓
CTA / thread-block tile: HBM traffic과 grid scheduling
  ↓
Warp or warp-group tile: shared-memory tile 분할
  ↓
MMA instruction tile: ISA-level matrix operation
  ↓
Thread / warp-group state: operand와 accumulator 관리
```

CUTLASS는 이 계층을 소프트웨어 추상화로 제공한다. Register blocking은 한 번 가져온 operand를 여러 FMA에 재사용하고 mainloop 동안 accumulator를 가까운 저장 공간에 둔다.

실제 저장 위치와 operand path는 아키텍처마다 다르다. 모든 세대를 “thread register fragment”로 설명하면 Blackwell의 TMEM 경로를 설명할 수 없다.

---

## 3. Tensor Core Pipeline

Tensor Core는 작은 matrix tile에

$$
D_{\text{frag}}\leftarrow A_{\text{frag}}B_{\text{frag}}+D_{\text{frag}}
$$

를 수행한다. 완성된 GEMM kernel은 이 instruction 외에도 HBM load, shared-memory layout, synchronization, tail handling, epilogue, store를 책임진다.

Operand path는 GPU 세대마다 다르다.

- Volta부터 Ampere까지 널리 사용된 warp-level MMA 경로에서는 shared-memory operand를 thread별 register fragment로 옮기고 accumulator도 register에 보관한다.
- **Hopper WGMMA**에서는 B를 shared-memory descriptor로 참조하고 구성에 따라 A는 shared memory 또는 register에서 공급하며 accumulator는 register에 둔다. [PTX ISA — WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma)
- **Blackwell SM100 `tcgen05.mma`**에서는 accumulator가 Tensor Memory(TMEM)에 저장된다. A는 shared memory 또는 TMEM, B는 shared memory에서 공급할 수 있다. Hopper의 register-accumulator 설명을 Blackwell에 그대로 적용할 수 없다. [NVIDIA CUTLASS — tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)

Epilogue는 accumulator를 output layout에 맞게 옮긴다. 여기에 bias, activation, dtype conversion 등을 적용한 뒤 global memory에 저장한다. cuBLASLt가 지원하는 epilogue를 사용하면 tuned mainloop를 유지하면서 intermediate store를 없앨 수 있다.

지원하지 않는 layout transform이나 irregular indexing을 mainloop에 넣으면 GEMM 자체가 느려질 수 있다. 이 경우 GEMM은 library에 남기고 인접 연산을 별도 kernel로 처리하는 편이 빠를 수 있다.

Precision은 input, compute/accumulator, output dtype으로 나눠 기록한다. BF16 input, FP32 accumulator, BF16 output은 하나의 조합이다. FP8과 block-scaled format은 scale value의 dtype과 scale granularity도 필요하다. 지원하는 조합은 GPU architecture와 library version에 따라 달라진다.

### Load와 compute 겹치기

![CUTLASS double-buffered software pipeline](./assets/cutlass-software-pipeline.webp)

*Global-to-shared load, shared-to-register load, math의 overlap을 보여 주는 legacy CUTLASS double-buffering schematic이다. Hopper WGMMA/TMA나 Blackwell pipeline을 정확히 묘사한 그림은 아니다. Source: [NVIDIA CUTLASS — Efficient GEMM in CUDA, Software Pipelining](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining)*

Software pipelining에서는 현재 tile을 계산하는 동안 다음 tile을 불러온다.

```text
load tile 0
wait tile 0

compute tile 0  ||  load tile 1
compute tile 1  ||  load tile 2
compute tile 2  ||  load tile 3
```

Double buffering과 multi-stage pipeline은 memory latency를 숨기는 대신 shared-memory 사용량과 pipeline state를 늘린다. Pipeline이 지나치게 깊으면 occupancy가 낮아지고 short-K workload에서는 setup cost만 늘 수 있다.

Ampere의 asynchronous copy, Hopper 이후의 TMA, 세대별 MMA/barrier는 모두 load와 compute를 겹치기 위한 장치다. 구현 방식은 서로 다르다.

Tile tail도 함께 봐야 한다. $M=130$이고 CTA tile의 M 크기가 64라면 세 번째 tile에서 유효한 row는 2개뿐이다. 이 경우 큰 tile의 nominal throughput보다 boundary waste와 부족한 CTA 수가 성능에 더 큰 영향을 줄 수 있다.

---

## 4. Shape에 scheduling

Scheduler를 고르기 전에 병렬성이 어느 축에서 부족한지 확인한다. Split-K와 Stream-K는 K 방향의 병렬성이 부족할 때 검토할 수 있는 방법이다.


| 관찰된 문제                          | 먼저 확인할 지표                        | Kernel-level 후보                           | 지불하는 비용                                        |
| ------------------------------- | -------------------------------- | ----------------------------------------- | ---------------------------------------------- |
| M/N이 작고 K가 길어 output tile이 부족함  | active CTA 수, absolute latency   | Split-K 또는 Stream-K 계열                    | partial reduction, workspace/atomic, 달라진 덧셈 순서 |
| Tile 수가 execution wave와 잘 맞지 않음 | 마지막 wave utilization             | 더 작은 tile 또는 dynamic/persistent scheduler | scheduler overhead, resource residency         |
| 동일 shape의 small GEMM 다수         | launch 비중, buffer stride         | strided batched GEMM                      | batch 내부 shape·layout 제약                       |
| 서로 다른 small GEMM 다수             | problem별 tile 수와 tail            | grouped persistent GEMM                   | metadata access, load imbalance                |
| $M\approx1$인 batch-1 linear     | memory bandwidth, weight traffic | GEMV/small-M kernel, weight prepacking    | 전용 layout과 추가 유지 비용                            |


Split-K는 같은 output tile의 K range를 여러 worker가 나눠 계산한다.

$$
P^{(s)}_{ij}=\sum_{k\in K_s}A_{ik}B_{kj},\qquad
P_{ij}=\sum_sP^{(s)}_{ij}
$$

Split-K는 K 방향의 병렬성을 늘리는 대신 부분합을 저장하고 다시 합친다. Stream-K 계열은 K tile 또는 MAC work를 더 유연하게 분배해 마지막 wave의 불균형을 줄인다.

기존 output tile만으로 GPU를 충분히 채울 수 있다면 두 방법 모두 손해가 될 수 있다. 추가 scheduler와 reduction 비용이 병렬화로 얻는 이득보다 커지기 때문이다.

Grouped GEMM은 서로 다른 problem을 한 persistent launch 안에서 worker CTA에 배정한다. CUTLASS grouped scheduler에도 metadata 탐색과 problem ordering 비용이 있다. Shape가 서로 다르다는 이유만으로 grouped GEMM이 항상 유리하지는 않다. [NVIDIA CUTLASS — Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)

Small-M skinny GEMM에서는 $M$ 방향의 병렬성과 weight reuse가 함께 줄어든다. $A\in\mathbb{R}^{M\times K}$, $B\in\mathbb{R}^{K\times N}$에서 $M$이 1에 가까워지면 output tile 수가 부족해진다. 같은 weight $B$를 여러 output row에서 재사용할 기회도 줄어든다.

A의 한 원소는 여전히 여러 N output에 사용된다. 이 병목을 “A reuse 부족”으로 설명하면 정확하지 않다.

Request batching은 여러 요청의 M을 합쳐 B/weight reuse와 병렬성을 늘린다. 대신 요청 대기 시간이 늘 수 있다. 이는 kernel scheduler가 아니라 latency와 throughput을 함께 다루는 시스템 수준의 선택이다.

Convolution 전체를 GEMM으로 볼 수도 없다. cuDNN의 일부 convolution은 implicit GEMM을 사용하지만 transform-based algorithm도 사용한다. Implicit GEMM은 실제 im2col matrix를 HBM에 materialize하지 않는다. [NVIDIA Convolutional Layers User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#convolution-algorithms)

---

## 5. 실제 workload에서 비교

Peak TFLOPS만으로 kernel을 고를 수는 없다. 실제 제품에서 사용하는 shape를 모으고 같은 조건에서 후보를 비교해야 한다. 빠른 후보는 수치 검증과 production replay를 다시 통과해야 한다.

### Shape manifest 만들기

평균 shape 하나로 축약하지 말고 제품 경로에서 다음 필드를 수집한다.

```text
name, M, N, K, batch/group, transA, transB,
A/B/C/D layout and dtype, compute type,
alpha, beta, epilogue, alignment,
workspace limit, frequency or probability
```

최소한 다음 regime이 실제 제품에 존재하는지 확인한다. 아래 숫자는 benchmark 결과가 아니라 분류를 설명하기 위한 예시 shape다.


| Regime              | 예시               | 이 shape가 드러내는 것                        |
| ------------------- | ---------------- | -------------------------------------- |
| Large square        | `4096×4096×4096` | steady-state compute와 pipeline         |
| Small-M, long-K     | `8×4096×16384`   | 부족한 output parallelism과 weight traffic |
| Tail-sensitive      | `130×4096×4096`  | CTA tile 경계 낭비                         |
| Short-K             | `4096×4096×64`   | load·epilogue 비중                       |
| Heterogeneous group | 실제 `(M,N,K)` 목록  | scheduler와 metadata imbalance          |


### 비교할 후보 좁히기

표준 연산은 cuBLAS를 기준점으로 삼는다. 같은 layout, epilogue, workspace 조건을 표현한 cuBLASLt heuristic 후보를 함께 비교한다. CUTLASS나 custom kernel은 두 API로 요구사항을 표현할 수 없거나 중요한 shape에서 같은 병목이 반복될 때 추가한다.

Workspace 상한과 end-to-end 측정 범위는 모든 후보에 동일하게 적용한다.

CUTLASS 후보를 탐색할 때는 profiler가 제공하는 verification, warm-up, 반복, CSV 출력을 사용할 수 있다. 다음 명령의 `4096³`은 실행 형식을 보여 주는 예시이며 성능 주장이 아니다.

```bash
./tools/profiler/cutlass_profiler \
  --operation=gemm \
  --m=4096 --n=4096 --k=4096 \
  --op_class=tensorop \
  --warmup-iterations=20 \
  --profiling-iterations=100 \
  --verification-enabled=true \
  --output=gemm-4096
```

Build에 포함된 kernel과 target architecture에 따라 실제 후보 집합은 달라진다. 가능한 option은 해당 binary의 `--operation=gemm --help`로 확인한다. [NVIDIA CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)

### Latency 옆에 원인도 남기기

- Small GEMM은 TFLOPS보다 median·tail **absolute latency**를 우선한다.
- 큰 GEMM은 latency와 achieved FLOP/s를 함께 보고 Nsight Compute에서 SM utilization, DRAM/L2 utilization 및 read/write byte를 확인한다.
- Register 수, spill, shared-memory 사용량, achieved occupancy로 custom tile의 비용을 추적한다.
- Fused epilogue나 quantized decode가 있으면 pure GEMM이 아니라 호출 전후의 end-to-end interval을 잰다.
- Warm-up과 profiling loop를 분리하고 allocation과 copy를 측정 구간 밖으로 옮긴다. Cache-hot 결과가 production을 대표하지 않으면 여러 tensor buffer를 rotation한다. 이는 NVIDIA의 공식 GEMM measurement guideline에도 명시된 조건이다. [NVIDIA CUTLASS — GEMM Performance Measurement Methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)

### 마지막은 수치 검증

Input, compute, output dtype에 맞는 reference와 tolerance를 기록한다. Split-K, Stream-K, fused epilogue, 낮은 precision은 reduction order나 rounding을 바꿀 수 있다. Drop-in reproducibility가 필요하면 cuBLAS의 조건을 확인하고 toolkit, GPU architecture, algorithm, workspace configuration을 함께 고정한다. Byte-exact가 요구사항이 아니라면 그보다 의미 있는 error bound를 사용한다. [NVIDIA cuBLAS — Results Reproducibility](https://docs.nvidia.com/cuda/cublas/#results-reproducibility)

마지막으로 실제 cache residency, stream concurrency, CUDA Graph 사용 여부, request mix에서 후보를 다시 실행한다. Microbenchmark의 이득이 이 환경에서도 남아 있어야 library보다 낮은 계층으로 내려갈 가치가 있다.

---

## References

1. NVIDIA, *cuBLAS 13.3 Documentation*: [https://docs.nvidia.com/cuda/cublas/](https://docs.nvidia.com/cuda/cublas/)
2. NVIDIA, *cuBLASLt API*: [https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
3. NVIDIA, *CUTLASS 4.6.1 Documentation*: [https://docs.nvidia.com/cutlass/latest/overview.html](https://docs.nvidia.com/cutlass/latest/overview.html)
4. NVIDIA, *Efficient GEMM in CUDA*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
5. NVIDIA, *CUTLASS GEMM API*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)
6. NVIDIA, *CUTLASS Profiler*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
7. NVIDIA, *GEMM Performance Measurement Methodology Guidelines*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
8. NVIDIA, *PTX ISA — WGMMA*: [https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma)
9. NVIDIA, *tcgen05 MMA Programming Guide*: [https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
10. NVIDIA, *Grouped Kernel Schedulers*: [https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
11. NVIDIA, *Convolutional Layers User's Guide*: [https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)
