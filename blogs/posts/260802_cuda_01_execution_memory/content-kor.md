## Abstract

CUDA 최적화는 thread를 많이 만드는 문제가 아니다. GPU가 어떤 단위로 instruction을 발행하고, 실행이 멈춘 동안 무엇으로 latency를 숨기며, 데이터가 HBM과 on-chip memory 사이를 어떻게 이동하는지를 함께 봐야 한다.

실제 성능을 좌우하는 질문은 대체로 다음과 같다.

- 한 warp의 lane들이 같은 control flow를 따르는가?
- 실행 가능한 warp가 충분해 memory latency를 숨길 수 있는가?
- global memory access가 coalesced되어 있는가?
- 같은 데이터를 register나 shared memory에서 재사용하는가?
- register와 shared memory 사용량이 SM residency를 과도하게 제한하지 않는가?

이 글은 CUDA kernel, thread, warp, block, SM, occupancy와 memory hierarchy를 하나의 실행 모델로 연결한다. 이후 다룰 GEMM, kernel fusion, FlashAttention의 기반이 되는 내용이다.

---

## 1. Execution Model

![CUDA grid와 thread block 구조](./assets/nvidia-grid-of-thread-blocks.webp)

*Source: [CUDA Programming Guide — Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)*

CUDA 프로그램은 CPU에서 실행되는 **host code**와 GPU에서 실행되는 **device code**로 나뉜다. GPU에서 병렬 실행되도록 선언한 device 함수가 **CUDA kernel**이다.

가장 단순한 vector addition kernel은 다음과 같다.

```cpp
__global__ void add(
    const float* x,
    const float* y,
    float* z,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        z[i] = x[i] + y[i];
    }
}
```

Kernel launch에는 일반 함수 호출과 달리 execution configuration이 붙는다.

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;

add<<<blocks, threads>>>(x, y, z, n);
```

`<<<blocks, threads>>>`는 하나의 **grid**와 그 안에 포함될 **thread block**의 수와 크기를 지정한다.

```text
Kernel launch
  └─ Grid
      ├─ Thread Block 0
      │   ├─ Thread 0
      │   ├─ Thread 1
      │   └─ ...
      ├─ Thread Block 1
      └─ ...
```

각 thread는 같은 kernel code를 실행하지만 `blockIdx`, `blockDim`, `threadIdx`를 통해 서로 다른 data index를 계산한다.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
z[i] = x[i] + y[i];
```

Thread 하나가 원소 하나를 처리하는 방식은 가장 단순한 mapping일 뿐 규칙은 아니다. 하나의 thread가 여러 원소를 순회할 수도 있고, 여러 thread가 하나의 output을 협력해 계산할 수도 있다.

**Thread Block.**

Block은 thread들이 협력할 수 있는 기본 범위다. 같은 block의 thread들은 다음 자원을 공유한다.

- shared memory
- block-level barrier인 `__syncthreads()`
- cooperative loading, reduction, scan

한 block은 실행되는 동안 하나의 SM에 resident한다. Block에 필요한 register와 shared memory도 해당 SM에서 할당되며, 이 자원 사용량이 동시에 resident할 수 있는 block 수를 제한한다.

**Grid.**

Grid는 한 번의 kernel launch가 생성한 전체 block 집합이다. 일반적인 kernel에서는 block 간 실행 순서를 가정할 수 없다. Block 0이 block 1보다 먼저 시작하거나 먼저 끝난다는 보장도 없다.

Block 사이에 global synchronization이 필요하다면 보통 다음 중 하나를 선택한다.

- kernel을 분리해 kernel boundary를 global barrier로 사용한다.
- cooperative launch를 사용한다.
- global atomic과 별도의 synchronization protocol을 설계한다.

대부분의 경우 kernel을 분리하는 방식이 가장 단순하고 검증하기 쉽다.

**Asynchronous Launch.**

Kernel launch는 기본적으로 host에 대해 asynchronous하다. CPU는 작업을 CUDA stream에 제출한 뒤 다음 host code를 계속 실행할 수 있다.

다만 다음 연산은 host가 GPU 작업의 완료를 기다리게 만들 수 있다.

- 명시적인 `cudaDeviceSynchronize()` 또는 stream synchronization
- synchronous host-to-device / device-to-host copy
- GPU 결과를 CPU scalar로 읽는 연산
- 동일 stream에서 선행 작업의 결과에 의존하는 후속 작업

Kernel 자체만이 아니라 launch 사이의 host synchronization과 stream dependency도 전체 latency에 포함된다.

---

## 2. Warp Execution

CUDA thread는 hardware에서 완전히 독립적으로 하나씩 실행되지 않는다. NVIDIA GPU는 32개 thread를 **warp**로 묶어 instruction을 발행한다.[1]

```text
Warp
 ├─ lane 0
 ├─ lane 1
 ├─ ...
 └─ lane 31
```

각 thread는 warp 안에서 lane ID를 가진다. Warp scheduler는 실행 가능한 warp를 선택하고, 해당 warp의 active lane에 instruction을 발행한다.

CUDA의 실행 모델을 **SIMT**, 즉 Single Instruction, Multiple Threads라고 부르는 이유도 여기에 있다. 같은 instruction을 실행하더라도 각 lane은 서로 다른 register 값과 memory address를 가질 수 있다.

```cpp
out[i] = x[i] + y[i];
```

```text
lane 0  -> out[0]  = x[0]  + y[0]
lane 1  -> out[1]  = x[1]  + y[1]
...
lane 31 -> out[31] = x[31] + y[31]
```

이 구조는 동일한 연산을 대량의 데이터에 적용하는 workload에서 높은 throughput을 낸다. 반대로 lane마다 control flow나 작업량이 크게 달라지면 warp의 실행 효율이 떨어진다.

**Warp Divergence.**

한 warp의 lane들이 서로 다른 branch를 선택하면 **warp divergence**가 발생한다.

```cpp
if (threadIdx.x % 2 == 0) {
    path_a();
} else {
    path_b();
}
```

개념적으로는 각 branch를 active mask를 바꾸어 실행한다.

```text
path_a
  even lanes: active
  odd lanes:  inactive

path_b
  even lanes: inactive
  odd lanes:  active
```

두 경로가 전체 lane에서 동시에 처리되는 것이 아니다. 서로 다른 경로를 순차적으로 수행하는 동안 해당 경로에 속하지 않는 lane은 inactive 상태가 된다. Branch가 길고 lane별 선택이 불규칙할수록 유효 throughput이 낮아진다.

그러나 `if`가 존재한다는 사실만으로 divergence가 생기지는 않는다.

```cpp
if (blockIdx.x < num_regular_blocks) {
    regular_path();
} else {
    tail_path();
}
```

같은 warp의 모든 lane이 동일한 조건을 선택하면 control flow는 uniform하다. 짧은 조건문은 compiler가 predication으로 바꾸기도 한다.

따라서 확인해야 할 것은 branch의 유무가 아니라 다음이다.

> 같은 warp의 lane들이 서로 다른 경로를 얼마나 자주 선택하며, 그 경로가 얼마나 오래 지속되는가?

Mesh processing, BVH traversal, sparse voxel, ray traversal처럼 입력 구조에 따라 loop count와 branch가 달라지는 kernel은 divergence에 특히 민감하다.

```cpp
while (node != nullptr) {
    if (intersects(node)) {
        node = node->child;
    } else {
        node = node->next;
    }
}
```

각 lane이 서로 다른 node를 방문하면 warp는 가장 오래 남아 있는 lane에 의해 진행 속도가 제한될 수 있다. 일반적인 완화 방법은 다음과 같다.

- spatially coherent한 query를 같은 warp에 배치한다.
- 비슷한 작업량의 primitive를 정렬하거나 bucketize한다.
- traversal을 persistent work queue로 재구성한다.
- common path와 exceptional path를 별도 kernel로 분리한다.
- warp-level compaction으로 active work를 다시 묶는다.

재정렬과 compaction에도 비용이 있으므로, divergence 감소가 실제 kernel duration 감소로 이어지는지는 profile로 확인해야 한다.

---

## 3. SM Scheduling

GPU의 실제 실행 자원은 **SM**, Streaming Multiprocessor다. 세부 구성은 GPU architecture마다 다르지만, 개념적으로 다음 자원을 포함한다.

```text
SM
 ├─ Warp schedulers
 ├─ CUDA cores
 ├─ Tensor cores
 ├─ Load/store units
 ├─ Special-function units
 ├─ Register file
 └─ Shared memory / L1
```

한 warp가 global memory load를 발행한 뒤 결과를 기다린다고 하자. GPU는 해당 warp만 붙잡고 기다리는 대신, 같은 SM에 resident한 다른 ready warp의 instruction을 발행한다.

```text
Warp 0: global load → wait
Warp 1: arithmetic
Warp 2: shared-memory load
Warp 3: Tensor Core instruction
Warp 0: data ready → resume
```

이 방식이 **latency hiding**이다. GPU가 높은 throughput을 얻으려면 두 수준의 병렬성이 필요하다.

- **Thread-level parallelism**: 충분한 block과 warp로 전체 SM을 채운다.
- **Instruction-level parallelism**: 한 thread나 warp 안에 독립적으로 발행할 instruction을 준비한다.

Thread 수가 많더라도 dependency chain이 길거나 memory access가 불규칙하면 ready warp가 부족해질 수 있다. 반대로 resident warp 수가 적어도 software pipeline과 instruction-level parallelism이 충분하면 높은 throughput을 유지할 수 있다.

**Occupancy.**

Occupancy는 SM에 실제로 resident한 active warp 수를 architecture가 허용하는 최대 active warp 수로 나눈 값이다.

$$
\text{occupancy}
=
\frac{\text{resident active warps per SM}}
{\text{maximum active warps per SM}}
$$

Occupancy는 주로 다음 자원에 의해 제한된다.

- registers per thread
- shared memory per block
- threads per block
- maximum blocks per SM
- maximum warps per SM

예를 들어 thread당 register 사용량이 증가하면 block 전체의 register allocation도 커진다. 그 결과 한 SM에 동시에 resident할 수 있는 block과 warp 수가 줄어들 수 있다.

하지만 높은 occupancy가 높은 성능을 보장하지는 않는다.[2] GEMM kernel은 많은 accumulator를 register에 유지하므로 occupancy가 낮을 수 있지만, 다음 조건을 만족하면 충분히 빠르다.

- Tensor Core utilization이 높다.
- shared memory와 register의 data reuse가 충분하다.
- software pipeline이 memory latency를 숨긴다.
- 독립적인 MMA instruction이 충분하다.

Occupancy를 높이기 위해 register 수를 무리하게 줄이면 accumulator가 local memory로 spill되어 오히려 느려질 수 있다.

Occupancy는 최적화 목표라기보다 진단 지표로 보는 편이 정확하다.

- 낮은 occupancy와 낮은 compute throughput이 함께 나타나면 parallelism 부족을 의심할 수 있다.
- 낮은 occupancy에서도 compute throughput이 높다면 문제가 아닐 수 있다.
- 높은 occupancy에서도 memory dependency 때문에 warp가 자주 stall할 수 있다.
- occupancy를 높인 뒤 register spill이 생기면 전체 성능은 악화될 수 있다.

따라서 occupancy는 다음 지표와 함께 봐야 한다.

- achieved memory bandwidth
- SM / Tensor Core utilization
- eligible warps per cycle
- warp stall reasons
- local load/store와 register spill
- absolute kernel duration

---

## 4. Memory Hierarchy

![CUDA device의 memory spaces와 접근 범위](./assets/nvidia-memory-spaces.webp)

*Source: [CUDA C++ Best Practices Guide — Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#device-memory-spaces)*

GPU memory는 대략 다음 계층으로 볼 수 있다.

```text
fast / small
────────────────────────
Register
Shared memory / L1
L2 cache
Global memory / HBM
Host memory
────────────────────────
slow / large
```

각 계층은 latency와 capacity뿐 아니라 scope, visibility, 관리 방식이 다르다.

| Memory | Scope | Managed by | Typical use |
|---|---|---|---|
| Register | thread | compiler | accumulator, address, temporary value |
| Shared memory | block | programmer | tile, reduction, transpose |
| L1 cache | SM | hardware | local reuse of global-memory data |
| L2 cache | device | hardware | cache shared across SMs |
| Global memory | device | allocation/API | tensors and large buffers |
| Local memory | logical thread scope | compiler | register spill, large local array |
| Constant memory | grid-wide read-only | programmer | small broadcast constants |
| Texture / read-only path | read-only workload | API/compiler | spatial locality, specialized addressing |

**Register.**

Register는 thread의 scalar, address, accumulator, 작은 fragment를 보관하는 가장 빠른 storage다.

```cpp
float acc = 0.0f;
float a = ...;
float b = ...;
acc += a * b;
```

`acc`, `a`, `b`는 일반적으로 register에 배치된다. Register는 programmer가 pointer로 직접 접근하는 memory가 아니라 compiler가 변수에 할당하는 제한된 SM resource다.

**Shared Memory.**

Shared memory는 같은 block의 thread가 공유하는 programmer-managed on-chip memory다.

```cpp
__shared__ float tile[32][33];
```

Programmer가 global memory의 데이터를 shared memory에 staging하고, 필요한 경우 barrier로 producer와 consumer를 동기화한다. Shared memory는 cache 역할뿐 아니라 transpose, reduction, data exchange를 위한 scratchpad 역할도 한다.

**L1 and L2.**

Global-memory access는 hardware cache를 통과할 수 있다. L1은 SM에 가깝고, L2는 device 전체 SM이 공유한다.

Cache는 반복 접근을 줄여 주지만 performance-critical kernel을 단순히 “cache가 처리할 것”이라고 가정해서는 안 된다. Working set이 크거나 access pattern이 streaming이면 cache hit rate가 낮을 수 있고, 여러 SM과 kernel이 같은 cache capacity를 경쟁한다.

**Global Memory and HBM.**

PyTorch tensor와 일반적인 CUDA allocation의 본체는 global memory에 있다. 용량과 bandwidth는 크지만 on-chip memory보다 latency가 높다.

현대 GPU에서는 Tensor Core throughput이 HBM bandwidth보다 더 빠르게 증가해 왔다. 따라서 같은 데이터를 HBM에서 반복해서 읽는 kernel은 arithmetic unit을 충분히 활용하기 어렵다. 고성능 kernel은 가능한 한 HBM traffic을 줄이고, 한 번 가져온 데이터를 shared memory와 register에서 반복 사용한다.

**Local Memory.**

`local memory`는 이름과 달리 빠른 on-chip storage가 아니다. Thread마다 논리적으로 private하지만 물리적으로는 device memory에 위치한다.

Local memory traffic은 주로 다음 상황에서 증가한다.

- register spill
- compile time에 index를 결정하기 어려운 큰 local array
- 과도하게 큰 thread-local state

Nsight Compute에서 local load/store가 눈에 띄면 register allocation과 spill부터 확인하는 편이 좋다.

---

## 5. Memory Access

![Warp의 adjacent thread가 연속된 global memory word에 접근하는 coalesced access](./assets/nvidia-coalesced-access.webp)

*Source: [CUDA C++ Best Practices Guide — Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)*

Global-memory optimization의 첫 번째 원칙은 한 warp의 요청을 가능한 적은 **memory transaction**으로 처리하는 것이다. 이를 **coalescing**이라고 한다.[3]

**Coalescing.**

다음 접근에서는 warp의 lane들이 연속된 FP32 값을 읽는다.

```cpp
float v = x[base + threadIdx.x];
```

```text
lane 0  -> x[0]
lane 1  -> x[1]
...
lane 31 -> x[31]
```

Warp 전체가 사용하는 데이터는 128 bytes다. 주소가 적절히 aligned되어 있다면 hardware는 이 요청을 소수의 연속 memory sector로 처리할 수 있다.

반면 stride가 크면 각 lane이 서로 다른 sector나 cache line을 건드린다.

```cpp
float v = x[threadIdx.x * stride];
```

```text
lane 0  -> x[0]
lane 1  -> x[1024]
lane 2  -> x[2048]
...
```

실제로 소비하는 값은 여전히 128 bytes지만, 이를 위해 훨씬 많은 bytes가 전송될 수 있다.

Coalescing은 단순히 index가 연속인지 확인하는 문제가 아니다. 다음 두 양을 비교해야 한다.

1. Warp가 실제로 소비한 bytes
2. 해당 요청을 처리하기 위해 memory subsystem이 전송한 bytes

$$
\text{memory efficiency}
=
\frac{\text{useful bytes}}
{\text{transferred bytes}}
$$

Efficiency가 낮으면 DRAM bandwidth가 peak에 도달하지 않았더라도 memory instruction 수나 transaction 수가 병목일 수 있다.

**Alignment and Vectorized Access.**

연속 접근도 시작 주소가 transaction boundary에서 어긋나면 추가 sector가 필요할 수 있다. CUDA allocation 자체는 충분한 alignment를 제공하지만 다음 경우에는 misalignment가 생길 수 있다.

- offset이 있는 tensor view
- 비정상적인 row stride
- custom packed format의 header 뒤에서 시작하는 payload
- 임의 offset을 더한 pointer arithmetic

한 thread가 여러 값을 한 instruction으로 읽는 vectorized load/store는 instruction 수와 address calculation을 줄일 수 있다.

```cpp
float4 v = reinterpret_cast<const float4*>(x)[i];
```

다만 pointer alignment와 tail 처리가 정확해야 한다. `float4`를 사용한다고 warp 전체의 access pattern이 자동으로 coalesced되는 것은 아니다.

**Data Layout.**

Memory layout은 warp가 같은 시점에 소비하는 field에 맞춰야 한다.

**Array of Structures**는 한 element의 속성을 함께 배치한다.

```cpp
struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

Vertex vertices[N];
```

Position, normal, UV를 항상 함께 읽는 kernel에는 자연스럽다. 그러나 position만 필요한 kernel도 전체 struct를 따라 이동하므로 불필요한 traffic이 생길 수 있다.

**Structure of Arrays**는 field별로 buffer를 분리한다.

```cpp
float3 positions[N];
float3 normals[N];
float2 uvs[N];
```

Position만 필요한 stage는 `positions`만 읽을 수 있다. 경우에 따라 component까지 분리할 수도 있다.

```cpp
float pos_x[N];
float pos_y[N];
float pos_z[N];
```

어느 layout이 항상 우월한 것은 아니다. 기준은 단순하다.

> Warp가 동시에 사용하는 데이터를 memory에서도 가깝게 배치한다.

3D pipeline은 stage마다 필요한 attribute가 다르므로, 하나의 범용 vertex struct보다 stage-specific compact buffer가 더 효율적인 경우가 많다.

---

## 6. On-Chip Reuse

Shared memory와 register의 핵심 역할은 HBM에서 가져온 데이터를 다시 사용해 **arithmetic intensity**를 높이는 것이다. 다만 두 자원 모두 SM capacity가 제한되어 있어 reuse와 residency 사이의 trade-off가 생긴다.

**Shared-Memory Tiling.**

Matrix multiplication에서 block이 A와 B의 tile을 shared memory에 올리면 같은 값을 여러 output 계산에 재사용할 수 있다.

```text
HBM: tile load once
        ↓
Shared memory: reuse across threads
        ↓
Register: accumulate outputs
```

Global-memory traffic은 줄고, 같은 bytes당 더 많은 arithmetic을 수행할 수 있다.

Shared memory는 layout conversion에도 유용하다. Matrix transpose를 global memory에서 직접 처리하면 read 또는 write 한쪽이 strided해지기 쉽다. Shared-memory tile을 staging buffer로 사용하면 양쪽 global access를 모두 coalesced하게 만들 수 있다.

```text
Global read:  row-major, coalesced
        ↓
Shared memory: transpose tile
        ↓
Global write: row-major, coalesced
```

**Bank Conflict.**

Shared memory는 여러 bank로 나뉘어 있어 warp의 lane들이 서로 다른 bank를 접근할 때 높은 throughput을 낸다. 서로 다른 주소가 같은 bank에 몰리면 **bank conflict**가 발생하며, 요청이 여러 단계로 serialize될 수 있다.

전형적인 transpose tile은 column access에서 문제가 생길 수 있다.

```cpp
__shared__ float tile[32][32];
```

Row stride 32가 bank mapping과 겹치면 column 방향 접근이 같은 bank에 집중된다. 흔한 해결책은 padding이다.

```cpp
__shared__ float tile[32][33];
```

논리적인 tile 크기는 그대로지만 physical row stride를 33으로 바꾸어 bank mapping을 분산한다.[3]

Global-memory coalescing과 shared-memory bank conflict는 서로 다른 문제다.

- **Coalescing**: L1/L2/HBM으로 향하는 warp transaction의 효율
- **Bank conflict**: shared memory 내부에서 lane access가 병렬 처리되는 효율

Tiled kernel은 두 조건을 모두 만족해야 한다.

**Register Pressure.**

Register는 가장 빠른 storage지만 SM마다 총량이 제한되어 있다. Thread당 register 수가 늘어나면 다음 trade-off가 생긴다.

```text
registers per thread ↑
        ↓
registers per block ↑
        ↓
resident blocks / warps ↓
        ↓
occupancy may decrease
```

반대로 register 수를 과도하게 줄이면 값이 local memory로 spill된다.

```text
insufficient registers
        ↓
local-memory spill
        ↓
additional load/store
        ↓
longer kernel duration
```

Register 최적화의 목표는 사용량 자체를 최소화하는 것이 아니다. 충분한 reuse와 instruction-level parallelism을 유지하면서 spill과 과도한 residency 감소를 피해야 한다.

Register pressure가 커지는 흔한 원인은 다음과 같다.

- 큰 local array
- aggressive loop unrolling
- thread당 많은 output tile
- 긴 live range
- 복잡한 address calculation
- fusion으로 늘어난 temporary value
- activation, quantization 등을 포함한 무거운 epilogue

개선 방향도 trade-off를 전제로 한다.

- variable live range를 줄인다.
- 큰 thread-local state를 shared memory로 옮길 수 있는지 검토한다.
- unroll factor를 줄인다.
- thread당 output tile을 줄인다.
- fusion boundary를 다시 나눈다.
- compiler register limit은 spill을 측정하면서 사용한다.

---

## 7. Communication

CUDA kernel의 비용은 arithmetic과 memory access만으로 결정되지 않는다. Thread 간 communication과 synchronization도 중요한 병목이 된다.

**Block Barrier.**

```cpp
__syncthreads();
```

`__syncthreads()`는 block의 participating thread가 barrier에 도달할 때까지 기다린다. Shared-memory tile을 cooperative하게 load한 뒤 소비하기 전에 주로 사용한다.

Barrier는 block 전체에서 일관된 control flow로 실행되어야 한다.

```cpp
if (threadIdx.x < 16) {
    __syncthreads();  // unsafe: not all threads participate
}
```

일부 thread만 barrier에 진입하면 deadlock 또는 undefined behavior가 발생할 수 있다. Barrier가 correctness에 필요하더라도 지나치게 자주 사용하면 ready warp를 줄이고 pipeline을 끊는다.

**Warp Shuffle.**

Warp 내부에서 register 값을 교환할 때는 shared memory 대신 shuffle instruction을 사용할 수 있다.

```cpp
float other = __shfl_down_sync(0xffffffff, value, offset);
```

Reduction, scan, broadcast에 유용하며 shared-memory write, barrier, read를 줄일 수 있다.

```text
register in one lane
        ↓ shuffle
register in another lane
```

다만 shuffle은 warp 내부 communication에만 사용할 수 있다. Warp를 넘어서는 exchange에는 shared memory나 다른 synchronization mechanism이 필요하다.

**Atomic Contention.**

여러 thread가 같은 address를 갱신해야 할 때 atomic operation을 사용한다.

```cpp
atomicAdd(counter, value);
```

Atomic은 correctness를 보장하지만 동일 address에 요청이 집중되면 serialization이 발생한다. 일반적인 해결책은 update를 계층적으로 aggregate하는 것이다.

```text
thread-local accumulation
        ↓
warp reduction
        ↓
block reduction
        ↓
one global atomic per block
```

Mesh rasterization이나 voxelization에서는 primitive를 spatial tile로 binning해 동일 voxel에 대한 contention을 줄이는 방식도 효과적이다.

---

## 8. Performance Model

Kernel optimization을 시작하기 전에 무엇이 한계인지 분류해야 한다. 일반적으로 병목은 **compute throughput**, **memory bandwidth**, **latency / issue efficiency** 중 하나에 가깝다.

**Compute-Bound.**

Compute-bound kernel은 arithmetic pipeline이나 Tensor Core throughput이 한계다. Arithmetic intensity와 compute utilization이 높고, memory bandwidth에는 상대적으로 여유가 있다.

대형 GEMM이 대표적이다. 이 경우에는 다음 변화가 중요하다.

- FLOP 자체를 줄인다.
- 더 효율적인 instruction path를 사용한다.
- Tensor Core tile과 data type을 최적화한다.
- pipeline dependency와 instruction issue를 개선한다.

**Memory-Bound.**

Memory-bound kernel은 HBM에서 데이터를 읽고 쓰는 속도가 한계다. 단순 elementwise operation, copy, activation처럼 arithmetic intensity가 낮은 연산이 여기에 가깝다.

이 경우 arithmetic instruction 몇 개를 줄이는 것보다 다음 변화가 더 효과적이다.

- intermediate tensor를 제거한다.
- kernel을 fuse한다.
- bytes per element를 줄인다.
- coalescing과 alignment를 개선한다.
- on-chip reuse를 늘린다.

**Latency- or Issue-Bound.**

Compute utilization과 DRAM throughput이 모두 낮은데 kernel이 느릴 수도 있다. 이 경우 peak FLOP나 bandwidth가 아니라 작업을 충분히 발행하지 못하는 것이 문제다.

- grid가 작아 전체 SM을 채우지 못한다.
- dependency chain이 길다.
- memory access가 불규칙하다.
- warp divergence가 크다.
- atomic contention이 심하다.
- 작은 kernel이 지나치게 많다.
- host launch gap이나 synchronization이 크다.

이 유형은 단순한 bandwidth 수치만 보면 원인을 놓치기 쉽다.

**Arithmetic Intensity and Roofline.**

Arithmetic intensity는 이동한 bytes당 수행한 연산량이다.

$$
I
=
\frac{\text{FLOPs}}
{\text{transferred bytes}}
$$

Roofline 관점에서 attainable performance는 compute peak와 memory bandwidth가 만드는 두 한계 중 작은 값에 제한된다.

$$
P_{\text{attainable}}
\leq
\min
\left(
P_{\text{compute peak}},
I \cdot B_{\text{memory}}
\right)
$$

여기서 `P_compute peak`는 최대 compute throughput, `B_memory`는 memory bandwidth다.

FP32 vector addition은 원소마다 두 번 load하고 한 번 store한다.

- `x`: 4-byte load
- `y`: 4-byte load
- `z`: 4-byte store
- addition: 1 FLOP

따라서 대략적인 arithmetic intensity는 다음과 같다.

$$
I
\approx
\frac{1\ \text{FLOP}}
{12\ \text{bytes}}
$$

매우 낮은 값이다. Add instruction을 조금 최적화하는 것보다 memory traffic을 줄이는 편이 중요하다.

Matrix multiplication은 같은 A와 B element를 여러 output에 재사용할 수 있다. Tiling이 잘 되어 있다면 HBM에서 한 번 가져온 데이터를 shared memory와 register에서 반복 사용하므로 arithmetic intensity가 높아지고 compute-bound에 가까워진다.

이 차이가 GEMM과 FlashAttention의 핵심이다. 두 연산 모두 단순히 FLOP를 빠르게 수행하는 것이 아니라, expensive data movement를 줄이고 on-chip reuse를 극대화한다.

---

## 9. Optimization Patterns

앞의 원리는 실제 kernel에서 몇 가지 반복되는 형태로 나타난다.

**Kernel Fusion.**

다음 elementwise pipeline을 각각 별도 kernel로 실행한다고 하자.

```python
u = x * scale
v = u + shift
out = relu(v)
```

별도 kernel에서는 intermediate tensor가 global memory에 materialize된다.

```text
Kernel 1: x, scale read → u write
Kernel 2: u, shift read → v write
Kernel 3: v read → out write
```

하나의 kernel로 fuse하면 intermediate traffic과 launch를 제거할 수 있다.

```cpp
float value = x[i] * scale[i] + shift[i];
out[i] = max(value, 0.0f);
```

```text
x, scale, shift read → out write
```

이 연산이 memory-bound라면 arithmetic 변화가 거의 없어도 큰 개선이 가능하다. 다만 fusion이 커질수록 register pressure, code size, scheduling complexity가 증가할 수 있으므로 무조건 많이 합치는 것이 정답은 아니다.

**Tiled Transpose.**

Naive matrix transpose는 read 또는 write 한쪽이 strided해지기 쉽다.

```cpp
out[col * height + row] = in[row * width + col];
```

Shared-memory tile을 사용하면 다음 순서로 양쪽 global access를 coalesced하게 만들 수 있다.

```text
1. Input row를 coalesced하게 read
2. Shared-memory tile에 store
3. Block barrier
4. Tile을 transpose된 index로 read
5. Output row에 coalesced하게 write
```

Bank conflict를 피하기 위해 tile에 padding을 둔다.

```cpp
__shared__ float tile[TILE][TILE + 1];
```

이 예제에는 CUDA memory optimization의 핵심 패턴이 모두 들어 있다.

- coalesced global access
- shared-memory staging
- block-level synchronization
- padding을 통한 bank-conflict avoidance

**Kernel Review.**

실제 kernel을 처음 볼 때는 다음 순서로 확인하면 원인을 좁히기 쉽다.

**Execution**

- Grid가 GPU 전체를 채울 만큼 큰가?
- Block size가 warp 단위와 잘 맞는가?
- Warp divergence와 thread별 workload imbalance가 큰가?
- Ready warp가 부족하거나 dependency chain이 긴가?

**Global Memory**

- Warp lane들이 인접한 address를 접근하는가?
- 불필요한 intermediate를 읽고 쓰는가?
- Misalignment나 큰 stride가 있는가?
- 같은 데이터를 HBM에서 반복해서 읽는가?

**Shared Memory**

- Block 내부 reuse가 shared-memory staging 비용보다 충분히 큰가?
- Bank conflict가 있는가?
- Barrier가 지나치게 많지 않은가?
- Shared-memory allocation이 residency를 제한하지 않는가?

**Register**

- Thread당 register 수가 과도한가?
- Local-memory spill이 발생하는가?
- 더 큰 register tile이 실제 reuse를 높이는가, occupancy만 낮추는가?

**Pipeline**

- 작은 kernel이 지나치게 많이 launch되는가?
- CPU synchronization이나 scalar readback이 끼어 있는가?
- Memory copy와 compute를 overlap할 수 있는가?
- Kernel fusion으로 intermediate traffic을 제거할 수 있는가?

---

## Summary

CUDA performance는 thread 수 하나로 설명되지 않는다.

1. Kernel launch는 grid를 만들고, grid는 여러 thread block으로 구성된다.
2. Block은 하나의 SM에 resident하며 shared memory와 block-level synchronization을 사용한다.
3. Thread는 warp 단위로 instruction을 발행하므로 lane별 control flow와 address pattern이 중요하다.
4. GPU는 ready warp를 교대로 실행해 memory와 pipeline latency를 숨긴다.
5. Occupancy는 latency hiding에 필요한 수단 중 하나이며, 그 자체가 최적화 목표는 아니다.
6. Global-memory traffic은 coalescing, alignment, compact layout으로 줄여야 한다.
7. Shared memory와 register는 HBM에서 가져온 데이터를 재사용해 arithmetic intensity를 높인다.
8. On-chip resource를 과도하게 사용하면 occupancy 감소, bank conflict, register spill이 발생할 수 있다.
9. 최적화 전에는 kernel이 compute-bound, memory-bound, latency-bound 중 어디에 가까운지 먼저 분류해야 한다.

다음 글에서는 이 원리들이 가장 정교하게 결합된 연산인 GEMM을 다룬다. Naive matrix multiplication이 shared-memory tiling, register blocking, Tensor Core, software pipeline을 거쳐 고성능 kernel로 바뀌는 과정을 살펴본다.

---

## References

1. NVIDIA, *CUDA Programming Guide*: https://docs.nvidia.com/cuda/cuda-programming-guide/
2. NVIDIA, *CUDA Best Practices Guide — Occupancy*: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
3. NVIDIA, *CUDA Best Practices Guide — Coalesced Access and Shared Memory*: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
4. NVIDIA, *CUDA Programming Guide — Asynchronous Data Copies*: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
