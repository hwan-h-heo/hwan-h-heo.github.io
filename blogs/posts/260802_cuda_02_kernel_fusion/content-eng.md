## Abstract

Splitting a CUDA computation across several kernels leaves values at the boundaries. If a later kernel needs one of those values, the producer must store the intermediate in global memory. The first cost that kernel fusion removes is not the kernel count itself, but the memory traffic caused by this materialization.

Once the intermediate is no longer stored, something else must retain it. A computation that finishes within one thread may need only a few registers. Sharing the value across threads introduces shuffles, shared memory, and barriers. Extending the scope across blocks brings back workspaces, atomics, or a separate reduction kernel.

This ownership boundary determines how far fusion can extend. First estimate the bytes that materialization costs. Then decide whether the intermediate belongs in a thread, warp, block, or GEMM tile, and measure the resources consumed by that choice. The best fused kernel is not the largest one. It fuses only while the new cost remains smaller than the cost it removes.

---

## 1. Materialization Cost

### Intermediates at Kernel Boundaries

In an operation graph, a producer creates a value and a consumer uses it. If they execute in separate kernels, the intermediate must outlive the producer. Materialization is the act of storing that value as a tensor in global memory.

![Unfused Nsight Systems trace with abs and sum running as separate kernels](./assets/nvidia-unfused-trace.webp)

*`abs` writes an output-sized intermediate, which `sum` then reads. The trace shows the kernel intervals but not the internal stages of the reduction. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 1](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

![Fused Nsight Systems trace with abs and sum combined into one kernel](./assets/nvidia-fused-trace.webp)

*The reduction computes `abs` internally and removes the per-element intermediate. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 2](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

```text
unfused: producer → global-memory materialization → consumer
fused:   producer → register / shared memory → consumer
```

Combining a producer and consumer in this way is vertical fusion. Horizontal fusion instead places small, independent operations in one kernel. It can share dispatch costs or input reads, but it is not the same as removing a producer–consumer intermediate.

A separate consumer may read an intermediate from L2 rather than HBM. Cache residency, however, depends on capacity, eviction, and traversal order. It is not equivalent to retaining a value in registers or shared memory within one kernel. Calculate logical traffic analytically, then inspect physical traffic with a profiler.

### Byte Budget

Suppose `x`, `scale`, and `shift` are all $N$-element tensors and each expression launches a separate kernel.

```python
u = x * scale
v = u + shift
out = relu(v)
```

If each element occupies $b$ bytes, let the tensor size be $S=Nb$. Ignoring cache reuse, the unfused path performs eight tensor-sized reads or writes.

$$
T_{\mathrm{unfused}} \approx 8S.
$$

After fusion, `u` and `v` become transient values.

```cpp
float value = x[i] * scale[i] + shift[i];
out[i] = fmaxf(value, 0.0f);
```

$$
T_{\mathrm{fused}} \approx 4S.
$$

Only three input reads and the final store remain. The $8S\rightarrow4S$ estimate excludes cache reuse. If `scale` or `shift` is a scalar, a short broadcast vector, or cache-resident data, actual traffic will be smaller. Strides and write allocation can increase it.

This calculation does not predict speedup. It only bounds the logical traffic that can be removed. If the byte count is small, or the profiler shows that caches already absorb most of it, fusion has little room to help.

### Launch Overhead and CUDA Graphs

When CPU submission and launch overhead dominate, a CUDA Graph may be the more direct solution. It prepares the workflow as an executable graph and reduces launch setup cost.

The operations still remain separate kernel nodes in a CUDA Graph, and their materialized intermediates do not disappear. Fusion removes data boundaries; a CUDA Graph reduces work-submission overhead. Both can be applied when both costs matter. This distinction follows the [CUDA Graph model in the CUDA 13.3 Programming Guide](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html).

The [VARCO3D 2.0 optimization](/blogs/posts/optimizing-sparse-3d-generation-inference/) exposed the same difference. Replacing `aten::gelu` with a standalone custom `gelu_tanh` kernel was slower than the eager path because the boundary that stored the GEMM output and then had GELU read and write it remained intact.

Moving bias and GELU into a cuBLASLt epilogue removed this round trip and replaced the standalone call with `_addmm_activation`. The custom GELU changed the consumer implementation; epilogue fusion removed the materialization between GEMM and GELU. The result was validated against a tolerance rather than treated as bitwise exact.

---

## 2. Intermediate Ownership

Removing a kernel boundary does not remove the value itself. Some execution unit must retain it until the consumer finishes. The scope of that ownership determines both the difficulty and the cost of fusion.

| Ownership scope | Representative operation | Where the intermediate lives | New cost |
| --- | --- | --- | --- |
| One thread | Elementwise chain | Registers | Live registers and instructions |
| Warp or block | Row reduction, normalization | Registers, shuffles, shared memory | Barriers and on-chip reduction |
| GEMM tile | Prologue, epilogue | Accumulators, operand tiles | Layout conversion and mainloop resource pressure |
| Multiple blocks | Large reduction, conflicting scatter | Partial workspace or global output | Atomics, another kernel, unspecified accumulation order |

### Thread-Local Ownership

In the following elementwise expression, one thread can process index $i$ from beginning to end.

$$
out_i=\operatorname{clamp}(x_i s_i+t_i,\ell,u)
$$

No other thread needs to observe the values between `x_i s_i`, `+t_i`, and `clamp`. If shapes, strides, aliasing, and side effects permit, the thread can keep them in registers. This is also a straightforward case for automatic compiler fusion.

[PyTorch Inductor](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html) primarily emits Triton kernels for such graphs on NVIDIA GPUs. The graph compiler chooses the boundary, while Triton provides the kernel language and compiler. [XLA](https://openxla.org/xla/architecture) and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) make similar decisions during their compilation stages.

A one-line Python expression does not guarantee one kernel. Inspect the generated graph or kernel trace.

### Warp and Block Ownership

LayerNorm, RMSNorm, and Softmax reduce many elements of a row into a statistic. One thread cannot own the complete result, so partial states must be combined with warp shuffles or shared memory. If the full row fits within a warp or block, the same kernel can finish both the reduction and normalization.

The Welford state for mean and variance can be represented as $(n,\mu,M_2)$. For two nonempty partial states $A$ and $B$, let $\delta=\mu_B-\mu_A$ and combine them as

$$
n=n_A+n_B,
\qquad
\mu=\mu_A+\delta\frac{n_B}{n},
$$

$$
M_2=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n}.
$$

Warps and blocks exchange only this small state, so no per-element intermediate must be written to global memory. If one block cannot own the entire row, however, the kernel may need to reread the input or leave a partial-result workspace. Atomics or another kernel may also be required. The combine rule follows the [parallel variance analysis by Chan, Golub, and LeVeque](https://doi.org/10.1080/00031305.1983.10483115).

FlashAttention fits the same model. Each score tile flows directly into online softmax and multiplication by $V$, while only the row state is carried forward. The full score and probability matrices disappear, but the tile and state that a CTA can hold become the unit of ownership.

### GEMM Tile Ownership

A linear layer often applies bias and activation after GEMM. Running all three operations as separate kernels leaves output-sized tensors between the stages. Epilogue fusion applies bias and activation before storing the GEMM result in global memory.

![CUTLASS GEMM hierarchy with an epilogue tile and epilogue functor](./assets/cutlass-gemm-epilogue.webp)

*Classic CUTLASS GEMM hierarchy. Per-thread accumulator fragments pass through an epilogue tile and functor before being stored in global memory. This figure describes the register-accumulator path in the cited CUTLASS model; the Blackwell SM100 path is distinguished below. Source: [NVIDIA CUTLASS — Efficient GEMM in CUDA, Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue)*

For the computation

$$
Z=\alpha AB+\beta C,
\qquad
D=\operatorname{GELU}(Z+b),
$$

let $S$ be the byte size of the $M\times N$ output. Exclude A, B, optional C, and the smaller broadcast bias, which both paths read in common. Running GEMM, bias, and activation separately produces approximately

$$
T_{\mathrm{unfused}}\approx5S
$$

of output and intermediate traffic. Over the same accounting scope, a fused epilogue can leave only the final store and reduce the total to $T_{\mathrm{fused}}\approx S$. During training, the kernel may return both the activation and pre-activation.

$$
D=\operatorname{GELU}(Z),
\qquad
\mathrm{Aux}=Z.
$$

Even when `Aux` remains for backward, the kernel boundary between GEMM and the activation is gone.

Accumulator placement depends on the architecture. CUTLASS paths commonly used through Hopper often pass MMA results to the epilogue as register fragments. The Blackwell SM100 TCGen05 path stores accumulators in TMEM; its epilogue loads a TMEM subtile into registers, applies the fused operations, and stores it through shared memory and TMA. Both paths transform the value before materializing the global output. The official [CUTLASS description of the SM100 epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#background-sm100-epilogue-flow) gives the detailed flow.

A prologue addresses the opposite GEMM boundary. For quantized weights,

$$
W=\operatorname{dequant}(W_q;s,z),
$$

the kernel can unpack, scale, and transform the layout while loading each operand tile rather than creating a dense temporary weight. The temporary disappears, but instruction count, register use, and pressure on operand delivery increase.

### Multi-Block and Irregular Ownership

Once ownership crosses a block boundary, fusion alone cannot easily contain all intermediate state. A large reduction must store block-level partial results and combine them later. A scatter in which several blocks update the same address requires atomics or a separate merge stage.

Consider a token-wise affine transform that selects a parameter row using `batch_ids`.

```python
scale_token = scale[batch_ids]
shift_token = shift[batch_ids]
out = x * scale_token + shift_token
```

In eager mode, two `[T,C]` gather results may be created before the affine operation. A compiler, DSL kernel, or fused custom operator can remove them by reading the selected parameter values while processing each token. Once indices are involved, however, read locality and write ownership depend on the data.

Before implementing the fusion, check whether channels are contiguous, nearby tokens reuse the same key, and sorting or binning costs can be shared with downstream stages. For a scatter, also ask whether equal keys can be reduced within a warp or block before reaching global atomics.

```cpp
atomicAdd(output + index[i], value[i]);
```

When updates converge on a hot destination, atomics serialize and the accumulation order becomes data-dependent. Warp aggregation, block-local partials, and spatial binning reduce global atomics at the cost of extra work or another materialization. Irregular gather/scatter does not automatically require custom CUDA. Compare graph compilers, GPU DSLs such as Triton, library primitives, and custom operators against the actual representation and conflict distribution.

---

## 3. Cost of Wider Ownership

### Live State and Synchronization

Fusion removes an intermediate by extending its live range. Within a thread, registers remain live for longer. Across a warp or block, shuffles, shared memory, and barriers appear. Once several blocks participate, at least one of a partial workspace, atomics, or another kernel is likely to return.

Higher register pressure can lower occupancy or cause local-memory spills. More shared memory reduces the number of resident blocks, and barriers hold consumers behind slower producers. Occupancy is not the objective by itself, but too few active warps make memory and instruction latency harder to hide.

Asynchronous copies and multistage pipelines can overlap data movement with computation and hide part of the latency. They also require buffers and pipeline state. Pipelining may conceal some added cost; it does not guarantee that fusion is profitable.

### Layout and the Tuned Mainloop

The producer and consumer do not necessarily prefer the same layout or tile shape. Removing one transform can turn contiguous access into strided access, making the kernel slower even as logical traffic falls. Measure consumer-oriented packing or a shared-memory transpose over the full path, including upstream conversion, tails, and alignment.

The performance of an existing tuned library kernel is even easier to overlook.

```text
fast library GEMM + small separate kernel
versus
custom fused GEMM with a slower mainloop
```

An epilogue supported by cuBLASLt or CUTLASS can preserve a tuned mainloop while removing the boundary. A broader custom fusion is worthwhile only when the reduction in traffic and launch cost exceeds the mainloop slowdown, workspace, dispatch, and backward-recomputation costs.

### Numerical Contract

Fusion can also remove an intermediate rounding point. An unfused BF16 GEMM rounds its FP32 accumulator when storing it as BF16. A fused epilogue can retain higher precision until the final store. FMA contraction and changes to the reduction tree can also alter finite-precision results.

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\not\equiv
\operatorname{fl}(a+\operatorname{fl}(b+c)).
$$

Define the required numerical contract before benchmarking.

| Contract | Requirement |
| --- | --- |
| Mathematical equivalence | The same operation in real arithmetic |
| Tolerance-based equivalence | A result within the stated absolute/relative error bounds |
| Determinism | The same result across repeated runs of the same implementation under the stated conditions |
| Byte-exact reference match | The same bits as the specified reference path |

These are different requirements, and satisfying one does not automatically satisfy the others. A result can be deterministic yet differ from the reference. It may also remain within tolerance while its low bits vary across runs. The chosen contract determines the permitted intermediate rounding, Tensor Core mode, FMA, reduction reassociation, and atomic order.

---

## 4. Fusion Boundaries

### Implementation Layers

Fusion does not have to begin with handwritten CUDA. Start with the highest-level mechanism that can express the boundary to be removed.

| Control surface | Useful when |
| --- | --- |
| Framework/compiler fusion | The graph and regular layout expose the complete boundary |
| CUDA Graph | Materialization may remain and launch/submission overhead dominates |
| CCCL algorithm: Thrust or CUB | A transform, reduce, scan, sort, or reduce-by-key matches a maintained primitive |
| cuBLASLt or a vendor fused operator | The operation after GEMM fits a supported epilogue and layout |
| CUTLASS / CuTe DSL / GPU kernel DSL | The tile dataflow or prologue/epilogue needs finer control |
| Custom CUDA operator | Ownership, indexing, atomics, or integration cannot be expressed effectively at a higher level |

CCCL includes Thrust and CUB. Thrust provides high-level parallel algorithms, while CUB provides device-, block-, and warp-level primitives such as `DeviceReduce`. Composing several primitives does not guarantee single-kernel fusion, and temporary storage and execution stages still count as costs. See the official [CCCL/CUB `DeviceReduce` documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html) for its scope.

### Validation Metrics

Compare the baseline and candidate over the same end-to-end region, shape distribution, dtype, and warm-up conditions. Each measurement should correspond to the cost that fusion is meant to reduce.

| Claim | What to inspect |
| --- | --- |
| The boundary disappeared | Kernel trace and launch count, absence of the target allocation, and expected logical intermediate write/read bytes |
| Physical traffic fell | Device-memory and L2 bytes/throughput and cache behavior in Nsight Compute **Memory Workload Analysis** |
| Resource costs did not erase the gain | Registers per thread, static/dynamic shared memory, theoretical/achieved occupancy, and local-memory traffic or spill instructions in **Launch Statistics** and **Occupancy** |
| The main computation did not slow down | Where applicable, GEMM/mainloop throughput, total kernel time, and end-to-end latency including casts, reorder, workspace, allocation, and dispatch |
| Correctness was preserved | Representative shapes, tail-heavy shapes, and inputs with concentrated conflicts under the selected numerical contract |

The section names in the table follow the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules). Raw metric identifiers vary by architecture and tool version, so they are presented at the section level.

Reject the candidate or narrow the scope of fusion if any of the following occurs:

- The target intermediate is still written and read.
- End-to-end improvement is within run-to-run noise, or an important shape becomes slower.
- Spills, shared-memory usage, occupancy loss, or a slower library mainloop consume the byte savings.
- Preprocessing, workspace, or backward recomputation merely moves the cost outside the measured region.
- The implementation violates the selected numerical contract.

The fact that operations can fit in one kernel is not evidence that the boundary should disappear. The full path must remain faster after accounting for the state and synchronization introduced by wider ownership.

---

## Fusion Criteria

Kernel fusion changes where intermediates live and how long they survive. First count the bytes spent on materialization, then identify the execution unit that can own each value until the consumer finishes. A thread-local chain can end in registers. A reduction needs cooperation across a warp or block. A GEMM prologue or epilogue consumes values inside a tile but must preserve the fast mainloop. Workspaces and atomics return when ownership spans blocks.

Only then should the implementation layer be chosen. Prefer a compiler or library when it can remove the boundary. A custom kernel becomes necessary when a higher layer cannot express the required ownership and dataflow. If a tuned library followed by a small kernel is faster end to end, keep the boundary. Judge fusion by the materialization that actually disappeared and the cost paid to remove it, not by the kernel count.

---

## References

The architecture-specific discussion was checked against the CUDA 13.3 Programming Guide and CUTLASS 4.6.1 as of August 2026. Recheck instruction paths and profiler details when targeting a different toolkit or GPU generation.

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
