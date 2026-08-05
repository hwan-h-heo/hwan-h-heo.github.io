## Abstract

In CUDA, GPU operations execute as functions called kernels. If `x * scale`, `+ shift`, and `ReLU` are computed separately, three kernels must be launched, and each intermediate tensor must be stored in global memory so that the next kernel can read it.

Kernel fusion combines consecutive operations into a single kernel, reducing both the memory traffic required to store and reload these intermediate tensors and the number of kernel launches. It is especially effective for elementwise operations, where computation is light but memory access and launch costs are significant.

Combining as many operations as possible is not always the right answer. Keeping intermediate values inside a thread or block for longer can increase synchronization, register, and shared-memory usage. It may also require giving up an already well-tuned library kernel, making the main computation slower instead.

This note focuses on vertical fusion, which combines a producer with the consumer of its value. It examines elementwise chains, GEMM prologues and epilogues, reductions, and indexed workloads in that order. For each case, it considers both the bytes saved and the new costs introduced, then asks how far fusion should extend to produce an actual performance improvement.

---

## 1. Intermediates Between Kernels

### Materialization and Two Types of Fusion

In an operation graph, a producer creates a value and a consumer uses it. When they execute in separate kernels, the intermediate value must survive after the producer finishes. Materialization is the process of storing that value as a tensor in global memory.

![Unfused Nsight Systems trace with abs and sum running as separate kernels](./assets/nvidia-unfused-trace.webp)

*`abs` writes an output-sized intermediate, which `sum` then reads. The trace shows the kernel intervals but not the internal stages of the reduction. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 1](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

![Fused Nsight Systems trace with abs and sum combined into one kernel](./assets/nvidia-fused-trace.webp)

*The reduction computes `abs` internally, removing the per-element intermediate. Source: [NVIDIA Technical Blog — Kernel Fusion, Figure 2](https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/)*

```text
unfused: producer → global-memory materialization → consumer
fused:   producer → register / shared memory → consumer
```

The fused path above is vertical fusion. The consumer uses the value produced by the producer directly from registers or shared memory, so the intermediate tensor disappears.

Horizontal fusion combines small, independent operations in a single kernel. Running several reductions together can share dispatch costs or input reads, but it does not eliminate a producer–consumer intermediate.

A separate consumer may read an intermediate from L2 rather than HBM. Cache residency, however, depends on capacity, eviction, and traversal order. It is not the same as keeping an intermediate in registers or shared memory within a kernel. Logical traffic should be calculated analytically, while actual traffic should be checked with a profiler.

### Byte Budget

Consider a synthetic elementwise chain in which `x`, `scale`, and `shift` are all $N$-element tensors and each expression launches a separate kernel.

```python
u = x * scale
v = u + shift
out = relu(v)
```

Let $S=Nb$, where each of the $N$ elements occupies $b$ bytes. Ignoring cache reuse, the unfused path performs eight tensor-sized reads or writes.

$$
T_{\mathrm{unfused}} \approx 8S.
$$

The fused kernel keeps `u` and `v` as transient values.

```cpp
float value = x[i] * scale[i] + shift[i];
out[i] = fmaxf(value, 0.0f);
```

$$
T_{\mathrm{fused}} \approx 4S,
$$

leaving three input reads and one final store. The $8S\rightarrow4S$ estimate excludes cache reuse. If `scale` or `shift` is a scalar, a short broadcast vector, or cache-resident data, the actual traffic will be smaller. Conversely, strides and write allocation can increase it.

The byte budget is useful for finding logical traffic that could be eliminated. It does not predict the actual speedup.

### How CUDA Graphs Differ

If CPU submission and launch overhead are the bottleneck, first consider a CUDA Graph, which prepares the workflow as an executable graph in advance and reduces launch setup costs.

In a CUDA Graph, the operations remain separate kernel nodes, and the materialized intermediates between them do not disappear. Fusion reduces data boundaries, whereas a CUDA Graph reduces work-submission costs, so the two techniques can also be applied together. This distinction follows the [CUDA Graph model in the CUDA 13.3 Programming Guide](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/04-special-topics/cuda-graphs.html).

The [VARCO3D 2.0 optimization](/blogs/posts/optimizing-sparse-3d-generation-inference/) also showed the difference between these approaches. Replacing `aten::gelu` with a standalone custom `gelu_tanh` kernel was slower than the eager path because the boundary that stored the GEMM output and then had GELU read and write it remained intact.

Moving bias and GELU into a cuBLASLt epilogue removed this round trip and replaced the standalone call with `_addmm_activation`. The custom GELU changed only the consumer, whereas epilogue fusion eliminated the boundary between GEMM and GELU. The result was validated against a tolerance rather than treated as bitwise exact.

---

## 2. Intermediate Ownership

Eliminating an intermediate requires knowing who owns the value until it is fully consumed. A computation completed by one thread and a computation to which several blocks contribute require different synchronization.


| Ownership scope | Typical boundary | Main added cost |
| --------------- | ---------------- | --------------- |
| One thread | Elementwise chain | Live registers and instruction count |
| Warp or block | Row reduction, normalization | Shuffle/shared memory and barriers |
| GEMM tile | Prologue or epilogue | Layout conversion and mainloop resource pressure |
| Multiple blocks | Large reduction or conflicting scatter | Workspace, atomics, or another kernel |


### Thread-Local Operations

For example,

$$
out_i=\operatorname{clamp}(x_i s_i+t_i,\ell,u)
$$

allows one thread to own index $i$ from beginning to end. If shapes, strides, aliasing, and side effects permit, a graph compiler can fuse the operations automatically.

[PyTorch Inductor](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_get_started.html) generates Triton code for graphs of this kind. The graph compiler determines the boundary, while Triton provides the kernel language and compiler. [XLA](https://openxla.org/xla/architecture) and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) make similar decisions during their compilation stages.

A one-line Python expression does not guarantee a single kernel. Inspect the generated graph or kernel trace.

### GEMM Prologues and Epilogues

A linear layer often applies bias and activation after GEMM. Running the three operations as separate kernels creates an output-sized tensor between stages. Epilogue fusion applies bias and activation before the GEMM result is stored in global memory, eliminating that intermediate.

![CUTLASS GEMM hierarchy with an epilogue tile and epilogue functor](./assets/cutlass-gemm-epilogue.webp)

*Classic CUTLASS GEMM hierarchy. Per-thread accumulator fragments pass through an epilogue tile and functor before being stored in global memory. This figure describes the register-accumulator path in the cited CUTLASS model; the Blackwell SM100 path is discussed separately below. Source: [NVIDIA CUTLASS — Efficient GEMM in CUDA, Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue)*

For the following computation,

$$
Z=\alpha AB+\beta C,
\qquad
D=\operatorname{GELU}(Z+b),
$$

let $S$ be the byte size of the $M\times N$ output. Exclude A, B, optional C, and the smaller broadcast bias, which both paths read in common. Running GEMM, bias, and activation separately produces approximately

$$
T_{\mathrm{unfused}}\approx5S
$$

of output and intermediate traffic. Within the same accounting scope, a fused epilogue can leave only the final store and reduce the total to $T_{\mathrm{fused}}\approx S$. During training, a kernel may return both the activation and pre-activation:

$$
D=\operatorname{GELU}(Z),
\qquad
\mathrm{Aux}=Z.
$$

`Aux` is reused during backward. Even though the auxiliary tensor remains, the kernel boundary between GEMM and the activation disappears.

Accumulator placement depends on the architecture. Traditional CUTLASS paths through Hopper commonly pass MMA results to the epilogue as register fragments, whereas the Blackwell SM100 TCGen05 path stores accumulators in TMEM. The epilogue loads a TMEM subtile into registers, applies the fusion, and then stores it through shared memory and TMA.

Both paths apply the transformation before materializing the global output, but the accumulator is not always held in thread registers. See the official [CUTLASS description of the SM100 epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/utils_sm100.html#background-sm100-epilogue-flow) for details.

A prologue addresses the other side of GEMM. For quantized weights,

$$
W=\operatorname{dequant}(W_q;s,z)
$$

the kernel can unpack, scale, and transform the layout while loading each operand tile instead of creating a dense temporary weight. This increases the instruction count, register usage, and pressure on operand delivery.

Asynchronous copies or pipelines can hide part of this cost. Pipelining, however, only hides latency; it does not guarantee that fusion will improve performance.

### Reduction

LayerNorm, RMSNorm, and Softmax reduce multiple elements into a single statistic. If one row fits within a warp or block, the same kernel can perform both the reduction and normalization.

If one block cannot own the complete row, the kernel may need to reread the input or use a small partial-result workspace, and atomics or an additional kernel may remain.

The minimum Welford state needed for mean and variance is $(n,\mu,M_2)$. For two nonempty partial states $A$ and $B$, let $\delta=\mu_B-\mu_A$ and combine them as follows:

$$
n=n_A+n_B,
\qquad
\mu=\mu_A+\delta\frac{n_B}{n},
$$

$$
M_2=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n}.
$$

These states can be combined in hierarchical warp and block reductions, but changing the floating-point evaluation order can also change the bit-level result. The combine rule follows the [parallel variance analysis by Chan, Golub, and LeVeque](https://doi.org/10.1080/00031305.1983.10483115).

If the consumer can use each tile immediately, eliminate the large per-element intermediate. If one block cannot own the result to completion, retain only a small partial workspace.

FlashAttention is a representative example. It passes each score tile directly to online softmax and $V$ multiplication, avoiding construction of the complete score and probability matrices.

### Indexed and Sparse Workloads

Once an index is introduced, read locality and write ownership depend on the data. Consider a token-wise affine transform that selects a parameter row using `batch_ids`.

```python
scale_token = scale[batch_ids]
shift_token = shift[batch_ids]
out = x * scale_token + shift_token
```

In eager mode, two `[T,C]` gather results may be created before the affine operation. A compiler, DSL kernel, or fused custom operator can remove these intermediates by reading the selected parameter values as it processes each token. Before choosing an implementation, check the following:

- Are channels contiguous, allowing adjacent lanes to access `x`, `out`, and the selected parameter row in a coalesced pattern?
- Do nearby tokens reuse the same batch or spatial key enough to recover the cost of grouping or shared-memory staging?
- Can the cost of sorting or binning be amortized across several downstream stages?
- For a scatter, can equal keys be reduced within a warp or block to lower the number of global atomics?

For a scatter such as

```cpp
atomicAdd(output + index[i], value[i]);
```

when updates converge on a hot destination, atomics serialize and the accumulation order also becomes data-dependent. Warp aggregation, block-local partials, and spatial binning reduce global atomics but may require additional work or a new materialization.

Irregular gather/scatter does not always require direct CUDA. Treat graph compilers, GPU DSLs such as Triton, library primitives, and custom CUDA operators as candidates, then choose according to the representation and conflict distribution.

Layout must be checked as well. If eliminating one transform changes contiguous access into strided access, the result can be slower. Measure consumer-oriented packing or a shared-memory transpose over the complete path, including upstream conversion, tails, and alignment.

---

## 3. Costs of Fusion

### Resource Usage and Library Performance

Fusion lengthens the live ranges of intermediate values. Higher register pressure can lower occupancy or cause local-memory spills, while increased shared-memory usage can reduce the number of resident blocks. Barriers may stall a warp or block, and the producer and consumer may prefer different tile shapes.

Occupancy is not itself the objective. However, too few active warps make it harder to hide memory and instruction latency.

The easiest cost to overlook is the performance of a tuned library kernel. Compare the following paths on the actual shapes:

```text
fast library GEMM + small separate kernel
versus
custom fused GEMM with a slower mainloop
```

An epilogue supported by cuBLASLt or CUTLASS can preserve a tuned mainloop. A broader custom fusion is worthwhile only when the reduction in traffic and launch costs exceeds the mainloop slowdown, workspace, dispatch, and backward-recomputation costs.

### Numerical Contract

Fusion can remove an intermediate rounding point. An unfused BF16 GEMM rounds its FP32 accumulator when storing it as BF16, whereas a fused epilogue retains higher precision until the final store. FMA contraction or a different reduction tree can also change finite-precision results.

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\not\equiv
\operatorname{fl}(a+\operatorname{fl}(b+c)).
$$

Define the required numerical contract before starting the benchmark.


| Contract | Requirement |
| -------- | ----------- |
| Mathematical equivalence | The same operation in real arithmetic |
| Tolerance-based equivalence | A result within the stated absolute/relative error bounds |
| Determinism | The same result across repeated runs of the same implementation under the stated conditions |
| Byte-exact reference match | The same bits as the specified reference path |


The four contracts are independent. A result can be deterministic yet differ from the reference, or remain within tolerance while its low bits vary across runs. The selected contract determines the permitted intermediate rounding, Tensor Core mode, FMA, reduction reassociation, and atomic order.

---

## 4. Implementation Choices and Validation

### Control Surface

There is more than one way to implement fusion. Start with the highest-level option that can express the boundary to be removed.


| Control surface | Useful when |
| --------------- | ----------- |
| Framework/compiler fusion | The graph and regular layout expose the complete boundary |
| CUDA Graph | Launch/submission overhead dominates and materialization may remain |
| CCCL algorithm: Thrust or CUB | A transform, reduce, scan, sort, or reduce-by-key matches a maintained primitive. Temporary storage or multiple launches may remain in the composition |
| cuBLASLt or a supported vendor fused operator | The GEMM post-operation fits a supported epilogue and layout |
| CUTLASS / CuTe DSL / GPU kernel DSL | The tile dataflow or prologue/epilogue requires finer control |
| Custom CUDA operator | Ownership, indexing, atomics, or integration cannot be expressed effectively at a higher level |


CCCL includes Thrust and CUB. Thrust provides high-level parallel algorithms, while CUB provides device-, block-, and warp-level primitives such as `DeviceReduce`. Temporary storage and execution strategy must also be counted as candidate costs, and composing several primitives does not guarantee single-kernel fusion. See the official [CCCL/CUB `DeviceReduce` documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html) for its scope.

### Validation and Rejection Criteria

Measure the baseline and candidate over the same end-to-end region, shape distribution, dtype, and warm-up conditions. Connect each measurement to the cost that fusion is intended to reduce.


| Claim | What to inspect |
| ----- | --------------- |
| The boundary disappeared | Kernel trace and launch count, absence of the target allocation, and expected logical intermediate write/read bytes |
| Physical traffic fell | Device-memory and L2 bytes/throughput and cache behavior in Nsight Compute **Memory Workload Analysis** |
| Resource costs did not erase the gain | Registers per thread, static/dynamic shared memory, theoretical/achieved occupancy, and local-memory traffic or spill instructions in **Launch Statistics** and **Occupancy** |
| The main computation did not slow down | Where applicable, GEMM/mainloop throughput, total kernel time, and end-to-end latency including casts, reorder, workspace, allocation, and dispatch |
| Correctness was preserved | Representative shapes, tail-heavy shapes, and inputs with concentrated conflicts under the selected numerical contract |


The section names in the table follow the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules). Raw metric identifiers can vary by architecture and tool version, so they are presented at the section level.

Reject the candidate or narrow the scope of fusion if any of the following applies:

- The target intermediate is still written and read.
- End-to-end improvement is within run-to-run noise or important shapes become slower.
- Spills, shared-memory usage, occupancy loss, or a slower library mainloop consume the byte savings.
- Preprocessing, workspace, or backward recomputation merely moves the cost outside the measured region.
- The implementation violates the selected numerical contract.

The goal of fusion is not to build the largest possible kernel. Fuse only far enough to remove the measured cost without creating a larger one.

---

## Closing

Kernel fusion changes the lifetime of intermediates. Find the value being materialized and identify the thread, warp, or block that owns it; this determines how far fusion can extend.

Thread-local elementwise chains are relatively simple, but GEMM prologues and epilogues must preserve a fast mainloop. Reductions expand ownership across multiple threads and blocks, while indexed and sparse workloads make locality and write conflicts vary with each input.

If launch submission is the bottleneck, a CUDA Graph may be sufficient. If a tuned library plus a small separate kernel is faster end to end, keep that boundary. A good fused kernel removes a specific materialization and passes both performance measurement and the numerical contract.

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
