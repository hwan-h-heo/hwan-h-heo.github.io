## Abstract

Scaled Dot-Product Attention, or SDPA, computes similarities between queries and keys, normalizes them with softmax, and uses the resulting weights to combine values. It is one of the most widely used operations in Transformers and other modern deep learning models.

The formula appears to be a simple sequence of two matrix multiplications with softmax in between. However, naive attention executes these as three separate stages, writing and rereading score and probability matrices whose size grows quadratically with the number of tokens. The memory traffic from these intermediates becomes a bottleneck for latency and memory usage in many workloads.

FlashAttention does not change the SDPA formula. It reorganizes the same computation by tiling attention and carrying row-wise statistics through online softmax, so the score and probability matrices never need to be materialized in HBM.

Start with how this reorganization reduces memory traffic in the attention forward and backward passes. The changes from FA1 through FA4 can then be compared through intermediate-state ownership and the hardware pipeline. The final section identifies the kernel that actually runs behind PyTorch and cuDNN SDPA backends.

---

## 1. Attention Memory Traffic

![FlashAttention's GPU memory hierarchy, tiled computation, and runtime compared with attention that materializes intermediates](./assets/flashattention-figure-1.webp)

*GPU memory hierarchy, FlashAttention's tiled dataflow, and runtime compared with a baseline that materializes intermediates. Hardware and measurements are from the 2022 FA1 evaluation environment. Source: [FlashAttention paper — Figure 1](https://arxiv.org/abs/2205.14135).*

For one attention head, let the inputs be

$$
Q \in \mathbb{R}^{N_q \times d},
\qquad
K \in \mathbb{R}^{N_k \times d},
\qquad
V \in \mathbb{R}^{N_k \times d_v}.
$$

The score, probability, and output are

$$
S=\frac{QK^\top}{\sqrt d}+B+M,
\qquad
P_{ij}=\frac{\exp(S_{ij})}{\sum_{k=0}^{N_k-1}\exp(S_{ik})},
\qquad
O=PV
$$

where $B$ is an optional bias and $M$ is an additive mask. In self-attention, $N_q=N_k=N$. A baseline that separates the three stages and materializes the intermediates executes as follows.

```text
QKᵀ → write S to HBM
S    → row-wise softmax → write P to HBM
P,V  → PV → write O
```

The two matrix multiplications cost $O(N_qN_kd)$ and $O(N_qN_kd_v)$, respectively, and each intermediate matrix occupies $O(N_qN_k)$ space.

FlashAttention removes the round trips that write $S$ and $P$ to HBM and read them back, but the quadratic compute of dense attention remains.

The IO analysis in the FA1 paper uses narrower conditions than the general notation above. It assumes single-head self-attention with $Q,K,V\in\mathbb R^{N\times d}$ and a two-level memory model consisting only of HBM and SRAM. The SRAM capacity in elements, $M_{\mathrm{SRAM}}$, is assumed to satisfy

$$
d \le M_{\mathrm{SRAM}} \le Nd
$$

Under these conditions, FA1 performs

$$
\Theta\!\left(\frac{N^2d^2}{M_{\mathrm{SRAM}}}\right)
$$

HBM element accesses, while the baseline that materializes intermediates performs $\Theta(Nd+N^2)$. Both are element-access counts in the algorithmic model; actual cache behavior and HBM transactions depend on the architecture. [The FA1 paper states both these assumptions and the scope of the lower bound](https://arxiv.org/abs/2205.14135).

This baseline does not describe the default behavior of current frameworks. As of August 2026, [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) selects a backend according to the input conditions, while [cuDNN SDPA](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html) can select FA2-based implementations automatically or explicitly. The same API call may execute a different kernel.

---

## 2. Tiled Online Softmax

Because each softmax probability depends on a reduction over the full row, softmax cannot be attached directly to a GEMM epilogue like an ordinary elementwise operation. FlashAttention instead carries the reduction result as state from tile to tile.

For one query row, let $A$ be the set of keys processed so far. Maintain three values:

$$
m_A=\max_{j\in A}s_j,
$$

$$
\ell_A=\sum_{j\in A}\exp(s_j-m_A),
$$

$$
o_A=\sum_{j\in A}\exp(s_j-m_A)v_j.
$$

$m_A$ is the running maximum, $\ell_A$ is the normalization sum, and $o_A$ is the value-weighted numerator. After processing the last key,

$$
O=\frac{o_A}{\ell_A}
$$

is obtained.

Let $A$ be the existing set of keys and $B$ a new block. The reference maximum for merging their local states is

$$
m=\max(m_A,m_B)
$$

Using the relation

$$
e^{s_j-m}=e^{s_j-m_A}e^{m_A-m},
$$

the other states can be merged as well:

$$
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,
$$

$$
o=e^{m_A-m}o_A+e^{m_B-m}o_B.
$$

An actual kernel updates the state directly from a score block $s^{(b)}$.

$$
m_b=\max_j s_j^{(b)},
\qquad
m_{\mathrm{new}}=\max(m_{\mathrm{old}},m_b),
$$

$$
\ell_{\mathrm{new}}
=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}\ell_{\mathrm{old}}
+\sum_j e^{s_j^{(b)}-m_{\mathrm{new}}},
$$

$$
o_{\mathrm{new}}
=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}o_{\mathrm{old}}
+\sum_j e^{s_j^{(b)}-m_{\mathrm{new}}}v_j.
$$

If the scores are divided into `[1, 2]` and `[3, 4]`, the denominator from the first block is multiplied by $e^{2-4}$.

$$
\ell=e^{-2}(e^{-1}+1)+(e^{-1}+1)=e^{-3}+e^{-2}+e^{-1}+1.
$$

This is the stable-softmax denominator obtained by subtracting 4 from the full row, and the merge formula itself is exact in real arithmetic.

The equations do not determine which loop, CTA, or warp owns this state. The implementation differences between generations arise from this state ownership and the pipeline around it.

---

## 3. FA1–FA4

Algorithm 1 in FA1 places K/V block column $j$ in the outer loop.

```text
for each K/V block j:
    load K_j, V_j once
    for each Q block i:
        load Q_i and partial O_i, m_i, l_i
        compute S_ij and update O_i, m_i, l_i
        store partial O_i, m_i, l_i
```

This reuses one K/V block across several Q blocks, but it also rereads Q and the normalized partial-output state from HBM on every outer-loop iteration. Thus, avoiding the storage of $S/P$ does not mean that every input is read only once.

FA2 swaps the two loops so that one CTA owns a query-row block. It traverses the K/V blocks while keeping Q, the online-softmax state, and the still-unnormalized output numerator on-chip.

```text
for each Q block i in one CTA:
    load Q_i
    m_i = -inf; l_i = 0; o_i = 0

    for each K/V block j:
        load K_j, V_j
        S_ij = Q_i K_j^T * scale + bias + mask
        update m_i, l_i, o_i

    O_i = o_i / l_i
    L_i = m_i + log(l_i)
    store O_i, L_i
```

FA1 stores a normalized partial output after each block, whereas FA2 retains the undivided $o_i$ and divides only once at the end, reducing work outside the matrix multiplications.

The partitioning scheme that assigns K/V to different warps and merges partial outputs in shared memory also changes. FA2 partitions Q rows and output slices across warps; when the layouts align, MMA fragments and row state remain in registers, leaving Q/K/V tiles and layout transformations in shared memory. Larger tiles increase both reuse and resource consumption.

Later versions begin with the same two questions: what bottleneck remains in the previous implementation, and which execution unit should own each state?


| Version | Bottleneck targeted by this version | Ownership or pipeline response | Numerical scope |
| --- | --- | --- | --- |
| FA1 | Baseline materialization of $S/P$ in HBM | K/V outer loop; repeated visits to Q/O row state | Exactly equivalent to dense attention in real arithmetic |
| FA2 | Occupancy, non-matmul work, inter-warp exchange | Q outer loop; CTA-owned row block; per-warp Q/output partitioning | Same mathematical function, different floating-point order |
| FA3 | Underused Hopper asynchronous units | TMA producers, WGMMA consumers, GEMM–softmax overlap | FP16/BF16 path and a separate FP8 path |
| FA4 | Blackwell Tensor Cores scaling faster than softmax and shared memory | Fully asynchronous MMA to TMEM, larger tiles, two exponential paths, conditional rescaling, 2-CTA MMA in backward | March 2026 v1 preprint; polynomial `exp2` is an explicit approximation |


FA3 separates producer warpgroups that issue TMA loads from consumers that execute WGMMA and softmax, overlapping data movement with computation and interleaving $QK^\top$, softmax, and $PV$.

Its FP8 forward path adds separate numerical processing. Before quantizing Q/K/V by blocks, it applies an orthogonal transform to Q and K, preserving $QK^\top$ while spreading outliers. Block quantization and incoherent processing reduce FP8 error and should be distinguished from the FP16/BF16 pipeline. [The FA3 paper likewise presents these as separate contributions](https://arxiv.org/abs/2407.08608).

FA4 is a March 2026 arXiv v1 preprint targeting Blackwell, not yet a standard implementation validated across multiple platforms.

It uses fully asynchronous MMA that stores accumulators in TMEM and moves output correction off the critical path. Most `exp2` values are computed with hardware `MUFU.EX2`, while only a tile-dependent subset is approximated with FMA polynomials; the paper uses approximation ratios of 10–25%.

Conditional rescaling delays rescaling the output accumulator until the running maximum grows beyond a threshold, while continuing to track the total scale required for final normalization. In backward, two CTAs execute MMA together to reduce shared-memory traffic and the number of atomic accumulations into $dQ$. [FA4 v1 describes the algorithm and the scope of its B200 experiments](https://arxiv.org/abs/2603.05451).

Masks and biases also affect the dataflow. Causal attention skips tiles that lie entirely in the future and applies predicates only to diagonal tiles, while a bias that can be expressed analytically is generated inside the score tile. An arbitrary dense $N_q\times N_k$ bias adds a large input stream and may become a new bottleneck.

---

## 4. Backward Recomputation and Determinism

Forward stores only $Q/K/V/O$ and one row-wise logsumexp value instead of the full $S/P$.

$$
L_i=m_i+\log\ell_i=\log\sum_k e^{S_{ik}}.
$$

Backward recomputes each unstored score tile, reconstructs the probabilities as

$$
P_{ij}=\exp(S_{ij}-L_i)
$$

and uses them immediately, repeating Tensor Core work instead of storing large activations.

Without dropout, if the output gradient is $dO$, then

$$
dV=P^\top dO,
\qquad
dP=dOV^\top
$$

The row-wise term needed by the softmax gradient can be obtained from the output already stored:

$$
D_i=\sum_{a=0}^{d_v-1}dO_{ia}O_{ia}=\sum_jP_{ij}dP_{ij}.
$$

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i),
$$

$$
dQ=\frac{dSK}{\sqrt d},
\qquad
dK=\frac{dS^\top Q}{\sqrt d}.
$$

The following is a representative **sequence-parallel, nondeterministic** mapping used in FA2; other backward schedules are also possible.

```text
for each K/V block j in one CTA:
    load K_j, V_j
    initialize dK_j, dV_j

    for each Q block i:
        load Q_i, dO_i, L_i, D_i
        recompute S_ij and P_ij = exp(S_ij - L_i)
        dV_j += P_ij^T dO_i
        dP_ij = dO_i V_j^T
        dS_ij = P_ij * (dP_ij - D_i)
        dK_j += dS_ij^T Q_i / sqrt(d)
        atomicAdd(dQ_i, dS_ij K_j / sqrt(d))

    store dK_j, dV_j
```

The CTA owns $dK_j$ and $dV_j$ through completion, so neither needs a cross-CTA reduction. Multiple K/V-column CTAs contribute to the same $dQ_i$, however, and the unspecified ordering of atomic adds makes this path nondeterministic even without dropout.

A deterministic implementation stores partial contributions in workspace or sums them in a fixed order, using additional memory and computation. The [official FlashAttention implementation](https://github.com/Dao-AILab/flash-attention) provides a separate deterministic backward and states that it is slower and uses more memory. cuDNN support varies by architecture and version.

Reproducing the dropout mask and making gradient accumulation deterministic are separate problems.

$$
P^{\mathrm{drop}}_{ij}=\frac{Z_{ij}}{1-p}P_{ij}
$$

Softmax normalization finishes before dropout, and $P^{\mathrm{drop}}$ is multiplied by $V$. Backward must regenerate the same $Z_{ij}$ used in forward.

An implementation that uses a counter-based RNG maps each **logical attention position** $(\text{batch},\text{head},i,j)$ to the same counter and random value. The exact counter mapping is a backend-specific implementation contract. If this mapping and the saved RNG state match, forward and backward may traverse tiles in different orders. Gradient accumulation can still remain nondeterministic even when the dropout mask is reconstructed correctly.

---

## 5. Performance Regimes and Backend Selection

The name FlashAttention alone does not identify the kernel that runs. Training, prefill, and decode have different primary bottlenecks, and the input layout and mask also affect backend selection.


| Regime | Dominant pressure | Implication for kernel/backend selection |
| --- | --- | --- |
| Training or long prefill | $S/P$ activation traffic, dense matmul, backward state | Fused training SDPA and recomputation often provide the main benefit |
| Short prefill or small batch/head count | Launch and setup cost, tile tails, insufficient parallel work | Compare framework/vendor heuristics and alternatives on the actual shape |
| Decode, $N_q\approx1$ | KV-cache bandwidth, paging, split-KV reduction, batching | Requires decode- and KV-cache-aware kernels; a training/prefill schedule does not solve these costs by itself |
| GQA/MQA | KV-head reuse, mapping multiple Q heads to fewer KV heads, reduction shape | Verify native grouped-head support and measure the balance between reuse and parallelism |
| Paged or variable-length inputs | Indirection, load balance, padding waste, physical layout | Use paged/ragged SDPA or serving kernels and verify support for the mask/layout combination |


![Forward-backward runtime and memory usage across attention implementations](./assets/flashattention-figure-3.webp)

*Forward–backward runtime and memory usage measured on A100 in the 2022 FA1 paper. This is not an FA2–FA4 comparison. The Linformer and block-sparse curves compute different approximate attention patterns. Source: [FlashAttention paper — Figure 3](https://arxiv.org/abs/2205.14135).*

Because backend support changes quickly, the related statements in this article were checked against the August 2026 documentation for PyTorch 2.13 and cuDNN 9.13.1, and the Dao-AILab repository.

Dtype, head dimension, mask, stride, GQA, paging, determinism, and GPU architecture may change the backend or trigger a fallback. In PyTorch, constrain the candidates with `torch.nn.attention.sdpa_kernel()`, then inspect the GPU events in a profiler trace to identify the kernel that actually ran.

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

prof.export_chrome_trace("sdpa-trace.json")
```

When the selected backend does not support a shape or dtype, constraining the candidates exposes the reason through an error or warning instead of silently accepting a fallback. In the Chrome trace, inspect the kernel events on the GPU track rather than stopping at the CPU-side SDPA operator name. Match the warm-up count and profiling region to the real benchmark.

---

## FlashAttention Path Selection

The common foundation of FlashAttention is the $(m,\ell,o)$ state carried from tile to tile. FA1 uses this state to eliminate HBM materialization of $S/P$, while FA2 lets a query CTA own the row state through completion. FA3 and FA4 reorganize producers, consumers, on-chip storage, and the computation pipeline for Hopper and Blackwell.

Sharing a version name does not guarantee identical numerical contracts or performance characteristics. FP8 quantization, polynomial `exp2`, atomic adds, dropout RNG mapping, and decode-oriented KV-cache scheduling must each be validated separately, and the backend should be identified from the kernel that actually runs rather than from the API name.

---

## References

1. Tri Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135)
2. Tri Dao, [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691)
3. Jay Shah et al., [*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*](https://arxiv.org/abs/2407.08608)
4. Ted Zadouri et al., [*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling* — arXiv v1, March 2026](https://arxiv.org/abs/2603.05451)
5. Dao-AILab, [*FlashAttention reference implementation*](https://github.com/Dao-AILab/flash-attention)
6. PyTorch, [`scaled_dot_product_attention` documentation — PyTorch 2.13](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
7. NVIDIA, [*cuDNN Scaled Dot Product Attention* — checked against cuDNN 9.13.1](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
