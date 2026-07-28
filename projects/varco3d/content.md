:::{.container .col-11 .varco-feature-strip}
![VARCO 3D 2.0 generated textured character](/blogs/posts/260727_sparse_3d/assets/varco3d2-result-a.png)
:::

:::{.container .portfolio-details-container .col-11}
:::{.row .gy-4}
:::{.col-lg-8}
:::{.portfolio-description}

## Project Overview

**VARCO 3D** is NC AI's production 3D generative AI service for creating textured meshes from native 3D generation models and GPU-first geometry processing.
My work spans the full generation stack: **mesh pre/post-processing**, **designing and training 1B+ parameter in-house 3D generative models from scratch**, and moving research-grade inference paths into production service.

:::{.mt-4}
**Core contributions**

- **Large-scale 3D model R&D**: designed and trained in-house 3D generative models at 1B+ parameter scale, covering the transition from VARCO 3D 1.0 to 2.0.
- **Production inference optimization**: analyzed the forward path, removed null-context attention arithmetic, and fused memory-bound paths with custom CUDA kernels and cuBLASLt epilogues.
- **Mesh post-processing**: implemented CUDA-based solid correction, QEM decimation, topology cleaning, and large-mesh UV unwrapping paths for production outputs.
- **Texture delivery**: built a CUDA rasterizer based texture back-projection pipeline that maps multi-view generated images into UV texture space.

:::
:::
:::

:::{.col-lg-4}
:::{.portfolio-info}

### Project Details

- **Role**: 3D Generative Model and Production Pipeline R&D
- **Category**: Commercial AI Service, 3D Generation
- **Organization**: NC AI
- **Model Scale**: 1B+ parameter in-house 3D generative models
- **Technology**: VecSet-based ShapeVAE, Dense DiT Denoiser, Sparse DC VAE, Sparse DiT Denoiser, CUDA, cuBLASLt, QEM, UV Unwrapping, CUDA Rasterization
- **Service URL**: [VARCO 3D](https://www.varco.ai/3d)
- **Latest Technical Writing**: [Sparse 3D Inference Optimization](/blogs/posts/optimizing-sparse-3d-generation-inference/)
- **Development Retrospective**: [Varco3D: A Year in Review](/blogs/posts/varco3d-a-year-in-review-2025-retrospective/)
- **Related Project**: [CaPa](/projects/capa/)

:::
:::
:::
:::

:::{.row .gx-5 .justify-content-center}
:::{.project-readable .portfolio-description}

## System Focus

The practical target of VARCO 3D is not just to synthesize geometry.
The service path has to produce a textured mesh that is fast to generate, stable to post-process, and usable by downstream 3D workflows.

I worked across three connected layers:

1. **Native 3D generation**: train large in-house geometry models rather than relying on slow per-asset SDS optimization.
2. **Production inference**: profile and rewrite the expensive denoising path so the model can run under service latency constraints.
3. **Mesh and texture processing**: convert generated geometry into cleaned, decimated, unwrapped, textured mesh assets.

## Architecture Shift

The model stack moved from dense latent generation to sparse active-structure generation.
That shift mattered because it changed both the training target and the serving bottleneck.

:::{.varco-version-stack}
:::{.varco-version-item}
### VARCO 3D 1.0

**VecSet-based ShapeVAE + Dense DiT Denoiser**

Dense geometry generation with lattice-conditioned refinement, focused on producing stable mesh structure from native 3D latents.
:::

:::{.varco-version-item}
### VARCO 3D 2.0

**Sparse DC VAE + Sparse DiT Denoiser**

Sparse latent generation over active 3D structure, built for higher-detail outputs and a more scalable production inference path.
:::
:::

The tradeoff is that sparse assets are less uniform at runtime.
Token count and active layout vary by input, so the serving path needed model-aware profiling and custom CUDA work rather than only generic graph-level acceleration.

## Production Inference Optimization

On the VARCO 3D 2.0 sparse denoiser, I profiled the forward path and identified null-context attention as arithmetic the model did not need to repeat.
The optimized path replaces unconditional cross-attention with a fixed-vector path, then fuses memory-bound tensor operations through custom CUDA kernels and cuBLASLt epilogues.

This was not a benchmark-only shortcut.
The optimized path entered production serving, with numerical validation and fallback rules defined around the fused kernels.
The full write-up covers the numerical validation contract, fallback rules, and latency results:

[Read the VARCO 3D 2.0 inference optimization write-up](/blogs/posts/optimizing-sparse-3d-generation-inference/)

## Mesh and Texture Pipeline

To extend geometry generation into textured mesh delivery, I implemented and optimized the post-generation pipeline around GPU execution:

- **CUDA-based QEM and topology cleaning** for stable simplification and production mesh repair.
- **Custom UDF kernels** for fast solid correction from generated geometry.
- **Flood-fill acceleration kernels** to make occupancy and solidness correction practical at service scale.
- **Optimized tile-based UV unwrapping** so algorithmic unwrap remains stable on million-face meshes.
- **CUDA rasterizer based back-projection** that uses UV IDs to project multi-view generated images directly into texture map space.

This layer is where the generated result becomes a deployable asset: cleaned geometry, controlled face count, valid UVs, and texture maps produced without a slow CPU-bound handoff.

## Result

:::{.varco-object-embed}
<video class="project-video varco-community-showcase" autoplay muted loop playsinline preload="metadata" poster="./assets/community-showcase-poster.webp" aria-label="Recent VARCO 3D community creations transitioning from geometry to textured assets">
    <source src="./assets/community-showcase.mp4" type="video/mp4">
</video>
:::

*Geometry-to-texture reel built from 10 recent public creations on [VARCO 3D Explore](https://3d.varco.ai/explore).*

:::
:::
