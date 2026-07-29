## System Focus

The practical target of VARCO 3D is not just to synthesize geometry.
The service path has to produce a textured mesh that is fast to generate, stable to post-process, and usable by downstream 3D workflows.

I worked across three connected layers:

1. **Native 3D generation** — train large in-house geometry models rather than relying on slow per-asset SDS optimization.
2. **Production inference** — profile and rewrite the expensive denoising path so the model can run under service latency constraints.
3. **Mesh and texture processing** — convert generated geometry into cleaned, decimated, unwrapped, textured mesh assets.

## Architecture Shift

The model stack moved from dense latent generation to sparse active-structure generation.
That shift mattered because it changed both the training target and the serving bottleneck.

### VARCO 3D 1.0

**VecSet-based ShapeVAE and Dense DiT Denoiser**

The first stack used dense geometry generation with lattice-conditioned refinement, focused on producing stable mesh structure from native 3D latents.

### VARCO 3D 2.0

**Sparse DC VAE and Sparse DiT Denoiser**

The second stack moved generation onto active sparse 3D structure for higher-detail outputs and a more scalable inference path.
Sparse assets are less uniform at runtime, however: token count and active layout vary by input.
Serving therefore needed model-aware profiling and custom CUDA work rather than only generic graph-level acceleration.

## Production Inference Optimization

On the VARCO 3D 2.0 sparse denoiser, I profiled the forward path and identified null-context attention as arithmetic the model did not need to repeat.
The optimized path replaces unconditional cross-attention with a fixed-vector path, then fuses memory-bound tensor operations through custom CUDA kernels and cuBLASLt epilogues.

This was not a benchmark-only shortcut.
The optimized path entered production serving with numerical validation and fallback rules around the fused kernels.

[Read the VARCO 3D 2.0 inference optimization write-up](/blogs/posts/optimizing-sparse-3d-generation-inference/)

## Mesh and Texture Delivery

To extend geometry generation into textured-mesh delivery, I implemented and optimized the post-generation pipeline around GPU execution:

- **CUDA-based QEM and topology cleaning** for stable simplification and production mesh repair.
- **Custom UDF kernels** for fast solid correction from generated geometry.
- **Flood-fill acceleration kernels** to make occupancy and solidness correction practical at service scale.
- **Optimized tile-based UV unwrapping** so algorithmic unwrap remains stable on million-face meshes.
- **CUDA rasterizer based back-projection** that uses UV IDs to project multi-view generated images directly into texture-map space.

This layer is where the generated result becomes a deployable asset: cleaned geometry, controlled face count, valid UVs, and texture maps produced without a slow CPU-bound handoff.

## Result

<video autoplay muted loop playsinline preload="metadata" poster="./assets/community-showcase-poster.webp" aria-label="Recent VARCO 3D community creations transitioning from geometry to textured assets">
  <source src="./assets/community-showcase.mp4" type="video/mp4">
</video>

The geometry-to-texture reel uses ten recent public creations from [VARCO 3D Explore](https://3d.varco.ai/explore).
