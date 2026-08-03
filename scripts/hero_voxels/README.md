# Hero surface-voxel hierarchy

This directory contains the deterministic offline preprocessing pipeline for the
portfolio Three.js hero. The browser never loads, parses, or voxelizes the source
GLB. It fetches one versioned JSON manifest and one compact binary hierarchy.

## Environment

Use Python 3.11–3.13 and `uv`. The lockfile pins the only runtime dependencies:

- `numpy==2.3.2`
- `trimesh==4.7.1`

```bash
uv sync --project scripts/hero_voxels
```

If `uv` is not on `PATH`, use `python -m uv` in place of `uv`.

## Canonical generation

Production uses a conservative 512^3 surface reference. It is collapsed first
to 256 and then to the displayed 128 finest level. The runtime hierarchy is
`8, 16, 32, 64, 128`, and every active index is retained at every displayed
level—there is no production leaf sampling.

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/build_voxel_hierarchy.py \
  --input assets/hero/03_remesh.glb \
  --output-dir assets/hero/voxel-hierarchy \
  --reference-resolution 512 \
  --resolutions 8 16 32 64 128 \
  --seed 7
```

Direct 256^3 reference voxelization is supported:

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/build_voxel_hierarchy.py \
  --input assets/hero/03_remesh.glb \
  --output-dir assets/hero/voxel-hierarchy \
  --reference-resolution 256 \
  --resolutions 8 16 32 64 128 \
  --seed 7
```

`--desktop-max-leaves` and `--mobile-max-leaves` remain available for unusual
constrained deployments. Their default `0` means retain all finest-level
indices. A positive value enables deterministic Morton-stratified sampling and
is not used by the committed assets.

Demonstration-parent caps default to `512, 900, 500, 250` on desktop and
`512, 300, 160, 80` on mobile. Rejected-ghost budgets default to 2,000 and 600.
Override these with `--desktop-demo-parents`, `--mobile-demo-parents`,
`--desktop-ghost-budget`, and `--mobile-ghost-budget`.

## GLB loading and normalization

The builder inspects every GLB primitive and rejects modes other than glTF
triangle mode 4. `trimesh` traverses every geometry-bearing scene node, resolves
its world transform, and bakes that transform into vertex positions. Indexed and
non-indexed triangle primitives, multiple nodes, meshes, and primitives are
supported. Cameras, lights, animation, and material appearance are ignored.
Non-finite and degenerate triangles are removed; generation fails when no valid
triangles remain.

All triangles share one AABB and are normalized with uniform scaling only:

```python
center = 0.5 * (bbox_min + bbox_max)
scale = 0.96 / max(bbox_max - bbox_min)
normalized = (vertices - center) * scale
```

The result fits inside `[-0.48, 0.48]^3`, centered in the voxel domain
`[-0.5, 0.5]^3`. Coordinates retain the glTF right-handed `xyz` order. Original
and normalized bounds, translation, uniform scale, axis convention, source
filename, and source SHA-256 are stored in the manifest.

## Surface voxelization and hierarchy

Voxelization is surface-only. Each triangle's clipped grid-space cell AABB is
enumerated in bounded chunks. Candidate cells are tested with deterministic
triangle/AABB separating axes: the triangle plane and nine edge/cell-axis cross
products, with box-axis overlap guaranteed by candidate enumeration. Intersecting
cells are accumulated sparsely; no dense 512^3 array or random sampling is used.

Reference coordinates are unique and lexicographically sorted. Hierarchy
derivation is strict:

```python
idx256 = unique(idx512 // 2)  # omitted when reference resolution is 256
idx128 = unique(idx256 // 2)
idx64  = unique(idx128 // 2)
idx32  = unique(idx64 // 2)
idx16  = unique(idx32 // 2)
idx8   = unique(idx16 // 2)
```

Each active parent stores one byte. Child `(dx, dy, dz)` uses:

```python
bit = dx * 4 + dy * 2 + dz
```

Bits 0–7 therefore enumerate `(000), (001), (010), (011), (100), (101),
(110), (111)`. Python validation and JavaScript ghost expansion use this exact
ordering.

The committed desktop and mobile variants retain identical full surface
coordinates. They differ only in the offline-selected demonstration parents and
rejected-ghost budgets. Demonstration candidates must have two to five active
children and are distributed in Morton order.

## Binary schema

The schema is `hero-voxel-hierarchy`, version 2. Large arrays are stored only in
the little-endian binary payload, each at a four-byte-aligned offset.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | 4 bytes | ASCII `HVOX` |
| 4 | `Uint32` | schema version |
| 8 | `Uint32` | level count |
| 12 | `Uint32` | header byte length (`16`) |

Each level points to packed `Uint16[count][3]` coordinates. Levels 8–64 also
contain `Uint8[count]` child masks and a small sorted `Uint32[]` array of
demonstration-parent indices. The final 128 level contains coordinates only.
Descriptors include type, byte offset, and byte length. The manifest records the
total binary size and SHA-256; unknown versions are rejected.

## Validation and determinism

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/validate_voxel_hierarchy.py \
  --manifest assets/hero/voxel-hierarchy/hero-voxels-desktop.json

uv run --project scripts/hero_voxels python scripts/hero_voxels/validate_voxel_hierarchy.py \
  --manifest assets/hero/voxel-hierarchy/hero-voxels-mobile.json
```

Validation covers version/header fields, typed lengths, bounds and alignment,
coordinate range/order/uniqueness, exact parent derivation, exact mask
reconstruction, full-level retention, demo indices, ghost budgets, file sizes,
and hashes.

Generate the same command into a second directory and pass its manifest through
`--compare-manifest` to prove byte-identical JSON and binary output. Manifests
omit timestamps and environment-dependent metadata; identical input bytes,
arguments, seed, Python range, and lockfile produce identical files.

## Runtime behavior

The production entry point defaults to the existing wave hero. The voxel hero is
an experimental one-shot intro loaded only when `?hero=voxel` is present; the
default page does not fetch a voxel manifest or binary. When selected,
`js/hero-voxel.js` chooses
the desktop or mobile/low-spec manifest, validates its schema and binary ranges,
and creates typed-array views directly over the fetched `ArrayBuffer`.

The opening draw is a fully occupied instanced 8^3 cube volume. A shader-driven
Manhattan/BFS-like wave begins at the visible upper-left `(x-, y+, z+)` corner and travels
toward the opposite corner. When the wave reaches a cell, non-surface voxels
shrink and disappear while surface voxels move from dark graphite to a quiet
silver-gray.

One additional screen-space `THREE.Points` draw places 110 deterministic ambient
particles across desktop, 58 on compact layouts, and 38 on constrained devices.
They use dim cool graphite rather than bright stars, drift only a few pixels, and
share the voxel dissolve envelope so no particle layer remains after the Wave
handoff.

The full `8→16→32→64→128` surface then refines without index sampling. At load
time, the runtime expands each binary parent coordinate and child mask into its
eight deterministic candidate slots using typed arrays. This does not infer
occupancy or voxelize geometry: the binary mask remains the sole source of each
candidate's retained/rejected state. No per-voxel JavaScript objects are created.

One shared unit cube and immutable instanced attributes are used. The shader
calculates `floor(child / 2)`, parent and child centers, movement, scale, opacity,
and a recursive start time. A coarse 8^3 corner rank chooses the region; each
subsequent octree offset supplies the BFS order inside that parent's group of
eight. Every local parent first becomes eight visible child slots; retained
children begin their own subdivision while that parent's rejection collapse is
still progressing. Confirmed coarse cells in the dense prune mesh shrink directly into
the first candidate level, so the initial prune and recursive cascade overlap
without a separate coarse-surface mesh swap. Per-frame JavaScript updates only
uniforms, camera state, and mesh visibility.

The recursive cascade uses a linear global clock rather than easing the complete
sequence. Every parent uses the same 0.16 normalized local duration, a constant
linear split speed, and a constant linear rejection-collapse speed. The minimum
upper-left child handoff begins when its parent is roughly 22% through the local
transition, while the octant BFS delays the farthest child to roughly 31%.
This keeps about three adjacent hierarchy levels visible and lets each reached
sub-volume descend toward the finest resolution before the broad front passes.
Small
parent-coherent offsets break up identical Manhattan-distance bands without
changing child ordering.

The dense prune mesh and all four candidate-level meshes remain present through
the prune-to-cascade handoff. Intermediate active voxels fade only when their own local
subdivision begins, so coarse and fine regions coexist along the moving front;
there is no whole-level mesh swap. This phase uses one ambient-particle draw,
one dense draw, four candidate draws, and the bounding box. The current `03_remesh.glb` hierarchy
contains 193 coarse parents and 235,232 resident candidates in total; the
largest single draw is 189,896 candidates for `64→128`. The completed 128-level
voxel silhouette then remains as the quiet final hold.

The production voxel intro lasts 6.15 seconds. The occupied volume reveals for 0.5
seconds, holds briefly, and then spends 2.91 seconds on a constant-speed
corner-to-corner prune. The 4.01-second recursive cascade begins with that prune
at 0.72 seconds. Its root
schedule uses the exact same corner rank and coordinate hash as classification,
so every confirmed coarse voxel starts refinement on the frame it appears rather
than waiting for a later resolution stage. Each lineage then continues through
all finer levels at 0.14-0.2 second handoff intervals. Motion completes at 4.73
seconds; the final voxel surface holds for 0.27 seconds and dissolves smoothly over 1.15
seconds. The voxel renderer is then disposed before the existing wave hero fades
in directly at its quiet ambient phase. The object remains axis-aligned. The submitted wide-screen
camera reference uses a 38.5-degree azimuth, 24-degree elevation, 42.5-degree
field of view, and distance 13.
The model moves along the camera's view-plane right axis, preserving its depth
and vertical center while placing it near the established wave hero's
right-center locus.

Experimental and debug parameters:

- `?hero=voxel` explicitly loads the experimental voxel implementation.
- `?hero=voxel&heroVoxelDebug=1` keeps the 6.15-second voxel sequence looping and
  shows phase, counts, rejected slots, draw calls, FPS, pause/play, and a timeline scrubber.
- `?hero=voxel&heroVoxelDebug=1&heroVoxelTime=4.86` opens on the brief final voxel hold.
- The default URL and `?hero=wave` load the production wave implementation.

DPR is capped at 1.5 desktop, 1.25 compact, and 1.0 constrained hardware. The
renderer pauses offscreen and while the document is hidden, and disposes all
observers, listeners, animation frames, geometries, materials, buffers, debug UI,
and renderer resources. Reduced motion shows the static full 128 voxel surface
with a static ambient particle field and without running the one-shot handoff.
Hierarchy failure loads the existing wave
hero; WebGL failure or context loss uses the static image fallback. A successful
one-shot disposes the voxel renderer before the ambient wave loop starts, so only
one animation loop is active.

## Using another GLB

Place the replacement GLB under ignored `assets/hero/`, update `--input`, and
rerun the canonical command. Review
the printed full counts before deploying. The static build intentionally excludes
the source GLB from `blogs/dist`; commit the regenerated JSON and binary files
together.

## Known limitations

- Conservative boundary intersections may thicken a surface by one reference
  voxel; the 512→256 collapse intentionally favors thin-part retention.
- Optional Morton sampling does not formally guarantee each disconnected
  component, which is why it is disabled for production.
- Transparent transitional instances are not individually depth-sorted.
- The full 128 mobile surface has more instances than the former sampled mobile
  asset; the halved maximum resolution and DPR caps keep the workload bounded.
