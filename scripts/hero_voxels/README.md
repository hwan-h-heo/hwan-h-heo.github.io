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
`8, 16, 32, 64, 128`. A deterministic camera-visibility pass selects the
finest leaves needed by the fixed desktop and mobile views, then derives every
coarser level from those selected leaves. There is no stochastic production
leaf sampling.

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/build_voxel_hierarchy.py \
  --input assets/hero/hero_test_2.glb \
  --output-dir assets/hero/voxel-hierarchy \
  --reference-resolution 512 \
  --resolutions 8 16 32 64 128 \
  --camera-visible-only \
  --visibility-raster-height 512 \
  --visibility-depth-layers 6 \
  --seed 7
```

Direct 256^3 reference voxelization is supported:

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/build_voxel_hierarchy.py \
  --input assets/hero/hero_test_2.glb \
  --output-dir assets/hero/voxel-hierarchy \
  --reference-resolution 256 \
  --resolutions 8 16 32 64 128 \
  --camera-visible-only \
  --visibility-raster-height 512 \
  --visibility-depth-layers 6 \
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

Reference coordinates are unique and lexicographically sorted. The complete
surface hierarchy is derived strictly as follows:

```python
idx256 = unique(idx512 // 2)  # omitted when reference resolution is 256
idx128 = unique(idx256 // 2)
idx64  = unique(idx128 // 2)
idx32  = unique(idx64 // 2)
idx16  = unique(idx32 // 2)
idx8   = unique(idx16 // 2)
```

For production, the finest `idx128` is projected as complete voxel-cell AABBs
through the exact standalone Note camera profiles. A deterministic 512px-high
software z-buffer samples the centered 16:9 desktop frame plus 16:9 and 4:3
compact frames. For each layout it evaluates a 3×3 grid across the bounded
azimuth/elevation orbit (`-10°, 0°, +10°`) at three camera radii. A six-voxel
depth slack preserves silhouette edges and nearby layers. The retained `idx128` is then the sole source for the
rendered `64, 32, 16, 8` levels; those levels are never culled independently.
The manifest records both complete and camera-retained counts and the complete
camera profile. Disable this optimization by omitting `--camera-visible-only`.

Each active parent stores one byte. Child `(dx, dy, dz)` uses:

```python
bit = dx * 4 + dy * 2 + dz
```

Bits 0–7 therefore enumerate `(000), (001), (010), (011), (100), (101),
(110), (111)`. Python validation and JavaScript ghost expansion use this exact
ordering.

The committed desktop and mobile variants use their respective responsive
camera profiles, so their retained coordinates differ slightly. Demonstration
candidates must have two to five active children and are distributed in Morton
order.

## Binary schema

The schema is `hero-voxel-hierarchy`, version 2. Large arrays are stored only in
the little-endian binary payload, each at a four-byte-aligned offset.

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | 4 bytes | ASCII `HVOX` |
| 4 | `Uint32` | schema version |
| 8 | `Uint32` | level count |
| 12 | `Uint32` | header byte length (`16`) |

Each level points to packed `Uint16[count][3]` camera-retained voxel coordinates. Levels 8–64 also
contain `Uint8[count]` child masks and a small sorted `Uint32[]` array of
demonstration-parent indices. The final 128 level contains coordinates only.
Descriptors include type, byte offset, and byte length. The manifest records the
total binary size and SHA-256; unknown versions are rejected. Source GLB vertex
and face arrays are never written to either the manifest or binary payload.

## Validation and determinism

```bash
uv run --project scripts/hero_voxels python scripts/hero_voxels/validate_voxel_hierarchy.py \
  --manifest assets/hero/voxel-hierarchy/hero-voxels-desktop.json

uv run --project scripts/hero_voxels python scripts/hero_voxels/validate_voxel_hierarchy.py \
  --manifest assets/hero/voxel-hierarchy/hero-voxels-mobile.json
```

Validation covers version/header fields, typed lengths, bounds and alignment,
coordinate range/order/uniqueness, exact parent derivation, exact mask
reconstruction, camera-retained counts, demo indices, ghost budgets, file
sizes, and hashes.

Generate the same command into a second directory and pass its manifest through
`--compare-manifest` to prove byte-identical JSON and binary output. Manifests
omit timestamps and environment-dependent metadata; identical input bytes,
arguments, seed, Python range, and lockfile produce identical files.

## Runtime behavior

The production entry point always loads the existing Wave hero and contains no
URL switch to the voxel experiment. Consequently, the home page never fetches a
voxel manifest or binary. `js/hero-voxel.js` is retained as source material for
a dedicated technical Note/demo; when explicitly mounted in that isolated context it chooses
the desktop or mobile/low-spec manifest, validates its schema and binary ranges,
and creates typed-array views directly over the fetched `ArrayBuffer`.

The opening draw is a fully occupied instanced 8^3 cube volume. A shader-driven
Manhattan/BFS-like wave begins at the visible upper-left `(x-, y+, z+)` corner and travels
toward the opposite corner. When the wave reaches a cell, non-surface voxels
shrink and disappear while surface voxels move from dark graphite to a quiet,
desaturated blue-gray sampled from the Wave palette.

A separate screen-space 2D canvas places 160 deterministic ambient particles
across desktop, 96 on compact layouts, and 60 on constrained devices. It sits
above the hero's readability gradient but below the copy; keeping these points
inside the voxel WebGL canvas would hide them beneath that near-opaque gradient.
They use visible but muted cool graphite rather than bright stars and drift only
a few pixels. Particle radii stay between approximately 0.44 and 1.02 CSS pixels;
count, opacity, and drift remain independent from that size. In the Note they
share the standalone dissolve and restart with the voxel loop; no Wave handoff
is loaded.

The camera-visible `8→16→32→64→128` surface then refines without index sampling. At load
time, the runtime expands each binary parent coordinate and child mask into its
eight deterministic candidate slots using typed arrays. This does not infer
occupancy or voxelize geometry: the binary mask remains the sole source of each
candidate's retained/rejected state. No per-voxel JavaScript objects are created.

One shared unit cube and immutable instanced attributes are used. The shader
calculates `floor(child / 2)`, parent and child centers, movement, scale, opacity,
and a recursive start time. Cube fill falls gradually from `0.88` at resolution
8 to `0.72` at resolution 128 so fine voxels retain visible separation. Both the
dense and candidate shaders add the same very low-amplitude cool-graphite to
muted-blue-gray positional gradient without changing the material family. A coarse
8^3 corner rank chooses the region; each
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
there is no whole-level mesh swap. This phase uses one lightweight 2D ambient
canvas, one dense WebGL draw, four candidate draws, and the bounding box. The current desktop
`hero_test_2.glb` hierarchy retains 206 coarse parents and 107,376 resident
candidates in total; the largest single draw is 78,592 candidates for
`64→128`. The completed desktop 128-level voxel silhouette uses 37,330 retained
leaves; mobile retains 37,637. Their binaries are 322,776 and 322,188 bytes.
As each finest child settles, its center joins a broad, low-frequency traveling
ripple of roughly one cell along the object rather than freezing during the final
hold. Reduced-motion mode
samples one static ripple state.

The production voxel intro lasts 8.12 seconds after a uniform 10% slowdown. The
occupied volume reveals for 0.66 seconds, holds briefly, and then spends 3.85
seconds on a constant-speed corner-to-corner prune. The 5.3-second recursive
cascade begins with that prune at 0.95 seconds. Its root
schedule uses the exact same corner rank and coordinate hash as classification,
so every confirmed coarse voxel starts refinement on the frame it appears rather
than waiting for a later resolution stage. Each lineage then continues through
all finer levels at roughly 0.19-0.26 second handoff intervals. Motion completes
at 6.25 seconds; the final voxel surface holds for 0.35 seconds and dissolves
over 1.52 seconds before the standalone loop restarts. The object remains
axis-aligned and centered in the full Note frame. The submitted camera reference
uses a 43-degree azimuth, 26.5-degree authored elevation, 41.5-degree field of
view, distance 9.8, zero view-plane offset, and vertical center -0.45. Runtime
defaults are read back from the loaded manifest, preventing authored JavaScript
constants from silently drifting away from the visibility envelope. Standalone
layout uses that zero offset directly and bypasses the former home hero's
responsive right-bias interpolation, so the tuner and embedded iframe share the
same framing at every size.

Pointer input orbits the camera around its actual submitted look target rather
than translating it in screen space. The two-axis pointer vector is normalized
to a strict 10-degree orbit radius, eased toward the pointer, and reset toward
the submitted view when the pointer leaves the iframe. The offline culling
profile conservatively covers the larger enclosing ±10-degree square.

The debug query values `heroVoxelDebug=1` and `heroVoxelTime=<seconds>` remain
inside the experimental module for an isolated demo harness, but the portfolio
home loader does not import that module for any query string. The default and
all query-string variants therefore use the production Wave implementation.

The Hierarchical Surface Decoding Note mounts the same module in a same-origin
iframe with `data-hero-voxel-standalone="true"`. In that mode the sequence loops
after its own dissolve, suppresses the Wave handoff, exposes a small Pause/Play
control, and falls back to the Note poster instead of importing the home hero.
The iframe reports its parent-page intersection state so rendering also pauses
when the reader scrolls past the live figure.

DPR is capped at 1.5 desktop, 1.25 compact, and 1.0 constrained hardware. The
renderer pauses offscreen and while the document is hidden, and disposes all
observers, listeners, animation frames, geometries, materials, buffers, debug UI,
and renderer resources. Reduced motion shows the static full 128 voxel surface
with a static ambient particle field and without running the one-shot handoff.
Hierarchy failure loads the existing wave
hero; WebGL failure or context loss uses the static image fallback. A successful
one-shot permits both lightweight loops only during the final 1.52-second visual
handoff, then disposes the voxel renderer and leaves only the ambient Wave loop.

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
- Camera visibility is tied to the documented responsive layout and strict
  ±10-degree orbit envelope. The 27 samples per aspect, `radius ±1.2`, and
  six-layer depth margin tolerate the authored interaction, but increasing that
  angular limit still requires hierarchy regeneration.
