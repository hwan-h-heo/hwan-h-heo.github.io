#!/usr/bin/env python3
"""Build deterministic surface-voxel hierarchy assets from a GLB scene."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh

from hierarchy_common import (
    BinaryWriter,
    PRODUCTION_RESOLUTIONS,
    VERSION,
    build_child_masks,
    collapse_to_resolution,
    derive_hierarchy,
    sample_morton_stratified,
    select_demo_parents,
    sha256_file,
    write_deterministic_json,
)


TARGET_EXTENT = 0.96
GRID_MINIMUM = -0.5
SAT_EPSILON = 1e-10
TRIANGLE_BLOCK_SIZE = 65_536


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-resolution", type=int, choices=(256, 512), default=512)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(PRODUCTION_RESOLUTIONS))
    parser.add_argument(
        "--desktop-max-leaves",
        type=int,
        default=0,
        help="Optional finest-level sampling budget; 0 retains every active voxel (default).",
    )
    parser.add_argument(
        "--mobile-max-leaves",
        type=int,
        default=0,
        help="Optional finest-level sampling budget; 0 retains every active voxel (default).",
    )
    parser.add_argument("--desktop-demo-parents", type=int, nargs=4, default=(512, 900, 500, 250))
    parser.add_argument("--mobile-demo-parents", type=int, nargs=4, default=(512, 300, 160, 80))
    parser.add_argument("--desktop-ghost-budget", type=int, default=2_000)
    parser.add_argument("--mobile-ghost-budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    if tuple(arguments.resolutions) != PRODUCTION_RESOLUTIONS:
        parser.error(f"--resolutions must be exactly {' '.join(map(str, PRODUCTION_RESOLUTIONS))}")
    for name in ("desktop_max_leaves", "mobile_max_leaves"):
        if getattr(arguments, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return arguments


def inspect_triangle_primitives(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("The GLB header is truncated.")
        magic, version, total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError("Only GLB-encoded glTF 2.0 assets are supported.")
        if total_length != path.stat().st_size:
            raise ValueError("The GLB declared length does not match the file size.")
        json_document = None
        while handle.tell() < total_length:
            chunk_header = handle.read(8)
            if len(chunk_header) != 8:
                raise ValueError("A GLB chunk header is truncated.")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            payload = handle.read(chunk_length)
            if len(payload) != chunk_length:
                raise ValueError("A GLB chunk is truncated.")
            if chunk_type == 0x4E4F534A:
                json_document = json.loads(payload.rstrip(b" \t\r\n\0").decode("utf-8"))
        if json_document is None:
            raise ValueError("The GLB has no JSON scene chunk.")

    modes: dict[str, int] = {}
    unsupported: list[str] = []
    for mesh_index, mesh in enumerate(json_document.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            mode = int(primitive.get("mode", 4))
            modes[str(mode)] = modes.get(str(mode), 0) + 1
            if mode != 4:
                unsupported.append(f"mesh {mesh_index}, primitive {primitive_index}, mode {mode}")
    if unsupported:
        raise ValueError("Unsupported non-triangle GLB primitives: " + "; ".join(unsupported))
    return modes


def load_scene_triangles(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    primitive_modes = inspect_triangle_primitives(path)
    loaded = trimesh.load_scene(path, process=False)
    if not isinstance(loaded, trimesh.Scene):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded

    triangle_batches: list[np.ndarray] = []
    invalid_count = 0
    degenerate_count = 0
    instance_count = 0
    geometry_names: set[str] = set()

    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
            continue
        if geometry.faces.shape[1] != 3:
            raise ValueError(f"Node {node_name!r} contains non-triangle faces.")

        vertices = trimesh.transform_points(np.asarray(geometry.vertices, dtype=np.float64), transform)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        triangles = vertices[faces]
        finite = np.all(np.isfinite(triangles), axis=(1, 2))
        invalid_count += int(np.count_nonzero(~finite))
        triangles = triangles[finite]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        nondegenerate = np.einsum("ij,ij->i", cross, cross) > 1e-28
        degenerate_count += int(np.count_nonzero(~nondegenerate))
        triangles = triangles[nondegenerate]
        if len(triangles):
            triangle_batches.append(triangles)
            instance_count += 1
            geometry_names.add(str(geometry_name))

    if not triangle_batches:
        raise ValueError("No valid triangle geometry remains after loading the complete GLB scene.")

    triangles = np.concatenate(triangle_batches, axis=0)
    metadata = {
        "geometryCount": len(geometry_names),
        "nodeInstanceCount": instance_count,
        "triangleCount": int(len(triangles)),
        "invalidTriangleCount": invalid_count,
        "degenerateTriangleCount": degenerate_count,
        "primitiveModes": primitive_modes,
    }
    return triangles, metadata


def normalize_triangles(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    flattened = triangles.reshape((-1, 3))
    bounds_min = flattened.min(axis=0)
    bounds_max = flattened.max(axis=0)
    extent = bounds_max - bounds_min
    maximum_extent = float(np.max(extent))
    if not math.isfinite(maximum_extent) or maximum_extent <= 0:
        raise ValueError("The combined mesh bounds have no finite non-zero extent.")
    center = 0.5 * (bounds_min + bounds_max)
    scale = TARGET_EXTENT / maximum_extent
    normalized = (triangles - center) * scale
    normalized_flat = normalized.reshape((-1, 3))
    normalized_min = normalized_flat.min(axis=0)
    normalized_max = normalized_flat.max(axis=0)
    return normalized, {
        "originalBounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "normalizedBounds": {"min": normalized_min.tolist(), "max": normalized_max.tolist()},
        "translation": (-center).tolist(),
        "uniformScale": float(scale),
        "targetExtent": TARGET_EXTENT,
        "domain": [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    }


def triangle_aabb_hits(triangle: np.ndarray, centers: np.ndarray) -> np.ndarray:
    relative = triangle[None, :, :] - centers[:, None, :]
    active = np.ones(len(centers), dtype=bool)
    edges = (
        triangle[1] - triangle[0],
        triangle[2] - triangle[1],
        triangle[0] - triangle[2],
    )
    box_axes = np.eye(3, dtype=np.float64)
    axes = [np.cross(edges[0], edges[1])]
    axes.extend(np.cross(edge, axis) for edge in edges for axis in box_axes)

    for axis in axes:
        if float(np.dot(axis, axis)) <= SAT_EPSILON:
            continue
        radius = 0.5 * float(np.sum(np.abs(axis)))
        projection = np.einsum("nvc,c->nv", relative, axis)
        separated = (projection.min(axis=1) > radius + SAT_EPSILON) | (
            projection.max(axis=1) < -radius - SAT_EPSILON
        )
        active &= ~separated
        if not np.any(active):
            break
    return active


def voxelize_surface(triangles: np.ndarray, resolution: int, quiet: bool) -> np.ndarray:
    grid_triangles = (triangles - GRID_MINIMUM) * resolution
    occupied: set[int] = set()
    report_interval = max(1, len(grid_triangles) // 20)

    for triangle_index, triangle in enumerate(grid_triangles):
        lower = np.floor(triangle.min(axis=0) - 1e-9).astype(np.int64)
        upper = np.floor(triangle.max(axis=0) + 1e-9).astype(np.int64)
        lower = np.clip(lower, 0, resolution - 1)
        upper = np.clip(upper, 0, resolution - 1)
        dimensions = upper - lower + 1
        candidate_count = int(np.prod(dimensions, dtype=np.int64))
        yz_count = int(dimensions[1] * dimensions[2])

        for start in range(0, candidate_count, TRIANGLE_BLOCK_SIZE):
            stop = min(start + TRIANGLE_BLOCK_SIZE, candidate_count)
            linear = np.arange(start, stop, dtype=np.int64)
            local_x = linear // yz_count
            remainder = linear - local_x * yz_count
            local_y = remainder // int(dimensions[2])
            local_z = remainder - local_y * int(dimensions[2])
            coordinates = np.column_stack((local_x, local_y, local_z)) + lower
            centers = coordinates.astype(np.float64) + 0.5
            hits = triangle_aabb_hits(triangle, centers)
            hit_coordinates = coordinates[hits]
            codes = (
                hit_coordinates[:, 0] * resolution * resolution
                + hit_coordinates[:, 1] * resolution
                + hit_coordinates[:, 2]
            )
            occupied.update(int(code) for code in codes)

        if not quiet and ((triangle_index + 1) % report_interval == 0 or triangle_index + 1 == len(grid_triangles)):
            percent = 100 * (triangle_index + 1) / len(grid_triangles)
            print(f"  voxelization {percent:5.1f}% | sparse cells {len(occupied):,}", flush=True)

    codes = np.fromiter(sorted(occupied), dtype=np.uint64, count=len(occupied))
    x = codes // (resolution * resolution)
    remainder = codes - x * resolution * resolution
    y = remainder // resolution
    z = remainder - y * resolution
    return np.column_stack((x, y, z)).astype(np.uint16)


def build_variant(
    name: str,
    full_levels: dict[int, np.ndarray],
    budget: int,
    demo_limits: tuple[int, ...],
    ghost_budget: int,
    seed: int,
    common_manifest: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    finest_resolution = PRODUCTION_RESOLUTIONS[-1]
    full_leaves = full_levels[finest_resolution]
    sampling_applied = 0 < budget < len(full_leaves)
    leaves = sample_morton_stratified(
        full_leaves,
        budget,
        seed,
        finest_resolution,
    ) if sampling_applied else full_leaves
    levels = derive_hierarchy(leaves)
    masks = {
        resolution: build_child_masks(levels[resolution], levels[resolution * 2], resolution)
        for resolution in PRODUCTION_RESOLUTIONS[:-1]
    }
    demos: dict[int, np.ndarray] = {}
    ghost_counts: dict[int, int] = {}
    for transition_index, resolution in enumerate(PRODUCTION_RESOLUTIONS[:-1]):
        demos[resolution], ghost_counts[resolution] = select_demo_parents(
            levels[resolution],
            masks[resolution],
            int(demo_limits[transition_index]),
            ghost_budget,
            seed + resolution,
        )

    writer = BinaryWriter(len(PRODUCTION_RESOLUTIONS))
    manifest_levels: dict[str, object] = {}
    maximum_transition_instances = 0
    cascade_candidate_instances = 0
    for resolution in PRODUCTION_RESOLUTIONS:
        indices_range = writer.add_array(levels[resolution], "<u2")
        level_entry: dict[str, object] = {
            "count": int(len(levels[resolution])),
            "indices": {
                **indices_range,
                "components": 3,
                "type": "uint16",
            },
            "childMasks": None,
            "demoParents": None,
            "transition": None,
        }
        if resolution != finest_resolution:
            mask_range = writer.add_array(masks[resolution], "u1")
            demo_range = writer.add_array(demos[resolution], "<u4")
            child_count = int(len(levels[resolution * 2]))
            ghost_count = ghost_counts[resolution]
            transition_instances = int(len(levels[resolution]) * 8)
            maximum_transition_instances = max(maximum_transition_instances, transition_instances)
            cascade_candidate_instances += transition_instances
            level_entry["childMasks"] = {**mask_range, "type": "uint8"}
            level_entry["demoParents"] = {
                **demo_range,
                "count": int(len(demos[resolution])),
                "type": "uint32",
            }
            level_entry["transition"] = {
                "activeChildren": child_count,
                "candidateChildren": int(len(levels[resolution]) * 8),
                "demonstrationParentCount": int(len(demos[resolution])),
                "rejectedGhostCount": ghost_count,
                "retentionRatio": child_count / max(len(levels[resolution]) * 8, 1),
            }
        manifest_levels[str(resolution)] = level_entry

    binary_name = f"hero-voxels-{name}.bin"
    manifest_name = f"hero-voxels-{name}.json"
    binary_path = output_dir / binary_name
    manifest_path = output_dir / manifest_name
    payload = writer.finish()
    binary_path.write_bytes(payload)

    manifest = {
        **common_manifest,
        "variant": name,
        "budget": {
            "finestLeaves": int(budget) if budget > 0 else None,
            "ghostInstancesPerTransition": int(ghost_budget),
        },
        "renderCounts": {str(resolution): int(len(levels[resolution])) for resolution in PRODUCTION_RESOLUTIONS},
        "sampling": {
            "applied": sampling_applied,
            "componentCoverage": (
                {
                    "available": False,
                    "reason": "Sparse connected-component labeling is intentionally omitted from the minimal dependency pipeline.",
                }
                if sampling_applied
                else {
                    "available": True,
                    "retained": "all active finest-level voxels",
                }
            ),
            "method": "Morton-order stratified selection" if sampling_applied else "none; full active indices retained at every displayed level",
            "seed": int(seed),
        },
        "binary": {
            "byteLength": len(payload),
            "headerByteLength": 16,
            "littleEndian": True,
            "sha256": sha256_file(binary_path),
            "url": binary_name,
        },
        "levels": manifest_levels,
        "runtime": {
            "maximumTransitionInstanceCount": maximum_transition_instances,
            "cascadeCandidateInstanceCount": cascade_candidate_instances,
            "estimatedGpuAttributeBytes": (
                cascade_candidate_instances * 7
                + PRODUCTION_RESOLUTIONS[0] ** 3 * 7
            ),
        },
    }
    write_deterministic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    arguments = parse_arguments()
    input_path = arguments.input.resolve()
    output_dir = arguments.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input GLB does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}")
    triangles, mesh_metadata = load_scene_triangles(input_path)
    normalized_triangles, normalization = normalize_triangles(triangles)
    print(
        f"Loaded {mesh_metadata['triangleCount']:,} valid triangles from "
        f"{mesh_metadata['nodeInstanceCount']} visible node instances."
    )
    print(f"Conservative surface voxelization at {arguments.reference_resolution}^3")
    reference_indices = voxelize_surface(
        normalized_triangles,
        arguments.reference_resolution,
        arguments.quiet,
    )
    leaves_256 = collapse_to_resolution(
        reference_indices,
        arguments.reference_resolution,
        256,
    )
    full_finest = collapse_to_resolution(
        leaves_256,
        256,
        PRODUCTION_RESOLUTIONS[-1],
    )
    full_levels = derive_hierarchy(full_finest)

    common_manifest = {
        "schema": "hero-voxel-hierarchy",
        "version": VERSION,
        "axisOrder": "xyz",
        "axisConvention": "glTF 2.0 right-handed source coordinates; normalized xyz coordinates are stored without axis permutation",
        "referenceResolution": int(arguments.reference_resolution),
        "productionResolutions": list(PRODUCTION_RESOLUTIONS),
        "source": {
            "file": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "normalization": normalization,
        "mesh": mesh_metadata,
        "fullCounts": {str(resolution): int(len(full_levels[resolution])) for resolution in PRODUCTION_RESOLUTIONS},
        "hierarchy": {
            "childBit": "dx * 4 + dy * 2 + dz",
            "derivation": "The reference is collapsed to 256, then to the displayed 128 finest level; every coarser level is unique(finer // 2).",
        },
        "voxelization": {
            "method": "Conservative triangle-AABB overlap using box, triangle-plane, and nine edge-cross-axis SAT tests",
            "surfaceOnly": True,
        },
    }

    desktop = build_variant(
        "desktop",
        full_levels,
        arguments.desktop_max_leaves,
        tuple(arguments.desktop_demo_parents),
        arguments.desktop_ghost_budget,
        arguments.seed,
        common_manifest,
        output_dir,
    )
    mobile = build_variant(
        "mobile",
        full_levels,
        arguments.mobile_max_leaves,
        tuple(arguments.mobile_demo_parents),
        arguments.mobile_ghost_budget,
        arguments.seed,
        common_manifest,
        output_dir,
    )

    print("Generated deterministic hierarchy assets:")
    for manifest in (desktop, mobile):
        counts = ", ".join(f"{resolution}:{manifest['renderCounts'][str(resolution)]:,}" for resolution in PRODUCTION_RESOLUTIONS)
        print(f"  {manifest['variant']}: {counts} | {manifest['binary']['byteLength']:,} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
