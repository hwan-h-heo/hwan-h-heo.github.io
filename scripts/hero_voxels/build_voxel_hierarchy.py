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
WORLD_SCALE = 4.05
CAMERA_AZIMUTH_DEGREES = 43.0
CAMERA_ELEVATION_DEGREES = 26.5
CAMERA_ORBIT_LIMIT_DEGREES = 10.0
CAMERA_ORBIT_SAMPLES_DEGREES = (-10.0, 0.0, 10.0)
CAMERA_PROFILES = {
    "desktop": {
        "distance": 9.8,
        "distanceEnvelope": 1.2,
        "fov": 41.5,
        "verticalCenter": -0.45,
        "layouts": (
            {"screenOffset": 0.0, "aspect": 16 / 9},
        ),
    },
    "mobile": {
        "distance": 9.8,
        "distanceEnvelope": 1.2,
        "fov": 41.5,
        "verticalCenter": -0.45,
        "layouts": (
            {"screenOffset": 0.0, "aspect": 16 / 9},
            {"screenOffset": 0.0, "aspect": 4 / 3},
        ),
    },
}


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
    parser.add_argument(
        "--camera-visible-only",
        action="store_true",
        help="Retain only finest voxels visible across the runtime camera-orbit envelope.",
    )
    parser.add_argument("--visibility-raster-height", type=int, default=512)
    parser.add_argument("--visibility-depth-layers", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    if tuple(arguments.resolutions) != PRODUCTION_RESOLUTIONS:
        parser.error(f"--resolutions must be exactly {' '.join(map(str, PRODUCTION_RESOLUTIONS))}")
    for name in ("desktop_max_leaves", "mobile_max_leaves"):
        if getattr(arguments, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if arguments.visibility_raster_height < 64:
        parser.error("--visibility-raster-height must be at least 64")
    if arguments.visibility_depth_layers < 0:
        parser.error("--visibility-depth-layers cannot be negative")
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


def camera_visibility_profile(name: str) -> dict[str, object]:
    settings = CAMERA_PROFILES[name]
    azimuth = math.radians(CAMERA_AZIMUTH_DEGREES)
    elevation = math.radians(CAMERA_ELEVATION_DEGREES)
    vertical_center = float(settings["verticalCenter"])
    target = np.array((0.0, vertical_center, 0.0), dtype=np.float64)
    distance = float(settings["distance"])
    distance_envelope = float(settings["distanceEnvelope"])
    horizontal_distance = distance * math.cos(elevation)
    base_camera = np.array((
        horizontal_distance * math.sin(azimuth),
        distance * math.sin(elevation),
        horizontal_distance * math.cos(azimuth),
    ), dtype=np.float64)
    base_offset = base_camera - target
    orbit_radius = float(np.linalg.norm(base_offset))
    orbit_azimuth = math.atan2(float(base_offset[0]), float(base_offset[2]))
    orbit_elevation = math.asin(float(base_offset[1]) / orbit_radius)
    views: list[dict[str, object]] = []
    for layout in settings["layouts"]:
        screen_offset = float(layout["screenOffset"])
        group = np.array((
            math.cos(azimuth) * screen_offset,
            vertical_center,
            -math.sin(azimuth) * screen_offset,
        ), dtype=np.float64)
        cameras = []
        for azimuth_offset in CAMERA_ORBIT_SAMPLES_DEGREES:
            for elevation_offset in CAMERA_ORBIT_SAMPLES_DEGREES:
                sample_azimuth = orbit_azimuth + math.radians(azimuth_offset)
                sample_elevation = orbit_elevation + math.radians(elevation_offset)
                for radius_offset in (-distance_envelope, 0.0, distance_envelope):
                    sample_radius = orbit_radius + radius_offset
                    sample_horizontal = sample_radius * math.cos(sample_elevation)
                    cameras.append(target + np.array((
                        sample_horizontal * math.sin(sample_azimuth),
                        sample_radius * math.sin(sample_elevation),
                        sample_horizontal * math.cos(sample_azimuth),
                    ), dtype=np.float64))
        views.append({
            "aspect": float(layout["aspect"]),
            "screenOffset": screen_offset,
            "groupPosition": group,
            "cameraPositions": cameras,
        })
    return {
        "settings": settings,
        "views": views,
        "lookTarget": target,
    }


def visible_voxels_for_camera(
    indices: np.ndarray,
    resolution: int,
    variant: str,
    raster_height: int,
    depth_layers: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Conservatively keep voxel AABBs visible in a small runtime camera envelope."""
    profile = camera_visibility_profile(variant)
    settings = profile["settings"]
    half_cell = WORLD_SCALE / resolution * 0.5
    local_centers = (
        (indices.astype(np.float64) + 0.5) / resolution - 0.5
    ) * WORLD_SCALE
    corner_signs = np.array(
        [
            (dx, dy, dz)
            for dx in (-1.0, 1.0)
            for dy in (-1.0, 1.0)
            for dz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    local_corners = local_centers[:, None, :] + corner_signs[None, :, :] * half_cell
    retained = np.zeros(len(indices), dtype=bool)
    tangent = math.tan(math.radians(float(settings["fov"])) * 0.5)
    depth_slack = depth_layers * WORLD_SCALE / resolution
    raster_sizes: list[list[int]] = []
    camera_sample_count = 0

    for view in profile["views"]:
        aspect = float(view["aspect"])
        raster_width = max(64, int(round(raster_height * aspect)))
        raster_sizes.append([raster_width, raster_height])
        corners = local_corners + view["groupPosition"][None, None, :]
        for camera in view["cameraPositions"]:
            camera_sample_count += 1
            forward = profile["lookTarget"] - camera
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float64))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            relative = corners - camera[None, None, :]
            depth = np.einsum("nvc,c->nv", relative, forward)
            projected_x = np.einsum("nvc,c->nv", relative, right) / (
                np.maximum(depth, 1e-8) * tangent * aspect
            )
            projected_y = np.einsum("nvc,c->nv", relative, up) / (
                np.maximum(depth, 1e-8) * tangent
            )
            lower_x = np.floor((projected_x.min(axis=1) * 0.5 + 0.5) * raster_width).astype(np.int64)
            upper_x = np.ceil((projected_x.max(axis=1) * 0.5 + 0.5) * raster_width).astype(np.int64) - 1
            lower_y = np.floor((projected_y.min(axis=1) * 0.5 + 0.5) * raster_height).astype(np.int64)
            upper_y = np.ceil((projected_y.max(axis=1) * 0.5 + 0.5) * raster_height).astype(np.int64) - 1
            near_depth = depth.min(axis=1)
            valid = (
                (near_depth > 0)
                & (upper_x >= 0)
                & (lower_x < raster_width)
                & (upper_y >= 0)
                & (lower_y < raster_height)
            )
            lower_x = np.clip(lower_x, 0, raster_width - 1)
            upper_x = np.clip(upper_x, 0, raster_width - 1)
            lower_y = np.clip(lower_y, 0, raster_height - 1)
            upper_y = np.clip(upper_y, 0, raster_height - 1)
            z_buffer = np.full((raster_height, raster_width), np.inf, dtype=np.float64)
            visible_candidates = np.flatnonzero(valid)

            for index in visible_candidates:
                region = z_buffer[
                    lower_y[index]:upper_y[index] + 1,
                    lower_x[index]:upper_x[index] + 1,
                ]
                np.minimum(region, near_depth[index], out=region)

            for index in visible_candidates:
                region = z_buffer[
                    lower_y[index]:upper_y[index] + 1,
                    lower_x[index]:upper_x[index] + 1,
                ]
                if np.any(near_depth[index] <= region + depth_slack):
                    retained[index] = True

    visible = np.asarray(indices[retained], dtype=np.uint16)
    if len(visible) == 0:
        raise ValueError(f"Camera visibility culling removed every {variant} finest voxel.")
    metadata = {
        "applied": True,
        "method": "Conservative perspective z-buffer over projected finest-level voxel AABBs across a bounded camera-orbit envelope",
        "rasterSizes": raster_sizes,
        "depthSlackVoxels": float(depth_layers),
        "cameraSampleCount": camera_sample_count,
        "camera": {
            "azimuthDegrees": CAMERA_AZIMUTH_DEGREES,
            "elevationDegrees": CAMERA_ELEVATION_DEGREES,
            "distance": float(settings["distance"]),
            "distanceEnvelope": float(settings["distanceEnvelope"]),
            "fovDegrees": float(settings["fov"]),
            "screenOffset": float(settings["layouts"][0]["screenOffset"]),
            "verticalCenter": float(settings["verticalCenter"]),
            "layouts": [
                {
                    "aspect": float(layout["aspect"]),
                    "screenOffset": float(layout["screenOffset"]),
                }
                for layout in settings["layouts"]
            ],
            "orbitLimitDegrees": CAMERA_ORBIT_LIMIT_DEGREES,
            "orbitSamplesDegrees": list(CAMERA_ORBIT_SAMPLES_DEGREES),
        },
        "inputFinestCount": int(len(indices)),
        "retainedFinestCount": int(len(visible)),
        "retentionRatio": len(visible) / max(len(indices), 1),
    }
    return visible, metadata


def build_variant(
    name: str,
    full_levels: dict[int, np.ndarray],
    budget: int,
    demo_limits: tuple[int, ...],
    ghost_budget: int,
    seed: int,
    common_manifest: dict[str, object],
    output_dir: Path,
    camera_visible_only: bool,
    visibility_raster_height: int,
    visibility_depth_layers: float,
) -> dict[str, object]:
    finest_resolution = PRODUCTION_RESOLUTIONS[-1]
    full_leaves = full_levels[finest_resolution]
    if camera_visible_only:
        render_leaves, visibility = visible_voxels_for_camera(
            full_leaves,
            finest_resolution,
            name,
            visibility_raster_height,
            visibility_depth_layers,
        )
    else:
        render_leaves = full_leaves
        visibility = {
            "applied": False,
            "method": "none; complete surface retained",
            "inputFinestCount": int(len(full_leaves)),
            "retainedFinestCount": int(len(full_leaves)),
            "retentionRatio": 1.0,
        }
    sampling_applied = 0 < budget < len(render_leaves)
    leaves = sample_morton_stratified(
        render_leaves,
        budget,
        seed,
        finest_resolution,
    ) if sampling_applied else render_leaves
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
                    "retained": (
                        "all camera-visible finest-level voxels"
                        if camera_visible_only
                        else "all active finest-level voxels"
                    ),
                }
            ),
            "method": "Morton-order stratified selection" if sampling_applied else "none; every selected finest index retained",
            "seed": int(seed),
        },
        "visibilityCulling": visibility,
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
            "derivation": (
                "The reference is collapsed to 256 and then 128; camera-visible 128 leaves are selected per variant, and every rendered coarser level is unique(finer // 2)."
                if arguments.camera_visible_only
                else "The reference is collapsed to 256, then to the displayed 128 finest level; every coarser level is unique(finer // 2)."
            ),
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
        arguments.camera_visible_only,
        arguments.visibility_raster_height,
        arguments.visibility_depth_layers,
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
        arguments.camera_visible_only,
        arguments.visibility_raster_height,
        arguments.visibility_depth_layers,
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
