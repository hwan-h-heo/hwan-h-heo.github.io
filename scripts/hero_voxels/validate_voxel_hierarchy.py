#!/usr/bin/env python3
"""Validate a generated hero voxel manifest and compact binary payload."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from hierarchy_common import (
    HEADER_SIZE,
    MAGIC,
    PRODUCTION_RESOLUTIONS,
    VERSION,
    coordinate_codes,
    expand_masks,
    lexsort_unique,
    sha256_file,
    valid_sha256,
)


DTYPES = {
    "uint8": np.dtype("u1"),
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--compare-manifest", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_array(payload: bytes, descriptor: dict[str, object], count: int, components: int = 1) -> np.ndarray:
    dtype_name = str(descriptor["type"])
    require(dtype_name in DTYPES, f"Unsupported array type {dtype_name!r}.")
    dtype = DTYPES[dtype_name]
    offset = int(descriptor["offset"])
    byte_length = int(descriptor["byteLength"])
    expected_length = count * components * dtype.itemsize
    require(offset % 4 == 0, f"Array offset {offset} is not 4-byte aligned.")
    require(byte_length == expected_length, f"Array at {offset} has byte length {byte_length}, expected {expected_length}.")
    require(offset >= HEADER_SIZE and offset + byte_length <= len(payload), f"Array at {offset} is outside the binary payload.")
    array = np.frombuffer(payload, dtype=dtype, count=count * components, offset=offset)
    return array.reshape((count, components)) if components != 1 else array


def load_and_validate(manifest_path: Path, print_report: bool = True) -> tuple[dict[str, object], bytes]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "hero-voxel-hierarchy", "Unknown manifest schema.")
    require(manifest.get("version") == VERSION, f"Manifest version must be {VERSION}.")
    require(manifest.get("axisOrder") == "xyz", "Axis order must be xyz.")
    require(tuple(manifest.get("productionResolutions", [])) == PRODUCTION_RESOLUTIONS, "Production resolutions are incomplete or reordered.")
    require(manifest.get("referenceResolution") in (256, 512), "Reference resolution must be 256 or 512.")

    binary = manifest["binary"]
    payload_path = manifest_path.parent / str(binary["url"])
    payload = payload_path.read_bytes()
    require(len(payload) == int(binary["byteLength"]), "Binary byte length does not match the manifest.")
    require(len(payload) >= HEADER_SIZE, "Binary payload is shorter than its header.")
    magic, version, level_count, header_size = struct.unpack_from("<4sIII", payload, 0)
    require(magic == MAGIC, "Binary magic is invalid.")
    require(version == VERSION, "Binary version is invalid.")
    require(level_count == len(PRODUCTION_RESOLUTIONS), "Binary level count is invalid.")
    require(header_size == HEADER_SIZE, "Binary header size is invalid.")
    require(bool(binary.get("littleEndian")), "Binary payload must declare little-endian encoding.")
    require(valid_sha256(binary.get("sha256")), "Generated binary SHA-256 is malformed.")
    require(sha256_file(payload_path) == binary["sha256"], "Generated binary SHA-256 does not match.")
    require(valid_sha256(manifest.get("source", {}).get("sha256")), "Source SHA-256 is malformed.")
    source_candidate = manifest_path.parent.parent / str(manifest["source"]["file"])
    if source_candidate.is_file():
        require(sha256_file(source_candidate) == manifest["source"]["sha256"], "Source GLB SHA-256 does not match.")

    levels: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    maximum_instances = 0
    cascade_candidate_instances = 0
    finest_resolution = PRODUCTION_RESOLUTIONS[-1]
    for resolution in PRODUCTION_RESOLUTIONS:
        entry = manifest["levels"].get(str(resolution))
        require(isinstance(entry, dict), f"Missing level {resolution}.")
        count = int(entry["count"])
        indices_descriptor = entry["indices"]
        require(indices_descriptor.get("type") == "uint16", f"Level {resolution} indices must be Uint16.")
        require(int(indices_descriptor.get("components", 0)) == 3, f"Level {resolution} indices must have three components.")
        coordinates = read_array(payload, indices_descriptor, count, 3)
        require(np.all(coordinates < resolution), f"Level {resolution} has out-of-range coordinates.")
        codes = coordinate_codes(coordinates, resolution)
        require(len(codes) == 0 or np.all(codes[1:] > codes[:-1]), f"Level {resolution} is not unique and lexicographically sorted.")
        require(np.array_equal(coordinates, lexsort_unique(coordinates)), f"Level {resolution} coordinate canonicalization failed.")
        levels[resolution] = coordinates

        if resolution == finest_resolution:
            require(entry.get("childMasks") is None, "The final level must not contain child masks.")
            continue

        mask_descriptor = entry.get("childMasks")
        require(isinstance(mask_descriptor, dict) and mask_descriptor.get("type") == "uint8", f"Level {resolution} child masks must be Uint8.")
        masks[resolution] = read_array(payload, mask_descriptor, count)
        require(np.all(masks[resolution] > 0), f"Level {resolution} contains an empty active-parent child mask.")

        demo_descriptor = entry.get("demoParents")
        require(isinstance(demo_descriptor, dict) and demo_descriptor.get("type") == "uint32", f"Level {resolution} demonstration indices must be Uint32.")
        demo_count = int(demo_descriptor["count"])
        demos = read_array(payload, demo_descriptor, demo_count)
        require(np.all(demos < count), f"Level {resolution} has an out-of-range demonstration parent.")
        require(len(demos) == 0 or np.all(demos[1:] > demos[:-1]), f"Level {resolution} demonstration parents are not sorted and unique.")
        populations = np.fromiter((int(value).bit_count() for value in masks[resolution][demos]), dtype=np.uint8, count=len(demos))
        require(np.all((populations >= 2) & (populations <= 5)), f"Level {resolution} has an uninformative demonstration mask.")
        ghost_count = int(np.sum(8 - populations))
        transition = entry["transition"]
        require(ghost_count == int(transition["rejectedGhostCount"]), f"Level {resolution} ghost count is inconsistent.")
        require(demo_count == int(transition["demonstrationParentCount"]), f"Level {resolution} demonstration count is inconsistent.")
        transition_instances = count * 8
        maximum_instances = max(maximum_instances, transition_instances)
        cascade_candidate_instances += transition_instances

    for resolution in PRODUCTION_RESOLUTIONS[:-1]:
        expected_parents = lexsort_unique(levels[resolution * 2] // 2)
        require(np.array_equal(levels[resolution], expected_parents), f"Level {resolution} is not exactly unique(level {resolution * 2} // 2).")
        expanded = expand_masks(levels[resolution], masks[resolution])
        require(np.array_equal(expanded, levels[resolution * 2]), f"Level {resolution} child masks do not reconstruct level {resolution * 2}.")

    finest_budget_value = manifest["budget"]["finestLeaves"]
    ghost_budget = int(manifest["budget"]["ghostInstancesPerTransition"])
    visibility = manifest.get("visibilityCulling", {"applied": False})
    visibility_applied = bool(visibility.get("applied"))
    if finest_budget_value is not None:
        require(
            len(levels[finest_resolution]) <= int(finest_budget_value),
            "Finest retained leaf count exceeds its budget.",
        )
    if visibility_applied:
        require(
            int(visibility.get("inputFinestCount", -1))
            == int(manifest["fullCounts"][str(finest_resolution)]),
            "Visibility input count does not match the full finest level.",
        )
        require(
            int(visibility.get("retainedFinestCount", -1)) == len(levels[finest_resolution]),
            "Visibility retained count does not match the rendered finest level.",
        )
        require(
            0 < len(levels[finest_resolution]) <= int(manifest["fullCounts"][str(finest_resolution)]),
            "Visibility culling retained an invalid finest-level count.",
        )
    if not bool(manifest["sampling"]["applied"]) and not visibility_applied:
        require(
            all(
                int(manifest["renderCounts"][str(resolution)])
                == int(manifest["fullCounts"][str(resolution)])
                for resolution in PRODUCTION_RESOLUTIONS
            ),
            "A full-level hierarchy must retain every active coordinate.",
        )
    for resolution in PRODUCTION_RESOLUTIONS:
        require(
            int(manifest["renderCounts"][str(resolution)])
            <= int(manifest["fullCounts"][str(resolution)]),
            f"Level {resolution} retains more coordinates than the complete hierarchy.",
        )
    for resolution in PRODUCTION_RESOLUTIONS[:-1]:
        require(int(manifest["levels"][str(resolution)]["transition"]["rejectedGhostCount"]) <= ghost_budget, f"Level {resolution} ghost count exceeds its budget.")
    require(maximum_instances == int(manifest["runtime"]["maximumTransitionInstanceCount"]), "Maximum transition instance count is inconsistent.")
    require(cascade_candidate_instances == int(manifest["runtime"]["cascadeCandidateInstanceCount"]), "Total cascade candidate count is inconsistent.")

    if print_report:
        print(f"validated {manifest_path.name} ({manifest['variant']})")
        print("level | active parents | candidate children | active children | retention ratio")
        for resolution in PRODUCTION_RESOLUTIONS:
            if resolution == finest_resolution:
                print(f"{resolution:>5} | {len(levels[resolution]):>14,} | {'-':>18} | {'-':>15} | {'-':>15}")
                continue
            active_children = len(levels[resolution * 2])
            candidates = len(levels[resolution]) * 8
            print(f"{resolution:>5} | {len(levels[resolution]):>14,} | {candidates:>18,} | {active_children:>15,} | {active_children / max(candidates, 1):>14.2%}")
        full_counts = ", ".join(
            f"{resolution}:{manifest['fullCounts'][str(resolution)]:,}"
            for resolution in PRODUCTION_RESOLUTIONS
        )
        retained_counts = ", ".join(
            f"{resolution}:{manifest['renderCounts'][str(resolution)]:,}"
            for resolution in PRODUCTION_RESOLUTIONS
        )
        print(f"full counts: {full_counts}")
        print(f"retained counts: {retained_counts}")
        print(f"binary size: {len(payload):,} bytes")
        print(f"maximum transition instances: {maximum_instances:,}")
        print(f"total resident cascade candidates: {cascade_candidate_instances:,}")
        print(f"estimated GPU attribute size: {int(manifest['runtime']['estimatedGpuAttributeBytes']):,} bytes")

    return manifest, payload


def main() -> int:
    arguments = parse_arguments()
    manifest, payload = load_and_validate(arguments.manifest)
    if arguments.compare_manifest:
        comparison_manifest, comparison_payload = load_and_validate(arguments.compare_manifest, print_report=False)
        require(arguments.manifest.read_bytes() == arguments.compare_manifest.read_bytes(), "Repeated-generation manifests are not byte-identical.")
        require(payload == comparison_payload, "Repeated-generation binaries are not byte-identical.")
        require(manifest == comparison_manifest, "Repeated-generation manifest values differ.")
        print("repeated generation: manifest and binary are byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
