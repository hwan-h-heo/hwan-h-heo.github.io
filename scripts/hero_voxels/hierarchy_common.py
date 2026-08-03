"""Shared deterministic hierarchy and binary-format helpers."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


MAGIC = b"HVOX"
VERSION = 2
HEADER_SIZE = 16
PRODUCTION_RESOLUTIONS = (8, 16, 32, 64, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexsort_unique(indices: np.ndarray) -> np.ndarray:
    array = np.asarray(indices, dtype=np.uint16).reshape((-1, 3))
    if len(array) == 0:
        return array
    order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
    ordered = array[order]
    keep = np.ones(len(ordered), dtype=bool)
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return ordered[keep]


def coordinate_codes(indices: np.ndarray, resolution: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    return values[:, 0] * resolution * resolution + values[:, 1] * resolution + values[:, 2]


def derive_hierarchy(
    finest_indices: np.ndarray,
    resolutions: tuple[int, ...] = PRODUCTION_RESOLUTIONS,
) -> dict[int, np.ndarray]:
    finest_resolution = resolutions[-1]
    levels: dict[int, np.ndarray] = {
        finest_resolution: lexsort_unique(finest_indices),
    }
    for resolution in reversed(resolutions[:-1]):
        if resolution * 2 not in levels:
            raise ValueError("Hierarchy resolutions must increase by powers of two.")
        levels[resolution] = lexsort_unique(levels[resolution * 2] // 2)
    return {resolution: levels[resolution] for resolution in resolutions}


def collapse_to_resolution(
    indices: np.ndarray,
    source_resolution: int,
    target_resolution: int,
) -> np.ndarray:
    if source_resolution < target_resolution or source_resolution % target_resolution != 0:
        raise ValueError("Target resolution must evenly divide the source resolution.")
    factor = source_resolution // target_resolution
    if factor & (factor - 1):
        raise ValueError("Resolution collapse factor must be a power of two.")
    return lexsort_unique(indices // factor)


def build_child_masks(parents: np.ndarray, children: np.ndarray, parent_resolution: int) -> np.ndarray:
    parent_codes = coordinate_codes(parents, parent_resolution)
    child_parents = children // 2
    child_parent_codes = coordinate_codes(child_parents, parent_resolution)
    positions = np.searchsorted(parent_codes, child_parent_codes)
    if np.any(positions >= len(parents)) or np.any(parent_codes[positions] != child_parent_codes):
        raise ValueError(f"A {parent_resolution * 2}-level child has no active {parent_resolution}-level parent.")

    offsets = (children % 2).astype(np.uint8)
    # Shared Python/JavaScript bit convention: dx * 4 + dy * 2 + dz.
    bits = offsets[:, 0] * 4 + offsets[:, 1] * 2 + offsets[:, 2]
    masks = np.zeros(len(parents), dtype=np.uint8)
    np.bitwise_or.at(masks, positions, np.left_shift(np.uint8(1), bits))
    return masks


def morton_codes(indices: np.ndarray, bits: int = 8) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    codes = np.zeros(len(values), dtype=np.uint64)
    for bit in range(bits):
        codes |= ((values[:, 0] >> bit) & 1) << (3 * bit + 2)
        codes |= ((values[:, 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((values[:, 2] >> bit) & 1) << (3 * bit)
    return codes


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def stratified_positions(length: int, count: int, seed: int) -> np.ndarray:
    if count >= length:
        return np.arange(length, dtype=np.int64)
    selected = np.empty(count, dtype=np.int64)
    for stratum in range(count):
        start = (stratum * length) // count
        end = ((stratum + 1) * length) // count
        selected[stratum] = start + splitmix64(seed ^ stratum) % max(end - start, 1)
    return selected


def sample_morton_stratified(
    indices: np.ndarray,
    budget: int,
    seed: int,
    resolution: int,
) -> np.ndarray:
    if len(indices) <= budget:
        return np.asarray(indices, dtype=np.uint16).copy()
    morton = morton_codes(indices)
    lex_codes = coordinate_codes(indices, resolution)
    order = np.lexsort((lex_codes, morton))
    positions = stratified_positions(len(indices), budget, seed)
    return lexsort_unique(indices[order[positions]])


def select_demo_parents(
    parents: np.ndarray,
    masks: np.ndarray,
    maximum: int,
    ghost_budget: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    populations = np.fromiter((int(value).bit_count() for value in masks), dtype=np.uint8, count=len(masks))
    informative = np.flatnonzero((populations >= 2) & (populations <= 5))
    if len(informative) == 0 or maximum <= 0 or ghost_budget <= 0:
        return np.empty(0, dtype=np.uint32), 0

    parent_morton = morton_codes(parents[informative], bits=8)
    order = np.lexsort((informative, parent_morton))
    candidates = informative[order]
    target = min(maximum, len(candidates))

    while target > 0:
        positions = stratified_positions(len(candidates), target, seed)
        selected = np.sort(candidates[positions]).astype(np.uint32)
        ghost_count = int(np.sum(8 - populations[selected]))
        if ghost_count <= ghost_budget:
            return selected, ghost_count
        target = min(target - 1, max(0, int(target * ghost_budget / max(ghost_count, 1))))

    return np.empty(0, dtype=np.uint32), 0


class BinaryWriter:
    def __init__(self, level_count: int) -> None:
        self.data = bytearray(struct.pack("<4sIII", MAGIC, VERSION, level_count, HEADER_SIZE))

    def align(self, alignment: int = 4) -> None:
        padding = (-len(self.data)) % alignment
        if padding:
            self.data.extend(b"\0" * padding)

    def add_array(self, array: np.ndarray, dtype: str) -> dict[str, int | str]:
        self.align(4)
        offset = len(self.data)
        encoded = np.asarray(array, dtype=np.dtype(dtype).newbyteorder("<")).tobytes(order="C")
        self.data.extend(encoded)
        return {
            "offset": offset,
            "byteLength": len(encoded),
        }

    def finish(self) -> bytes:
        self.align(4)
        return bytes(self.data)


def write_deterministic_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def expand_masks(parents: np.ndarray, masks: np.ndarray) -> np.ndarray:
    children: list[tuple[int, int, int]] = []
    for parent, mask in zip(parents, masks, strict=True):
        px, py, pz = (int(value) for value in parent)
        for bit in range(8):
            if int(mask) & (1 << bit):
                children.append((
                    px * 2 + ((bit >> 2) & 1),
                    py * 2 + ((bit >> 1) & 1),
                    pz * 2 + (bit & 1),
                ))
    return lexsort_unique(np.asarray(children, dtype=np.uint16).reshape((-1, 3)))


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def popcount_sum(values: Iterable[int]) -> int:
    return sum(int(value).bit_count() for value in values)
