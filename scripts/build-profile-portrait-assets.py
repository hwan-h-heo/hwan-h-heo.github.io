#!/usr/bin/env python3
"""Build the cleaned portrait and masked scalar depth assets used on the home page."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLOR_SOURCE = REPO_ROOT / "assets" / "profile4.png"
DEFAULT_ALPHA_SOURCE = REPO_ROOT / "assets" / "profile4-author.png"
DEFAULT_PORTRAIT_OUTPUT = REPO_ROOT / "assets" / "profile4-portrait.png"
DEFAULT_DEPTH_OUTPUT = REPO_ROOT / "assets" / "profile4-depth.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--color-source", type=Path, default=DEFAULT_COLOR_SOURCE)
    parser.add_argument("--alpha-source", type=Path, default=DEFAULT_ALPHA_SOURCE)
    parser.add_argument("--portrait-output", type=Path, default=DEFAULT_PORTRAIT_OUTPUT)
    parser.add_argument("--depth-output", type=Path, default=DEFAULT_DEPTH_OUTPUT)
    return parser.parse_args()


def largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("The alpha source does not contain a foreground component.")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255


def fill_enclosed_holes(binary: np.ndarray) -> np.ndarray:
    flood = binary.copy()
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return cv2.bitwise_or(binary, cv2.bitwise_not(flood))


def repair_alpha(alpha: np.ndarray) -> tuple[np.ndarray, int]:
    original_foreground = (alpha >= 8).astype(np.uint8) * 255
    component = largest_component(original_foreground)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = fill_enclosed_holes(closed)
    safe_interior = cv2.erode(filled, kernel, iterations=1) > 0

    repaired = np.where(component > 0, alpha, 0).astype(np.uint8)
    before = repaired.copy()
    repaired[safe_interior] = 255
    repaired[filled == 0] = 0
    repaired = np.clip((repaired.astype(np.float32) - 12.0) * (255.0 / 243.0), 0, 255).astype(np.uint8)
    repaired_pixels = int(np.count_nonzero(repaired != before))
    return repaired, repaired_pixels


def resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(alpha, mode="L").resize(size, Image.Resampling.LANCZOS)
    return np.array(resized, dtype=np.uint8)


def inverse_turbo(depth_rgb: np.ndarray) -> np.ndarray:
    values = np.arange(256, dtype=np.uint8).reshape(256, 1)
    turbo_rgb = cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(256, 3)[:, ::-1].astype(np.int32)
    scalar = np.empty(depth_rgb.shape[:2], dtype=np.uint8)

    for y, row in enumerate(depth_rgb.astype(np.int32)):
        distances = ((row[:, None, :] - turbo_rgb[None, :, :]) ** 2).sum(axis=2)
        scalar[y] = distances.argmin(axis=1).astype(np.uint8)

    return cv2.GaussianBlur(scalar, (0, 0), sigmaX=1.15, sigmaY=1.15)


def normalize_depth(depth: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, float, float]:
    foreground = alpha >= 128
    values = depth[foreground]
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((depth.astype(np.float32) - low) / max(high - low, 1.0), 0, 1)
    normalized[~foreground] = 0
    return (normalized * 255).astype(np.uint8), float(low), float(high)


def save_assets(args: argparse.Namespace) -> None:
    color = np.array(Image.open(args.color_source).convert("RGB"), dtype=np.uint8)
    alpha_source = np.array(Image.open(args.alpha_source).convert("RGBA"), dtype=np.uint8)[..., 3]
    repaired_alpha, repaired_pixels = repair_alpha(alpha_source)
    alpha = resize_alpha(repaired_alpha, (color.shape[1], color.shape[0]))

    portrait = np.dstack([color, alpha])
    args.portrait_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(portrait, mode="RGBA").save(args.portrait_output, optimize=True)

    depth_rgb = np.array(Image.open(args.depth_source).convert("RGB"), dtype=np.uint8)
    if depth_rgb.shape[:2] != color.shape[:2]:
        depth_rgb = cv2.resize(depth_rgb, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_CUBIC)
    scalar = inverse_turbo(depth_rgb)
    scalar, low, high = normalize_depth(scalar, alpha)
    depth_asset = np.dstack([scalar, scalar, scalar, alpha])
    args.depth_output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_asset, mode="RGBA").save(args.depth_output, optimize=True)

    print(f"Repaired alpha pixels: {repaired_pixels}")
    print(f"Normalized inverse-Turbo depth range: {low:.1f}–{high:.1f}")
    print(f"Wrote portrait: {args.portrait_output}")
    print(f"Wrote depth: {args.depth_output}")


if __name__ == "__main__":
    save_assets(parse_args())
