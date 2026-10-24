"""Render enlarged static variants of the article's original Manim scenes.

Run with the original project's Manim environment. QUAD_REMESH_MANIM_SOURCE
can point to another checkout of quad_remesh_blog. No encoded video is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCE_DIR = Path(os.environ.get(
    "QUAD_REMESH_MANIM_SOURCE",
    Path.home() / "Documents/manim_skill/quad_remesh_blog",
)).expanduser().resolve()
sys.path.insert(0, str(SOURCE_DIR))

from manim import DOWN, RIGHT, Group, __version__ as manim_version, config
from blog_scenes import (
    BlogCarrierPatch,
    BlogCutCharts,
    BlogIntegerSeam,
    BlogPairingVsLattice,
)


class ArticleStill:
    """Reframe the source scene's completed vector objects before rasterizing."""

    remove_context_inset = False
    stack_panels = False
    panel_gap = None
    max_pixel_width = None

    def construct(self):
        super().construct()
        if not config.save_last_frame:
            raise ValueError("Static article scenes must be rendered with -s")

        # These regions belong to the video title, narration, and carrier inset.
        # Keep the original panel labels, equations, geometry, and correspondence.
        remove = [
            mob for mob in self.mobjects
            if mob.get_center()[1] > 2.8
            or mob.get_center()[1] < -3.05
            or (self.remove_context_inset and mob.get_center()[0] < -5.2)
        ]
        self.remove(*remove)
        content = Group(*self.mobjects)
        content.clear_updaters(recursive=True)
        if self.panel_gap is not None:
            # Keep each panel's geometry and annotations together. The divider
            # stays at x=0; its two inner margins are half the requested gap.
            left = Group(*[mob for mob in self.mobjects if mob.get_center()[0] < 0])
            right = Group(*[mob for mob in self.mobjects if mob.get_center()[0] > 0])
            left.shift(RIGHT * (-self.panel_gap / 2 - left.get_right()[0]))
            right.shift(RIGHT * (self.panel_gap / 2 - right.get_left()[0]))
        if self.stack_panels:
            left = Group(*[mob for mob in self.mobjects if mob.get_center()[0] < 0])
            right = Group(*[mob for mob in self.mobjects if mob.get_center()[0] >= 0])
            Group(left, right).arrange(DOWN, buff=0.45)
        width = content.width + 0.64
        height = content.height + 0.56
        pixels = int(os.environ.get("QUAD_REMESH_STILL_WIDTH", "1600"))
        if self.max_pixel_width:
            pixels = min(pixels, self.max_pixel_width)
        pixel_height = 2 * math.ceil(pixels * height / width / 2)
        self.camera.frame_width = width
        self.camera.frame_center = content.get_center()
        self.camera.reset_pixel_shape(pixel_height, pixels)


class PairingStill(ArticleStill, BlogPairingVsLattice):
    panel_gap = 1.0


class CarrierStill(ArticleStill, BlogCarrierPatch):
    pass


class ChartSeamStill(ArticleStill, BlogCutCharts):
    remove_context_inset = True


class IntegerTransitionStill(ArticleStill, BlogIntegerSeam):
    pass


class IntegerTransitionMobileStill(ArticleStill, BlogIntegerSeam):
    stack_panels = True
    max_pixel_width = 800


FIGURES = [
    ("PairingStill", "pairing-vs-lattice", "BlogPairingVsLattice"),
    ("CarrierStill", "carrier-patch", "BlogCarrierPatch"),
    ("ChartSeamStill", "cut-charts", "BlogCutCharts"),
    ("IntegerTransitionStill", "integer-seam", "BlogIntegerSeam"),
    ("IntegerTransitionMobileStill", "integer-seam-mobile", "BlogIntegerSeam"),
]


def main():
    from PIL import Image

    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--scene", action="append", choices=[row[0] for row in FIGURES])
    parser.add_argument("--work-dir", type=Path,
                        default=repo / "_workspace/2026-10-15-quad-remesh-stills")
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "blogs/posts/260818_quad_remesh/assets/manim/stills")
    args = parser.parse_args()
    if args.width < 640 or not 1 <= args.quality <= 100:
        parser.error("Use a width of at least 640px and a quality from 1 to 100")
    args.work_dir = args.work_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [row for row in FIGURES if not args.scene or row[0] in args.scene]
    env = {
        **os.environ,
        "QUAD_REMESH_MANIM_SOURCE": str(SOURCE_DIR),
        "QUAD_REMESH_STILL_WIDTH": str(args.width),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    media = args.work_dir / "renders"
    command = [
        sys.executable, "-m", "manim", "-s", "--renderer", "cairo",
        "--config_file", str(SOURCE_DIR / "manim.cfg"),
        "--media_dir", str(media), "--verbosity", "WARNING",
        str(Path(__file__).resolve()), *[row[0] for row in selected],
    ]
    if not args.export_only:
        with (args.work_dir / "render.log").open("w") as log:
            subprocess.run(command, cwd=SOURCE_DIR, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    records = []
    for scene, slug, original in selected:
        matches = list((media / "images").rglob(f"{scene}_ManimCE_v{manim_version}.png"))
        if len(matches) != 1:
            raise ValueError(f"Expected one native Manim image for {scene}: {matches}")
        target = args.output_dir / f"{slug}.webp"
        with Image.open(matches[0]) as rendered:
            image = rendered.convert("RGB")
            image.save(target, "WEBP", quality=args.quality, method=6)
            size = image.size
        record = {
            "scene": scene,
            "source_scene": original,
            "image": target.name,
            "width": size[0],
            "height": size[1],
            "bytes": target.stat().st_size,
            "quality": args.quality,
        }
        records.append(record)
        print(f"{slug}: {size[0]}x{size[1]}, {record['bytes'] / 1024:.1f} KiB", flush=True)

    manifest = args.output_dir / "manifest.json"
    if args.scene and manifest.exists():
        by_scene = {row["scene"]: row for row in json.loads(manifest.read_text())["figures"]}
        by_scene.update({row["scene"]: row for row in records})
        records = [by_scene[scene] for scene, _, _ in FIGURES if scene in by_scene]
    provenance = {
        "method": "Native Manim save_last_frame rendering of reframed source scenes; no video capture",
        "renderer": str(Path(__file__).relative_to(repo)),
        "source_project": "quad_remesh_blog",
        "manim_version": manim_version,
        "source_sha256": {
            name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
            for name in ["blog_scenes.py", "blog_geometry.py", "common.py", "geometry.py",
                         "carrier_patch.py", "scene_patch_helpers.py", "scenes_foundations.py",
                         "scenes_explained.py", "scenes_extraction.py", "teaching_solvers.py"]
        },
        "figures": records,
    }
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
