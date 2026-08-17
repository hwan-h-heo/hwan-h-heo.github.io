#!/usr/bin/env python3
"""Build the CaPa portfolio cover from the official pipeline artwork."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (1800, 600)
SOURCE_URL = "https://ncsoft.github.io/CaPa/assets/pipeline.png"
SOURCE_CROPS = {
    "geometry": (3980, 70, 4780, 940),
    "multiview": (1740, 1190, 2440, 1980),
    "texture": (190, 1040, 900, 1985),
}
STAGES = (
    {
        "key": "geometry",
        "center_x": 270,
        "top": 30,
        "max_size": (390, 455),
        "label": "3D Geometry",
        "label_size": 34,
    },
    {
        "key": "multiview",
        "center_x": 900,
        "top": 8,
        "max_size": (500, 505),
        "label": "High-Quality MV images w/o Janus",
        "label_size": 31,
    },
    {
        "key": "texture",
        "center_x": 1530,
        "top": 0,
        "max_size": (440, 520),
        "label": "Final Textured Mesh",
        "label_size": 34,
    },
)
LABEL_COLOR = (31, 41, 55, 235)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root / "projects" / "capa" / "assets" / "remote-adac548a479a.png",
        help=f"Official CaPa pipeline artwork ({SOURCE_URL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "projects" / "capa" / "assets" / "portfolio-cover.webp",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def crop_alpha_content(source: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    crop = source.crop(box)
    content_box = crop.getchannel("A").getbbox()
    if not content_box:
        raise ValueError(f"Crop {box} does not contain visible pixels.")
    return crop.crop(content_box)


def fit_content(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(max_size, Image.Resampling.LANCZOS)
    return fitted


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start_x: int,
    end_x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    draw.line((start_x, y, end_x, y), fill=color, width=3)
    draw.polygon(
        ((end_x, y), (end_x - 15, y - 10), (end_x - 15, y + 10)),
        fill=color,
    )


def render_cover(source: Image.Image) -> Image.Image:
    cover = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cover)

    draw_arrow(draw, 500, 640, 286, (100, 116, 139, 150))
    draw_arrow(draw, 1175, 1300, 286, (22, 127, 148, 175))

    for stage in STAGES:
        content = fit_content(
            crop_alpha_content(source, SOURCE_CROPS[stage["key"]]),
            stage["max_size"],
        )
        position = (
            stage["center_x"] - content.width // 2,
            stage["top"],
        )
        cover.alpha_composite(content, position)

        font = load_font(stage["label_size"])
        label_box = draw.textbbox((0, 0), stage["label"], font=font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (stage["center_x"] - label_width // 2, 538),
            stage["label"],
            font=font,
            fill=LABEL_COLOR,
        )

    return cover


def main() -> None:
    args = parse_args()
    source = Image.open(args.source).convert("RGBA")
    if source.size != (4833, 2118):
        raise ValueError(f"Unexpected CaPa pipeline size {source.size}; expected (4833, 2118).")

    cover = render_cover(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.casefold() == ".webp":
        cover.save(args.output, "WEBP", quality=92, method=6, exact=True)
    else:
        cover.save(args.output)
    print(f"CaPa portfolio cover: {args.output}")


if __name__ == "__main__":
    main()
