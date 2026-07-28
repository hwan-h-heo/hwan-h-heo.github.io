#!/usr/bin/env python3
"""Build the VARCO 3D community geometry-to-texture showcase video."""

from __future__ import annotations

import argparse
import io
import json
import ssl
import subprocess
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import certifi
except ImportError:
    certifi = None


SITE_ORIGIN = "https://3d.varco.ai"
API_ENDPOINT = f"{SITE_ORIGIN}/api/trpc/task.getSharedTasks"
CANVAS_SIZE = (1920, 1080)
FPS = 30
DEFAULT_SCENE_LAYOUT = (2, 2, 2, 2, 1, 1)
SCENE_SECONDS = {2: 1.7, 1: 2.0}
SCENE_TRANSITION_SECONDS = 0.2
TILE_SIZES = {2: 760, 1: 880}
TILE_GAP_X = 32
GRID_Y = {2: 190, 1: 158}
HERO_ASSET_TITLES = ("Feathered Skull Shaman", "Monstrous Feathered Claw")
ACCENT = (53, 207, 255)
BACKGROUND_TOP = (7, 9, 11)
BACKGROUND_BOTTOM = (17, 20, 24)
USER_AGENT = "hwan-h-heo.io VARCO 3D showcase builder"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where() if certifi else None)


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    default_assets = repository_root / "projects" / "varco3d" / "assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        default=",".join(str(count) for count in DEFAULT_SCENE_LAYOUT),
        help="Comma-separated asset count per scene; supported scene sizes are 1 and 2.",
    )
    parser.add_argument("--request-limit", type=int, default=60)
    parser.add_argument("--output", type=Path, default=default_assets / "community-showcase.mp4")
    parser.add_argument("--poster", type=Path, default=default_assets / "community-showcase-poster.webp")
    parser.add_argument("--manifest", type=Path, default=default_assets / "community-showcase.json")
    return parser.parse_args()


def request_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45, context=SSL_CONTEXT) as response:
        return json.load(response)


def absolute_asset_url(path: str) -> str:
    return urllib.parse.urljoin(f"{SITE_ORIGIN}/", path)


def fetch_recent_assets(count: int, request_limit: int) -> list[dict[str, str]]:
    payload = {
        "0": {
            "isFeaturedOnly": False,
            "limit": request_limit,
            "offset": 0,
            "sortBy": "sharedAt",
            "outputTypes": [],
        }
    }
    query = urllib.parse.urlencode({
        "batch": "1",
        "input": json.dumps(payload, separators=(",", ":")),
    })
    response = request_json(f"{API_ENDPOINT}?{query}")
    tasks = response[0]["result"]["data"]["tasks"]

    selected: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for item in tasks:
        task = item.get("task") or {}
        output = task.get("output") or {}
        title = str(task.get("title") or "Untitled").strip()
        title_key = title.casefold()
        thumbnail = output.get("thumbnail")
        overlay_thumbnail = output.get("overlayThumbnail")
        if task.get("type") != "GENERATE_TEXTURED_MESH":
            continue
        if not thumbnail or not overlay_thumbnail or title_key in seen_titles:
            continue

        selected.append({
            "id": task["id"],
            "title": title,
            "creator": str(item.get("name") or ""),
            "sharedAt": task.get("sharedAt") or "",
            "thumbnail": absolute_asset_url(thumbnail),
            "overlayThumbnail": absolute_asset_url(overlay_thumbnail),
            "exploreUrl": f"{SITE_ORIGIN}/assets/{task['id']}",
        })
        seen_titles.add(title_key)
        if len(selected) == count:
            break

    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} recent assets with both thumbnails; requested {count}.")
    return selected


def arrange_assets(assets: list[dict[str, str]], layout: tuple[int, ...]) -> list[dict[str, str]]:
    hero_assets = []
    for title in HERO_ASSET_TITLES:
        hero = next((asset for asset in assets if asset["title"] == title), None)
        if not hero:
            raise RuntimeError(f'Could not find hero asset "{title}" in the recent Explore response.')
        hero_assets.append(hero)

    regular_count = sum(layout) - len(hero_assets)
    regular_assets = [
        asset
        for asset in assets
        if asset["title"] not in HERO_ASSET_TITLES
    ][:regular_count]
    if len(regular_assets) < regular_count:
        raise RuntimeError(f"Only found {len(regular_assets)} non-hero assets; requested {regular_count}.")
    return regular_assets + hero_assets


def download_image(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def crop_tile(image: Image.Image, tile_size: int) -> Image.Image:
    return ImageOps.fit(
        image,
        (tile_size, tile_size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def download_asset_pair(asset: dict[str, str]) -> dict[str, object]:
    return {
        **asset,
        "geometryImage": download_image(asset["overlayThumbnail"]),
        "textureImage": download_image(asset["thumbnail"]),
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/SFNS.ttf"),
        Path("/System/Library/Fonts/SFCompact.ttf"),
        Path("/System/Library/Fonts/SFNSMono.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    if bold:
        candidates.insert(0, Path("/System/Library/Fonts/SFCompact.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SUBTITLE = load_font(17)
FONT_LABEL_TWO = load_font(28, bold=True)
FONT_LABEL_SINGLE = load_font(31, bold=True)
FONT_META = load_font(14)
FONT_PAGE = load_font(16)


def vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        amount = y / max(1, height - 1)
        color = tuple(round(top[channel] + (bottom[channel] - top[channel]) * amount) for channel in range(3))
        for x in range(width):
            pixels[x, y] = color
    return image


BACKGROUND = vertical_gradient(CANVAS_SIZE, BACKGROUND_TOP, BACKGROUND_BOTTOM)


def truncate_text(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), value, font=font)[2] <= max_width:
        return value
    shortened = value
    while shortened and draw.textbbox((0, 0), f"{shortened}…", font=font)[2] > max_width:
        shortened = shortened[:-1]
    return f"{shortened.rstrip()}…"


def rounded_tile(image: Image.Image, tile_size: int) -> Image.Image:
    rounded = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile_size - 1, tile_size - 1), radius=24, fill=255)
    rounded.paste(image.convert("RGBA"), (0, 0), mask)
    return rounded


def add_tile_label(image: Image.Image, title: str, tile_size: int, asset_count: int) -> Image.Image:
    labeled = image.convert("RGBA")
    overlay = Image.new("RGBA", labeled.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    label_height = 104 if asset_count == 2 else 116
    font = FONT_LABEL_TWO if asset_count == 2 else FONT_LABEL_SINGLE
    for offset in range(label_height):
        alpha = round(185 * (offset / label_height) ** 1.65)
        overlay_draw.line(
            (0, tile_size - label_height + offset, tile_size, tile_size - label_height + offset),
            fill=(2, 4, 6, alpha),
        )
    safe_title = truncate_text(overlay_draw, title, font, tile_size - 64)
    overlay_draw.text((30, tile_size - 60), safe_title, fill=(244, 247, 250, 245), font=font)
    return Image.alpha_composite(labeled, overlay)


def prepare_scenes(assets: list[dict[str, str]], layout: tuple[int, ...]) -> list[list[dict[str, object]]]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        downloaded = list(executor.map(download_asset_pair, assets))
    scenes: list[list[dict[str, object]]] = []
    asset_index = 0
    for asset_count in layout:
        tile_size = TILE_SIZES[asset_count]
        scene = []
        for asset in downloaded[asset_index:asset_index + asset_count]:
            metadata = {
                key: value
                for key, value in asset.items()
                if key not in {"geometryImage", "textureImage"}
            }
            geometry = crop_tile(asset["geometryImage"], tile_size)
            texture = crop_tile(asset["textureImage"], tile_size)
            scene.append({
                **metadata,
                "geometryTile": rounded_tile(
                    add_tile_label(geometry, asset["title"], tile_size, asset_count),
                    tile_size,
                ),
                "textureTile": rounded_tile(
                    add_tile_label(texture, asset["title"], tile_size, asset_count),
                    tile_size,
                ),
            })
        scenes.append(scene)
        asset_index += asset_count
    return scenes


def tile_position(index: int, asset_count: int, tile_size: int) -> tuple[int, int]:
    total_width = asset_count * tile_size + max(0, asset_count - 1) * TILE_GAP_X
    start_x = round((CANVAS_SIZE[0] - total_width) / 2)
    return start_x + index * (tile_size + TILE_GAP_X), GRID_Y[asset_count]


def draw_header(image: Image.Image, page_index: int, page_count: int) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((64, 42, 102, 47), radius=3, fill=ACCENT)
    draw.text((64, 58), "RECENT COMMUNITY CREATIONS", font=FONT_TITLE, fill=(244, 247, 250))
    draw.text((64, 110), "VARCO 3D 2.0  ·  GEOMETRY → TEXTURE", font=FONT_SUBTITLE, fill=(142, 151, 163))
    source = "3D.VARCO.AI/EXPLORE"
    source_width = draw.textbbox((0, 0), source, font=FONT_META)[2]
    draw.text((CANVAS_SIZE[0] - 64 - source_width, 64), source, font=FONT_META, fill=(142, 151, 163))
    page_label = f"{page_index + 1:02d} / {page_count:02d}"
    page_width = draw.textbbox((0, 0), page_label, font=FONT_PAGE)[2]
    draw.text((CANVAS_SIZE[0] - 64 - page_width, 102), page_label, font=FONT_PAGE, fill=ACCENT)


def smoothstep(value: float) -> float:
    bounded = max(0.0, min(1.0, value))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def reveal_progress(local_time: float, tile_index: int) -> float:
    start = 0.22 + tile_index * 0.08
    return smoothstep((local_time - start) / 0.42)


def render_scene(scene_assets: list[dict[str, object]], scene_index: int, scene_count: int, local_time: float) -> Image.Image:
    frame = BACKGROUND.copy().convert("RGBA")
    draw_header(frame, scene_index, scene_count)
    asset_count = len(scene_assets)
    tile_size = TILE_SIZES[asset_count]

    for index, asset in enumerate(scene_assets):
        x, y = tile_position(index, asset_count, tile_size)
        geometry = asset["geometryTile"]
        texture = asset["textureTile"]
        frame.alpha_composite(geometry, (x, y))

        progress = reveal_progress(local_time, index)
        reveal_width = round(tile_size * progress)
        if reveal_width > 0:
            frame.alpha_composite(texture.crop((0, 0, reveal_width, tile_size)), (x, y))

        border = ImageDraw.Draw(frame)
        border.rounded_rectangle(
            (x, y, x + tile_size - 1, y + tile_size - 1),
            radius=24,
            outline=(255, 255, 255, 32),
            width=2,
        )
        if 0 < reveal_width < tile_size:
            edge_x = x + reveal_width
            border.rectangle((edge_x - 8, y + 2, edge_x + 8, y + tile_size - 3), fill=(53, 207, 255, 24))
            border.line((edge_x, y + 2, edge_x, y + tile_size - 3), fill=(226, 249, 255, 235), width=2)

    return frame.convert("RGB")


def render_frame(scenes: list[list[dict[str, object]]], layout: tuple[int, ...], time_seconds: float) -> Image.Image:
    scene_index = 0
    scene_start = 0.0
    for index, asset_count in enumerate(layout):
        scene_duration = SCENE_SECONDS[asset_count]
        if time_seconds < scene_start + scene_duration or index == len(layout) - 1:
            scene_index = index
            break
        scene_start += scene_duration

    asset_count = layout[scene_index]
    scene_duration = SCENE_SECONDS[asset_count]
    content_duration = scene_duration - SCENE_TRANSITION_SECONDS
    local_time = time_seconds - scene_start
    if local_time < content_duration:
        return render_scene(scenes[scene_index], scene_index, len(scenes), local_time)

    next_scene_index = (scene_index + 1) % len(scenes)
    transition = smoothstep((local_time - content_duration) / SCENE_TRANSITION_SECONDS)
    outgoing = render_scene(scenes[scene_index], scene_index, len(scenes), content_duration)
    incoming = render_scene(scenes[next_scene_index], next_scene_index, len(scenes), 0)
    return Image.blend(outgoing, incoming, transition)


def encode_video(scenes: list[list[dict[str, object]]], layout: tuple[int, ...], output: Path) -> float:
    duration = sum(SCENE_SECONDS[asset_count] for asset_count in layout)
    frame_count = round(duration * FPS)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
        "-framerate",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_number in range(frame_count):
            frame = render_frame(scenes, layout, frame_number / FPS)
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the showcase video.")
    return duration


def write_manifest(assets: list[dict[str, str]], layout: tuple[int, ...], duration: float, path: Path) -> None:
    payload = {
        "source": f"{SITE_ORIGIN}/explore",
        "sort": "sharedAt",
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "sceneLayout": list(layout),
        "durationSeconds": duration,
        "assets": assets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    layout = tuple(int(value.strip()) for value in args.layout.split(",") if value.strip())
    if not layout or any(asset_count not in TILE_SIZES for asset_count in layout):
        raise ValueError("--layout must contain only scene sizes 1 and 2.")

    recent_assets = fetch_recent_assets(sum(layout) + 8, args.request_limit)
    assets = arrange_assets(recent_assets, layout)
    scenes = prepare_scenes(assets, layout)
    duration = encode_video(scenes, layout, args.output)
    first_scene_content_duration = SCENE_SECONDS[layout[0]] - SCENE_TRANSITION_SECONDS
    poster_frame = render_scene(scenes[0], 0, len(scenes), first_scene_content_duration)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    poster_frame.save(args.poster, "WEBP", quality=88, method=6)
    write_manifest(assets, layout, duration, args.manifest)

    print(f"Built {args.output} from {len(assets)} recent shared assets in scenes {layout}.")
    print(f"Poster: {args.poster}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
