#!/usr/bin/env python3
"""
Generates the mod's cover/thumbnail image for the Mod Manager listing (src/cover.jpg), referenced by manifest.sii's
`icon:` field. SCS requires a 276x162 JPEG.

DriveDogs cover layout (profile memory drivedogs-family.md): logo top-left, title + amber subtitle in the middle,
amber version pill top-right, the game badge bottom-left (a per-game colour pill, the game read from
pack.config.json) and one subject drawn small and faded into the background, lower right: for Pets a seated dog,
rendered from the imported German shepherd's sit pose (build/import, posed offline with its own animation by
pma_pose.py). Composited at 4x (1104x648) then downscaled for anti-aliasing.

Copied from the ETS2 Pets repo's tools/generate_cover.py (ADR 0001); only the subject's source differs: ETS2 reads
willet's seated shepherd from the extracted Workshop item, this repo poses the imported pet instead.

Reads the version from src/manifest.sii, so the cover always shows the version being packed - pack.config.json's
prePack runs this before every pack.

Usage: python tools/generate_cover.py
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GAME, IMPORT, REPO  # noqa: E402
from pma_pose import posed_triangles  # noqa: E402
from pmg_preview import render  # noqa: E402

LOGO_PATH = REPO / "logo.png"
OUT_PATH = REPO / "src" / "cover.jpg"
MANIFEST_PATH = REPO / "src" / "manifest.sii"
SUBJECT_MODEL = IMPORT / "pets/out/vehicle/truck/upgrade/interior_decors/drivedogs/pets/gsd/gsd_sit.pmd"

SCALE = 4
FINAL_SIZE = (276, 162)
CANVAS_SIZE = (FINAL_SIZE[0] * SCALE, FINAL_SIZE[1] * SCALE)

FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_REG = r"C:\Windows\Fonts\arial.ttf"

BG_TOP = (10, 12, 16)
BG_BOTTOM = (24, 20, 14)
TITLE_COLOR = (255, 255, 255)
SUBTITLE_COLOR = (255, 196, 84)
SUBJECT_COLOR = (255, 196, 84)
SUBJECT_OPACITY = 0.55
BADGE_COLOR = (255, 196, 84)
BADGE_TEXT_COLOR = (20, 16, 10)
# Game badge (bottom-left): one colour per game so ETS2 and ATS covers differ at thumbnail size.
GAME_BADGES = {"ets2": ("ETS2", (40, 110, 210)), "ats": ("ATS", (200, 45, 45))}
GAME_TEXT_COLOR = (255, 255, 255)


def read_version():
    m = re.search(r'^\s*package_version:\s*"([^"]+)"', MANIFEST_PATH.read_text(encoding="utf-8"), re.MULTILINE)
    if not m:
        raise SystemExit(f"No package_version in {MANIFEST_PATH}")
    return m.group(1)


def read_game():
    """The game from pack.config.json (the pack's own setting), so the badge can't disagree with the pack."""
    game = json.loads((REPO / "pack.config.json").read_text(encoding="utf-8-sig")).get("game", GAME)
    if game not in GAME_BADGES:
        raise SystemExit(f"pack.config.json game {game!r}: no badge defined")
    return game


def make_background(size):
    w, h = size
    img = Image.new("RGB", size, BG_BOTTOM)
    px = img.load()
    for y in range(h):
        t = y / (h - 1)
        color = tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))
        for x in range(w):
            px[x, y] = color
    return img


def subject_image(height):
    """The imported German shepherd in its sit pose (frame 0 of its animation), head to the right (full strength; the
    ramp fades its back toward the title), as an amber silhouette with soft shading, cropped to its outline."""
    pma = SUBJECT_MODEL.with_suffix(".pma")
    if not (SUBJECT_MODEL.with_suffix(".pmg").is_file() and pma.is_file()):
        raise SystemExit(f"cover subject {SUBJECT_MODEL} is missing (run tools/import_ets2.py first)")
    triangles = posed_triangles(SUBJECT_MODEL, pma, 0)
    shaded = render(triangles, size=height * 2, yaw_deg=150.0, pitch_deg=10.0)
    shaded = shaded.crop(shaded.getchannel("A").getbbox())
    shaded = shaded.resize((int(shaded.width * height / shaded.height), height), Image.LANCZOS)
    grey = shaded.convert("L")
    tinted = Image.merge("RGB", [grey.point(lambda v, c=c: int(c * min(1.0, 0.5 + 0.5 * v / 180))) for c in SUBJECT_COLOR])
    tinted.putalpha(shaded.getchannel("A"))
    return tinted


def add_subject(img):
    """The dog small and faded in the lower right, fading out toward the text on its left (alpha ramp)."""
    w, h = img.size
    dog = subject_image(int(h * 0.52))
    alpha = dog.getchannel("A").point(lambda a: int(a * SUBJECT_OPACITY))
    ramp = Image.linear_gradient("L").rotate(90, expand=True).resize(dog.size)  # 255 at the right edge
    ramp = ramp.point(lambda v: int(90 + v * 165 / 255))  # 35 % at the text side, full at the right
    dog.putalpha(Image.composite(alpha, Image.new("L", dog.size, 0), ramp))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    layer.paste(dog, (int(w * 0.97 - dog.width), int(h * 0.985 - dog.height)), dog)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def paste_logo(img):
    logo = Image.open(LOGO_PATH).convert("RGBA")
    target_h = int(img.height * 0.62)
    logo = logo.resize((int(logo.width * target_h / logo.height), target_h), Image.LANCZOS)
    pad = int(img.height * 0.06)
    pos = (pad, pad)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_logo = Image.new("RGBA", logo.size, (0, 0, 0, 255))
    shadow_logo.putalpha(logo.split()[3].point(lambda a: a * 160 // 255))
    shadow.paste(shadow_logo, (pos[0] + 6, pos[1] + 6), shadow_logo)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    base = Image.alpha_composite(img.convert("RGBA"), shadow)
    base.paste(logo, pos, logo)
    return base.convert("RGB")


def fit_font(draw, text, font_path, max_width, start_size, min_size=10):
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, min_size)


def draw_with_outline(draw, pos, text, font, fill, outline=(0, 0, 0), width=2):
    x, y = pos
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def add_text(img):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x_start = w * 0.40
    max_width = w * 0.97 - x_start
    title_lines = ["DriveDogs:", "Pets"]
    subtitle = "Pets for the Truck Cab"

    title_size = int(h * 0.135)
    for line in title_lines:
        title_size = min(title_size, fit_font(draw, line, FONT_BOLD, max_width, title_size).size)
    title_font = ImageFont.truetype(FONT_BOLD, title_size)
    subtitle_font = fit_font(draw, subtitle, FONT_REG, max_width, int(h * 0.06))

    line_bbox = draw.textbbox((0, 0), "Ag", font=title_font)
    line_h = (line_bbox[3] - line_bbox[1]) * 1.15
    y = h * 0.45 - (line_h * len(title_lines) + line_h * 0.9) / 2
    for line in title_lines:
        draw_with_outline(draw, (x_start, y), line, title_font, TITLE_COLOR, width=3)
        y += line_h
    y += line_h * 0.15
    draw_with_outline(draw, (x_start, y), subtitle, subtitle_font, SUBTITLE_COLOR, width=2)
    return img


def pill(img, text, fill, text_fill, anchor):
    """A rounded pill holding `text`; anchor = ("right", "top") or ("left", "bottom") corner of the cover."""
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    font = ImageFont.truetype(FONT_BOLD, int(h * 0.06))
    bbox = draw.textbbox((0, 0), text, font=font)
    pad_x, pad_y, margin = h * 0.02, h * 0.015, h * 0.03
    pw, ph = (bbox[2] - bbox[0]) + 2 * pad_x, (bbox[3] - bbox[1]) + 2 * pad_y
    x0 = w - margin - pw if anchor[0] == "right" else margin
    y0 = margin if anchor[1] == "top" else h - margin - ph
    draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=ph / 2, fill=(*fill, 235))
    draw.text((x0 + pad_x - bbox[0], y0 + pad_y - bbox[1]), text, font=font, fill=text_fill)
    return img


def main():
    version, game = read_version(), read_game()
    img = make_background(CANVAS_SIZE)
    img = add_subject(img)
    img = paste_logo(img)
    img = add_text(img)
    img = pill(img, f"v{version}", BADGE_COLOR, BADGE_TEXT_COLOR, ("right", "top"))
    label, colour = GAME_BADGES[game]
    img = pill(img, label, colour, GAME_TEXT_COLOR, ("left", "bottom"))
    img = img.resize(FINAL_SIZE, Image.LANCZOS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH, "JPEG", quality=92)
    print(f"Wrote {OUT_PATH} ({img.size[0]}x{img.size[1]}, v{version}, {label})")


if __name__ == "__main__":
    main()
