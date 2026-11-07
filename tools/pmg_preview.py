"""Render a shaded preview image of an ETS2 model (.pmd/.pmg, PMG version 0x15).

Only geometry is read: piece positions and triangle indices. Materials are not decoded, so every piece is
drawn in one neutral colour, except pieces whose material is a shadow or other non-visual helper, which are
skipped. That is enough to tell parts apart in the catalog without a game or Blender.

Copied unchanged from the ETS2 Pets repo's tools/pmg_preview.py (ADR 0001).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PMG_VERSION = 0x15
_HEADER = struct.Struct("<B3s5iQ4f3f3f10i")  # 0x70 bytes
_PIECE = struct.Struct("<5i4f3f3f5i4ii")  # 100 bytes
# Materials whose pieces are not visible surfaces (shadow casters, collision helpers).
_HIDDEN_MATERIAL = re.compile(r"shadow|collision|invisible", re.IGNORECASE)


def _material_paths(pmd: bytes) -> list[str]:
    """Material paths of a .pmd in declaration order (piece material_index indexes this list)."""
    return [m.decode("ascii", "replace") for m in re.findall(rb"/[\x21-\x7e]+?\.mat", pmd)]


def read_triangles(pmd_path: Path) -> np.ndarray | None:
    """Return an (N, 3, 3) array of visible triangles, or None when the model can't be read."""
    pmg_path = pmd_path.with_suffix(".pmg")
    if not pmg_path.is_file():
        return None
    data = pmg_path.read_bytes()
    if len(data) < _HEADER.size or data[1:4] != b"gmP" or data[0] != PMG_VERSION:
        return None
    header = _HEADER.unpack_from(data, 0)
    piece_count, pieces_offset = header[2], header[21]
    materials = _material_paths(pmd_path.read_bytes()) if pmd_path.is_file() else []

    triangles = []
    for index in range(piece_count):
        piece = _PIECE.unpack_from(data, pieces_offset + index * _PIECE.size)
        edge_count, vertex_count, material_index = piece[0], piece[1], piece[4]
        stride, position_offset, index_offset = piece[15], piece[16], piece[-1]
        if material_index < len(materials) and _HIDDEN_MATERIAL.search(materials[material_index]):
            continue
        if vertex_count == 0 or edge_count < 3 or stride < 12:
            continue
        raw = np.frombuffer(data, dtype=np.uint8, count=vertex_count * stride, offset=position_offset)
        positions = raw.reshape(vertex_count, stride)[:, :12].copy().view("<f4").reshape(vertex_count, 3)
        indices = np.frombuffer(data, dtype="<u2", count=edge_count, offset=index_offset)
        indices = indices[: len(indices) // 3 * 3].reshape(-1, 3)
        if indices.size and indices.max() < vertex_count:
            triangles.append(positions[indices])
    return np.concatenate(triangles) if triangles else None


def view_of(triangles: np.ndarray, yaw_deg: float = -35.0, pitch_deg: float = 20.0) -> np.ndarray:
    """The triangles in render()'s camera space (x right, y up, z toward the camera)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    rot_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    return triangles @ (rot_x @ rot_y @ np.diag([-1.0, 1.0, -1.0])).T


def framing(frames: list[np.ndarray], yaw_deg: float = -35.0, pitch_deg: float = 20.0) -> tuple:
    """One fixed framing (low, high in camera x / y) that holds every given frame: pass it to render() for each frame
    of an animation, so the picture does not re-fit (zoom / shift) as a head or tail changes the outline (2026-10-15:
    the review GIFs re-fitted every frame - the still body seemed to shake with willet's head loop)."""
    xy = np.concatenate([view_of(t, yaw_deg, pitch_deg)[:, :, :2].reshape(-1, 2) for t in frames])
    return xy.min(axis=0), xy.max(axis=0)


def render(triangles: np.ndarray, size: int = 256, yaw_deg: float = -35.0, pitch_deg: float = 20.0,
           frame: tuple | None = None) -> Image.Image:
    """Orthographic, flat-shaded render from the front-left and slightly above (game space: +Y up, -Z forward).
    `frame` (framing()): a fixed picture area; None = fitted to these triangles (as before)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    rot_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    # Turn the truck 180 degrees about Y so its front (-Z) faces the camera (a rotation, not a mirror).
    view = triangles @ (rot_x @ rot_y @ np.diag([-1.0, 1.0, -1.0])).T

    normals = np.cross(view[:, 1] - view[:, 0], view[:, 2] - view[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    view, normals = view[keep], normals[keep] / lengths[keep, None]
    if not len(view):
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    light = np.array([-0.4, 0.7, 0.6])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ light)  # two-sided: mod meshes often have flipped faces

    scale_up = 2  # supersample, then downscale for anti-aliasing
    canvas = size * scale_up
    xy = view[:, :, :2]
    low, high = frame if frame is not None else (xy.reshape(-1, 2).min(axis=0), xy.reshape(-1, 2).max(axis=0))
    extent = max(high - low) or 1.0
    pad = canvas * 0.06
    factor = (canvas - 2 * pad) / extent
    centre = (low + high) / 2
    screen = (xy - centre) * factor
    screen[:, :, 1] *= -1
    screen += canvas / 2

    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    order = np.argsort(view[:, :, 2].mean(axis=1))  # painter's algorithm: far (most negative z) first
    base = np.array([176, 184, 196])
    for i in order:
        colour = tuple(int(c) for c in base * shade[i]) + (255,)
        draw.polygon([tuple(p) for p in screen[i]], fill=colour)
    return image.resize((size, size), Image.LANCZOS)


def piece_and_locator_counts(pmd_path: Path) -> tuple[int, int] | None:
    """(geometry pieces, locators) of a PMG 0x15 model; a model with locators and no pieces is mount points only."""
    pmg_path = pmd_path.with_suffix(".pmg")
    if not pmg_path.is_file():
        return None
    data = pmg_path.read_bytes()
    if len(data) < _HEADER.size or data[1:4] != b"gmP" or data[0] != PMG_VERSION:
        return None
    header = _HEADER.unpack_from(data, 0)
    return header[2], header[6]


def render_model(pmd_path: Path, out_path: Path, size: int = 256) -> bool:
    """Render pmd_path to out_path (PNG). Returns False when the model can't be read."""
    triangles = read_triangles(pmd_path)
    if triangles is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render(triangles, size).save(out_path, optimize=True)
    return True
