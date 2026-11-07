"""UV checks on exported models (2026-10-20, 8.0.2): every pose up to 8.0.1 shipped with UV (0, 0) on every vertex
(no UV link on its SCS materials, see blender/scs.link_uvs) and nothing noticed - in game each material was one flat
colour. These read the UVs back from a .pim (SCS Blender Tools' text export) or a converted .pmg (PMG 0x15) so a lock
and a pack can refuse a model whose UVs are all zero.

A range is one per piece: (vertices, u min, u max, v min, v max, vertices at exactly (0, 0)).

Copied unchanged from the ETS2 Pets repo's tools/petlib/uvs.py (ADR 0001).
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pma_pose import _HEADER, _PIECE  # noqa: E402

_ROW = re.compile(r"&([0-9a-f]{8})\s+&([0-9a-f]{8})")


def _range(uv: np.ndarray) -> tuple:
    if not len(uv):
        return (0, 0.0, 0.0, 0.0, 0.0, 0)
    zeros = int(np.count_nonzero((uv[:, 0] == 0) & (uv[:, 1] == 0)))
    return (len(uv), float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()), float(uv[:, 1].max()), zeros)


def pim_uv_lines(lines: list[str]) -> set[int]:
    """Indices of the data rows of every `_UV0` stream in a .pim's lines."""
    rows, inside = set(), False
    for i, line in enumerate(lines):
        if line.strip() == 'Tag: "_UV0"':
            inside = True
        elif inside and line.strip() == "}":
            inside = False
        elif inside and _ROW.search(line):
            rows.add(i)
    return rows


def pim_uv_ranges(path: Path) -> list[tuple]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = pim_uv_lines(lines)
    out, block = [], []
    for i, line in enumerate(lines):
        if i in rows:
            u, v = _ROW.search(line).groups()
            block.append((struct.unpack(">f", bytes.fromhex(u))[0], struct.unpack(">f", bytes.fromhex(v))[0]))
        elif block:
            out.append(_range(np.array(block)))
            block = []
    if block:
        out.append(_range(np.array(block)))
    return out


def pmg_uv_ranges(path: Path) -> list[tuple]:
    pmg = path.read_bytes()
    if pmg[1:4] != b"gmP" or pmg[0] != 0x15:
        raise SystemExit(f"{path}: not a PMG 0x15")
    h = _HEADER.unpack_from(pmg, 0)
    out = []
    for i in range(h[2]):
        p = _PIECE.unpack_from(pmg, h[21] + i * _PIECE.size)
        verts, stride, pos_off, uv_off = p[1], p[15], p[16], p[18]
        if uv_off < 0:  # a piece without UVs
            out.append(_range(np.zeros((0, 2))))
            continue
        raw = np.frombuffer(pmg, np.uint8, count=verts * stride, offset=pos_off).reshape(verts, stride)
        col = uv_off - pos_off
        out.append(_range(raw[:, col:col + 8].copy().view("<f4").reshape(verts, 2)))
    return out


def flat(ranges: list[tuple]) -> bool:
    """True when a piece of the model has all its vertices on one UV point (in game: one texel of its texture, a flat
    colour) or no UVs. A model without pieces draws nothing (willet's mount shells carry only locators): not flat."""
    return any(r[0] == 0 or (r[1] == r[2] and r[3] == r[4]) for r in ranges)


def describe(ranges: list[tuple]) -> str:
    return " | ".join(f"{n} v u {u0:.2f}..{u1:.2f} v {v0:.2f}..{v1:.2f} ({z} at 0,0)"
                      for n, u0, u1, v0, v1, z in ranges)
