"""Scale an ETS2 skinned model (PMG 0x15) and its animation (PMA v5) uniformly about the model origin.

A uniform shrink by s is the same transform applied around every bone: every translation-like value (vertex
positions, bone bind matrices' translation, bone local translations, animation frame translations, root motion,
bounds) is multiplied by s and every rotation, stretch and bone scale stays as it is. Skinning then gives exactly
s x the original pose, in every frame. Normals and tangents don't change. Used for willet's dogs (derived parts).

Layouts as in ConverterPIX (mwl4) structs/pmg_0x15.h and structs/pma_0x05.h.

Copied unchanged from the ETS2 Pets repo's tools/pmg_scale.py (ADR 0001).
"""
from __future__ import annotations

import struct

import numpy as np

_HEADER = struct.Struct("<B3s5iQ4f3f3f10i")  # 0x70 bytes
_PIECE_SIZE = 100
_LOCATOR_SIZE = 44
_BONE_SIZE = 200
_PMA_HEADER = struct.Struct("<IHHIQ3ff3f3ff5i")  # 84 bytes
_PMA_FRAME_SIZE = 56
_PMA_MOVEMENT = 1 << 1
PMG_VERSION = 0x15
PMA_VERSION = 5


def _scale3(buf: bytearray, at: int, s: float) -> None:
    struct.pack_into("<3f", buf, at, *(v * s for v in struct.unpack_from("<3f", buf, at)))


def _scale1(buf: bytearray, at: int, s: float) -> None:
    struct.pack_into("<f", buf, at, struct.unpack_from("<f", buf, at)[0] * s)


def _matrix(buf: bytes, at: int) -> np.ndarray:
    return np.array(struct.unpack_from("<16f", buf, at), dtype=np.float64).reshape(4, 4)


def scale_pmg(data: bytes, s: float) -> bytes:
    """Return a copy of a PMG 0x15 model scaled by s about its origin (skinned or not)."""
    if len(data) < _HEADER.size or data[1:4] != b"gmP" or data[0] != PMG_VERSION:
        raise ValueError("not a PMG 0x15 model")
    h = _HEADER.unpack_from(data, 0)
    piece_count, bone_count, locator_count = h[2], h[4], h[6]
    skeleton_offset, locators_offset, pieces_offset = h[18], h[20], h[21]
    out = bytearray(data)
    _scale3(out, 0x20, s)  # bounding-sphere centre
    _scale1(out, 0x2C, s)  # bounding diagonal
    _scale3(out, 0x30, s)  # bounding-box min
    _scale3(out, 0x3C, s)  # bounding-box max
    for b in range(bone_count):
        at = skeleton_offset + b * _BONE_SIZE
        bind, inverse = _matrix(out, at + 8), _matrix(out, at + 72)
        # translation sits in elements 3, 7, 11 (last row 0 0 0 1); the two matrices are each other's inverse
        if not (np.allclose(bind[3], [0, 0, 0, 1], atol=1e-5) and np.allclose(bind @ inverse, np.eye(4), atol=1e-3)):
            raise ValueError(f"bone {b}: unexpected bind matrix layout")
        for m in (at + 8, at + 72):
            for k in (3, 7, 11):
                _scale1(out, m + 4 * k, s)
        _scale3(out, at + 168, s)  # local translation
        if not np.allclose(_matrix(out, at + 8) @ _matrix(out, at + 72), np.eye(4), atol=1e-3):
            raise ValueError(f"bone {b}: scaled bind matrices are no longer inverse")
    for i in range(locator_count):
        _scale3(out, locators_offset + i * _LOCATOR_SIZE + 8, s)
    scaled: set[int] = set()
    for i in range(piece_count):
        base = pieces_offset + i * _PIECE_SIZE
        vertices = struct.unpack_from("<i", out, base + 4)[0]
        stride, position_offset = struct.unpack_from("<2i", out, base + 60)
        _scale3(out, base + 20, s)  # piece centre
        _scale1(out, base + 32, s)  # piece diagonal
        _scale3(out, base + 36, s)  # piece min
        _scale3(out, base + 48, s)  # piece max
        if position_offset in scaled or vertices <= 0:
            continue
        scaled.add(position_offset)
        positions = np.ndarray((vertices, 3), dtype="<f4", buffer=out, offset=position_offset, strides=(stride, 4))
        positions *= np.float32(s)
    return bytes(out)


def scale_pma(data: bytes, s: float) -> bytes:
    """Return a copy of a PMA v5 animation with every translation scaled by s (pairs with scale_pmg)."""
    if len(data) < _PMA_HEADER.size or struct.unpack_from("<I", data, 0)[0] != PMA_VERSION:
        raise ValueError("not a PMA v5 animation")
    h = _PMA_HEADER.unpack_from(data, 0)
    frames, flags, bones = h[1], h[2], h[3]
    frames_offset, delta_trans_offset = h[18], h[19]
    if frames_offset + frames * bones * _PMA_FRAME_SIZE > len(data):
        raise ValueError("frame table runs past the end of the file")
    out = bytearray(data)
    _scale3(out, 20, s)  # bounding-sphere centre
    _scale1(out, 32, s)  # bounding-sphere radius
    _scale3(out, 36, s)  # bounding-box min
    _scale3(out, 48, s)  # bounding-box max
    for f in range(frames * bones):
        _scale3(out, frames_offset + f * _PMA_FRAME_SIZE + 32, s)
    if flags & _PMA_MOVEMENT:
        for f in range(frames):
            _scale3(out, delta_trans_offset + f * 12, s)
    return bytes(out)


def rename_hookup(data: bytes, name: str) -> bytes:
    """Return a copy of a mesh-less shell PMG whose locators all name the hookup `name` (the string pool holds just
    the one hookup name, as willet's shells do)."""
    if len(data) < _HEADER.size or data[1:4] != b"gmP" or data[0] != PMG_VERSION:
        raise ValueError("not a PMG 0x15 model")
    h = list(_HEADER.unpack_from(data, 0))
    piece_count, locator_count, locators_offset = h[2], h[6], h[20]
    pool_offset, pool_size = h[22], h[23]
    if piece_count or h[25] or h[27] or pool_offset + pool_size != len(data):
        raise ValueError("not a mesh-less shell (pieces or vertex data after the string pool)")
    pool = name.encode("ascii") + b"\0"
    pool += b"\0" * (-len(pool) % 4)
    out = bytearray(data[:pool_offset]) + pool
    for i in range(locator_count):
        at = locators_offset + i * _LOCATOR_SIZE + 40
        if struct.unpack_from("<i", out, at)[0] >= 0:
            struct.pack_into("<i", out, at, 0)
    h[23] = len(pool)
    h[24] = h[26] = pool_offset + len(pool)  # empty vertex and index pools sit at the end
    _HEADER.pack_into(out, 0, *h)
    return bytes(out)


_PART = struct.Struct("<Q4i")  # name token, piece count, first piece, locator count, first locator


def pair_shell(members: list[tuple[bytes, str]]) -> bytes:
    """Return a mesh-less shell PMG that places several animated hookups at once (2026-11-01, duo pilot: two pets in
    one slot). `members` = [(shell PMG, hookup name)]: copies of ONE shell model, each with its locators already turned
    and moved for that member (pmg_shift.turn_locators). Every part keeps its name and gets, in member order, each
    member's locators of that part, its hookup locators naming the member's hookup; the string pool holds the names.
    One member is a single placed pet (2026-11-01, the table items)."""
    if not members:
        raise ValueError("a pair shell needs a member")
    shells = []
    for data, _ in members:
        if len(data) < _HEADER.size or data[1:4] != b"gmP" or data[0] != PMG_VERSION:
            raise ValueError("not a PMG 0x15 model")
        h = _HEADER.unpack_from(data, 0)
        if h[2] or h[4] or h[25] or h[27]:
            raise ValueError("not a mesh-less shell (pieces, bones or vertex data)")
        parts = [_PART.unpack_from(data, h[19] + i * _PART.size) for i in range(h[3])]
        locators = [data[h[20] + i * _LOCATOR_SIZE:h[20] + (i + 1) * _LOCATOR_SIZE] for i in range(h[6])]
        shells.append((h, parts, locators))
    h0, parts0, _ = shells[0]
    for h, parts, _ in shells[1:]:
        if [(p[0], p[3]) for p in parts] != [(p[0], p[3]) for p in parts0]:
            raise ValueError("the members' shells are not copies of one shell model (parts / locators differ)")
    pool, names = b"", []
    for _, name in members:
        names.append(len(pool))
        pool += name.encode("ascii") + b"\0"
    pool += b"\0" * (-len(pool) % 4)
    parts_out, locators_out = bytearray(), bytearray()
    count = 0
    for k, part in enumerate(parts0):
        first = count
        for (_, parts, locators), name_at in zip(shells, names):
            for loc in locators[parts[k][4]:parts[k][4] + parts[k][3]]:
                loc = bytearray(loc)
                if struct.unpack_from("<i", loc, 40)[0] >= 0:
                    struct.pack_into("<i", loc, 40, name_at)
                locators_out += loc
                count += 1
        parts_out += _PART.pack(part[0], part[1], part[2], count - first, first)
    h = list(h0)
    h[6] = count
    h[18] = h[19] = _HEADER.size                       # no skeleton; the parts first
    h[20] = h[19] + len(parts_out)                     # locators
    h[21] = h[22] = h[20] + len(locators_out)          # no pieces; the string pool
    h[23] = len(pool)
    h[24] = h[26] = h[22] + len(pool)                  # empty vertex and index pools at the end
    h[25] = h[27] = 0
    return _HEADER.pack(*h) + bytes(parts_out) + bytes(locators_out) + pool
