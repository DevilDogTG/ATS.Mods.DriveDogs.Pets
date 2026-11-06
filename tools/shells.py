"""Our own mesh-less mount shells (ADR 0001): a model with no geometry whose locators place animated hookups (the pets)
in a cab slot or on a dashboard table. The game puts the shell at the slot's locator; each shell locator names one
pet's animated hookup and sits where that pet goes, in the slot's frame.

Layout as willet's ETS2 shells (Workshop 2997276736), which the ETS2 pack turns per pose: PMG 0x15 with one part
"defaultpart" holding the locators, a string pool with the hookup names, no pieces, bones or vertex data; PMD 0x04 with
one look, one variant and that part visible. ATS cabs are left-hand drive only: no `_uk` shells.

A pet's locator is the ETS2 placement of the same pose (tools/build_pack.py turned_shell / pair_shell there, on a
shell locator at the origin): turned by the mount's `turn` about the vertical axis, moved by its `move` in the turned
frame, and by `offset` in the slot's frame (a pair member's `shift`, a truck's floor lift or seat centre).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

HEADER = struct.Struct("<B3s5iQ10f10i")  # PMG 0x15 header, 112 bytes
PART = struct.Struct("<Q4i")              # name token, piece count, first piece, locator count, first locator
LOCATOR = struct.Struct("<Q3ff4fi")       # name token, position, scale, rotation (w, x, y, z), hookup offset
PMD_HEADER = struct.Struct("<16I")        # PMD 0x04 header, 64 bytes
PMG_VERSION = 0x15
PMD_VERSION = 4
FLT_MAX = 3.4028234663852886e+38
SHELL_ROOT = "/vehicle/truck/upgrade/interior_decors/drivedogs"  # <name>/shell.pmd + .pmg, beside the imported pets
TOKEN_CHARS = "\0" + "0123456789abcdefghijklmnopqrstuvwxyz_"


def token_value(name: str) -> int:
    """An SCS token: at most 12 of [0-9a-z_]."""
    if not 0 < len(name) <= 12 or any(c not in TOKEN_CHARS[1:] for c in name):
        raise ValueError(f"{name!r}: not a token (1-12 of 0-9, a-z, _)")
    return sum(TOKEN_CHARS.index(c) * 38 ** i for i, c in enumerate(name))


@dataclass(frozen=True)
class Pet:
    """One shell locator: the animated hookup it places, its position (slot frame, metres) and rotation (w, x, y, z)."""
    hookup: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


def yaw(deg: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) of a turn by `deg` about the vertical (+Y) axis."""
    half = math.radians(deg) / 2
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Vector v rotated by the unit quaternion q = (w, x, y, z)."""
    w, qx, qy, qz = q
    tx, ty, tz = 2 * (qy * v[2] - qz * v[1]), 2 * (qz * v[0] - qx * v[2]), 2 * (qx * v[1] - qy * v[0])
    return (v[0] + w * tx + qy * tz - qz * ty, v[1] + w * ty + qz * tx - qx * tz, v[2] + w * tz + qx * ty - qy * tx)


def place(hookup: str, turn: float = 0.0, move=None, offset=(0.0, 0.0, 0.0)) -> Pet:
    """A pet placed as its ETS2 mount places it: turned by `turn` degrees, moved by `move` in its own (turned) frame
    and by `offset` in the slot's frame."""
    q = yaw(turn)
    lx, ly, lz = rotate(q, tuple(move) if move else (0.0, 0.0, 0.0))
    return Pet(hookup, (offset[0] + lx, offset[1] + ly, offset[2] + lz), q)


def pose_pets(pose: dict, offset=(0.0, 0.0, 0.0)) -> list[Pet]:
    """The pet of one imported pose (models.json entry), moved by `offset` (slot frame)."""
    mount = pose["mount"]
    return [place(pose["hookup"], mount.get("turn", 0), mount.get("move"), offset)]


def pair_pets(pair: dict, offset=(0.0, 0.0, 0.0)) -> list[Pet]:
    """The members of one imported pair or table item (pairs.json entry): each placed by its own mount, moved by its
    `shift` and by `offset` (slot frame)."""
    return [place(m["hookup"], m["mount"].get("turn", 0), m["mount"].get("move"),
                  tuple(s + o for s, o in zip(m["shift"], offset))) for m in pair["members"]]


def shell_files(name: str, pets: list[Pet]) -> tuple[str, dict[str, bytes]]:
    """(the .pmd path a def names, {game path: bytes}) of the shell `name` placing `pets`."""
    pmd = f"{SHELL_ROOT}/{name}/shell.pmd"
    return pmd, {pmd: shell_pmd(), pmd[:-4] + ".pmg": shell_pmg(pets)}


def shell_pmg(pets: list[Pet]) -> bytes:
    """A mesh-less shell PMG: one part "defaultpart" with a locator "default" per pet, naming its hookup."""
    if not pets:
        raise ValueError("a shell places at least one pet")
    pool, at = b"", {}
    for pet in pets:
        token_value(pet.hookup)  # hookup units are tokens too
        if pet.hookup not in at:
            at[pet.hookup] = len(pool)
            pool += pet.hookup.encode("ascii") + b"\0"
    pool += b"\0" * (-len(pool) % 4)
    parts = PART.pack(token_value("defaultpart"), 0, 0, len(pets), 0)
    locators = b"".join(LOCATOR.pack(token_value("default"), *p.position, 1.0, *p.rotation, at[p.hookup])
                        for p in pets)
    parts_off = HEADER.size
    locators_off = parts_off + len(parts)
    pool_off = locators_off + len(locators)
    end = pool_off + len(pool)
    header = HEADER.pack(PMG_VERSION, b"gmP", 0, 1, 0, 0, len(pets), 0,
                         0.0, 0.0, 0.0, 0.0, FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX,
                         parts_off, parts_off, locators_off, pool_off, pool_off, len(pool), end, 0, end, 0)
    return header + parts + locators + pool


def shell_pmd() -> bytes:
    """The PMD of a one-part shell: look "default", variant "default", the part visible, no materials."""
    looks = struct.pack("<Q", token_value("default"))
    variants = struct.pack("<Q", token_value("default"))
    part_attributes = struct.pack("<2i", 0, 1)                         # the part's attributes: [0, 1)
    attributes = struct.pack("<Q2i", token_value("visible"), 0, 0)    # "visible", type int, value offset 0
    values = struct.pack("<i", 1)                                     # variant "default": visible = 1
    look_off = PMD_HEADER.size
    variant_off = look_off + len(looks)
    part_attr_off = variant_off + len(variants)
    attr_off = part_attr_off + len(part_attributes)
    value_off = attr_off + len(attributes)
    material_off = value_off + len(values)
    header = PMD_HEADER.pack(PMD_VERSION, 0, 1, 0, 1, 1, 1, len(values), 0,
                             look_off, variant_off, part_attr_off, value_off, attr_off, material_off, material_off)
    return header + looks + variants + part_attributes + attributes + values
