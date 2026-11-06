"""Checks of tools/shells.py. Run: python tools/test_shells.py

The byte layouts are willet's one-part ETS2 shell (toybed/shiba_bed.pmd / .pmg), which our writer reproduced byte for
byte on 2026-11-06; the placements are ETS2 9.2.0's (all 54 poses and 5 pairs matched its stage that day).
"""
from __future__ import annotations

import math
import struct
import sys

import shells

SHIBA_PMD = bytes.fromhex(
    "04000000000000000100000000000000010000000100000001000000040000000000000040000000480000005000000068000000"
    "580000006c0000006c00000000e7c3731500000000e7c373150000000000000001000000ceb971ed0a000000000000000000000001000000")


def close(a, b, tol=1e-6) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_pmd() -> None:
    assert shells.shell_pmd() == SHIBA_PMD


def test_pmg_layout() -> None:
    data = shells.shell_pmg([shells.Pet("shiba_bed", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))])
    assert len(data) == 192 and data[:4] == b"\x15gmP"
    h = shells.HEADER.unpack_from(data, 0)
    assert h[2:7] == (0, 1, 0, 0, 1)                      # no pieces, one part, no bones, one locator
    assert data[h[21]:h[21] + h[23]] == b"shiba_bed\0\0\0"  # string pool, padded to 4
    assert h[24] == h[26] == len(data) and h[25] == h[27] == 0


def test_pair_pool() -> None:
    pets = [shells.Pet("dd_tuay_stb", (0.1, 0, 0), shells.yaw(90)), shells.Pet("dd_boon_bed", (-0.1, 0, 0), shells.yaw(-90)),
            shells.Pet("dd_tuay_stb", (0, 0, 0.2), shells.yaw(0))]
    data = shells.shell_pmg(pets)
    h = shells.HEADER.unpack_from(data, 0)
    pool = data[h[21]:h[21] + h[23]]
    assert pool == b"dd_tuay_stb\0dd_boon_bed\0"
    hookups = [shells.LOCATOR.unpack_from(data, h[20] + i * shells.LOCATOR.size)[-1] for i in range(3)]
    assert hookups == [0, 12, 0]


def test_place() -> None:
    pet = shells.place("dd_x", 90, (0.0, 0.0, -1.0), (0.5, 0.1, 0.0))
    # a quarter turn left (+Y) sends the pose's -z (forward) to -x
    assert close(pet.position, (-0.5, 0.1, 0.0)) and close(pet.rotation, (math.sqrt(0.5), 0, math.sqrt(0.5), 0))
    pair = {"members": [{"hookup": "dd_a", "mount": {"turn": -90, "move": [0.06, 0.0, 0.18]}, "shift": [0.225, 0, 0]}]}
    (member,) = shells.pair_pets(pair, (0.0, 0.05, 0.0))
    assert close(member.position, (0.225 - 0.18, 0.05, 0.06))


def test_bad_hookup() -> None:
    try:
        shells.shell_pmg([shells.Pet("Dog-1", (0, 0, 0), (1, 0, 0, 0))])
    except ValueError:
        return
    raise AssertionError("a hookup name that is no token must be refused")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"shells: {len(tests)} checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
