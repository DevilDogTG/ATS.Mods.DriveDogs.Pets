"""Measure where the pets sit in the ATS cabs (extracted base game, tools/cabs.py); changes nothing unless --write.

  seat   the co-driver seat cushion of every ATS truck (each interior, its variant's parts): its centre in the toyseat
         slot frame (x forward, z sideways) - fixes.yaml seat_centre, where build_pack moves seat poses.
  floor  the floor under every ATS truck's toybig locator (each interior): the highest flat upward surface over the
         locator within 20 cm of it - fixes.yaml slot_floor, what build_pack raises floor pets by.
  bed    each bed pose / pair (or one token) as its shell places it at frame 0, laid on every ATS bunk (each interior
         with a toybed locator): the share of its footprint off the mattress (the flat surface at the locator's height
         reachable from the locator without crossing anything 3-40 cm above it). --sweep: per truck, the one shift
         across the bunk that fits all its bed pets best - fixes.yaml bed_shift. Pets are never re-posed here, only
         moved as a whole.

seat / floor / bed --sweep with --write rewrite that block of fixes.yaml (a truck's value is the mean over its interiors). The
measuring code (cushion, floor, mattress) is copied from the ETS2 Pets repo's tools/pose_check.py `seat`, `floor` and
`bed` (ADR 0001); the cab lists are the ATS interiors instead of hand-kept lists, LHD only (ATS has no UK cabs).

Usage: python tools/cab_fit.py seat [--write]
       python tools/cab_fit.py floor [--write]
       python tools/cab_fit.py bed [<token>]
       python tools/cab_fit.py bed --sweep [--write]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cabs import Interior, Model, base, interiors  # noqa: E402
from common import IMPORT, REPO  # noqa: E402
from pma_pose import posed_triangles  # noqa: E402
from shells import pair_pets, pose_pets  # noqa: E402

FIXES = REPO / "fixes.yaml"


def _rotate(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    w, v = q[0], q[1:]
    return p + 2 * np.cross(v, np.cross(v, p) + w * p)


def _surface_points(tris: np.ndarray, step: float, seed: int = 1) -> np.ndarray:
    """Points spread over triangles about `step` apart (vertices included)."""
    rng = np.random.default_rng(seed)
    pts = [tris.reshape(-1, 3)]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    k = np.clip((np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2 / step ** 2).astype(int), 0, 300)
    for n in range(1, 301):
        sel = k >= n
        if not sel.any():
            break
        r1, r2 = rng.random((sel.sum(), 1)), rng.random((sel.sum(), 1))
        flip = (r1 + r2) > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        pts.append(a[sel] + (b[sel] - a[sel]) * r1 + (c[sel] - a[sel]) * r2)
    return np.concatenate(pts)


def _label(interior: Interior) -> str:
    return f"{Path(interior.model.path).stem} / {interior.variant or '-'}"


# ----------------------------------------------------------------------------------------------- seat centres
SEAT_CELL, SEAT_HALF, SEAT_MIN_AREA = 0.01, 0.5, 0.12


def cushion(interior: Interior, seat: Model | None = None) -> tuple[float, float, float] | None:
    """The co-driver seat cushion around the interior's toyseat locator (with `seat`, a co-driver seat accessory, at
    its codrv_seat locator), in the slot frame: (x centre, z centre, area m2) of the cells reachable from the locator
    whose highest surface is 6 cm below to 10 cm above it (higher: the backrest, the armrest; lower: off the seat).
    None without a toyseat locator."""
    slot = interior.slot("toyseat")
    if slot is None:
        return None
    loc, q = slot
    r = _rotate(q, np.eye(3)).T
    t = (interior.triangles(seat) - loc) @ r
    c = t.mean(1)
    half, cell = SEAT_HALF, SEAT_CELL
    t = t[(np.abs(c[:, 0]) < half + 0.1) & (np.abs(c[:, 2]) < half + 0.1) & (c[:, 1] > -0.4) & (c[:, 1] < 0.8)]
    p = _surface_points(t, 0.008)
    n = int(2 * half / cell)
    top = np.full((n, n), -9.0)
    i, j = ((p[:, 0] + half) / cell).astype(int), ((p[:, 2] + half) / cell).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    np.maximum.at(top, (i[ok], j[ok]), p[ok, 1])
    seat = (top > -0.06) & (top < 0.10)
    si, sj = np.nonzero(seat)
    if not len(si):
        return None
    k = ((si - n // 2) ** 2 + (sj - n // 2) ** 2).argmin()
    seen = np.zeros_like(seat)
    stack = [(si[k], sj[k])]
    while stack:
        a, b = stack.pop()
        if 0 <= a < n and 0 <= b < n and seat[a, b] and not seen[a, b]:
            seen[a, b] = True
            stack += [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
    a, b = np.nonzero(seen)
    x, z = a * cell - half, b * cell - half
    return (x.min() + x.max()) / 2, (z.min() + z.max()) / 2, len(a) * cell * cell


def seats(cabs: dict[str, list[Interior]]) -> dict[str, list[float]]:
    found = {}
    for truck, rows in cabs.items():
        good = []
        for interior, seat in [(i, s) for i in rows for s in (i.seats or [None])]:
            g = cushion(interior, seat)
            ok = g is not None and g[2] >= SEAT_MIN_AREA
            label = _label(interior) + (f" + {Path(seat.path).stem}" if seat else "")
            print(f"  {truck:26s} {label:44s} "
                  + ("no toyseat locator / cushion" if g is None else f"centre ({g[0]:+.3f}, {g[1]:+.3f}) {g[2]:.2f} m2"
                     + ("" if ok else "  (too small: left out)")))
            if ok:
                good.append(g)
        if good:
            xs, zs = [g[0] for g in good], [g[1] for g in good]
            found[truck] = [round(float(np.mean(xs)), 3), round(float(np.mean(zs)), 3)]
            print(f"  {truck:26s} -> [{found[truck][0]:+.3f}, {found[truck][1]:+.3f}]  (spread x "
                  f"{(max(xs) - min(xs)) * 100:.1f} cm, z {(max(zs) - min(zs)) * 100:.1f} cm over {len(good)})")
    return found


# ------------------------------------------------------------------------------------------------- floor slot
def floor_height(interior: Interior) -> float | None:
    """The floor under the toybig locator: the highest flat upward surface over it (5 cm square) within 20 cm of its
    height, relative to it (m). None without a toybig locator; 0 when no floor is found."""
    slot = interior.slot("toybig")
    if slot is None:
        return None
    loc, _ = slot
    t = interior.triangles() - loc
    t = t[(np.abs(t[:, :, 0]).min(1) < 1.0) & (np.abs(t[:, :, 2]).min(1) < 1.0) & (t[:, :, 1].min(1) < 1.4)]
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    flat = np.abs(nrm[:, 1]) > 0.9 * np.linalg.norm(nrm, axis=1)
    over = ((t[:, :, 0].min(1) < 0.05) & (t[:, :, 0].max(1) > -0.05)
            & (t[:, :, 2].min(1) < 0.05) & (t[:, :, 2].max(1) > -0.05))
    ys = t.mean(1)[flat & over, 1]
    ys = ys[np.abs(ys) < 0.2]
    return float(ys.max()) if ys.size else 0.0


def floors(cabs: dict[str, list[Interior]]) -> dict[str, float]:
    found = {}
    for truck, rows in cabs.items():
        got = []
        for interior in rows:
            fy = floor_height(interior)
            print(f"  {truck:26s} {_label(interior):34s} "
                  + ("no toybig locator" if fy is None else f"floor {fy * 100:+5.1f} cm"))
            if fy is not None:
                got.append(fy)
        if got:
            found[truck] = round(float(np.mean(got)), 3)
            print(f"  {truck:26s} -> {found[truck] * 100:+.1f} cm  (spread {(max(got) - min(got)) * 100:.1f} cm "
                  f"over {len(got)})")
    return found


# ------------------------------------------------------------------------------------------------------- beds
BED_CELL = 0.02
BED_HALF = np.array([1.4, 0.9])


def mattress(interior: Interior) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(rotation, flat, free) grids over the cab's x/z around the toybed locator (BED_HALF, BED_CELL): flat at the
    locator's height, and the mattress = flat cells reachable from the locator without crossing anything 3-40 cm above
    it (walls, cupboards, the seat backs). None without a toybed locator."""
    slot = interior.slot("toybed")
    if slot is None:
        return None
    loc, q = slot
    cell, half = BED_CELL, BED_HALF
    t = interior.triangles() - loc
    n = (2 * half / cell).astype(int)
    flat = np.zeros(n, bool)
    block = np.zeros(n, bool)
    normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    lens = np.linalg.norm(normals, axis=1)
    ok = lens > 1e-9
    t, normals = t[ok], normals[ok] / lens[ok, None]
    near = (np.abs(t[:, :, 0]).min(1) < half[0] + 0.3) & (np.abs(t[:, :, 2]).min(1) < half[1] + 0.3)
    t, normals = t[near], normals[near]

    def mark(grid, tri):  # every cell a triangle's projection (x, z) covers, edges included
        for a, b in ((0, 1), (1, 2), (2, 0)):
            for s in np.linspace(0, 1, max(2, int(np.linalg.norm(tri[b] - tri[a]) / cell * 2) + 2)):
                p = tri[a] + (tri[b] - tri[a]) * s
                i, j = int((p[0] + half[0]) / cell), int((p[2] + half[1]) / cell)
                if 0 <= i < n[0] and 0 <= j < n[1]:
                    grid[i, j] = True
        lo = np.floor((tri[:, [0, 2]].min(0) + half) / cell).astype(int)
        hi = np.ceil((tri[:, [0, 2]].max(0) + half) / cell).astype(int)
        a, b, c = tri[0][[0, 2]], tri[1][[0, 2]], tri[2][[0, 2]]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            return
        for i in range(max(lo[0], 0), min(hi[0], n[0])):
            for j in range(max(lo[1], 0), min(hi[1], n[1])):
                px, pz = -half[0] + (i + 0.5) * cell, -half[1] + (j + 0.5) * cell
                l1 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (pz - c[1])) / den
                l2 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (pz - c[1])) / den
                if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
                    grid[i, j] = True

    for tri, nrm in zip(t, normals):
        ys = tri[:, 1]
        if abs(nrm[1]) > 0.8 and np.abs(ys).max() < 0.05:
            mark(flat, tri)
        elif ys.max() > 0.03 and ys.min() < 0.40:
            mark(block, tri)
    free = np.zeros(n, bool)
    start = (int(half[0] / cell), int(half[1] / cell))
    stack = [start] if flat[start] and not block[start] else []
    while stack:
        i, j = stack.pop()
        if not (0 <= i < n[0] and 0 <= j < n[1]) or free[i, j] or not flat[i, j] or block[i, j]:
            continue
        free[i, j] = True
        stack += [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    return q, flat, free


def off_bed(pet_pts: np.ndarray, q: np.ndarray, free: np.ndarray) -> float:
    """The share of the pet's footprint (slot frame points, laid on the bunk) off the mattress."""
    cell, half, n = BED_CELL, BED_HALF, free.shape
    dog = _rotate(q, pet_pts)
    di = ((dog[:, 0] + half[0]) / cell).astype(int).clip(0, n[0] - 1)
    dj = ((dog[:, 2] + half[1]) / cell).astype(int).clip(0, n[1] - 1)
    foot = np.zeros(n, bool)
    foot[di, dj] = True
    return float((foot & ~free).sum() / max(foot.sum(), 1))


def bed_items() -> dict[str, np.ndarray]:
    """Every bed pose / pair: its points (frame 0) as its shell places them in the toybed slot frame."""
    poses = json.loads((IMPORT / "pets/models.json").read_text(encoding="utf-8"))
    pairs = json.loads((IMPORT / "pairs/pairs.json").read_text(encoding="utf-8"))
    files = {p["hookup"]: (p["model"], p["animation"]) for p in poses}
    for pair in pairs:
        for m in pair["members"]:
            if "model" in m:
                files[m["hookup"]] = (m["model"], m["animation"])
    out = {}
    for item, pets in ([(p, pose_pets(p)) for p in poses] + [(p, pair_pets(p)) for p in pairs]):
        mount = item["mount"]
        if mount.get("addon") or mount.get("to_slot", mount.get("slot")) != "toybed":
            continue
        pts = []
        for pet in pets:
            model, animation = files[pet.hookup]
            root = IMPORT / ("pets" if pet.hookup in {p["hookup"] for p in poses} else "pairs") / "out"
            tris = posed_triangles(root / model.lstrip("/"), root / animation.lstrip("/"), 0).reshape(-1, 3)
            pts.append(_rotate(np.array(pet.rotation), tris) + np.array(pet.position))
        out[item["token"]] = np.concatenate(pts)
    return out


def beds(cabs: dict[str, list[Interior]], only: str | None) -> None:
    items = bed_items()
    if only:
        if only not in items:
            raise SystemExit(f"{only}: not a bed pose / pair ({', '.join(sorted(items))})")
        items = {only: items[only]}
    bunks = [(truck, interior, m) for truck, rows in cabs.items() for interior in rows
             if (m := mattress(interior)) is not None]
    print(f"% of each bed pet's footprint off the mattress, per bunk ({len(bunks)} interiors with a toybed locator):")
    for token, pts in items.items():
        shares = [(truck, interior, off_bed(pts, q, free), free.any()) for truck, interior, (q, _, free) in bunks]
        worst = sorted((s for s in shares if s[3]), key=lambda s: -s[2])
        bad = [s for s in worst if s[2] > 0.02]
        print(f"  {token:10s} off > 2 % on {len(bad)}/{len(worst)} bunks; worst: "
              + ", ".join(f"{t} {Path(i.model.path).stem}/{i.variant} {s:.0%}" for t, i, s, _ in worst[:4]))
    missing = sorted({f"{t} {Path(i.model.path).stem}/{i.variant}" for t, i, (_, _, free) in bunks if not free.any()})
    if missing:
        print(f"  no mattress found at the locator: {len(missing)}: " + "; ".join(missing))


BED_SHIFTS = range(-6, 22, 2)  # cm along the slot's z (across the bunk)


def bed_sweep(cabs: dict[str, list[Interior]]) -> dict[str, float]:
    """One shift across the bunk (slot z) per truck for all its bed pets: where a truck's mattress is not centred on
    its toybed locator (the big dogs hang off one edge), the shift in BED_SHIFTS with the lowest worst share off the
    mattress over all bed pets and bunks (whole %), the smallest one on a tie. Trucks whose pets all fit (<= 2 % off)
    keep 0."""
    items = bed_items()
    found = {}
    for truck, rows in cabs.items():
        bunks = [m for interior in rows if (m := mattress(interior)) is not None and m[2].any()]
        if not bunks:
            continue

        def worst(dz: float) -> tuple[float, str]:
            shares = [(off_bed(pts + np.array([0.0, 0.0, dz / 100]), q, free), token)
                      for token, pts in items.items() for q, _, free in bunks]
            return max(shares)

        now = worst(0)
        if now[0] <= 0.02:
            print(f"  {truck:26s} all bed pets fit ({len(bunks)} bunks, worst {now[0]:.0%} {now[1]})")
            continue
        best = min(BED_SHIFTS, key=lambda dz: (round(worst(dz)[0] * 100), abs(dz)))
        after = worst(best)
        found[truck] = best / 100
        print(f"  {truck:26s} worst {now[0]:4.0%} ({now[1]}) -> shift z {best:+d} cm: worst {after[0]:4.0%} ({after[1]}),"
              f" {len(bunks)} bunks")
    return found


# ------------------------------------------------------------------------------------------------- fixes.yaml
def write_block(key: str, comment: list[str], lines: list[str]) -> None:
    """Replace fixes.yaml's `key` block (its comment above it included) with this one."""
    text = FIXES.read_text(encoding="utf-8") if FIXES.is_file() else "# Measured fixes applied by tools/build_pack.py.\nschema: 1\n"
    text = re.sub(rf"\n(#[^\n]*\n)*{key}:.*?(?=\n#|\n[a-z_]+:|\Z)", "", text, flags=re.DOTALL).rstrip("\n") + "\n"
    FIXES.write_text(text + "\n" + "\n".join(comment + [f"{key}:"] + lines) + "\n", encoding="utf-8", newline="\n")
    print(f"  written to {FIXES}")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("seat", "floor", "bed"):
        print(__doc__)
        return 2
    cabs = interiors(base())
    mode, write = argv[1], "--write" in argv
    if mode == "seat":
        found = seats(cabs)
        if write:
            write_block("seat_centre", [
                "# seat_centre: where each truck's co-driver seat cushion is, in its toyseat slot frame (metres: x forward,",
                "# z sideways; ATS cabs are left-hand drive only): seat poses are moved there (build_pack, their shell",
                "# locators), after their mount `move` brought each pose's rest point to the slot. Mean over the truck's",
                "# interiors, measured with `python tools/cab_fit.py seat --write`."],
                [f"  {t}: [{x:.3f}, {z:.3f}]" for t, (x, z) in found.items()])
    elif mode == "floor":
        found = floors(cabs)
        if write:
            write_block("slot_floor", [
                "# slot_floor: metres a part placed in a slot is raised on a truck whose floor is not at that slot's",
                "# locator height (build_pack: those trucks get their own shell copies). Mean over the truck's interiors,",
                "# measured with `python tools/cab_fit.py floor --write`."],
                ["  toybig:"] + [f"    {t}: {v:.3f}" for t, v in found.items()])
    elif "--sweep" in argv:
        found = bed_sweep(cabs)
        if write:
            write_block("bed_shift", [
                "# bed_shift: metres bed pets are moved across the bunk (toybed slot z) on a truck whose mattress is not",
                "# centred on its toybed locator (build_pack: those trucks get their own shell copies). One shift per",
                "# truck for all bed pets, measured with `python tools/cab_fit.py bed --sweep --write`."],
                [f"  {t}: {v:.3f}" for t, v in found.items()])
    else:
        beds(cabs, next((a for a in argv[2:] if not a.startswith("--")), None))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
