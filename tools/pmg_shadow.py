"""A still, shadow-casting copy of a placed pet inside its mesh-less shell (2026-11-07, brains plan shadow-proxy: the
shadow pilot).

A pet is an animated hookup (skinned model + loop) named by a shell locator: in the cab it receives the sun's shadow
(the A-pillar's falls on it) but casts none. SCS casts shadows from geometry with a shadow-only material
(eut2.shadowonly: drawn into the shadow map, never seen; the Scania S 2024e chassis chs_4x2 has one). This puts a
copy of each member - its frame-0 posed mesh, welded - into the shell, one piece per hookup locator, placed as the
game places the hookup (locator position + rotation). Two kinds:
  only    eut2.shadowonly - the candidate: the pet as before, plus its shadow
  statue  the same copy drawn grey and visible, the locators' hookups dropped (no pet) - the diagnostic: does any
          shell geometry cast in the cab, and does the copy match the pet?

Contact patch (2026-11-08, 9.2.1 round 2; brains plan pets-pack subplans/table-shadow): the game's shadow-map bias drops
the bottom slice of every caster, so the sun shadow starts a few cm off the paws and the pet looks afloat. As SiSL's
table cat (workshop 645553604, cat2.sisl_tr: a quad 1.4 mm over its base, eut2.decalshadow + a soft footprint, white =
dark), each copy may get a quad on the table under it with its own baked contact shadow (contact_map: the ambient
occlusion of the copy's down-facing triangles on the table plane), material eut2.decalshadow (= eut2.shadowmap.decal.over,
an 8-bit shadow sampler; written RGBA, the value in every channel). Quad uv as SiSL's: v (rows) along +x, u (columns)
along -z, stride 36 as the statue.

Layouts: PMG 0x15 as pmg_scale / pma_pose. A shadow-only piece as SCS's own (chs_4x2 piece 18): stride 28 = position
3f, normal 3f, rgba 4 x u8 (0 0 0 255), no UVs (mask 0xFFFFFFFF, 0 channels); the statue adds one UV channel (stride
36, as chs_4x2 piece 0: uv after rgba, mask 0xFFFFFFF0) and rgba 128 (= 1.0, no baked shade). u16 indices, SCS winding
(clockwise: the outward normal is -cross(b - a, c - a)). PMD 0x04 (vanilla dashbrd_set_1.pmd): 16 u32 header
(version, materials, looks, pieces, variants, parts, attributes, attribute values size, material block size, then the
offsets of looks, variants, part attributes, attribute values, attributes, material offsets, material data); the
materials are look_count x material_count u32 absolute file offsets, then the null-terminated paths.

Copied unchanged from the ETS2 Pets repo's tools/pmg_shadow.py (ADR 0001).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pma_pose import _HEADER, _PIECE, compose, frame_locals, read_bones
from pmg_scale import _LOCATOR_SIZE, _PART, PMG_VERSION

_PMD = struct.Struct("<16I")
PMD_VERSION = 4
WELD = 1e-4  # metres: vertices this close are one (the pieces' seams, split normals / UVs)
FOLDER = "/vehicle/truck/upgrade/interior_decors/drivedogs/shadow"
MATERIALS = {
    "only": (f"{FOLDER}/proxy_only.mat",
             'effect : "eut2.shadowonly.rfx" {\n\tshadowmap_bias : 0.000000\n}\n'),
    "statue": (f"{FOLDER}/proxy_statue.mat",
               'effect : "eut2.dif.spec.rfx" {\n\tadditional_ambient : 0.000000\n'
               "\tdiffuse : { 0.600000 , 0.600000 , 0.600000 }\n\treflection : 0.000000\n"
               "\tshininess : 20.000000\n\tspecular : { 0.000000 , 0.000000 , 0.000000 }\n"
               '\ttexture : "texture_base" {\n\t\tsource : "/material/vehicle/helpers/umat_difspec.tobj"\n'
               "\t\tsampler : default\n\t}\n}\n"),
}
CONTACT_REACH = 0.08    # metres: triangles farther than this from a table point darken it not at all
CONTACT_MARGIN = 0.05   # metres: the quad = the copy's x / z box + this on every side (the patch is 0 at its edge)
CONTACT_LIFT = 0.0014   # metres over the table (SiSL's quad)
CONTACT_SINK = 0.003    # metres: the bake's table plane under the table top (the soles count, no holes)
CONTACT_GRID = 128      # bake samples per side, upscaled to CONTACT_SIZE
CONTACT_SIZE = 256      # texture side, px
CONTACT_DARK = 0.85     # the darkest the patch gets (SiSL's ~0.86)
CONTACT_GAIN = 1.5      # occlusion x this before the floor and the gamma (1.0 / 0.75: a ring too faint past the paws)
CONTACT_FLOOR = 0.01    # occlusion under this is 0 (the patch ends; a gamma would lift the far field everywhere)
CONTACT_GAMMA = 0.6     # < 1 widens the soft part (2.5 / 0.5 tried: a blob)


def contact_material(hookup: str) -> tuple[str, str, str]:
    """(material path, its text, its texture's .tobj path) of a hookup's contact patch."""
    tobj = f"{FOLDER}/contact_{hookup}.tobj"
    return (f"{FOLDER}/contact_{hookup}.mat",
            'effect : "eut2.decalshadow.rfx" {\n\tdiffuse : { 1.000000 , 1.000000 , 1.000000 }\n'
            f'\ttexture : "texture_base" {{\n\t\tsource : "{tobj}"\n\t\tsampler : default\n\t}}\n}}\n', tobj)


def posed_mesh(pmd: Path, pma: Path, frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A skinned model's mesh at one frame of its loop, every piece in one: (vertices (n, 3), triangles (m, 3)),
    welded (WELD) and without degenerate triangles. As pma_pose.posed_triangles, indexed."""
    pmg = pmd.with_suffix(".pmg").read_bytes()
    h = _HEADER.unpack_from(pmg, 0)
    bones = read_bones(pmg)
    world = compose(bones, frame_locals(pma.read_bytes(), frame, bones))
    skin = np.stack([w @ b["inverse"] for w, b in zip(world, bones)])
    verts, tris, base = [], [], 0
    for i in range(h[2]):
        p = _PIECE.unpack_from(pmg, h[21] + i * _PIECE.size)
        edges, count, stride, pos_off, bi_off, bw_off, idx_off = p[0], p[1], p[15], p[16], p[22], p[23], p[24]
        raw = np.frombuffer(pmg, dtype=np.uint8, count=count * stride, offset=pos_off).reshape(count, stride)
        pos = raw[:, :12].copy().view("<f4").reshape(count, 3).astype(np.float64)
        bi = np.lib.stride_tricks.as_strided(np.frombuffer(pmg, np.uint8, count * stride, bi_off), (count, 4),
                                             (stride, 1)).astype(int)
        bw = np.lib.stride_tricks.as_strided(np.frombuffer(pmg, np.uint8, count * stride, bw_off), (count, 4),
                                             (stride, 1)).astype(np.float64) / 255.0
        homo = np.c_[pos, np.ones(count)]
        posed = sum(bw[:, k, None] * np.einsum("vij,vj->vi", skin[bi[:, k]], homo)[:, :3] for k in range(4))
        idx = np.frombuffer(pmg, dtype="<u2", count=edges, offset=idx_off).astype(int)
        verts.append(posed)
        tris.append(idx[: len(idx) // 3 * 3].reshape(-1, 3) + base)
        base += count
    v, t = np.concatenate(verts), np.concatenate(tris)
    keys, first, remap = np.unique(np.round(v / WELD).astype(np.int64), axis=0, return_index=True,
                                   return_inverse=True)
    t = remap.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return v[first], t


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, outward for SCS's clockwise winding."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = -np.cross(b - a, c - a)
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, tris[:, k], face)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > 1e-12, n / np.maximum(length, 1e-12), [0.0, 1.0, 0.0])


def contact_map(verts: np.ndarray, tris: np.ndarray, table: float | None = None) -> tuple[Image.Image, list[float]]:
    """A copy's contact patch on the table (at height `table` in its hookup model's frame; default its lowest point,
    above it for a pet set into the table), in that frame: (shade, an L image CONTACT_SIZE square - 255 = darkest,
    rows along +x, columns along -z; quad [x_lo, x_hi, z_lo, z_hi, y]). Shade = CONTACT_DARK x o ^ CONTACT_GAMMA,
    o = (CONTACT_GAIN x occlusion - CONTACT_FLOOR) / (1 - CONTACT_FLOOR) in 0..1; occlusion = the sum over the
    triangles facing a table point of their form factor onto it (area cos cos / pi r^2) x (1 - r / CONTACT_REACH)^2,
    on CONTACT_GRID^2 points, upscaled and softened. Raises if the patch is not 0 at the quad's edge (it would show as
    a square)."""
    y0 = float(verts[:, 1].min() if table is None else table)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = -np.cross(b - a, c - a)
    double = np.linalg.norm(face, axis=1)
    keep = double > 1e-12
    area, normal, centre = double[keep] / 2, face[keep] / double[keep, None], ((a + b + c) / 3)[keep]
    lo, hi = verts[:, [0, 2]].min(0), verts[:, [0, 2]].max(0)
    mid, half = (lo + hi) / 2, float((hi - lo).max() / 2 + CONTACT_MARGIN)
    step = 2 * half / CONTACT_GRID
    zs = mid[1] + half - (np.arange(CONTACT_GRID) + 0.5) * step  # columns along -z
    occlusion = np.zeros((CONTACT_GRID, CONTACT_GRID))
    for row in range(CONTACT_GRID):
        x = mid[0] - half + (row + 0.5) * step
        points = np.stack([np.full(CONTACT_GRID, x), np.full(CONTACT_GRID, y0 - CONTACT_SINK), zs], 1)
        d = centre[None] - points[:, None]
        r2 = (d ** 2).sum(-1)
        r = np.sqrt(r2)
        up = d[..., 1] / r                         # cos at the table point (normal +y)
        facing = -(d * normal[None]).sum(-1) / r   # cos at the triangle (outward normal toward the point)
        fall = np.clip(1 - r / CONTACT_REACH, 0, 1) ** 2
        occlusion[row] = np.where((up > 0) & (facing > 0), area[None] * up * facing * fall / (np.pi * r2), 0).sum(1)
    level = np.clip((CONTACT_GAIN * occlusion - CONTACT_FLOOR) / (1 - CONTACT_FLOOR), 0, 1)
    shade = CONTACT_DARK * level ** CONTACT_GAMMA
    image = Image.fromarray(np.round(shade * 255).astype(np.uint8)).resize((CONTACT_SIZE, CONTACT_SIZE), Image.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(CONTACT_SIZE / CONTACT_GRID))
    edge = np.asarray(image)
    if max(edge[0].max(), edge[-1].max(), edge[:, 0].max(), edge[:, -1].max()) > 0:
        raise ValueError("contact patch not 0 at the quad's edge: raise CONTACT_MARGIN")
    return image, [float(mid[0] - half), float(mid[0] + half), float(mid[1] - half), float(mid[1] + half),
                   y0 + CONTACT_LIFT]


def _quad(quad: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A contact quad: vertices (4, 3), uvs (4, 2) and its two triangles - SiSL's corner order, facing up."""
    x0, x1, z0, z1, y = quad
    verts = np.array([[x0, y, z0], [x1, y, z0], [x0, y, z1], [x1, y, z1]])
    return verts, np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2], [1, 3, 2]])


def _rotate(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Points p (n, 3) rotated by the unit quaternion q = (w, x, y, z) (pose_check._rotate, for arrays)."""
    w, v = q[0], np.asarray(q[1:], dtype=np.float64)
    return p + 2 * np.cross(v, np.cross(v, p) + w * p)


def _bounds(points: np.ndarray) -> tuple[list[float], float, list[float], list[float]]:
    """(sphere centre, sphere radius, box min, box max): the sphere about the box centre."""
    lo, hi = points.min(0), points.max(0)
    centre = (lo + hi) / 2
    return list(centre), float(np.linalg.norm(points - centre, axis=1).max()), list(lo), list(hi)


def _pool_string(data: bytes, pool_offset: int, at: int) -> str:
    end = data.index(b"\0", pool_offset + at)
    return data[pool_offset + at:end].decode("ascii")


def shell_with_proxy(shell: bytes, meshes: dict[str, tuple[np.ndarray, np.ndarray]], kind: str,
                     contact: dict[str, list[float]] | None = None) -> tuple[bytes, int]:
    """A mesh-less shell PMG (pmg_scale.pair_shell) with a piece at every hookup locator: the mesh of the hookup it
    names (`meshes`: hookup name -> (vertices, triangles) in that hookup model's frame) placed by the locator, material
    0. kind `statue` also drops the locators' hookups (the copy is drawn instead of the pet). `contact` (hookup name ->
    quad, contact_map) puts that hookup's contact patch after its copy: a quad piece placed the same way, material 1 +
    its place in sorted(contact) (pmd_with_materials). Returns (PMG, pieces)."""
    contact = contact or {}
    order = sorted(contact)
    if kind not in MATERIALS:
        raise ValueError(f"unknown shadow kind {kind!r} (one of {', '.join(MATERIALS)})")
    if len(shell) < _HEADER.size or shell[1:4] != b"gmP" or shell[0] != PMG_VERSION:
        raise ValueError("not a PMG 0x15 model")
    h = list(_HEADER.unpack_from(shell, 0))
    if h[2] or h[4] or h[25] or h[27]:
        raise ValueError("not a mesh-less shell (pieces, bones or vertex data)")
    parts = [list(_PART.unpack_from(shell, h[19] + i * _PART.size)) for i in range(h[3])]
    locators = [bytearray(shell[h[20] + i * _LOCATOR_SIZE:h[20] + (i + 1) * _LOCATOR_SIZE]) for i in range(h[6])]
    pool = shell[h[22]:h[22] + h[23]]
    uv = kind == "statue"
    stride = 36 if uv else 28
    pieces = []  # (vertex bytes, index bytes, vertex count, index count, bounds, stride, uv, material)
    used = set()
    for part in parts:
        first = len(pieces)
        for loc in locators[part[4]:part[4] + part[3]]:
            name_at = struct.unpack_from("<i", loc, 40)[0]
            if name_at < 0:
                continue
            name = _pool_string(shell, h[22], name_at)
            if name not in meshes:
                raise ValueError(f"no proxy mesh for hookup {name}")
            verts, tris = meshes[name]
            if len(verts) > 0xFFFF:
                raise ValueError(f"{name}: {len(verts)} vertices, more than u16 indices hold")
            q = np.array(struct.unpack_from("<4f", loc, 24), dtype=np.float64)
            at = np.array(struct.unpack_from("<3f", loc, 8), dtype=np.float64)
            placed = _rotate(q, verts) + at
            normals = _rotate(q, vertex_normals(verts, tris))
            columns = [placed.astype("<f4").view(np.uint8).reshape(-1, 12),
                       normals.astype("<f4").view(np.uint8).reshape(-1, 12),
                       np.tile(np.array([128, 128, 128, 255] if uv else [0, 0, 0, 255], np.uint8), (len(verts), 1))]
            if uv:  # a planar map over the copy's own box: real UVs on a flat-grey texture
                lo, hi = verts.min(0), verts.max(0)
                st = (verts[:, [0, 1]] - lo[[0, 1]]) / np.maximum(hi[[0, 1]] - lo[[0, 1]], 1e-6)
                columns.append(np.ascontiguousarray(st, dtype="<f4").view(np.uint8).reshape(-1, 8))
            index = tris.astype("<u2").tobytes()
            index += b"\0" * (-len(index) % 4)
            pieces.append((np.hstack(columns).tobytes(), index, len(verts), tris.size, _bounds(placed), stride, uv, 0))
            if uv:
                struct.pack_into("<i", loc, 40, -1)
            if name in contact:
                corners, st, square = _quad(contact[name])
                placed = _rotate(q, corners) + at
                columns = [placed.astype("<f4").view(np.uint8).reshape(-1, 12),
                           _rotate(q, np.tile([0.0, 1.0, 0.0], (4, 1))).astype("<f4").view(np.uint8).reshape(-1, 12),
                           np.tile(np.array([128, 128, 128, 255], np.uint8), (4, 1)),
                           st.astype("<f4").view(np.uint8).reshape(-1, 8)]
                pieces.append((np.hstack(columns).tobytes(), square.astype("<u2").tobytes(), 4, square.size,
                               _bounds(placed), 36, True, 1 + order.index(name)))
                used.add(name)
        part[1], part[2] = len(pieces) - first, first
    if not pieces:
        raise ValueError("the shell has no hookup locator")
    if set(order) - used:
        raise ValueError(f"contact patch for {', '.join(sorted(set(order) - used))}: no locator names it")
    parts_out = b"".join(_PART.pack(*p) for p in parts)
    locators_out = b"".join(bytes(loc) for loc in locators)
    pieces_at = _HEADER.size + len(parts_out) + len(locators_out)
    pool_at = pieces_at + len(pieces) * _PIECE.size
    vertex_at = pool_at + len(pool)
    index_at = vertex_at + sum(len(p[0]) for p in pieces)
    table, vertex_pool, index_pool = b"", b"", b""
    for vbytes, ibytes, count, edges, (centre, radius, lo, hi), stride, uv, material in pieces:
        pos = vertex_at + len(vertex_pool)
        table += _PIECE.pack(edges, count, -16 if uv else -1, 1 if uv else 0, material, *centre, radius, *lo, *hi,
                             stride, pos, pos + 12, pos + 28 if uv else -1, pos + 24, -1, -1, -1, -1,
                             index_at + len(index_pool))
        vertex_pool += vbytes
        index_pool += ibytes
    points = np.array([c for p in pieces for c in (p[4][2], p[4][3])])
    centre, _, lo, hi = _bounds(points)
    radius = max(float(np.linalg.norm(np.array(p[4][0]) - centre)) + p[4][1] for p in pieces)
    h[2] = len(pieces)
    h[8:18] = [*centre, radius, *lo, *hi]
    h[18] = h[19] = _HEADER.size                       # no skeleton; the parts first
    h[20] = h[19] + len(parts_out)
    h[21] = pieces_at
    h[22], h[23] = pool_at, len(pool)
    h[24], h[25] = vertex_at, len(vertex_pool)
    h[26], h[27] = index_at, len(index_pool)
    return _HEADER.pack(*h) + parts_out + locators_out + table + pool + vertex_pool + index_pool, len(pieces)


def pmd_with_material(pmd: bytes, material: str, pieces: int) -> bytes:
    """A mesh-less shell's PMD with one material (the same path in every look) and `pieces` pieces."""
    return pmd_with_materials(pmd, [material], pieces)


def pmd_with_materials(pmd: bytes, materials: list[str], pieces: int) -> bytes:
    """A mesh-less shell's PMD with these materials (the same paths in every look; the table look-major, as vanilla
    dashtable_01_eu.pmd) and `pieces` pieces."""
    h = list(_PMD.unpack_from(pmd, 0))
    if h[0] != PMD_VERSION:
        raise ValueError("not a PMD 0x04")
    if h[1] or h[3] or h[14] != len(pmd) or h[15] != len(pmd):
        raise ValueError("not a mesh-less shell PMD (materials, pieces or data after the material block)")
    looks = h[2]
    paths = [m.encode("ascii") + b"\0" for m in materials]
    block = b"".join(paths)
    data_at = len(pmd) + 4 * looks * len(paths)
    h[1], h[3], h[8] = len(paths), pieces, len(block) * looks
    h[15] = data_at
    starts = np.cumsum([0] + [len(p) for p in paths[:-1]])
    offsets = struct.pack(f"<{looks * len(paths)}I", *(int(data_at + k * len(block) + s)
                                                      for k in range(looks) for s in starts))
    return _PMD.pack(*h) + pmd[_PMD.size:] + offsets + block * looks


def save_proxy(path: Path, meshes: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"{k}.{part}": a for k, (v, t) in meshes.items() for part, a in (("v", v), ("t", t))})


def load_proxy(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path) as z:
        names = sorted({key.rsplit(".", 1)[0] for key in z.files})
        return {n: (z[f"{n}.v"], z[f"{n}.t"]) for n in names}
