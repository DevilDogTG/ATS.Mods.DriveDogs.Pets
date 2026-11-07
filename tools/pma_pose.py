"""Pose a skinned ETS2 model (PMG 0x15) with a frame of its animation (PMA v5) and return the posed triangles, so the
converted game files can be checked offline (renders via pmg_preview.render) without the game.

Layouts as in ConverterPIX (mwl4) structs/pmg_0x15.h and structs/pma_0x05.h:
  PMG bone (200 bytes): name token u64, bind matrix 16f (+8, model space, translation at elements 3/7/11), its inverse
    (+72), bind local transform: stretch quat (+136), rotation quat (+152), translation 3f (+168), scale 3f (+180),
    determinant sign f (+192), parent index i8 (+196).
  PMG piece (100 bytes): stride (+60), position (+64), bone indices (+88, 4 x u8), bone weights (+92, 4 x u8),
    index buffer (+96).
  PMA v5: frames u16 (+4), bones u32 (+8), offsets of lengths (+64, f32 per frame), bones (+68, one u8 per channel:
    the PMG bone index it drives), frames (+72): per frame and channel 56 bytes: stretch quat, rotation quat,
    translation 3f, scale 3f.
Quaternions are stored w, x, y, z. A bone's local matrix is T * R * S, S the scale along the stretch quat's axes; its
model-space matrix is the parent's times its local. check_bind() confirms the reading on a model: the bind locals,
composed, must reproduce every bind matrix.

Usage: python tools/pma_pose.py <model.pmd> <anim.pma> <frame> <out.png>   (frame -1 = bind pose)

Copied unchanged from the ETS2 Pets repo's tools/pma_pose.py (ADR 0001).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_HEADER = struct.Struct("<B3s5iQ4f3f3f10i")
_PIECE = struct.Struct("<5i4f3f3f5i4ii")
_PMA = struct.Struct("<IHHIQ3ff3f3ff5i")


def _quat(w, x, y, z) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def local_matrix(stretch, rotation, translation, scale, sign=1.0) -> np.ndarray:
    so = _quat(*stretch)
    s = so @ np.diag(scale) @ so.T
    m = np.eye(4)
    m[:3, :3] = _quat(*rotation) @ s * (1 if sign >= 0 else 1)
    m[:3, 3] = translation
    return m


def read_bones(pmg: bytes) -> list[dict]:
    h = _HEADER.unpack_from(pmg, 0)
    count, offset = h[4], h[18]
    bones = []
    for b in range(count):
        at = offset + b * 200
        bones.append({
            "token": struct.unpack_from("<Q", pmg, at)[0],
            "bind": np.array(struct.unpack_from("<16f", pmg, at + 8)).reshape(4, 4),
            "inverse": np.array(struct.unpack_from("<16f", pmg, at + 72)).reshape(4, 4),
            "stretch": struct.unpack_from("<4f", pmg, at + 136),
            "rotation": struct.unpack_from("<4f", pmg, at + 152),
            "translation": struct.unpack_from("<3f", pmg, at + 168),
            "scale": struct.unpack_from("<3f", pmg, at + 180),
            "sign": struct.unpack_from("<f", pmg, at + 192)[0],
            "parent": struct.unpack_from("<b", pmg, at + 196)[0],
        })
    return bones


def compose(bones: list[dict], locals_: list[np.ndarray]) -> list[np.ndarray]:
    world: list[np.ndarray] = []
    for b, local in zip(bones, locals_):
        world.append(local if b["parent"] < 0 else world[b["parent"]] @ local)
    return world


def check_bind(bones: list[dict]) -> float:
    """Largest difference between the composed bind locals and the stored bind matrices (should be ~0)."""
    world = compose(bones, [local_matrix(b["stretch"], b["rotation"], b["translation"], b["scale"], b["sign"])
                            for b in bones])
    return max(float(np.abs(w - b["bind"]).max()) for w, b in zip(world, bones))


def frame_locals(pma: bytes, frame: int, bones: list[dict]) -> list[np.ndarray]:
    """Local matrices of every PMG bone at one animation frame; bones the animation doesn't drive keep their bind
    local."""
    h = _PMA.unpack_from(pma, 0)
    frames, count, bones_off, frames_off = h[1], h[3], h[17], h[18]
    if not 0 <= frame < frames:
        raise ValueError(f"frame {frame} outside 0..{frames - 1}")
    indices = struct.unpack_from(f"<{count}B", pma, bones_off)  # one byte per channel: the PMG bone it drives
    locals_ = [local_matrix(b["stretch"], b["rotation"], b["translation"], b["scale"], b["sign"]) for b in bones]
    for k, index in enumerate(indices):
        at = frames_off + (frame * count + k) * 56
        so, rot = struct.unpack_from("<4f", pma, at), struct.unpack_from("<4f", pma, at + 16)
        tr, sc = struct.unpack_from("<3f", pma, at + 32), struct.unpack_from("<3f", pma, at + 44)
        if index < len(locals_):
            locals_[index] = local_matrix(so, rot, tr, sc)
    return locals_


def posed_triangles(pmd: Path, pma: Path | None, frame: int) -> np.ndarray:
    pmg = pmd.with_suffix(".pmg").read_bytes()
    h = _HEADER.unpack_from(pmg, 0)
    bones = read_bones(pmg)
    if pma is None or frame < 0:
        skin = [np.eye(4) for _ in bones]
    else:
        world = compose(bones, frame_locals(pma.read_bytes(), frame, bones))
        skin = [w @ b["inverse"] for w, b in zip(world, bones)]
    skin_arr = np.stack(skin) if skin else np.zeros((0, 4, 4))
    tris = []
    for i in range(h[2]):
        p = _PIECE.unpack_from(pmg, h[21] + i * _PIECE.size)
        edges, verts, stride, pos_off, bi_off, bw_off, idx_off = p[0], p[1], p[15], p[16], p[22], p[23], p[24]
        raw = np.frombuffer(pmg, dtype=np.uint8, count=verts * stride, offset=pos_off).reshape(verts, stride)
        pos = raw[:, :12].copy().view("<f4").reshape(verts, 3).astype(np.float64)
        if bones and bi_off >= 0 and bw_off >= 0:
            bi = np.frombuffer(pmg, dtype=np.uint8, count=verts * stride, offset=bi_off)
            bi = np.lib.stride_tricks.as_strided(bi, (verts, 4), (stride, 1)).astype(int)
            bw = np.frombuffer(pmg, dtype=np.uint8, count=verts * stride, offset=bw_off)
            bw = np.lib.stride_tricks.as_strided(bw, (verts, 4), (stride, 1)).astype(np.float64) / 255.0
            homo = np.c_[pos, np.ones(verts)]
            out = np.zeros((verts, 3))
            for k in range(4):
                m = skin_arr[bi[:, k]]  # (verts, 4, 4)
                out += bw[:, k, None] * np.einsum("vij,vj->vi", m, homo)[:, :3]
            pos = out
        idx = np.frombuffer(pmg, dtype="<u2", count=edges, offset=idx_off)
        idx = idx[: len(idx) // 3 * 3].reshape(-1, 3)
        tris.append(pos[idx])
    return np.concatenate(tris)


def main(argv: list[str]) -> int:
    from pmg_preview import render
    pmd, pma, frame, out = Path(argv[0]), Path(argv[1]), int(argv[2]), Path(argv[3])
    bones = read_bones(pmd.with_suffix(".pmg").read_bytes())
    print(f"bones {len(bones)}, bind check max error {check_bind(bones):.2e}")
    tris = posed_triangles(pmd, pma, frame)
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    print(f"posed size x {hi[0] - lo[0]:.3f} y {hi[1] - lo[1]:.3f} z {hi[2] - lo[2]:.3f}, min y {lo[1]:.3f}")
    render(tris, 512).save(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
