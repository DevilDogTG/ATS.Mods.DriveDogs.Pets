"""The ATS cabs the pets are placed in, read from the extracted base game (build.yaml base_game.version).

Every truck's interiors come from its interior defs (def/vehicle/truck/<truck>/interior/*.sii: model + variant); their
models are in the trucks' DLC archives and are fetched on first use (ref_fetch). Per interior:
  slot     the locator of a slot: ATS names some with a suffix (toybed_82, toybig_1, set_dashbrd1), so the name is
           matched by prefix; an interior can hold several in parts its variants switch (T680 cab_b / cab_c), so the one
           in a part its variant shows wins.
  surface  the triangles of the parts its variant shows (shadow / collision materials left out), in the cab's frame.
           Only the interior's own model: an ATS interior folder holds the day cab and every sleeper at the same
           coordinates, so reading the whole folder (as the ETS2 pose_check does) would stack them.
  seats    the truck's co-driver seat accessories (def .../accessory/codrv_seat/, T680 2022, VNL 2018 / 2025, W900,
           5700XE, VNR Electric): their models, placed by the game at the interior's codrv_seat locator. Those cabs
           have no seat cushion of their own around the toyseat slot.

Used by tools/cab_fit.py (measures seat / floor / bed for fixes.yaml).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import GAME, read_yaml, reference_root, ref_exists, ref_fetch, ref_path
from megapack import base_dir
from pmg import Pmg, token

_HIDDEN_MATERIAL = re.compile(r"shadow|collision|invisible", re.IGNORECASE)
_INTERIOR = re.compile(r"accessory_interior_data\s*:\s*([\w.]+)\s*\{(.*?)\n\}", re.DOTALL)


def base() -> Path:
    config = read_yaml("build.yaml")
    path = base_dir(reference_root(GAME), config["base_game"]["version"])
    if not (path / "_index.sqlite").is_file():
        raise SystemExit(f"{path}: no extracted ATS base {config['base_game']['version']} (extract-reference base)")
    return path


def _value(body: str, key: str) -> str | None:
    m = re.search(rf'^\s*{key}\s*:\s*"?([^"\n]+?)"?\s*$', body, re.MULTILINE)
    return m.group(1) if m else None


def visible_parts(pmd: bytes, variant: str | None) -> list[bool] | None:
    """Which parts a PMD 0x04 variant shows (its `visible` attribute; a part without one is shown). None when the
    variant is not in the model (then every part counts)."""
    h = struct.unpack_from("<16I", pmd, 0)
    _, _, _, _, nvar, npattr, nattr, valsize, _, _, var_off, pattr_off, attrval_off, attr_off = h[:14]
    variants = [token(struct.unpack_from("<Q", pmd, var_off + 8 * i)[0]) for i in range(nvar)]
    if not variant or variant.lower() not in variants:
        return None
    row = attrval_off + variants.index(variant.lower()) * valsize
    attrs = [struct.unpack_from("<Qii", pmd, attr_off + 16 * i) for i in range(nattr)]
    out = []
    for p in range(npattr):
        first, end = struct.unpack_from("<2i", pmd, pattr_off + 8 * p)
        shown = True
        for name, _, off in attrs[first:end]:
            if token(name) == "visible":
                shown = bool(struct.unpack_from("<i", pmd, row + off)[0])
        out.append(shown)
    return out


@dataclass
class Model:
    path: str  # game path of the .pmd
    pmg: Pmg
    materials: list[str]

    def triangles(self, shown: list[bool] | None = None) -> np.ndarray:
        """(N, 3, 3) visible triangles of the parts `shown` (None: all), in the model's frame."""
        data, out = self.pmg.data, []
        for index, (_, piece_count, piece_first, _, _) in enumerate(self.pmg.parts):
            if shown is not None and not shown[index]:
                continue
            for piece in self.pmg.pieces[piece_first:piece_first + piece_count]:
                edges, verts, material, stride, positions, indices = (piece[0], piece[1], piece[4], piece[15],
                                                                      piece[16], piece[24])
                if verts == 0 or edges < 3 or stride < 12:
                    continue
                if material < len(self.materials) and _HIDDEN_MATERIAL.search(self.materials[material]):
                    continue
                raw = np.frombuffer(data, dtype=np.uint8, count=verts * stride, offset=positions)
                pos = raw.reshape(verts, stride)[:, :12].copy().view("<f4").reshape(verts, 3)
                idx = np.frombuffer(data, dtype="<u2", count=edges, offset=indices)[: edges // 3 * 3].reshape(-1, 3)
                if idx.size and idx.max() < verts:
                    out.append(pos[idx])
        return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _rotate(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    w, v = q[0], q[1:]
    return p + 2 * np.cross(v, np.cross(v, p) + w * p)


@dataclass
class Interior:
    truck: str
    unit: str
    variant: str | None
    model: Model
    shown: list[bool]  # per part
    seats: list[Model]  # the truck's co-driver seat accessories ([] when the interior has its own seat)

    @property
    def pmg(self) -> Pmg:
        return self.model.pmg

    def slot(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(position, rotation w x y z) of the slot's locator (name or name + suffix) in a part this variant shows;
        None when it has none there (a day cab variant of a model whose sleeper parts hold the toybed)."""
        for index, (_, _, _, count, first) in enumerate(self.pmg.parts):
            if not self.shown[index]:
                continue
            for loc in self.pmg.locators[first:first + count]:
                if loc.name == name or (loc.name.startswith(name) and not loc.name[len(name):].isalpha()):
                    return np.array(loc.position), np.array(loc.rotation)
        return None

    def triangles(self, seat: Model | None = None) -> np.ndarray:
        """(N, 3, 3) visible triangles of the parts this variant shows, with a co-driver seat accessory placed at
        the codrv_seat locator when given, in the cab's frame."""
        tris = self.model.triangles(self.shown)
        slot = self.slot("codrv_seat") if seat is not None else None
        if slot is not None:
            loc, q = slot
            tris = np.concatenate([tris, _rotate(q, seat.triangles().reshape(-1, 3)).reshape(-1, 3, 3) + loc])
        return tris


def _model(root: Path, path: str, cache: dict[str, tuple[Model, bytes]]) -> tuple[Model, bytes]:
    if path not in cache:
        rel = path.lstrip("/")[:-4]
        pmd = ref_path(root, rel + ".pmd").read_bytes()
        materials = [m.decode("ascii", "replace") for m in re.findall(rb"/[!-~]+?\.mat", pmd)]
        cache[path] = (Model(path, Pmg.read(ref_path(root, rel + ".pmg")), materials), pmd)
    return cache[path]


def interiors(root: Path) -> dict[str, list[Interior]]:
    """Every ATS truck's interiors whose model the base game has: truck id -> [Interior]."""
    found: dict[str, list[tuple[str, str, str | None]]] = {}
    seats: dict[str, list[str]] = {}
    for truck_dir in sorted((root / "def/vehicle/truck").iterdir()):
        for sii in sorted((truck_dir / "interior").glob("*.sii")) if truck_dir.is_dir() else []:
            for m in _INTERIOR.finditer(sii.read_text(encoding="utf-8-sig", errors="replace")):
                model = _value(m.group(2), "model")
                if model and model.endswith(".pmd") and ref_exists(root, model.lstrip("/")[:-4] + ".pmg"):
                    found.setdefault(truck_dir.name, []).append((m.group(1), model, _value(m.group(2), "variant")))
        for sii in sorted((truck_dir / "accessory/codrv_seat").glob("*.sii")) if truck_dir.is_dir() else []:
            model = _value(sii.read_text(encoding="utf-8-sig", errors="replace"), "interior_model")
            if model and model.endswith(".pmd") and ref_exists(root, model.lstrip("/")[:-4] + ".pmg"):
                seats.setdefault(truck_dir.name, []).append(model)
    models = sorted({model for rows in found.values() for _, model, _ in rows} | {m for v in seats.values() for m in v})
    ref_fetch(root, [model.lstrip("/")[:-4] + suffix for model in models for suffix in (".pmg", ".pmd")])
    cache: dict[str, tuple[Model, bytes]] = {}
    out: dict[str, list[Interior]] = {}
    for truck, rows in found.items():
        truck_seats = [_model(root, m, cache)[0] for m in sorted(set(seats.get(truck, [])))]
        for unit, path, variant in rows:
            model, pmd = _model(root, path, cache)
            shown = visible_parts(pmd, variant) or [True] * model.pmg.part_count
            out.setdefault(truck, []).append(Interior(truck, unit, variant, model, shown, truck_seats))
    return out
