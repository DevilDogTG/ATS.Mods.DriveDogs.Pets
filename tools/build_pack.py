"""Stage the ATS Pets pack: the imported ETS2 pets placed in every ATS truck that has their slot (ADR 0001).

From the import (tools/import_ets2.py, checked first) this
  1. ships the imported game files as built (pet models, animations, textures, icons, pair hookup copies),
  2. writes one animated_hookup unit per pose and per scaled pair member, and the icon materials,
  3. writes our own mount shell (tools/shells.py) per pose and pair, placing each pet as its ETS2 mount does,
  4. writes one accessory def per pose / pair on every ATS truck of the base game that has its slot
     (def/vehicle/truck/<truck>/accessory/<slot>/), and the dashboard-table items (mount `addon`) as addon hookups,
  5. checks the stage: every referenced path exists in the stage or the game, every SII parses, every shell names a
     staged hookup.

Copied in parts from the ETS2 Pets repo's tools/build_pack.py (hookup units, icon materials, addon storage, stage
check). The ETS2 build writes its defs from willet's per-truck mounts; this one writes its own (no willet file ships).

Output: build/pack/ (src/ + staged files, packed by the pack-mod skill with -Root build/pack) and
build/pack-report.md. Exits non-zero when the import is not as recorded or the stage check fails.

Usage: python tools/build_pack.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, GAME, IMPORT, REPO, read_yaml, reference_root, ref_exists, ref_list  # noqa: E402
from import_ets2 import run_check  # noqa: E402
from megapack import base_dir  # noqa: E402
from shells import HEADER, pair_pets, pose_pets, shell_files  # noqa: E402
from sii_text import check_structure  # noqa: E402

STAGE = BUILD / "pack"
REPORT = BUILD / "pack-report.md"
_ICON = re.compile(r'^\s*icon\s*:\s*"([^"]+)"', re.MULTILINE)
_HOOKUP = re.compile(r"^\s*animated_hookup\s*:\s*([\w.]+)", re.MULTILINE)
_ANY_PATH = re.compile(rb"(?<![A-Za-z0-9_./-])/[A-Za-z0-9_./-]+?\.(?:pmd|pmg|pma|pmc|mat|tobj|dds|sii|sui)(?![A-Za-z0-9_])")


def crlf(lines: list[str]) -> bytes:
    return "\r\n".join(lines).encode("utf-8")


def game_path(root: Path, file: Path) -> str:
    return "/" + file.relative_to(root).as_posix()


def truck_slots(base: Path) -> dict[str, set[str]]:
    """ATS truck -> the accessory slots it has (a def under def/vehicle/truck/<truck>/accessory/<slot>/)."""
    trucks: dict[str, set[str]] = defaultdict(set)
    for rel in ref_list(base, "def/vehicle/truck/", ".sii"):
        parts = rel.split("/")
        if len(parts) == 7 and parts[4] == "accessory":
            trucks[parts[3]].add(parts[5])
    return trucks


def hookup_unit(hookup: str, model: str, animation: str) -> bytes:
    return crlf(["SiiNunit", "{", f"animated_hookup : {hookup}", "{", f'\tmodel: "{model}"',
                 f'\tanimation: "{animation}"', "}", "}", ""])


def icon_material(icon: str) -> bytes:
    """The truck-shop icon material the game looks up for a def's `icon: "<name>"` (/material/ui/accessory/<name>.mat)."""
    return f'material : "ui"\r\n{{\r\n\ttexture : "{icon}.tobj"\r\n\ttexture_name : "texture"\r\n}}\r\n'.encode()


def part_def(unit: str, name: str, icon: str, model: str, config: dict) -> bytes:
    """An interior accessory placing a shell in a truck slot (the game puts the model at the slot's locator)."""
    return crlf(["SiiNunit", "{", f"accessory_addon_int_data : {unit}", "{", f'\tname: "{name}"',
                 f"\tprice: {config['price']}", f"\tunlock: {config['unlock']}", f'\ticon: "{icon}"',
                 "\tpart_type: aftermarket", "", f'\texterior_model: "{model}"', f'\tinterior_model: "{model}"',
                 "\tlook: default", "\tvariant: default", "}", "}", ""])


def addon_unit(unit: str, name: str, icon: str, suitable_for: str, model: str, config: dict) -> list[str]:
    """An addon hookup placing a shell on another accessory's slot_N locators (set_dashbrd: a dashboard table)."""
    return [f"accessory_hookup_int_data : {unit}", "{", f'\tname: "{name}"', f'\ticon: "{icon}"',
            f'\tsuitable_for[]: "{suitable_for}"', f'\tmodel: "{model}"', "\tvariant: default",
            f"\tprice: {config['price']}", f"\tunlock: {config['unlock']}", "}", ""]


def items(poses: list[dict], pairs: list[dict]) -> list[dict]:
    """Every part the pack offers: one per pose and one per pair / table item, with the pets its shell places."""
    out = [{"token": p["token"], "name": f"{p['name_display']} ({p['label']})", "icon": p["icon"],
            "mount": p["mount"], "pets": pose_pets(p), "kind": "pose"} for p in poses]
    out += [{"token": p["token"], "name": f"{p['name_display']} ({p['label']})", "icon": p["icon"],
             "mount": p["mount"], "pets": pair_pets(p), "kind": "pair"} for p in pairs]
    return out


def stage_file(path: str, data: bytes | Path, owner: str, owners: dict[str, str], clashes: list[str]) -> None:
    """Write one file into the stage; the same path from two owners is a clash unless the bytes are equal."""
    content = data.read_bytes() if isinstance(data, Path) else data
    target = STAGE / path.lstrip("/")
    if path in owners:
        if target.read_bytes() != content:
            clashes.append(f"{path} ({owners[path]}, {owner})")
        return
    owners[path] = owner
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def shell_hookups(data: bytes) -> list[str]:
    h = HEADER.unpack_from(data, 0)
    return [s.decode("ascii") for s in data[h[21]:h[21] + h[23]].split(b"\0") if s]


def verify_stage(base: Path) -> list[str]:
    """Every staged SII parses, every def icon has its material, every absolute path inside a staged text, material,
    tobj or model file exists in the stage or the game, and every shell names a staged animated hookup."""
    problems = []
    hookups = set()
    for file in sorted(STAGE.rglob("*.sii")):
        hookups.update(_HOOKUP.findall(file.read_text(encoding="utf-8-sig", errors="replace")))
    for file in sorted(STAGE.rglob("*")):
        path = game_path(STAGE, file)
        if file.suffix in (".sii", ".sui"):
            text = file.read_text(encoding="utf-8-sig", errors="replace")
            problems += [f"{path}: {p}" for p in check_structure(text)]
            for icon in _ICON.findall(text) if path.startswith("/def/") else []:
                mat = f"material/ui/accessory/{icon}.mat"
                if not (STAGE / mat).is_file() and not ref_exists(base, mat):
                    problems.append(f"{path}: icon {icon} -> /{mat}")
        if file.suffix == ".pmg" and path.endswith("/shell.pmg"):
            problems += [f"{path}: hookup {h} has no animated_hookup unit"
                         for h in shell_hookups(file.read_bytes()) if h not in hookups]
        if file.suffix not in (".sii", ".sui", ".mat", ".tobj", ".pmd"):
            continue
        for ref in sorted(set(_ANY_PATH.findall(file.read_bytes()))):
            rel = ref.decode("ascii").lstrip("/")
            if not (STAGE / rel).is_file() and not ref_exists(base, rel):
                problems.append(f"{path} -> /{rel}")
    return problems


def main() -> int:
    if run_check():
        raise SystemExit("the import is not as import.json records (see above)")
    config = read_yaml("build.yaml")
    base = base_dir(reference_root(GAME), config["base_game"]["version"])
    if not (base / "_index.sqlite").is_file():
        raise SystemExit(f"{base}: no extracted ATS base {config['base_game']['version']} (extract-reference base)")
    poses = json.loads((IMPORT / "pets/models.json").read_text(encoding="utf-8"))
    pairs = json.loads((IMPORT / "pairs/pairs.json").read_text(encoding="utf-8"))
    trucks = truck_slots(base)

    ship: dict[str, tuple[bytes | Path, str]] = {}  # game path -> (content, owner)
    for part in ("pets", "pairs"):
        for file in sorted((IMPORT / part / "out").rglob("*")):
            if file.is_file():
                ship[game_path(IMPORT / part / "out", file)] = (file, f"import {part}")
    for pose in poses:
        ship[f"/unit/hookup/vehicle/drivedogs_{pose['hookup']}.sii"] = (
            hookup_unit(pose["hookup"], pose["model"], pose["animation"]), "hookups")
    for pair in pairs:
        for m in pair["members"]:
            if "model" in m:  # a member scaled for the pair: its own hookup copy (imported model + animation)
                ship[f"/unit/hookup/vehicle/drivedogs_{m['hookup']}.sii"] = (
                    hookup_unit(m["hookup"], m["model"], m["animation"]), "hookups")
    for icon in sorted({p["icon"] for p in poses} | {p["icon"] for p in pairs}):
        ship[f"/material/ui/accessory/{icon}.mat"] = (icon_material(icon), "icons")

    notes: dict[str, list[str]] = defaultdict(list)
    addon_units: list[str] = []
    for item in items(poses, pairs):
        mount = item["mount"]
        model, files = shell_files(item["token"] + ("_addon" if mount.get("addon") else ""), item["pets"])
        for path, data in files.items():
            ship[path] = (data, f"shell {item['token']}")
        hookups = ", ".join(p.hookup for p in item["pets"])
        if mount.get("addon"):
            addon_units += addon_unit(f"{item['token']}.drivedogs.addon_hookup", item["name"], item["icon"],
                                      mount["addon"], model, config)
            notes["table items (addon hookups)"].append(f"{item['token']}: suitable_for {mount['addon']} ({hookups})")
            continue
        slot = mount.get("to_slot", mount["slot"])
        fitted = sorted(t for t, slots in trucks.items() if slot in slots)
        for truck in fitted:
            ship[f"/def/vehicle/truck/{truck}/accessory/{slot}/drivedogs_pet.{item['token']}.sii"] = (
                part_def(f"{item['token']}.{truck}.{slot}", item["name"], item["icon"], model, config), "defs")
        notes[f"{item['kind']}s in {slot}"].append(f"{item['token']}: {len(fitted)} trucks ({hookups})")
    if addon_units:
        ship["/def/vehicle/addon_hookups/drivedogs_pets.sui"] = (crlf(addon_units + [""]), "table items")
        ship["/def/vehicle/addon_hookup_storage.drivedogs_pets.sii"] = (
            crlf(["SiiNunit", "{", '@include "addon_hookups/drivedogs_pets.sui"', "}", ""]), "table items")

    # Stage: src/ (our own files) plus every shipped file. build/pack is generated by this script only.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(REPO / "src", STAGE)
    owners: dict[str, str] = {}
    clashes: list[str] = []
    replaced = []
    for path, (data, owner) in sorted(ship.items()):
        if ref_exists(base, path):
            replaced.append(path)
        stage_file(path, data, owner, owners, clashes)
    problems = verify_stage(base)
    if not (STAGE / "cover.jpg").is_file():
        notes["warnings"].append("src/cover.jpg is missing (manifest icon)")

    lock = json.loads((REPO / "import.json").read_text(encoding="utf-8"))["ets2"]
    by_owner: dict[str, int] = defaultdict(int)
    for owner in owners.values():
        by_owner[owner.split(" ")[0]] += 1
    lines = ["# ATS Pets pack report", "",
             f"Game base: ATS {config['base_game']['version']}. Pets: ETS2 Pets {lock['version']} "
             f"({lock['commit'][:7]}). Staged at `build/pack/`.", "",
             f"- poses: {len(poses)}, pairs / table items: {len(pairs)}",
             f"- files staged: {len(owners)} ({', '.join(f'{k} {v}' for k, v in sorted(by_owner.items()))})", "",
             "## Trucks and slots", ""]
    for truck in sorted(trucks):
        used = sorted(s for s in trucks[truck] if s in {i["mount"].get("to_slot", i["mount"]["slot"])
                                                        for i in items(poses, pairs)})
        lines.append(f"- `{truck}`: {', '.join(used) or '(no pet slot)'}")
    lines.append("")
    for title, entries in sorted(notes.items()):
        lines += [f"## {title} ({len(entries)})", ""] + [f"- {e}" for e in entries] + [""]
    for title, entries in (("Shipped files that replace a game file", replaced), ("Path clashes", clashes),
                           ("Stage problems", problems)):
        if entries:
            lines += [f"## {title} ({len(entries)})", ""] + [f"- `{e}`" for e in entries] + [""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    fitted_trucks = sum(1 for t in trucks.values() if t & {"toybed", "toyseat", "toybig"})
    print(f"poses {len(poses)}, pairs {len(pairs)}, trucks with a pet slot {fitted_trucks}; "
          f"{len(owners)} files staged in {STAGE}; report: {REPORT}")
    for warning in notes.get("warnings", []):
        print(f"warning: {warning}")
    if problems or clashes or replaced:
        print(f"FAILED: {len(problems)} stage problems, {len(clashes)} clashes, {len(replaced)} game files replaced "
              "(see report)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
