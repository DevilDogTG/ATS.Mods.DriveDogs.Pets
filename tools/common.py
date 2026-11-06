"""Shared helpers of the ATS Pets tools: repo paths, the build configs and the scs-mod-developer host API.

Copied (trimmed) from the ETS2 Pets repo's tools/common.py, never imported across repos (ADR 0001). Host paths
(extracted-reference root, tools) come from this machine's scs-mod-developer host file through the `megapack` host
API; no tool parses that file or holds a machine path.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path.home() / ".agent-brains/profiles/scs-mod-developer/skills/megapack/scripts"))
from megapack import (reference_root, ref_exists, ref_fetch, ref_list, ref_path,  # noqa: E402,F401
                      repo_game, tool_path)

REPO = Path(__file__).resolve().parent.parent
GAME = repo_game(REPO)
if GAME != "ats":
    raise SystemExit(f"these tools build the ATS pack; pack.config.json says game {GAME!r} (copied to the wrong repo?)")

BUILD = REPO / "build"
IMPORT = BUILD / "import"  # the ETS2 build output this pack carries, checked against import.json


def read_yaml(name: str) -> dict:
    path = REPO / name
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def text_sha1(path: Path) -> str:
    """sha1 of a text file with LF line endings (git may check it out with either)."""
    return sha1(path.read_bytes().replace(b"\r\n", b"\n"))


def manifest_version(manifest: Path) -> str:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition(":")
        if key == "package_version":
            return value.strip().strip('"')
    raise SystemExit(f"{manifest}: no package_version")


def split_unit(name: str, vehicle: str | None = None) -> tuple[str, str | None, str]:
    """(part token, vehicle id, slot) of an accessory unit name.

    Names end with the vehicle id and the slot; everything before is the part token, which can itself contain
    dots. Pass `vehicle` (the def's folder name) when known: mod vehicle ids can have three segments. Without it
    the id is taken as the two segments before the slot, which holds for every SCS truck. Addon hookups are
    <token>.addon_hookup and belong to no vehicle.
    """
    if name.endswith(".addon_hookup"):
        return name[: -len(".addon_hookup")], None, "addon_hookup"
    if vehicle:
        head, _, slot = name.rpartition(".")
        if head.endswith("." + vehicle):
            return head[: -len(vehicle) - 1], vehicle, slot
        at = (head + ".").find("." + vehicle + ".")
        if at >= 0:
            return head[:at] + head[at + len(vehicle) + 1:], vehicle, slot
    pieces = name.split(".")
    if len(pieces) >= 4:
        return ".".join(pieces[:-3]), ".".join(pieces[-3:-1]), pieces[-1]
    return pieces[0], None, pieces[-1]
