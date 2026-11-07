"""Check a packed .scs against the stage it was packed from (scs-mod-developer rule): extract it with scs_extractor,
then file counts must match and every non-texture file must be byte-identical. scs_packer stores a texture beside the
.tobj that uses it, named after that .tobj, so a .dds kept elsewhere moves: those moves are listed, not failed.
Every model (.pmg) must have real UVs (8.0.2: up to 8.0.1 every one had UV (0, 0) everywhere - one flat colour per
material in game).

Usage: python tools/verify_pack.py output/local/drivedog_pets_v<version>.scs

Copied unchanged from the ETS2 Pets repo's tools/verify_pack.py (ADR 0001).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from common import REPO
from petlib import uvs

STAGE = REPO / "build" / "pack"
EXTRACT = REPO / "build" / "verify_extract"


def main(argv: list[str]) -> int:
    archive = Path(argv[0]).resolve()
    if archive.read_bytes()[:4] != b"SCS#":
        raise SystemExit(f"{archive} is not a HashFS archive")
    shutil.rmtree(EXTRACT, ignore_errors=True)
    EXTRACT.mkdir(parents=True)  # scs_extractor writes nothing, silently, into a missing folder
    proc = subprocess.run(["scs_extractor", str(archive), str(EXTRACT)], capture_output=True, text=True)
    staged = {p.relative_to(STAGE).as_posix() for p in STAGE.rglob("*") if p.is_file()}
    packed = {p.relative_to(EXTRACT).as_posix() for p in EXTRACT.rglob("*") if p.is_file()}
    if not packed:
        raise SystemExit(f"extraction gave no files: {proc.stdout}{proc.stderr}")
    moved = sorted(staged - packed)
    odd = [p for p in moved + sorted(packed - staged) if not p.endswith(".dds")]
    differ = [p for p in sorted(staged & packed) if not p.endswith((".dds", ".tobj"))
              and (STAGE / p).read_bytes() != (EXTRACT / p).read_bytes()]
    # 2026-10-20 (8.0.2): every model up to 8.0.1 shipped with UV (0, 0) on every vertex - one flat colour in game
    flat = [p for p in sorted(packed) if p.endswith(".pmg") and uvs.flat(uvs.pmg_uv_ranges(EXTRACT / p))]
    models = sum(p.endswith(".pmg") for p in packed)
    print(f"stage {len(staged)} files, archive {len(packed)}; textures moved beside their .tobj: {len(moved)}; "
          f"models with real UVs: {models - len(flat)} of {models}")
    for p in odd + differ:
        print(f"  PROBLEM {p}")
    for p in flat:
        print(f"  PROBLEM {p}: no real UVs ({uvs.describe(uvs.pmg_uv_ranges(EXTRACT / p))})")
    ok = len(staged) == len(packed) and not odd and not differ and not flat
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
