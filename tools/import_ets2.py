"""Import the pets of one ETS2 Pets release (ADR 0001): the game-neutral output of the ETS2 build, its credits, and a
manifest that pins them.

Copied from a clean ETS2 Pets checkout whose build is the one it packed (every imported file byte-identical to its
`build/pack` stage):
  build/pets/out/**, build/pets/models.json    (gather_pets: locked poses, materials, icons)
  build/pairs/out/**, build/pairs/pairs.json   (pairs: pair / table items, their icons and scaled hookups)
into build/import/ (git-ignored), plus src/CREDITS.txt and the "Credits:" lines of src/description.txt. import.json
(committed) records the ETS2 version and commit and a sha1 per file; the ATS version must equal the ETS2 version.

Usage:
  python tools/import_ets2.py <ets2-checkout>   import (replaces build/import/, writes import.json and the credits)
  python tools/import_ets2.py --check           exit 1 when build/import/, the credits or the version disagree with
                                                import.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import IMPORT, REPO, manifest_version, sha1, text_sha1

ETS2_REPO = "DevilDogTG/ETS2.Mods.DriveDog.Pets"
PARTS = {"pets": ("out", "models.json"), "pairs": ("out", "pairs.json")}  # build/<part>/ -> what is imported
LOCK = REPO / "import.json"
CREDITS = REPO / "src" / "CREDITS.txt"
DESCRIPTION = REPO / "src" / "description.txt"
MANIFEST = REPO / "src" / "manifest.sii"


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def credit_lines(description: str) -> list[str]:
    rows = description.replace("\r\n", "\n").split("\n")
    start = rows.index("Credits:")
    end = next((i for i in range(start + 1, len(rows)) if not rows[i].startswith("- ")), len(rows))
    return rows[start + 1:end]


def with_credits(description: str, lines: list[str]) -> str:
    rows = description.replace("\r\n", "\n").split("\n")
    start = rows.index("Credits:")
    end = next((i for i in range(start + 1, len(rows)) if not rows[i].startswith("- ")), len(rows))
    rows[start + 1:end] = lines
    return "\r\n".join(rows)


def source_files(build: Path) -> dict[str, Path]:
    """Imported path (relative to build/import) -> the ETS2 build file."""
    files = {}
    for part, (folder, index) in PARTS.items():
        files[f"{part}/{index}"] = build / part / index
        for f in sorted((build / part / folder).rglob("*")):
            if f.is_file():
                files[f"{part}/{folder}/{f.relative_to(build / part / folder).as_posix()}"] = f
    return files


def check_source(ets2: Path) -> tuple[str, str, dict[str, Path]]:
    """(version, commit, files) of a checkout fit to import from; refuses anything else."""
    if not (ets2 / ".git").exists():
        raise SystemExit(f"{ets2}: not a git checkout")
    if not git(ets2, "remote", "get-url", "origin").removesuffix(".git").endswith(ETS2_REPO):
        raise SystemExit(f"{ets2}: origin is not {ETS2_REPO}")
    if json.loads((ets2 / "pack.config.json").read_text(encoding="utf-8")).get("game") != "ets2":
        raise SystemExit(f"{ets2}: pack.config.json is not the ETS2 pack")
    dirty = git(ets2, "status", "--porcelain")
    if dirty:
        raise SystemExit(f"{ets2}: uncommitted changes, import only from a clean commit:\n{dirty}")
    version, commit = manifest_version(ets2 / "src" / "manifest.sii"), git(ets2, "rev-parse", "HEAD")
    build = ets2 / "build"
    stage = build / "pack"
    if not (stage / "manifest.sii").is_file() or manifest_version(stage / "manifest.sii") != version:
        raise SystemExit(f"{ets2}: build/pack is not a {version} stage (build and pack the ETS2 release first)")
    files = source_files(build)
    for rel in PARTS:
        if not (build / rel / PARTS[rel][1]).is_file():
            raise SystemExit(f"{ets2}: build/{rel}/{PARTS[rel][1]} is missing")
    stale = [rel for rel, f in files.items() if not rel.endswith(".json")
             and f.read_bytes() != (stage / rel.split("/", 2)[2]).read_bytes()]
    if stale:
        raise SystemExit(f"{ets2}: {len(stale)} build files differ from the {version} stage (stale build?), "
                         f"e.g. {stale[:3]}")
    return version, commit, files


def run_import(ets2: Path) -> int:
    version, commit, files = check_source(ets2)
    shutil.rmtree(IMPORT, ignore_errors=True)
    lock = {"ets2": {"repo": ETS2_REPO, "version": version, "commit": commit}, "files": {}}
    for rel, src in files.items():
        target = IMPORT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        target.write_bytes(data)
        lock["files"][rel] = sha1(data)
    CREDITS.write_bytes((ets2 / "src" / "CREDITS.txt").read_bytes())
    lines = credit_lines((ets2 / "src" / "description.txt").read_bytes().decode("utf-8"))
    DESCRIPTION.write_bytes(with_credits(DESCRIPTION.read_bytes().decode("utf-8"), lines).encode("utf-8"))
    lock["credits"] = {"CREDITS.txt": text_sha1(CREDITS), "description": lines}
    LOCK.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"imported ETS2 Pets {version} ({commit[:7]}): {len(files)} files, "
          f"{sum(f.stat().st_size for f in files.values()) / 1e6:.1f} MB -> {IMPORT.relative_to(REPO)}")
    if manifest_version(MANIFEST) != version:
        print(f"note: src/manifest.sii is {manifest_version(MANIFEST)}; set package_version to {version}")
    return 0


def run_check() -> int:
    if not LOCK.is_file():
        print("no import.json: run python tools/import_ets2.py <ets2-checkout>")
        return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ets2 = lock["ets2"]
    problems = []
    have = {f.relative_to(IMPORT).as_posix() for f in IMPORT.rglob("*") if f.is_file()} if IMPORT.is_dir() else set()
    missing = [rel for rel in lock["files"] if rel not in have]
    changed = [rel for rel in lock["files"] if rel in have and sha1((IMPORT / rel).read_bytes()) != lock["files"][rel]]
    extra = sorted(have - set(lock["files"]))
    for name, rels in (("missing", missing), ("changed", changed), ("not in import.json", extra)):
        if rels:
            problems.append(f"{len(rels)} files {name} under build/import, e.g. {rels[:3]}")
    if not CREDITS.is_file() or text_sha1(CREDITS) != lock["credits"]["CREDITS.txt"]:
        problems.append("src/CREDITS.txt is not the imported one")
    if credit_lines(DESCRIPTION.read_bytes().decode("utf-8")) != lock["credits"]["description"]:
        problems.append("the Credits: lines of src/description.txt are not the imported ones")
    if manifest_version(MANIFEST) != ets2["version"]:
        problems.append(f"src/manifest.sii is {manifest_version(MANIFEST)}, the import is ETS2 {ets2['version']}")
    for p in problems:
        print(p)
    if missing or changed or extra:
        print(f"re-import: python tools/import_ets2.py <ETS2 Pets checkout at {ets2['commit'][:7]}, built and packed>")
    if not problems:
        print(f"import ok: ETS2 Pets {ets2['version']} ({ets2['commit'][:7]}), {len(lock['files'])} files")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        return run_check()
    if len(argv) == 1 and not argv[0].startswith("-"):
        return run_import(Path(argv[0]).resolve())
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
