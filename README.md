# DriveDogs: Pets (ATS)

American Truck Simulator (1.62) twin of DriveDogs: Pets for ETS2 (`DevilDogTG/ETS2.Mods.DriveDog.Pets`): animated dogs
and cats in the cab toy slots (bed, co-driver seat, cab floor) and on the dashboard tables, on every ATS truck that has
the slot. Personal use only, never published: every model belongs to its authors and is credited in `src/CREDITS.txt`
and the mod description. **The GitHub repo must stay Private.**

Pets are made only in the ETS2 repo. This repo imports one ETS2 release's built pets and places them in ATS cabs with
its own mount shells and truck defs ([ADR 0001](docs/adr/0001-import-ets2-build-own-shells.md)); the ATS version is the
version of the ETS2 release it imports.

## What's in the repo

| Path | What |
|---|---|
| `import.json` | the imported ETS2 release: version, commit, sha1 of every imported file, the credits |
| `tools/import_ets2.py` | imports from a clean, built and packed ETS2 Pets checkout into `build/import/` (`--check` verifies it) |
| `tools/common.py` | repo paths, host API (`megapack`), refuses to run unless `pack.config.json` says `ats` |
| `src/` | manifest, description, CREDITS.txt (imported) |
| `docs/adr/` | decision records |

Never in the repo: the imported pet files and everything built (`build/`, `output/`), the extracted base game (host
reference root).

## Building

1. Import (once per ETS2 release; the ETS2 checkout must be clean and its `build/pack` the stage of that release):
   `python tools/import_ets2.py <ETS2 Pets checkout>`, then set `src/manifest.sii` `package_version` to its version.
2. Pack with the `pack-mod` skill (`pack.config.json` prePack checks the import first).

## Requirements

- Python 3.10+ with PyYAML; the scs-mod-developer profile (host file, `megapack` host API, `scs_packer`).
- The ATS base game extracted to this host's reference root (`extract-reference`).
