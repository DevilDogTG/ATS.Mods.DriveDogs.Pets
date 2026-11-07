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
| `tools/build_pack.py` | stages `build/pack/`: imported pets, hookup units, our shells, a def per pose / pair on every ATS truck with its slot, the dashboard-table items; checks the stage (`build/pack-report.md`) |
| `tools/shells.py` | our locator-only mount shells (`tools/test_shells.py` checks them) |
| `fixes.yaml` | measured per-truck fixes: `seat_centre` (seat poses onto each truck's co-driver cushion), `slot_floor` (floor pets onto each truck's floor) |
| `tools/cab_fit.py`, `tools/cabs.py` | measures the ATS cabs (base game interiors by variant, co-driver seat accessories): `seat` / `floor` (`--write`: `fixes.yaml`), `bed` (each bed pet on every bunk, % off the mattress) |
| `tools/generate_cover.py` | `src/cover.jpg` (the Mod Manager icon): the DriveDogs cover with the ATS badge and the imported shepherd's sit pose (`pma_pose.py` + `pmg_preview.py`) |
| `build.yaml` | the ATS base version the trucks and slots are read from, shop price / unlock |
| `tools/common.py`, `tools/sii_text.py` | repo paths, host API (`megapack`), refuses to run unless `pack.config.json` says `ats`; SII unit scanner |
| `src/` | manifest, description, cover, CREDITS.txt (imported) |
| `docs/adr/` | decision records |

Never in the repo: the imported pet files and everything built (`build/`, `output/`), the extracted base game (host
reference root).

## Building

1. Import (once per ETS2 release; the ETS2 checkout must be clean and its `build/pack` the stage of that release):
   `python tools/import_ets2.py <ETS2 Pets checkout>`, then set `src/manifest.sii` `package_version` to its version.
2. Pack with the `pack-mod` skill, `-Root build/pack` (`pack.config.json` prePack checks the shells, draws the cover
   with the manifest version, then `tools/build_pack.py` checks the import and stages the pack). Test builds stay in
   `output/local/`.

## Requirements

- Python 3.10+ with PyYAML; the scs-mod-developer profile (host file, `megapack` host API, `scs_packer`).
- The ATS base game extracted to this host's reference root (`extract-reference`).
