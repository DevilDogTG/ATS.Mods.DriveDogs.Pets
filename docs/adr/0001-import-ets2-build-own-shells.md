# ADR 0001: pets come from the ETS2 build at a pinned commit; ATS writes its own mount shells

- Status: accepted
- Date: 2026-11-06

## Context

DriveDogs Pets is made in the ETS2 repo (`DevilDogTG/ETS2.Mods.DriveDog.Pets`): base models, poses, locks, icons and
animations live there (its ADR 0001), and an ATS release carries the version of the ETS2 release it imports. Models,
materials, textures, icons and animated hookups are game-neutral (same engine). What places a pet in a cab is not: the
ETS2 build mounts every pose through willet's per-truck defs and locator-only shells (Workshop 2997276736), and willet's
mod exists only for ETS2 and knows no ATS truck. Copying the ETS2 locks (Git LFS, 2.3 GB) and converting them here would
duplicate the pet-making toolchain for no gain.

## Decision

1. **Import the ETS2 build output.** An import tool copies the game-neutral output of the ETS2 build (gather_pets:
   `build/pets/out` + `models.json`; pairs: `build/pairs/out` + `pairs.json`) for one ETS2 commit. The files stay
   git-ignored; the repo commits only an import manifest: ETS2 version (`src/manifest.sii`), ETS2 commit, and a sha1
   per file. The build reads the import only after checking it against the manifest. No Git LFS in this repo.
2. **Own shells.** The ATS repo writes its own mesh-less mount shells (locator-only `.pmd`/`.pmg` per pose and slot,
   placed from the ATS slot locators) and its own truck accessory defs. No willet file ships, so willet is credited
   only where an imported pet is made from his models (the ETS2 credits chain).
3. Code the ATS build needs is copied from the ETS2 repo's tools with a wrong-game guard (`pack.config.json`
   `"game": "ats"`), never imported across repos. Pet-making tools are not copied.

## Consequences

- A pet change reaches ATS only by importing a newer ETS2 build; nothing here can re-pose or edit a pet.
- The ETS2 build has no release tags and its 9.2.0 stage sits in a worktree, so the import records the commit of the
  checkout it reads and refuses a dirty one; reproducing an import means rebuilding ETS2 at that commit.
- A small PMD/PMG writer for locator-only models is new code here; ATS LHD-only cabs mean no `_uk` shells.
