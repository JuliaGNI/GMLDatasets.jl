# Known issues

### K1 · The `DataLoader` constructor for labelled images is type piracy

- **location:** `src/data_loader.jl:29`
- **evidence:** `Aqua.test_piracies(GMLDatasets)` fails with one result; `test/quality/aqua.jl` marks
  it `broken = true`. The method adds to `GeometricMachineLearning.DataLoader` for argument types
  that this package does not own.
- **kind:** defect
- **found:** issue #29, part M29 of the test-suite migration

## Upstream

### K2 · Aqua's `persistent_tasks` check cannot run, because `InternedStrings` has no `Project.toml`

- **location:** `test/quality/aqua.jl:13`
- **evidence:** `Aqua.has_persistent_tasks(Base.PkgId(GMLDatasets))` with Aqua 0.8.18 throws
  "Unable to locate Project.toml in …/InternedStrings/JTzem". The dependency comes in through
  `MLDatasets` and `Pickle`. `test/quality/aqua.jl` marks the check `broken = true`.
- **kind:** upstream
- **found:** issue #30, part M29 of the test-suite migration

## Documentation

### K3 · The CHANGELOG states a narrower `GeometricMachineLearning` bound than `Project.toml` has

- **location:** `CHANGELOG.md:57`
- **evidence:** the `[Unreleased]` bullet "`[compat]` widens …" says
  `GeometricMachineLearning = "0.6, 0.7"`; `Project.toml:18` has `GeometricMachineLearning = "0.6, 0.7, 0.8"`.
- **kind:** found late
- **found:** #19 (2026-09-27, commit d92e3d5), which widened the bound in `Project.toml` and left
  the CHANGELOG line from #18 unchanged
