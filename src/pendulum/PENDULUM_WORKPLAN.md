# Workplan: Integrate a 4D Pendulum Dataset

Note: try to update all Julia packages to the latest version before you start. GO should be 0.4 and GML should be 0.5.

## Progress record (August 18, 2026)

### Execution pass update — final local validation

- Re-ran the focused pendulum suite after adding the required learning-path
  coverage: `26` tests pass, including `DataLoader` construction, a `4×n`
  `SymplecticAutoencoder` forward pass, and one CPU optimizer step.
- Re-ran the full package suite: MNIST/classification tests (`3147`),
  docstrings (`2`), and pendulum tests (`26`) all pass.
- Parsed all three purpose-specific Julia workflows, passed `bash -n` for
  `scripts/pendulum/run_sae.sh`, generated a temporary serialized dataset,
  generated a temporary `1400×1000` PNG, and completed a one-epoch CPU SAE
  smoke with a temporary `9.3 KB` HDF5 model.
- The pendulum-only documentation source passes doctests, cross-references,
  and HTML rendering when built from a temporary source tree with
  `checkdocs=:none`; the resulting `pendulum.html` and
  `pendulum_implementation.html` are present.
- The complete documentation build remains blocked by the pre-existing MNIST
  documentation path described below. No generated dataset, model, plot, or
  documentation build output was added to the repository.

### MNIST documentation failure findings and repair plan

- The full `docs/make.jl` build reaches the MNIST examples and fails while
  reading the cached MNIST gzip data with `GZip.ZError(-5, "buffer error")`.
  This indicates a truncated or corrupt local download/cache entry rather
  than a pendulum implementation failure; the package test suite's MNIST
  utility tests still passes.
- The same build reports four missing tutorial images:
  `mnist_visualization_light.png`, `mnist_visualization_dark.png`,
  `mnist_training_loss_light.png`, and `mnist_training_loss_dark.png`.
  These are referenced by `docs/src/mnist/mnist_tutorial.md` but are absent
  from the checkout.
- Recommended repair sequence: remove only the corrupt MNIST cache entry,
  re-download the dataset through the configured `DataDeps` source, and verify
  the archive checksum before rebuilding. Then either restore the four images
  from their source experiment or replace the references with reproducible
  generated figures. Finally, run `julia --project=docs docs/make.jl` in a
  clean environment with the dataset available locally or through a
  documented CI cache; do not make the pendulum docs build depend on a live
  MNIST download.
- No MNIST files were changed in this pendulum pass because repairing the
  cache or reconstructing missing figures would be unrelated scope and could
  overwrite a maintainer's local dataset state.

### Execution pass update

- Reconfirmed the integration target is the parent `GMLDatasets.jl` checkout.
- The current shell resolves Julia as `1.13.0-rc1`; the package remains declared
  for Julia `1.11`, so validation must distinguish implementation failures from
  toolchain or resolver failures.
- The local `GeometricMachineLearning` checkout is `0.5.0-DEV` and declares
  `GeometricOptimizers = "0.3"`; the local `GeometricOptimizers` checkout is
  `0.2.0`. No dependency pin or source change is justified until the resolver
  state is tested again.
- The next execution order is now: (1) attempt environment resolution without
  changing dependency declarations, (2) run the focused pendulum tests and
  workflow syntax/smoke checks, (3) fix only defects exposed by those checks,
  (4) review generated-file policy and prototype classification, and (5) run
  broader package/docs validation if the environment permits.

### Session continuation (August 18, 2026)

- Confirmed the already-instantiated environment resolves
  `GeometricOptimizers 0.3.1` with the `GeometricMachineLearning 0.5.0-DEV`
  `main` source; no dependency declarations were changed in this session.
- Fixed `scripts/pendulum/train_sae.jl` for the current HDF5 extension API by
  loading `HDF5` explicitly and importing `save` from
  `AbstractNeuralNetworks`. The previous `GeometricMachineLearning.save(...)`
  call had no matching method.
- The one-epoch CPU SAE smoke now passes and writes a temporary 9.3 KB HDF5
  artifact. A separate Julia process successfully loads that artifact with the
  matching `SymplecticAutoencoder` architecture.
- The package test target completed successfully: MNIST/classification tests
  passed (`3147`), docstrings passed (`2`), and pendulum tests passed (`20`).
- The non-interactive plotting smoke passed and wrote a temporary
  `1400×1000` PNG; `bash -n scripts/pendulum/run_sae.sh` also passed.
- The first documentation command used the package project and failed because
  `Documenter` is only declared in `docs/Project.toml`. Retrying with the docs
  project then failed because the local `GMLDatasets` path dependency is not
  instantiated/installed in that environment. No docs dependency files were
  changed.

### Final validation pass (August 18, 2026)

- Resolved and precompiled the existing `docs/Project.toml` environment with
  local `GMLDatasets`, `GeometricMachineLearning 0.5.0-DEV`, and
  `GeometricOptimizers 0.3.1`.
- The isolated pendulum documentation build passed, including doctests,
  cross-references, autodocs, and HTML rendering.
- The full documentation build remains blocked by pre-existing MNIST
  documentation requirements: the index doctest downloads MNIST from an
  unavailable network endpoint, and the tutorial references two absent PNG
  assets. The pendulum page itself is not implicated.
- Re-ran the focused pendulum tests (`20` passing), parsed all workflow
  scripts, ran the SAE smoke, checked `run_sae.sh` with `bash -n`, and passed
  `git diff --check`.
- Added root-level ignore rules for the generated pendulum `.jls` dataset and
  `.h5` model outputs. Prototype files remain retained pending maintainer
  review; no generated artifact is checked in.
- Several Julia commands initially hit `EPERM` on global depot lock/cache
  files; rerunning with depot access allowed the tests, SAE load, and plotting
  checks to complete. Treat this as an environment/connection issue, not a
  pendulum failure.

### Latest handoff status

- `Pkg.instantiate()` succeeded with the current registry and resolved
  `GeometricOptimizers 0.3.1` alongside the `GeometricMachineLearning 0.5.0-DEV`
  `main` source. No fallback pin or dependency declaration change was needed.
- The focused pendulum tests pass: 20 tests covering conversions, trajectory
  shape/time behavior, energy drift, dataset layout, and invalid arguments;
  the full package test target also passed in the current environment.
- All new Julia scripts parse successfully and `run_sae.sh` passes `bash -n`.
- The dataset-generation smoke script passes and writes a temporary serialized
  dataset.
- The SAE smoke exposed and fixed two current-API issues in
  `scripts/pendulum/train_sae.jl`: use `Adam()` with `Optimizer(...;
  step_size=1f-3)`, and use a `Float64` network to match the loader data.
- The SAE smoke and clean-process model loading now pass after the HDF5 save
  fix described above.
- Plotting and shell-wrapper syntax checks now pass.
- Documenter validation remains outstanding because the docs environment needs
  its local `GMLDatasets` path dependency instantiated; network/depot access is
  unreliable, so resume with the existing environment before attempting any
  package updates.

### Completed so far

#### Phase 1 — Repository and prototype audit

- Located the integration target at the parent `GMLDatasets.jl` checkout.
- Confirmed package metadata: Julia `1.11` compatibility, Documenter navigation
  in `docs/make.jl`, and existing `GeometricMachineLearning.DataLoader`
  conventions.
- Confirmed there is no applicable `AGENTS.md` in the repository tree.
- Audited all pendulum prototypes under `src/pendulum/`, including trajectory,
  SAE, reduced-model, CUDA, plotting, animation, and ensemble scripts.
- Recorded the migration classification in
  `src/pendulum/PROTOTYPE_AUDIT.md`; prototypes are retained for historical
  research because they are untracked and have not yet undergone a complete
  reference audit.

#### Phase 2 — Core dataset implementation

- Added `src/pendulum.jl` and included it from `src/GMLDatasets.jl`.
- Added `PendulumTrajectory` and `PendulumDataset` public types.
- Added deterministic `pendulum_trajectory` generation using
  `GeometricProblems.Pendulum` and `GeometricIntegrators.Gauss`.
- Added `pendulum_dataset` for collections of `(angle, momentum)` initial
  conditions.
- Added explicit angular/Euclidean conversions with documented coordinate
  ordering `(q₁, q₂, p₁, p₂)`.
- Added canonical and Euclidean Hamiltonian evaluation through
  `pendulum_energy`.
- Added `pendulum_matrix` for per-trajectory or concatenated `4×n` data.
- Added `pendulum_data_loader` as the isolated
  `GeometricMachineLearning.DataLoader` adapter.
- Added `GeometricProblems` and `GeometricIntegrators` to `Project.toml`.

#### Phase 3 — Tests and documentation

- Added focused tests in `test/pendulum.jl` and registered them in
  `test/runtests.jl`.
- Added the public user guide `docs/src/pendulum.md` and implementation note
  `docs/src/pendulum_implementation.md`.
- Added the pendulum page to the Documenter navigation in `docs/make.jl`.

#### Phase 4 — Reproducible workflows

- Added `scripts/pendulum/generate_dataset.jl` for deterministic serialized
  dataset generation.
- Added `scripts/pendulum/train_sae.jl` with configurable seed, epoch count,
  CPU SAE training, and non-overwriting HDF5 output.
- Added `scripts/pendulum/plot_dataset.jl` for non-interactive phase-portrait
  generation without retraining.
- Added strict wrapper `scripts/pendulum/run_sae.sh` with output-overwrite
  protection and run metadata reporting.

#### Validation and bug record

- Added `src/pendulum/PENDULUM_BUG_FIXES.md` documenting all implementation and
  regression-test bugs found and fixed so far.
- Fixed offset-array indexing from `GeometricSolutions` when extracting
  trajectory data.
- Fixed the uninferable unused type parameter in `PendulumDataset`.
- Fixed a regression assertion that compared a vector with a `2×1` matrix.
- Passed standalone CPU smoke checks for trajectory generation, coordinate
  round-tripping, energy behavior, and dataset matrix layout.
- Parsed all new Julia workflow scripts successfully and passed `bash -n` on
  the shell wrapper.
- Passed a one-epoch CPU SAE smoke, clean-process HDF5 model loading, the
  plotting smoke, and the full package test target.

### Current blockers and remaining work

- Complete the Documenter build from `docs/Project.toml` after making the
  local `GMLDatasets` path dependency available to that environment; avoid
  changing dependency declarations solely to work around the current depot or
  network state.
- Review the new training script against the exact resolved
  `GeometricMachineLearning` API and optional HDF5 extension before a long run.
- Decide, with maintainer review, which copied prototype files can be deleted;
  no prototype has been removed yet.
- Review the complete diff, dependency versions, generated-file ignore rules,
  and any desired checked-in model artifact before external repository actions.

### Refined completion gates

- Do not claim dependency completion from a resolver failure; record the exact
  Julia version, package source, and conflicting version bounds instead.
- A pendulum implementation pass is complete only after the focused tests,
  script parsing, and a short CPU workflow smoke check either pass or have a
  clearly documented optional-dependency blocker.
- Do not run long SAE training, delete prototype files, create artifacts, or
  perform commits/remote actions without a separate maintainer decision.


## Objective

Add a maintainable pendulum example to `GMLDatasets.jl`. The dataset should
generate trajectories of a pendulum represented in four Euclidean phase-space
coordinates and provide data suitable for training a symplectic autoencoder
(SAE) from four dimensions to two.

The implementation must be reproducible, testable, documented, and independent
of long-running interactive sessions.

## Current state and scope

This directory currently contains several prototype scripts, including:

- `pendulum.jl` and `pendulum_4d.jl` for trajectory generation;
- `pendulum_4d_train_sae.jl` and related scripts for SAE experiments;
- reduced-model training, plotting, animation, and CUDA variants.

These files must be inventoried before implementation. The final package
should not preserve duplicate or obsolete prototypes merely because they were
copied into this directory. Existing scripts may provide reusable algorithms,
but production functionality belongs in the package source tree, tests, and
small purpose-specific scripts.

## Constraints and decisions

1. **Inspect before modifying.** First identify the target `GMLDatasets.jl`
   checkout, its package layout, supported Julia versions, dependency policy,
   documentation structure, and existing dataset conventions.
2. **Keep long jobs out of interactive sessions.** Author executable shell
   scripts for dependency checks, tests, training, and artifact generation.
   The maintainer runs long commands locally and returns their output for
   analysis.
3. **Separate reusable code from experiments.** Dataset generation and data
   conversion must be importable package functionality. Training and plotting
   scripts should call that functionality rather than duplicate it.
4. **Make randomness reproducible.** Expose or document seeds and ensure tests
   do not depend on uncontrolled global random state.
5. **Avoid premature dependency coupling.** Use `GeometricProblems` and
   `GeometricIntegrators` where required by the existing project design.
   Treat `GeometricMachineLearning` as a temporary dependency only for the
   `DataLoader`/SAE demonstration, pending confirmation of the package roadmap.
6. **Gate external actions.** Commit, push, pull-request creation, issue
   creation, and deletion of copied files happen only after local review and
   explicit confirmation.

## Phase 1: Repository and prototype audit

### 1.1 Locate the integration target

- Confirm the `GMLDatasets.jl` repository root and branch.
- Read applicable `AGENTS.md`, `Project.toml`, `Manifest.toml`, CI, and
  `docs/` instructions.
- Record supported Julia versions and the commands used by CI.
- Inspect existing datasets, exports, naming conventions, and public API
  patterns.

### 1.2 Audit the pendulum directory

- Read every existing pendulum script, not only the main trajectory script.
- Identify duplicated coordinate transforms, Hamiltonians, parameter defaults,
  integrator setup, plotting code, and training logic.
- Record which scripts are prototypes, which are still useful, and which are
  candidates for removal after migration.
- Check whether any files, data, figures, or model parameters were copied in
  from another project and whether they are tracked or generated.

### 1.3 Write an implementation note

Before coding, document the selected API and answer these questions:

- What is the canonical dataset type or return value?
- What are the exact meanings and shapes of the four coordinates?
- Are trajectories returned per initial condition, concatenated, or both?
- Which parameters are configurable (`length`, `mass`, `gravity`, timestep,
  time span, initial conditions, and integrator)?
- What numerical and physical invariants can be tested reliably?
- Where does `DataLoader` integration belong, and can it be isolated behind an
  optional example dependency?

**Gate:** do not begin implementation until the target package layout and API
are known.

## Phase 2: Implement the dataset

### 2.1 Add core functionality

- Implement all reusable functionality in the agreed source file/module,
  initially named `pendulum_4d.jl` if that matches package conventions.
- Use `GeometricProblems.Pendulum` and `GeometricIntegrators` consistently with
  existing datasets.
- Implement conversion between angular/canonical coordinates and the four
  Euclidean coordinates used by the SAE.
- Provide deterministic trajectory generation with validated dimensions,
  time grid, parameters, and initial conditions.
- Provide a documented way to generate a collection of trajectories and to
  convert it into the layout expected by downstream learning code.
- Validate invalid arguments early with useful errors.

### 2.2 Keep the API numerically explicit

- State coordinate ordering in documentation and tests.
- State whether endpoints are included and how `nsteps` is defined.
- Preserve element type where practical and avoid accidental CPU/GPU
  assumptions in the dataset layer.
- Do not embed plotting, training, animation, or file-writing behavior in the
  dataset implementation.

**Gate:** a small, non-training smoke test can generate one trajectory and
verify its shape, time grid, finite values, and approximate energy behavior.

## Phase 3: Tests

Add tests following the package’s existing test organization. At minimum,
cover:

1. default and custom parameter construction;
2. coordinate conversion round trips where mathematically applicable;
3. trajectory dimensions, time spacing, endpoint behavior, and element type;
4. reproducibility for a fixed seed or fixed initial-condition collection;
5. finite values and an appropriate energy-conservation tolerance;
6. multiple initial conditions and batch/concatenation behavior;
7. `DataLoader` construction and the expected input dimension;
8. a minimal neural-network/SAE forward pass and one short training step.

Tests must remain short and CPU-compatible. Do not put the full SAE training
run in the test suite. If `DataLoader` or SAE tests require an optional
dependency, follow the project’s established optional-test convention rather
than making all dataset tests depend on it.

**Gate:** run the focused tests first, then the package test suite using the
repository’s documented command. Do not broaden unrelated fixes when failures
are discovered.

## Phase 4: Training and plotting workflow

Create purpose-specific scripts under the project’s established `scripts/`
directory (or document why another location is required):

### 4.1 Training script

- Generate or load the canonical dataset.
- Train the SAE and, if in scope, the reduced model.
- Accept configuration through clearly documented constants or command-line
  arguments.
- Set and report a seed.
- Save model parameters, training metadata, and loss history in a stable,
  versioned artifact format supported by the project (HDF5 if that is the
  established convention).
- Fail clearly when required optional dependencies or output directories are
  unavailable.

### 4.2 Plotting script

- Load checked-in or user-provided trained parameters.
- Recreate representative trajectories and latent/reconstruction plots.
- Write figures to an ignored/generated output directory.
- Avoid retraining and avoid requiring CUDA.
- Support a non-interactive backend suitable for documentation CI.

### 4.3 Execution wrappers

Author executable shell wrappers for long-running training and any expensive
artifact generation. Each wrapper should:

- use strict shell error handling;
- record the command, Julia project, seed, and output location;
- avoid overwriting artifacts unless explicitly requested; and
- print enough progress information to diagnose failures.

**Gate:** run a short smoke configuration before any full local training.

## Phase 5: Documentation and checked-in artifacts

- Add a user-facing page to the existing Documenter.jl navigation.
- Explain the dataset, coordinate convention, parameters, installation or
  optional-dependency requirements, and basic usage.
- Reuse the plotting workflow for figures, but never train during Documenter
  CI.
- Train locally with a recorded configuration and verify that the saved model
  can be loaded from a clean Julia process.
- Check in only the small, necessary parameter artifact(s), with stable names
  and documentation of their provenance. Do not check in generated caches,
  large raw trajectories, animations, or build output.
- Ensure documentation builds without a display and without CUDA.

**Gate:** run the focused documentation build and confirm that all referenced
artifacts exist in a fresh checkout.

## Phase 6: Cleanup and review

- Compare every prototype file with the new implementation and classify it as
  migrated, retained as an example, or obsolete.
- Remove copied files only after confirming they are not referenced by tests,
  documentation, scripts, or package metadata.
- Review dependency changes, public exports, artifact sizes, generated-file
  ignore rules, and reproducibility instructions.
- Inspect the complete diff and run the focused tests, package tests, and docs
  checks that are feasible without long-running training.

## Phase 7: External repository actions

Only after local validation and maintainer approval:

1. Commit the implementation with a message that identifies AI assistance if
   AI was used.
2. Push the branch and open a pull request describing the implementation,
   tests, artifacts, and any AI contribution.
3. Open or update the relevant GitHub issue stating that
   `GeometricMachineLearning` should cease to be a required dependency once
   `DataLoader` moves to its planned package. Check existing GML issues first
   to avoid duplicating an issue.

Do not claim completion of these steps until the corresponding remote links
and statuses have been verified.

## Definition of done

- The dataset is part of the package’s normal source and export structure.
- The documented API generates correct 4D pendulum data reproducibly.
- Focused tests cover generation, physical/numerical sanity, loading, and a
  minimal SAE interaction.
- Training and plotting are separate, reproducible, CPU-safe workflows.
- Documentation builds without training and uses verified parameters.
- Obsolete copied files and generated outputs are removed or ignored.
- The final diff, dependency changes, and external actions have been reviewed.
