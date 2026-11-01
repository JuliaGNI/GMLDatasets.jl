# Changelog

All notable changes to GMLDatasets.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

## [Unreleased]

### Added

- **`onehotbatch(S, target)`**, which encodes the labels in a chosen element type. The one-argument
  form is unchanged and is now a forwarder to it, so the encoding keeps the label type by default.
  The output uses the same backend as `target`.

- **A testset pinning the preprocessing contract of `split_and_flatten` and `onehotbatch`** —
  ordering, shape, element type and backend — against independent reference implementations.

- **`scripts/revision/`, the offline experiment harness** the paper revision's measurement,
  statistics and reproducibility requirements depend on. `run_experiments.sh` drives MNIST,
  Fashion-MNIST, the pendulum SAE and the retraction benchmark non-interactively, over ten seeds in
  full mode; `run_in_screen.sh` detaches it so a dropped SSH connection cannot kill an eight-hour
  run. Every stage streams to a durable log, records its status in `stages.csv`, and the whole run
  directory is packaged as a `.tar.gz` with an immediately verified SHA-256 alongside the Julia,
  CUDA and Git environments it ran in.

  What is new in the trainers, rather than around them: the image trainer takes a dataset and a seed
  list, so a repetition is paired across configurations and across MNIST and Fashion-MNIST, and it
  gains the `scalar-moment-adam` baseline — `ScalarMomentAdam` ([li2020efficient], Algorithm 2) on
  the Stiefel leaves and ordinary `Adam` on the Euclidean ones. Both trainers write machine-readable
  run records and per-step or per-epoch loss curves, which is what the statistics in the paper are
  computed from rather than from parsed report prose.

  **The baseline is one method object, not one adapter per trainer.** `ScalarMomentAdam` accepts a
  single `StiefelManifold` on purpose — a scalar second moment is a statement about one manifold —
  and both networks here are mixed trees, so it has to be assembled around the released method.
  That assembly is `GeometricOptimizers.CompositeMethod`, upstream, and
  `scripts/revision/scalar_moment_adam.jl` is the whole of what is local about it: the coefficients,
  and which ‖·‖² the second moment accumulates. What each trainer still owns is its *step loop*,
  because the two drive different ones — this repository's image trainer drives
  `GeometricOptimizers` directly (`scripts/geometric_optimizers/leaf_composite.jl`) and the pendulum
  trainer goes through `GeometricMachineLearning`'s training loop.
  The pendulum trainer defaults to 1000 epochs, configurable via `SAE_N_EPOCHS`.

  **The retraction benchmark is not here.** It is `scripts/retraction_records.jl` in
  `GeometricOptimizers`, beside the algorithms, the reference and the seeded lift sweep it measures,
  sharing that sweep with the `retraction_accuracy.jl` the package's own published figures come
  from. This harness supplies the provenance stamp and validates what comes back, which is the part
  that belongs to a *bundle's* contract rather than to the measurement.

  **The decomposed timings are the point of schema 4.** Gradient/AD, optimizer-state/direction and
  retraction/application are measured as mutually exclusive intervals through the
  `GeometricOptimizers` step observer, synchronising the device at every boundary, after an
  identically seeded warm-up step is discarded. `scripts/revision/README.md` states exactly what
  falls inside each category and what falls outside all three — the three totals are optimizer-step
  components and are not expected to add up to `total_seconds`.

  **Two gaps, both deliberate.** The pendulum trainer records end-to-end time, host allocation and
  GC time but *not* the decomposed phase timings, so it cannot carry a direction/retraction cost
  claim yet. And the Riemannian gradient of a device-resident point arrives on the host, for a
  defect in the packages that produce the gradient
  ([`GeometricMachineLearning` #258](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/258),
  [`AbstractNeuralNetworks` #39](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/issues/39)),
  and GeometricOptimizers 0.9 refuses a host gradient at a device point, so the pendulum stage and
  the environment preflight do not run on a CUDA host until those are fixed (`KNOWN_ISSUES.md`,
  K5). The image stages never take that path and are unaffected.

- **`scripts/revision/capture_source.jl`**, the harness's one provenance capture: a repository's
  commit, a reproducible patch that includes untracked files, the patch's SHA-256, and a cross-check
  that a tree `git status` calls dirty produced a non-empty patch. The runner calls it for both
  repositories a bundle records and passes its output to the retraction benchmark, so the
  mechanism whose whole purpose is to have exactly one copy has exactly one.

- **`.github/workflows/Scripts.yml`**, which runs the two harness suites that need
  `GeometricOptimizers` against `scripts/Project.toml`. The other two need nothing but `SHA` and
  Base and run from `test/runtests.jl`, inside the package suite. `CI.yml` is byte-identical in
  every repository by policy and cannot gain a step; its own header prescribes this escape hatch.
  Not a required check: the scripts environment resolves the newest releases its `[compat]` admits,
  so the job can go red for a release made upstream.

### Changed

- **GMLDatasets requires GeometricMachineLearning 0.9** (`[compat]` `"0.9"`, from
  `"0.6.1, 0.7, 0.8"`). GML 0.9 trains with `TrainingOptimizer` and `train!` and no longer defines
  `Optimizer` or `GradientOptimizer`, so the two tests that trained a network follow it:
  `test/mnist_utils.jl` takes one `GradientMethod` step with
  `GeometricOptimizers.optimization_step!` and still asserts that the step lowers the loss, and
  `test/pendulum.jl` trains the symplectic autoencoder for one epoch with `train!`. `src/` is
  unchanged. `test/Project.toml` bounds `NeuralNetworkParameters` to `"0.5"`, which GML 0.9 needs,
  and adds `GeometricOptimizers`, whose binding inside GML is not public, with no bound of its own.
  Under `NeuralNetworkParameters` 0.5 a function differentiated with respect to a parameter set is
  called with the `NetworkParameters` itself, so the `NamedTuple` arm of the annotation in
  `test/mnist_utils.jl` goes. `scripts/Project.toml` takes GML from the registry at `"0.9"` and no
  longer names a `[sources]` entry for it; delete a `scripts/Manifest.toml` written before this
  change.

  The MNIST tutorial follows the same API: it builds each optimizer with
  `TrainingOptimizer(nn.params; algorithm, linesearch)`, its training calls are `train!`, and it
  takes `accuracy` from `ReducedComplexityModeling`, which owns it in GML 0.9, so
  `docs/Project.toml` adds `ReducedComplexityModeling` at `"0.5"`. Its figures still come from the
  committed weights and loss histories.

  Six `[compat]` floors rise, because GML 0.9.0 and its dependencies cannot install beside the old
  ones: `AbstractNeuralNetworks` to `"0.9.2"` (from `"0.7.2, 0.8, 0.9"`) and `GeometricEquations` to
  `"0.21.5"` (from `"0.21"`), which GML 0.9.0 requires; `GeometricIntegrators` to `"0.18.6"` (from
  `"0.18.2"`), the first release on `GeometricBase` 0.15, which `GeometricEquations` 0.21.5
  requires; `GeometricProblems` to `"0.9.1"` (from `"0.8, 0.9"`) and `GeometricSolutions` to
  `"0.6.6"` (from `"0.6.5"`) for the same reason; and `KernelAbstractions` to `"0.9.38, 0.10"`
  (from `"0.9.15, 0.10"`), which `GeometricOptimizers` 0.9 requires.

  These bounds replace every bound of the same dependencies that the entries below name: for GML,
  `"0.6, 0.7"` in the four-dependency widening, `"0.6, 0.7, 0.8"` (#19, which has no entry of its
  own), and `"0.6.1, 0.7, 0.8"` after the floor raise; for `NeuralNetworkParameters`, the
  `"0.2.2, 0.3"` of `test/Project.toml`; the `AbstractNeuralNetworks`, `GeometricIntegrators`,
  `GeometricProblems`, `GeometricSolutions` and `KernelAbstractions` bounds of the widening and of
  the floor raises; and the `"0.21"` that `GeometricEquations` takes as a direct dependency. They
  close `KNOWN_ISSUES.md` K3.

- **Text that GML 0.9 made stale now states the current API.** The learning-rate comments of the
  five MNIST scripts in `scripts/geometric_optimizers/` say that `Optimizer` defaults to
  `Static(1e-3)` only for the Adam methods and to an expanding `Backtracking` search for
  `GradientMethod` and `MomentumMethod`, as `GeometricOptimizers` 0.9.2 does; the scripts' code is
  unchanged. `docs/src/running_the_experiments.md` lists the `MNIST_CONFIGURATIONS` keys that
  `mnist_cuda_repetitions.jl` accepts, with `geometric-adam-cayley` as the default and the two
  aliases. `KNOWN_ISSUES.md` K1 names `ReducedComplexityModeling` as the owner of `DataLoader`. The
  docs drop the unused `GeometricMachineLearning` InterLinks entry, its committed inventory and the
  `using GeometricMachineLearning` in `docs/make.jl`; the tutorials still load the package, so
  `docs/Project.toml` keeps it.

- **`scripts/gml/transformer_mnist.jl` and `transformer_fashion_mnist.jl` run on GML 0.9, and the
  documentation describes its API.** The two scripts build each optimizer with
  `TrainingOptimizer(nn.params; algorithm, linesearch)` from `Adam()`, `GradientMethod()` and
  `MomentumMethod(; α = T(0.5))`, train with `train!`, and take `accuracy` from
  `ReducedComplexityModeling`, which `scripts/Project.toml` adds at `"0.5"`. `MNIST_N_EPOCHS` sets
  their number of epochs (default 500), so `MNIST_N_EPOCHS=2` is a smoke run. `KNOWN_ISSUES.md` K4
  keeps the four `scripts/geometric_optimizers/` scripts only. *Running the Experiments* states the
  GeometricOptimizers 0.9 defaults and constructors, and *The Pendulum Data Set* links `DataLoader`
  to its docstring in the `ReducedComplexityModeling` manual, whose inventory `docs/make.jl` reads,
  with a committed copy in `docs/inventories/`.

- **Aqua's `persistent_tasks` check runs, and is no longer marked broken.** `InternedStrings` 0.7.1,
  which the tests reach through `MLDatasets` and `Pickle`, has a `Project.toml`, so the check runs
  and passes; marked broken, it failed `Pkg.test()` as an unexpected pass on a fresh resolve.
  `test/Project.toml` bounds `InternedStrings` to `"0.7.1"`, because `Pickle` still admits 0.7.0,
  which has no `Project.toml` and makes the check error. This closes `KNOWN_ISSUES.md` K2; the
  upstream fault is #30.

- **The scripts environment and the revision harness run on GeometricOptimizers 0.9.1 and the
  `TrainingOptimizer` of GeometricMachineLearning 0.9.** The Scripts workflow failed to resolve,
  because `scripts/Project.toml` bounded GeometricOptimizers to 0.8 while the 0.9 line was what the
  harness needs; the bounds are now GeometricOptimizers `"0.9.1"` and NeuralNetworkParameters
  `"0.5"`, which GML 0.9 needs. `GeometricOptimizers` comes from the registry and names no
  source: 0.9.1 is the first release that carries `CompositeMethod`, which the
  `scalar-moment-adam` baseline is built from. `GeometricMachineLearning` comes from the registry
  too (see the entry on GML 0.9 above). An environment for a benchmark that measures optimizer
  steps no longer follows a branch of the package it measures, so a number is read back from the
  release the run bundle's manifest records.

  The harness follows the 0.9 API. A method takes its element type from its coefficients, so
  `Adam(T; …)`, `ScalarMomentAdam(T; …)` and `MomentumMethod(β)` become `Adam(; …)`,
  `ScalarMomentAdam(; …)` and `MomentumMethod(; α = β)`, and `Adam()` serves a `Float32` set, which
  `Optimizer` converts. `LeafComposite` gives each leaf's `Optimizer` the method `leafmethod` picks
  for it, because an `Optimizer` takes no `CompositeMethod`. The pendulum trainer builds a
  `TrainingOptimizer` on the network's parameters and trains with `train!`, at the configured rate
  as before.

- **The repository moves from the JuliaGNI organisation to JuliaGML.** The documentation now
  deploys to `https://JuliaGML.github.io/GMLDatasets.jl`, and the README badge links there. The
  source links of the documentation and the README install command name
  `github.com/JuliaGML/GMLDatasets.jl`. GitHub redirects the old repository URL; it does not
  redirect the old documentation site.

- **The README install instructions no longer develop `GeometricMachineLearning` from `main`.** They
  install GMLDatasets from its repository and let GML resolve from the registry. The paragraph
  described a `[sources]` pin that `Project.toml` no longer has, and GML `main` no longer had the
  `Optimizer` API that the tests and the MNIST tutorial used then.

- **The documentation builds against the registered GeometricMachineLearning, not its `main`
  branch.** `docs/Project.toml` no longer pins GML to `main` under `[sources]`; that pin was for the
  time before 0.5 was registered. GML `main` has since removed `Optimizer` in favour of
  `TrainingOptimizer` and `train!`, so the MNIST tutorial failed with
  ``UndefVarError: `Optimizer` not defined``, while the package and its tests were on GML 0.6–0.8.
  The docs now resolve the same GML release as the package.

- **`Pkg.test()` no longer runs the doctests.** `test/quality/doctests.jl` is now the group
  `doctests`, which an empty `ARGS` does not run; `Pkg.test(test_args = ["doctests"])` runs it. In
  CI the Doctests job stays their runner, so the test matrix no longer runs them a second time.

- **`test/docstrings.jl` moves to `test/integration/docstrings.jl`**, because the test convention
  keeps a test file at the top level of `test/` only where it mirrors `src/<name>.jl`, and there is
  no `src/docstrings.jl`. The file tests the docstring examples of `onehotbatch` and
  `split_and_flatten`, both in `src/mnist_utils.jl`, but `test/mnist_utils.jl` already holds the
  tests of that file. Its content and its `@safetestset` label are unchanged.

- **CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of `Julia min`, and a test
  job saves the Julia cache only when it succeeds.**

- **The preprocessing figure of the MNIST tutorial is drawn with CairoMakie during the docs
  build**, in pure Julia. A `@setup` block of the tutorial includes
  `docs/src/mnist/mnist_visualization.jl` and saves one PNG per Documenter theme. The figure keeps
  the layout of the TikZ original: the digit with its patch grid, the 16 patches, the flattened
  patches, the flattened image with the red grid, and the transformer block with its arrows and
  labels. No LaTeX runs, and no image is committed.

  The tutorial wraps each themed image, this figure and the training-loss plot, in Documenter's
  `docs-light-only` or `docs-dark-only` container, so the page shows one image per theme. Before,
  it showed both images of each pair in either theme. The new `test/quality/mnist_figure.jl`, in
  the `core` group, checks both from the tutorial's source.

  `.github/workflows/Documenter.yml` is the canonical workflow again: the steps that installed TeX
  and ran `make` on the figure are gone.

- **The test suite follows the shared test convention.** `test/runtests.jl` defines `GROUPS` and
  lists one `@safetestset` per test file: `core` holds `quality/aqua.jl`, `mnist_utils.jl`,
  `docstrings.jl`, `pendulum.jl` and `quality/mnist_figure.jl`, and `slow` holds
  `quality/doctests.jl`. Empty `ARGS` runs both groups. The three existing test files keep their
  paths and their content.

  The test dependencies move from `[extras]` and `[targets]` of `Project.toml` into a new
  `test/Project.toml`. `NeuralNetworkParameters`, `Random`, `SafeTestsets`, `Test` and `Zygote`
  take their bounds with them unchanged, so the root `[compat]` loses those five entries. The
  package's own dependencies that the tests load (`AbstractNeuralNetworks`,
  `GeometricMachineLearning`, `GeometricProblems`) have no entry there, because the root bounds
  apply, and the new `Aqua` and `Documenter` have none either.

  The new `test/quality/aqua.jl` runs `Aqua.test_all`. One check is marked broken, with its issue:
  `piracies`, for the `DataLoader` constructor for labelled images (#29). The new
  `test/quality/doctests.jl` runs `doctest(GMLDatasets)`, as the Doctests job of CI does. No JET
  file is added.

- **The scripts environment resolved `GeometricMachineLearning` from `main`, not from the registry,
  until GML 0.9 was registered** (see the entry on GML 0.9 above). The harness needs the optimizer
  step observer and `PhaseTimer` of
  [GeometricOptimizers #78](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/78), the backend
  fixes of [#79](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/79),
  [#84](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/84) and
  [#85](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/85), and the 0.8 seams above.
  `scripts/Manifest.toml` stays untracked: every run bundle already carries the manifest that run
  resolved.

  Two of those fixes are what `scripts/revision/check_environment.jl` probes, because no version
  number can express them: that an optimizer cache and state can be built for a parameter set living
  on the GPU, and that the Riemannian gradient of a device-resident point lands on the device. Both
  once cost a run its pendulum stage after the image stages had already spent their hours — the
  image trainer keeps its parameters in a **host** container and copies to the device inside `∇F!`,
  so nothing before the pendulum stage ever builds a device-resident cache. The gradient probe fails
  on a CUDA host while GML #258 is open (K5).

- **`[compat]` widens to the current releases of four dependencies**, as one change rather than four:
  `AbstractNeuralNetworks = "0.7, 0.8"`, `GeometricMachineLearning = "0.6, 0.7"`,
  `GeometricProblems = "0.8, 0.9"` and `NeuralNetworkParameters = "0.2.2, 0.3"`. These arrived as four
  separate CompatHelper pull requests, and separately none of them can go green: they are one
  dependency family, released together, so resolving any one of the new majors drags the others in
  with it. Combining them is what lets a single CI run answer whether the package works against the
  current ecosystem.

  The `NeuralNetworkParameters` floor stayed at **0.2.2**, not the bare `0.2` two of those pull
  requests proposed. `test/mnist_utils.jl` annotated against `ParameterSet`, which 0.2.2 added, so a
  `0.2` floor would have advertised support for versions that cannot run the test suite.

- **The parameter-set annotation in `test/mnist_utils.jl` admitted a `NamedTuple`** until the move
  to GML 0.9 above. Under
  `GeometricMachineLearning` 0.7, `NeuralNetworkParameters`' `ZygoteRules.pullback` seeds the reverse
  pass with the wrapped `NamedTuple` rather than the container, so `loss_dl` is called with one while
  differentiating and the old `Union{Tuple, NetworkParameters}` was a `MethodError`.

- **`[compat]` raises four floors to the lowest versions that resolve**, a compat-only change with
  no change in behaviour: `GeometricIntegrators = "0.18.2"`, `GeometricSolutions = "0.6.5"`,
  `KernelAbstractions = "0.9.15"` and `MLDatasets = "0.7.13"`, from `0.18`, `0.6`, `0.9` and `0.7`.
  Under the CI Downgrade job, which forces every dependency to its lower bound, these four resolved
  to the newer versions above instead, so the old floors advertised versions that the dependency
  graph cannot select. The upper bounds are unchanged.

- **Julia 1.12 is the floor** (`[compat] julia = "1.12"`, from 1.11), in both `Project.toml` and
  `scripts/Project.toml`. A caller on Julia 1.11 can no longer install this version.

- **The `AbstractNeuralNetworks` lower bound rises from `0.7` to `0.7.2`.** `test/Project.toml`
  bounded `NeuralNetworkParameters` to `"0.2.2, 0.3"`, while `AbstractNeuralNetworks` 0.7.0 and
  0.7.1 require `NeuralNetworkParameters` 0.1 and 0.2. The `Downgrade` job resolved the root
  Project.toml at 0.7.0, so `Pkg.test()` stopped on an empty intersection between the manifest and
  the test environment. It failed the same way on `main` before the raise.

- **The `GeometricMachineLearning` lower bound rises from `0.6` to `0.6.1`.** `GeometricMachineLearning`
  became `"0.6.1, 0.7, 0.8"`. With the `AbstractNeuralNetworks` bound at 0.7.2, the
  `Downgrade` job resolved the root at `GeometricMachineLearning` 0.6.1 and named 0.6.0 as
  unreachable.

### Removed

- **The TikZ sources of the MNIST figure**: `docs/src/mnist/Makefile`, `final_image.tex`,
  `mnist_visualization.tex` and `mnist_visualization_dark.tex`, and the `.gitignore` rules for
  their intermediate files.

- **`.github/workflows/TagBot.yml`.** This package is **not registered in General**, so TagBot had
  nothing to do: it exists to create a GitHub release once a registry pull request merges, and no
  registry pull request will ever merge for a package that is not there. Verified against this
  package's own identity — UUID `676c5310-75a3-4edf-a7af-a2dbff91674b` appears nowhere in
  `Registry.toml`, and neither does the name. `Experiments/CLAUDE.md:53` states the general case in
  one line: nothing under `Experiments/` is registered.

  It was the only `TagBot.yml` under `Experiments/`. **The installer did not put it there and does
  not skip `Experiments/`:** `install-workflows.sh:80` iterates `Packages/*/` and `Experiments/*/`
  alike, and gates TagBot at `:94` on `kind = package`, exactly as its own header at `:28` says.
  What let the file survive is that `install_file` only ever *copies* — nothing removes a file the
  template set no longer covers — and `verify-workflows.jl:146` skips the TagBot comparison for
  experiments rather than asserting its absence. So there is no recurring cause: the deletion is
  final, and the installer will not restore it.

  This is the deletion `Experiments/CLAUDE.md:21-25` asks for, where the file is named as "drift
  rather than intent". Nothing in CI referenced it, and it never triggered on `push` or
  `pull_request`, so it can never have been a required check. If this package is ever registered,
  the fix is to bring it into the installer's scope, **not** to hand-copy
  `Knowledge/AI/githooks/workflows/TagBot.yml` back — hand-copying is the drift the installer
  exists to prevent.

### Added

- **`src/pendulum.jl`**, new: `pendulum`, `angular_to_euclidean`, `euclidean_to_angular` and
  `pendulum_energy`, with `docs/src/pendulum.md` and `scripts/pendulum/`. A mathematical pendulum,
  integrated symplectically and lifted into four dimensions, as a data set for the symplectic
  autoencoders — small, deterministic and with nothing to download.

  It is a thin layer, in the same sense `src/datasets.jl` is one over `MLDatasets`. `pendulum` is
  `GeometricProblems.Pendulum.hodeensemble` composed with `GeometricIntegrators.integrate` and
  returns the `GeometricSolutions.EnsembleSolution` unchanged, rather than a trajectory and data-set
  type of its own; `pendulum_energy` on canonical coordinates is
  `GeometricProblems.Pendulum.hamiltonian` broadcast, rather than a second copy of that formula; and
  there is no data loader function, because the array `angular_to_euclidean` returns is already what
  `GeometricMachineLearning.DataLoader`'s tensor constructor reads. Compare
  `GeometricMachineLearning`'s own symplectic-autoencoder tutorial, which is `hodeproblem`,
  `integrate`, `DataLoader` and nothing else.

  What is genuinely new is the **lift**. `angular_to_euclidean` replaces the angle by the position of
  the bob in the plane,

  ```
  q = (l sin θ, l cos θ),   p = (pθ cos θ / l, −pθ sin θ / l),
  ```

  which is a symplectomorphism onto its image, and that image is the tangent bundle of the circle of
  radius `l`: two dimensions of data sitting inside four, held there by ‖q‖ = l and q·p = 0. A
  `SymplecticAutoencoder` therefore has a curved submanifold to find rather than a linear subspace,
  which is the point of the example. `pendulum_energy` carries the same Hamiltonian into those
  coordinates, `H = ‖p‖²/2m + m·g·q₂`, and the tests check that it agrees with `GeometricProblems`'
  to the last bit at parameters that are *not* the defaults — the one place where dropping `l`, `m`
  or `g` somewhere in the lift would otherwise go unnoticed.

  On a solution, `angular_to_euclidean` reads `l` off the problem that solution was integrated from
  rather than assuming it is one. The canonical coordinates come out of `GeometricSolutions` as
  `OffsetVector`s indexed from zero, since a solution counts the initial condition as step 0; they
  are collected on the way out so that nothing downstream ever sees a zero-based axis.

  The Euclidean data are one `4 × n_t × n` array with rows `(q₁, q₂, p₁, p₂)`, and not the
  `(q = …, p = …)` named tuple that `DataLoader` also accepts, deliberately: the named tuple builds a
  loader, but training through it hits a missing `ZygotePullback` method for `AutoEncoderLoss` in
  `GeometricMachineLearning`. The stacked layout is what the symplectic architectures assume anyway —
  the first half of the rows is `q`, the second half is `p` — and one epoch of training is part of
  the test suite so that a change of layout cannot pass silently.

  `GeometricProblems`' Hamiltonian is `p²/(2ml²) + mgl·cos(q)`, with a **plus** in front of the
  potential, so the pendulum hangs down at `θ = π` and stands upright at `θ = 0`. That is the
  opposite of the convention most textbooks use and it is what every choice of initial conditions
  here depends on, so the documentation says so in a box rather than in passing.

  `scripts/pendulum/train_sae.jl` trains on the separatrix-relative grid of `momentum_fractions`
  below, including librating and rotating trajectories on both sides of the separatrix. Although the
  standard reduction cannot use a globally real-valued angular coordinate, a bounded cylinder can be
  represented differently in two dimensions — for example as an annulus. The SAE experiment therefore
  tests whether a learned reduction can represent the separatrix without reproducing `(θ, pθ)`. It
  uses the deeper 12,000-epoch architecture that demonstrated this behavior experimentally, prefers
  CUDA when available, falls back to the host, and saves the trained parameters and loss curve with
  HDF5.

- **`pendulum(; momentum_fractions)` samples the pendulum data set relative to the separatrix, and
  `separatrix_momentum` is exported.** `separatrix_momentum(θ, parameters)` is the momentum
  `m·l·√(2g·l·(1 − cos θ))` that puts the pendulum exactly on the separatrix at angle `θ`. Given a
  vector of fractions `f`, `pendulum` keeps its `qsamples` angles and gives each of them the
  momentum `f · separatrix_momentum(θ)`, so `|f| < 1` librates and `|f| > 1` rotates at *every*
  angle, and the sign of `f` picks the direction of rotation. A Cartesian grid cannot express that:
  a fixed `pθ` crosses the separatrix as `θ` varies, so no rectangle in `(θ, pθ)` controls how
  close to the separatrix the data comes, and the band of rotating orbits nearest it is left empty.
  `pmin`, `pmax` and `psamples` are unused with `momentum_fractions`; an angle grid that contains
  the unstable equilibrium, where the separatrix momentum vanishes, raises an `ArgumentError`
  rather than collapsing every trajectory onto the fixed point. `GeometricEquations` becomes a
  direct dependency (`"0.21"`), because the ensemble is assembled from `equation`,
  `EnsembleProblem` and the exported `Pendulum.hodeproblem`, rather than from the vector fields
  that `GeometricProblems` keeps private.

  `scripts/pendulum/train_sae.jl` trains on the grid that replaces a rectangle with data that
  straddles the separatrix under control: ten angles between the stable and the unstable
  equilibrium times nineteen fractions, 190 trajectories and 76 190 snapshots over `(0, 40)` with
  `Δt = 0.1`, against 100 trajectories and 10 100 snapshots over `(0, 10)` on the default grid. The
  longer `timespan` goes with the denser approach to the separatrix, because a librating orbit's
  period `4K(√((1+H)/2))` diverges there: `f = 3/4` already has period 11.9, longer than the
  `(0, 10)` the script used to integrate over, so the orbits nearest the separatrix did not close
  even once and their level sets were only ever partially traced. Over `(0, 40)` every trajectory
  closes at least twice, apart from the separatrix itself, which has no period. Six fractions
  between `|f| = 1` and `|f| = 2` fill the band `H ∈ (1, 1.60)` that the figure's original set
  jumped across, and four between `3/4` and `1` fill `H ∈ (0.913, 1)`.

### Changed

- **The MNIST scripts keep the parameters in a `NetworkParameters` rather than a bare `NamedTuple`.**
  `GeometricOptimizers` 0.7.0 takes a whole set of parameters only as a container: the alias it used to
  accept, `ArrayNamedTuple{T,S} = NamedTuple{S,<:Tuple{Vararg{AbstractArray{T}}}}`, was an alias for
  `Base.NamedTuple`, so every method on it was a method on a `Base` type — which is what made it
  unfixable in place and what caused four reachable `_copyto!` ambiguities upstream.

  The change is the wrap in `initial_parameters` and the type annotations that follow it. The wrap
  **shares the leaf arrays**, and the container forwards `keys`, `values`, `ps[i]`, `ps.field` and
  `length`, so `regroup`, `F`, `∇F!`, `flatten_parameters!` and the by-hand flat-vector indexing are
  unchanged — the flat layout is still read off `parameterlayout`, which is what makes the script's
  ranges and the optimizer's flattening agree by construction.

  All **five** scripts, not the two the ecosystem plan listed: `mnist.jl`, `mnist_cuda.jl`,
  `mnist_cuda_repetitions.jl`, `mnist_metal.jl` and `mnist_metal_short.jl`. `predict` and
  `network_loss` keep their `::NamedTuple` annotations — those take the *regrouped* parameters, which
  are a `NamedTuple` of vectors of matrices and not a parameter set at all.

## [0.1.0]

Initial release. Nothing here is new code — it is the MLDatasets-dependent material extracted from
`GeometricMachineLearning` and `GeometricOptimizers`, both of which are libraries for scientific
machine learning and neither of which should pull in an image-dataset package to document itself.

### Added

- **`src/mnist_utils.jl`**, moved from `GeometricMachineLearning/src/data_loader/mnist_utils.jl`:
  `onehotbatch`, `split_and_flatten` and the index arithmetic behind them (`patch_index`,
  `within_patch_index`, `index_conversion`). These were exported by `GeometricMachineLearning` and
  are now exported here instead.

- **The classification `DataLoader` constructor**, moved from
  `GeometricMachineLearning/src/data_loader/data_loader.jl`:
  `DataLoader(images::AbstractArray{T,3}, labels::AbstractVector)`. Its docstring already described
  itself as "tailored towards being used with the package MLDatasets.jl", which is what made it the
  natural cut. It has to live outside `GeometricMachineLearning` in any case: it is the only way to
  build a `DataLoader` whose input is `Float32` and whose output is `Int`, since the generic
  tensor–tensor constructor requires a single shared element type.

  Its docstring said MNIST images "are of size ``49\times49``", which they are not — they are
  ``28\times28`` and it is the *patches* that are 49-element vectors. Corrected in the move.

- **`src/datasets.jl`**, new: `mnist`, `fashion_mnist`, `mnist_data_loader` and
  `fashion_mnist_data_loader`. The two `*_data_loader` functions replace the
  `MLDatasets.MNIST(split=:train)[:]` / `DataLoader(...)` pair that was hand-copied into every one of
  the eleven scripts collected here.

- **`scripts/gml/`**, moved from `GeometricMachineLearning/scripts/`: `transformer_mnist.jl`,
  `transformer_fashion_mnist.jl` and `convert_jld2_to_h5.jl`, plus `plot_mnist_results.jl`, which is
  `transformer_bfgs.jl` turned into the plotting half of the pair (see below).

  `autoencoder.jl`, `classifier.jl`, `mnist_grassmann.jl` and `transformer_analysis.jl` were moved
  with them and then dropped. None of the four could run. The first three were last modified in
  July 2023 and called `init_optimizer_cache`, which no longer exists, and `optimization_step!` with
  its 2023 five-argument signature. `transformer_analysis.jl` was in the same state in four
  different ways — `Classification`, which is `ClassificationLayer` now; `DataLoader(images, labels)`
  without importing this package's constructor for it; `initialparameters(backend, T, model)`, which
  `AbstractNeuralNetworks` 0.6.4 has no method for; and `optimize_for_one_epoch!` without its `loss`
  and `λY` arguments — and what it did was a strict subset of `transformer_mnist.jl`: the same two
  configurations, no saving, and a `TODO` about the predictor that was never acted on.

  The first three were the only reason `Lux` and `Flux` were in `scripts/Project.toml` — for
  `Lux.setup`/`Lux.apply`, which the `NeuralNetwork` API replaced, and for `Flux.flatten` and
  `Flux.onehotbatch`, which are a `reshape` and this package's own `onehotbatch`. `Lux` and `Flux`
  are gone from the scripts environment with them, which also takes a `LuxCore` → `FluxExt`
  precompilation failure out of it.

  `transformer_bfgs.jl` was dead in the same four ways and is **`plot_mnist_results.jl`** instead of
  being dropped, because it is the only thing that draws the loss-curve figures. Three problems, one
  change:

  1. **It could not do BFGS.** Its first two curves were `BFGSOptimizer()`, GML's own manifold BFGS,
     deleted when the optimizer layer moved to `GeometricOptimizers` on the stated grounds that GO's
     `BFGS` replaces it. It does not — GO's is a quasi-Newton method whose cache holds an
     inverse-Hessian approximation sized by the *flattened* parameters, and GML's per-leaf update
     path cannot drive it. The figure legends said `BFGS` regardless, and the summary text named a
     `MomentumOptimizer` run that was a second copy of the `Adam` one.
  2. **It retrained in order to plot.** Four configurations of 500 epochs, to draw two loss curves —
     so a figure could not be redrawn without repeating the run that produced it.
  3. **It duplicated `transformer_mnist.jl`.** Roughly sixty lines of data loading, model
     construction and training loop, verbatim, differing only in the output stage.

  All three go away by reading the run instead of performing it. `transformer_mnist.jl` already
  writes the four loss arrays, the four wall-clock times and the four test accuracies to
  `mnist_parameters.jld2` — everything the figures and the summary need — so the new script opens
  that file and draws. Its four series are then whatever the run actually was, and the labels say so.

  This is the split `scripts/geometric_optimizers/distill_mnist_results.jl` already uses and the
  reason `docs/src/homogeneous_spaces_experiment.md` plots from checked-in CSVs: the documentation
  figures need neither a GPU nor a rerun, and neither should these. The results file is an argument
  defaulting to `mnist_parameters.jld2`, and the output names follow it, so
  `fashion_mnist_parameters.jld2` produces `fashion_mnist_*.png` and `transformer_fashion_mnist.jl`
  gains figures it never had.

  It reads the four datasets it needs rather than the whole file: `JLD2.load` would also pull in the
  `nn*weights`, and reconstructing a `NeuralNetworkParameters` of `StiefelManifold`s needs
  `GeometricMachineLearning` loaded purely to be discarded. The script depends on `CairoMakie` and
  `JLD2` and nothing else.

  **Both surviving training scripts needed repairs before either of them ran**, and each fault was
  carried over from GML rather than introduced here:

  - `GradientOptimizer(T(0.001))` and `MomentumOptimizer(T(0.001), T(0.5))` were `MethodError`s.
    A `GeometricOptimizers` method only produces a direction, so `GradientMethod` takes no
    arguments and `MomentumMethod` takes only the momentum coefficient; the step size is
    `Optimizer`'s `step_size` keyword, which is what `docs/src/mnist/mnist_tutorial.md` already did
    and what the scripts do now.
  - `save("mnist_parameters.jld2", …)` raised `UndefVarError`. `GeometricMachineLearning` and
    `AbstractNeuralNetworks` both export a `save`, so with `JLD2` also loaded the bare name is
    ambiguous and resolves to nothing. It is `JLD2.save` now.
  - `total_time = init_time - time()` is the elapsed time *negated*, so every duration these scripts
    reported and wrote into `mnist_parameters.jld2` was negative.
  - `text_string = "n_epochs: …\n"` was followed by the four result lines on the next line, with no
    `*` joining them to the first. The assignment therefore ended at the first line and the four
    lines were evaluated and discarded, so the report — printed and written to `measure_times…` —
    was one line long and contained none of the results it exists for.
  - `backend = CUDABackend()` was unconditional, with a `backend == CUDABackend() ? cu : identity`
    beside it that could only ever take the first branch. On a machine without CUDA the scripts died
    at the first `NeuralNetwork(model, backend, T)`, which is also what made them untestable. The
    choice is `CUDA.functional()` now and the host is a real fallback.

  Verified by running `transformer_mnist.jl` to completion at `L = 1` and `n_epochs = 1` on the
  host — four trainings, `mnist_parameters.jld2` written, a four-line report with positive times —
  and then `plot_mnist_results.jl` on the `.jld2` it produced, which draws both figures at the same
  byte sizes the training-and-plotting version did.

- **`scripts/geometric_optimizers/`**, moved from `GeometricOptimizers/scripts/`: `mnist.jl` and its
  `_cuda`, `_cuda_repetitions`, `_metal` and `_metal_short` variants, plus
  `distill_mnist_results.jl`, `metal_memory_probe.jl` and the two shell drivers. These spell the
  neural network out by hand rather than using `GeometricMachineLearning`, because
  `GeometricMachineLearning` depends on `GeometricOptimizers` and the dependency cannot be inverted.
  `Metal` is deliberately **not** a dependency of `scripts/Project.toml`: it does not resolve on
  Linux and would leave the whole environment unprecompilable there. Mac users add it by hand.

- **`docs/src/mnist/`**, the MNIST tutorial moved from
  `GeometricMachineLearning/docs/src/tutorials/mnist/`, including the pre-trained parameters
  (`mnist_nn1.h5`–`mnist_nn4.h5`, `mnist_metadata.h5`) so the page reports real numbers without
  retraining, and the TikZ sources for the patch-splitting figure.

- **`docs/src/homogeneous_spaces_experiment.md`**, the numerical experiment moved from
  `GeometricOptimizers/docs/src/manifold_optimizers.md`, with the three CSVs it plots. The theory
  half of that page stayed in `GeometricOptimizers`, where it documents that package's own
  algorithm.

- **`docs/src/running_the_experiments.md`**, the operational half of the former
  `GeometricOptimizers/MNIST_PORT.md`: what the four configurations are for, why one of them is
  expected to fail, and the Metal unified-memory handling the GPU scripts need.

### Notes

- **The docstring reference lives on one page.** `index.md` is an `@autodocs` over the whole module
  and `mnist/mnist_tutorial.md` closed with an `@docs` block naming five of the same docstrings.
  Documenter does not object, but it resolves a duplicated docstring to whichever copy it saw last,
  so every `@ref` in the package — including the ones on `index.md` itself, three paragraphs above
  its own reference section — pointed into the tutorial. The tutorial's block is gone and a sentence
  pointing at the reference is in its place.

- **`julia = "1.11"`, and the CI matrix starts there.** The `[sources]` blocks that pin
  `GeometricMachineLearning` to `main` are a Julia 1.11 feature; Pkg 1.10 ignores them, falls back to
  the registered 0.4.8 and cannot then satisfy the `"0.5"` bound, so a 1.10 job could only ever fail
  to resolve. Both the bound and the matrix entry go back to 1.10 when 0.5 is registered and the
  `[sources]` blocks go away — which is also what `README.md` says.

- **The `[sources]` URLs are the canonical ones.** All three pointed at
  `https://github.com/JuliaGNI/GeometricML`, which is a redirect to
  `https://github.com/JuliaGNI/GeometricMachineLearning.jl`.

- **`.github/workflows/Documenter.yml` no longer calls `Pkg.develop(path = pwd())`.**
  `docs/Project.toml` already has `GMLDatasets = {path = ".."}` under `[sources]`, and `Pkg.develop`
  overwrites that with an absolute path — which it duly did, on the machine this was verified on.

- **`docs/src/mnist/mnist_visualization.jl` no longer opens 34 image viewers.** Each of its four
  `Figure()`s was followed by `display(fig)` — the GLMakie idiom of opening a window and then drawing
  into it live. Under CairoMakie there is no interactive backend, so `display` falls through to the
  file-based show stack, which writes a temporary image and hands it to the system viewer: 34 Preview
  windows on macOS on every documentation build, and every one of them blank, because the call sits
  between `Figure()` and the `plot_image!` that fills it.

  The render pass itself is **not** removable, which is the trap here. Each figure is saved as the
  child scene of its `Axis` rather than as the figure — that is what makes the images tightly cropped
  — and the child scene of a figure that has never been rendered has no content, so dropping
  `display` outright makes `CairoMakie.save` write a fully transparent image. `Makie.colorbuffer` is
  the same render pass without the display stack; the 34 images it produces are byte-identical to the
  ones `display` produced. `Makie.update_state_before_display!` is not a substitute despite the name
  — it leaves the scenes empty.

  `docs/src/mnist/Makefile` drops the `-i` from the `julia` invocation, and the script drops the
  `exit()` that only existed to escape the REPL that `-i` opened.
