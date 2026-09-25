# Changelog

All notable changes to GMLDatasets.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

## [Unreleased]

### Added

- **`onehotbatch(S, target)`**, which encodes the labels in a chosen element type. The one-argument
  form is unchanged and is now a forwarder to it, so the encoding keeps the label type by default.

  A trainer wants `Float32` targets and gets `UInt8` ones from an `UInt8` label vector, which then
  promotes the whole loss on first contact. Every caller had been converting afterwards, allocating a
  second array of the same size to do it; the element type belongs where the array is allocated.
  `KernelAbstractions.zeros(backend, S, …)` is the one line that changes, so the output still lives
  on the same backend as `target`.

- **A testset pinning the preprocessing contract of `split_and_flatten` and `onehotbatch`** —
  ordering, shape, element type and host backend — against the reference implementations that the
  MNIST repetition trainer in `scripts/` used to carry inline. That trainer imports both from this
  package now, and this testset is what keeps the package from drifting away from the script that
  depends on it: without it the drift is invisible here and shows up as a wrong number eight hours
  into a run.

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
  claim yet. And the Riemannian gradient of a device-resident point currently reaches the device
  through a **temporary shim** in `GeometricOptimizers`, for a defect in the packages that produce
  the gradient ([`GeometricMachineLearning` #258](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/258),
  [`AbstractNeuralNetworks` #39](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/issues/39)).
  That shim moves the gradient across per manifold leaf per step, inside the region the phase timer
  attributes to the step, so **a pendulum timing published from a run carrying it is an upper bound
  rather than a measurement.** The image stages never take that path and are unaffected.

- **`scripts/revision/capture_source.jl`**, the harness's one provenance capture: a repository's
  commit, a reproducible patch that includes untracked files, the patch's SHA-256, and a cross-check
  that a tree `git status` calls dirty produced a non-empty patch. The runner calls it for both
  repositories a bundle records and passes its output to the retraction benchmark, so the
  mechanism whose whole purpose is to have exactly one copy has exactly one.

- **`.github/workflows/Scripts.yml`**, which runs the two harness suites that need
  `GeometricOptimizers` against `scripts/Project.toml`. The other two need nothing but `SHA` and
  Base and run from `test/runtests.jl`, inside the package suite. `CI.yml` is byte-identical in
  every repository by policy and cannot gain a step; its own header prescribes this escape hatch.
  Not a required check: the scripts environment resolves two moving branches on purpose, so the job
  can go red for a change made upstream.

### Changed

- **The scripts environment resolves `GeometricMachineLearning` and `GeometricOptimizers` from
  `main`.** The harness needs the optimizer step observer and `PhaseTimer` of
  [GeometricOptimizers #78](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/78), the backend
  fixes of [#79](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/79),
  [#84](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/84) and
  [#85](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/85), and the 0.8 seams above.
  `scripts/Manifest.toml` stays untracked: a manifest pinning two moving branch commits is stale the
  day after it is committed, and every run bundle already carries the manifest that run resolved.
  Refresh with `Pkg.update`, never `Pkg.resolve` — to `resolve` a `rev = "main"` source is a fixed
  pin, so it calls a stale commit satisfiable and leaves the environment silently behind.

  Two of those fixes are what `scripts/revision/check_environment.jl` probes, because no version
  number can express them: that an optimizer cache and state can be built for a parameter set living
  on the GPU, and that the Riemannian gradient of a device-resident point lands on the device. Both
  once cost a run its pendulum stage after the image stages had already spent their hours — the
  image trainer keeps its parameters in a **host** container and copies to the device inside `∇F!`,
  so nothing before the pendulum stage ever builds a device-resident cache. The preflight asserts
  the property the harness needs, not the shim that currently provides it, so it stays correct when
  the shim retires.

- **`[compat]` widens to the current releases of four dependencies**, as one change rather than four:
  `AbstractNeuralNetworks = "0.7, 0.8"`, `GeometricMachineLearning = "0.6, 0.7"`,
  `GeometricProblems = "0.8, 0.9"` and `NeuralNetworkParameters = "0.2.2, 0.3"`. These arrived as four
  separate CompatHelper pull requests, and separately none of them can go green: they are one
  dependency family, released together, so resolving any one of the new majors drags the others in
  with it. Combining them is what lets a single CI run answer whether the package works against the
  current ecosystem.

  The `NeuralNetworkParameters` floor stays at **0.2.2**, not the bare `0.2` two of those pull
  requests proposed. `test/mnist_utils.jl` annotates against `ParameterSet`, which 0.2.2 added, so a
  `0.2` floor would advertise support for versions that cannot run the test suite.

- **The parameter-set annotation in `test/mnist_utils.jl` admits a `NamedTuple`.** Under
  `GeometricMachineLearning` 0.7, `NeuralNetworkParameters`' `ZygoteRules.pullback` seeds the reverse
  pass with the wrapped `NamedTuple` rather than the container, so `loss_dl` is called with one while
  differentiating and the old `Union{Tuple, NetworkParameters}` was a `MethodError`.

- **The documentation workflow builds the TikZ figure before Documenter runs.** The MNIST tutorial's
  patch-splitting figure is TikZ, and Documenter's cross-reference check needs the `.png` files to
  exist. The job installs `texlive-xetex`, `texlive-pictures` and `poppler-utils`, then runs
  `make all -C docs/src/mnist` after `julia-buildpkg` — after, because the Makefile's `images` target
  runs in the root environment that `buildpkg` has just instantiated. This is now the fourth
  repository that legitimately keeps its own `Documenter.yml`, and the file's header says so.

### Fixed

- **The pendulum stage never resumed.** Its restart test read `configuration_key`, `repetition` and
  `seed` out of `pendulum-runs.csv` with `awk -F,`, but the display name between them is a quoted
  field containing a comma — `"Geometric Adam (Stiefel, Cayley retraction)"` — so every later column
  was shifted by one and the test compared `transport` against a repetition number. It could not
  match, so an interrupted pendulum matrix re-ran every seed it had already completed, silently and
  at full cost. The runner now asks `validate_run_artifacts.jl --list-complete` for the completed
  jobs, which parses the CSV properly.

- **The pendulum trainer quoted a learning rate it never applied.** `Optimizer(method, network;
  retraction = cayley)` takes its step size from the *method*'s default — `1e-3` for `Adam`, `1e-2`
  for the rest — while every run record and every HDF5 attribute reported the configured
  `SAE_STEP_SIZE`. The four configurations were therefore not compared at the rates the comparison
  says they were. The rate is now passed explicitly.

### Removed

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

  `scripts/pendulum/train_sae.jl` trains on `pendulum()`'s default bounded grid, including librating
  and rotating trajectories on both sides of the separatrix. Although the standard reduction cannot
  use a globally real-valued angular coordinate, a bounded cylinder can be represented differently
  in two dimensions — for example as an annulus. The SAE experiment therefore tests whether a
  learned reduction can represent the separatrix without reproducing `(θ, pθ)`. It uses the deeper
  12,000-epoch architecture that demonstrated this behavior experimentally, prefers CUDA when
  available, falls back to the host, and saves the trained parameters and loss curve with HDF5.

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
