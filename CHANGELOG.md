# Changelog

All notable changes to GMLDatasets.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

## [Unreleased]

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

- **The revision harness pins `GeometricOptimizers` at `ae50ece`**, the head of
  [PR #79](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/79), which is the observer of
  PR #78 plus the two fixes the GPU runs needed.

  First, `similar` of a horizontal lift allocated on the host regardless of where its argument lived,
  and because the four-argument optimizer-cache constructors bind their three gradient blocks to a
  single type, that was a `MethodError` at `Optimizer(Adam(), network)` for any device-resident
  network. It stopped the `pendulum-seed-1234` stage of run `20260903T125418Z_smoke` on the RTX 4090;
  the four image stages of the same run passed, because `mnist_cuda_repetitions.jl` keeps its
  parameters in a **host** container and copies to the device inside `∇F!`, so nothing before the
  pendulum stage ever built a device-resident cache.

  Second, once that was out of the way the same stage failed one layer deeper, at the first
  `optimization_step!` of runs `20260903T185459Z_smoke` and `20260903T191704Z_smoke`: the pullback
  returns a gradient leaf on the host for a device-resident `StiefelManifold` weight, so `rgrad`'s
  `∇L' * Y.A` was a CPU `gemm!` handed a device pointer
  (`ArgumentError: Illegal conversion of a CUDA.DeviceMemory to a Ptr{Float32}`).

  That second fix is a **temporary shim** in `GeometricOptimizers` for a defect in the packages that
  produce the gradient, filed as
  [`GeometricMachineLearning` #258](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/258)
  and [`AbstractNeuralNetworks` #39](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/issues/39).
  While it is in place it moves the gradient across per manifold leaf per step, inside the region the
  phase timer attributes to the step, so **the pendulum stage's decomposed timings are an upper bound
  rather than a measurement** and its `optimizer_state_direction_seconds` in particular carry that
  transfer. Any pendulum timing published from a run on this pin has to say so, or wait for the
  upstream fix. The image stages never take this path and are unaffected.

  `scripts/revision/check_environment.jl` now builds an optimizer cache and state for a 4 × 2
  device-resident `StiefelManifold` parameter set and calls `rgrad` on it with a host gradient, next
  to the `PhaseTimer` check and for the same reason: a version number cannot express any of these
  properties. Both failures are now preflight errors at second zero rather than ones that land after
  the image stages have already spent their hours. The `rgrad` check asserts the presence of the
  temporary shim, so it retires with the shim rather than outliving it.

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
