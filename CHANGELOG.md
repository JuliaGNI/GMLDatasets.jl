# Changelog

All notable changes to GMLDatasets.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

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
