# GMLDatasets

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/GMLDatasets.jl/latest/)

Dataset-backed demonstrations for [GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl)
and [GeometricOptimizers](https://github.com/JuliaGNI/GeometricOptimizers.jl).

Both of those packages are libraries for *scientific* machine learning and neither should depend on
an image-dataset package to document itself. This package holds everything that does: the
[MLDatasets](https://github.com/JuliaML/MLDatasets.jl) glue, the MNIST and Fashion-MNIST
demonstrations, and the numerical experiment from
[brantner2023generalizing](https://arxiv.org/abs/2305.16901) that shows manifold optimization making
a vision transformer trainable at all.

## What is in here

`src/` is a thin layer between `MLDatasets` and `GeometricMachineLearning.DataLoader`:

```julia
using GMLDatasets

dl      = mnist_data_loader(:train; patch_length = 7)
dl_test = mnist_data_loader(:test;  patch_length = 7)
```

`mnist_data_loader` cuts each ``28\times28`` image into 16 patches of ``7\times7``, flattens each
patch into a column, and one-hot encodes the labels — the *time series* format a transformer wants.
The pieces are also available on their own: `split_and_flatten`, `onehotbatch`, `mnist`,
`fashion_mnist`, and a `DataLoader(images, labels)` constructor.

`scripts/` holds the training runs, split by which package they exercise:

- `scripts/gml/` — written against `GeometricMachineLearning`, so they get `DataLoader`,
  `ClassificationTransformer` and the optimizers from the library. The training runs write a `.jld2`
  and `plot_mnist_results.jl` draws the loss-curve figures from it, so a figure can be redrawn
  without repeating four configurations of 500 epochs.
- `scripts/geometric_optimizers/` — written against `GeometricOptimizers` alone, with the neural
  network spelled out by hand. `GeometricMachineLearning` depends on `GeometricOptimizers`, so these
  cannot use it. Host, CUDA and Metal variants.

`docs/` builds the MNIST tutorial and the figures for the numerical experiment. The figures are drawn
from CSVs checked in under `docs/src/data/`, so building the documentation needs neither a GPU nor a
rerun.

## Installation

Not registered, and it needs `GeometricMachineLearning` at `0.5`, which is not tagged yet — the
newest registered version still defines `split_and_flatten` and `onehotbatch` itself and would
collide with this package. That is also why Julia **1.11** is the floor: `Project.toml` pins
`GeometricMachineLearning` to `main` through a `[sources]` block, which is a 1.11 feature, and Pkg
1.10 ignores it and resolves to the registered 0.4.8 instead. Both go back to 1.10 when `0.5` lands.
So until it does:

```julia
using Pkg
Pkg.develop(url = "https://github.com/JuliaGNI/GeometricMachineLearning.jl")
Pkg.develop(url = "https://github.com/JuliaGNI/GMLDatasets.jl.git")
```

or, working from local checkouts side by side:

```julia
using Pkg
Pkg.activate("GMLDatasets")
Pkg.develop(path = "../GeometricMachineLearning")
```

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
