# GMLDatasets

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/GMLDatasets.jl/latest/)

Dataset-backed demonstrations for [GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl)
and [GeometricOptimizers](https://github.com/JuliaGNI/GeometricOptimizers.jl).

Both of those packages are libraries for *scientific* machine learning and neither should depend on
an image-dataset package to document itself. This package holds everything that does: the
[MLDatasets](https://github.com/JuliaML/MLDatasets.jl) glue, the MNIST and Fashion-MNIST
demonstrations, the numerical experiment from
[brantner2023generalizing](https://arxiv.org/abs/2305.16901) that shows manifold optimization making
a vision transformer trainable at all, and a symplectically integrated pendulum data set for the
autoencoders.

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

The pendulum half is the same kind of thin layer, over
[GeometricProblems](https://github.com/JuliaGNI/GeometricProblems.jl) and
[GeometricIntegrators](https://github.com/JuliaGNI/GeometricIntegrators.jl) instead:

```julia
using GeometricMachineLearning

dl = DataLoader(angular_to_euclidean(pendulum()); autoencoder = true)
```

`pendulum` integrates a grid of initial conditions with a symplectic method and returns the
`EnsembleSolution`; `angular_to_euclidean` lifts the two canonical coordinates into the four
Euclidean ones, where the data lie on the tangent bundle of a circle — two dimensions inside four,
which is a submanifold worth asking a `SymplecticAutoencoder` to find. `euclidean_to_angular`
inverts the lift and `pendulum_energy` evaluates the Hamiltonian in either set of coordinates. There
is no data loader of its own: the array is already in the shape `DataLoader` reads.

`scripts/` holds the training runs, split by which package they exercise:

- `scripts/gml/` — written against `GeometricMachineLearning`, so they get `DataLoader`,
  `ClassificationTransformer` and the optimizers from the library. The training runs write a `.jld2`
  and `plot_mnist_results.jl` draws the loss-curve figures from it, so a figure can be redrawn
  without repeating four configurations of 500 epochs.
- `scripts/geometric_optimizers/` — written against `GeometricOptimizers` alone, with the neural
  network spelled out by hand. `GeometricMachineLearning` depends on `GeometricOptimizers`, so these
  cannot use it. Host, CUDA and Metal variants.
- `scripts/pendulum/` — `train_sae.jl` trains a symplectic autoencoder on the pendulum data set and
  `plot_dataset.jl` draws the data set and the integrator's energy error. Training covers both sides
  of the separatrix, uses CUDA when available, and saves its result with HDF5; neither script needs a
  download.

`docs/` builds the MNIST tutorial, the pendulum page and the figures for the numerical experiment.
The figures are drawn from CSVs checked in under `docs/src/data/` and the pendulum page integrates
its own data at build time in a fraction of a second, so building the documentation needs neither a
GPU nor a rerun.

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

## License

MIT, see [LICENSE.md](LICENSE.md).
