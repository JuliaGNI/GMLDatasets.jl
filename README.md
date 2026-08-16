# GMLDatasets

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/GMLDatasets.jl/latest/)

Dataset-backed demonstrations for [GeometricMachineLearning](https://github.com/JuliaGNI/GeometricML)
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
  `ClassificationTransformer` and the optimizers from the library.
- `scripts/geometric_optimizers/` — written against `GeometricOptimizers` alone, with the neural
  network spelled out by hand. `GeometricMachineLearning` depends on `GeometricOptimizers`, so these
  cannot use it. Host, CUDA and Metal variants.

`docs/` builds the MNIST tutorial and the figures for the numerical experiment. The figures are drawn
from CSVs checked in under `docs/src/data/`, so building the documentation needs neither a GPU nor a
rerun.

## Installation

Not registered, and it needs `GeometricMachineLearning` at `0.5`, which is not tagged yet — the
newest registered version still defines `split_and_flatten` and `onehotbatch` itself and would
collide with this package. So until `0.5` lands:

```julia
using Pkg
Pkg.develop(url = "https://github.com/JuliaGNI/GeometricML")
Pkg.develop(url = "https://github.com/JuliaGNI/GMLDatasets.git")
```

or, working from local checkouts side by side:

```julia
using Pkg
Pkg.activate("GMLDatasets")
Pkg.develop(path = "../GeometricMachineLearning")
```

## License

MIT, see [LICENSE.md](LICENSE.md).
