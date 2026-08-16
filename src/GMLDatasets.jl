"""
    GMLDatasets

Dataset-backed demonstrations for `GeometricMachineLearning` and `GeometricOptimizers`.

Both of those packages are libraries for scientific machine learning and neither should depend on an
image-dataset package to document itself. Everything that does lives here: the
[`MLDatasets`](https://github.com/JuliaML/MLDatasets.jl) glue, the MNIST and Fashion-MNIST
demonstrations and the numerical experiment from [brantner2023generalizing](@cite).

The entry points are [`mnist_data_loader`](@ref) and [`fashion_mnist_data_loader`](@ref), which hand
back a `GeometricMachineLearning.DataLoader` in the *time series* format a transformer wants:

```julia
using GMLDatasets

dl      = mnist_data_loader(:train; patch_length = 7)
dl_test = mnist_data_loader(:test;  patch_length = 7)
```
"""
module GMLDatasets

using AbstractNeuralNetworks: networkbackend
using KernelAbstractions

# `DataLoader` is imported rather than `using`ed because this package adds a constructor to it — the
# one for labelled image data, which cannot live in `GeometricMachineLearning` (see `data_loader.jl`).
import GeometricMachineLearning: DataLoader
import MLDatasets

export DataLoader
export split_and_flatten, onehotbatch
export mnist, fashion_mnist
export mnist_data_loader, fashion_mnist_data_loader

include("mnist_utils.jl")
include("data_loader.jl")
include("datasets.jl")

end
