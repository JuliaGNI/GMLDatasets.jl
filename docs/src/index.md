```@meta
CurrentModule = GMLDatasets
```

# GMLDatasets

Dataset-backed demonstrations for
[GeometricMachineLearning](https://juliagni.github.io/GeometricMachineLearning.jl/latest/) and
[GeometricOptimizers](https://juliagni.github.io/GeometricOptimizers.jl/latest/).

Both of those packages are libraries for *scientific* machine learning and neither should depend on
an image-dataset package to document itself. This one does: it holds the
[`MLDatasets`](https://github.com/JuliaML/MLDatasets.jl) glue, the MNIST and Fashion-MNIST
demonstrations, and the numerical experiment of [brantner2023generalizing](@cite) that shows manifold
optimization making a vision transformer trainable at all.

## The data loaders

[`mnist_data_loader`](@ref) and [`fashion_mnist_data_loader`](@ref) hand back a
`GeometricMachineLearning.DataLoader` in the *time series* format a transformer consumes — each
``28\times28`` image cut into 16 patches of ``7\times7``, each patch flattened into a column, the
labels one-hot encoded:

```@example index
using GMLDatasets

dl = mnist_data_loader(:train; patch_length = 7, suppress_info = true)
(dl.input_dim, dl.input_time_steps, dl.n_params, dl.output_dim)
```

The pieces are available on their own too: [`mnist`](@ref) and [`fashion_mnist`](@ref) return the raw
images and labels, [`split_and_flatten`](@ref) cuts an image into flattened patches and
[`onehotbatch`](@ref) encodes the labels.

## Where to go next

- [MNIST Tutorial](@ref) trains a vision transformer with weights on the Stiefel manifold, using
  `GeometricMachineLearning`, and compares four optimizers on it.
- [The Numerical Experiment on Homogeneous Spaces](@ref) carries the figures for a 500-epoch run of
  that experiment against `GeometricOptimizers` — training loss, test accuracy and drift off the
  manifold.
- [Running the Experiments](@ref) is the operational side: what each script in `scripts/` does, what
  the four configurations are for, and the device-memory handling the GPU runs need.

## Library

```@autodocs
Modules = [GMLDatasets]
```
