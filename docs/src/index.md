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
demonstrations, the numerical experiment of [brantner2023generalizing](@cite) that shows manifold
optimization making a vision transformer trainable at all, and a symplectically integrated pendulum
data set for the autoencoders.

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

## The pendulum

[`pendulum`](@ref) integrates a grid of pendulum trajectories and [`angular_to_euclidean`](@ref)
lifts them into ``\mathbb{R}^4``, where two dimensions of data sit on a curved submanifold of four —
which is what a symplectic autoencoder is for. There is no data loader of its own here; the array is
already in the shape `DataLoader` reads:

```@example index
solution = pendulum()
dl = DataLoader(angular_to_euclidean(solution); autoencoder = true, suppress_info = true)
(dl.input_dim, dl.input_time_steps, dl.n_params)
```

## Where to go next

- [The Pendulum Data Set](@ref) is the pendulum in full: the sign convention, the lift into
  ``\mathbb{R}^4``, and the energy check that says the integrator did its job.
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
