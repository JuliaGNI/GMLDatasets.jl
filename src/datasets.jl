@doc raw"""
    mnist(split = :train)

Return the MNIST images and labels for `split`, which is either `:train` or `:test`.

This is `MLDatasets.MNIST(split = split)[:]` and nothing more — the images come back as a
``28\times28\times{}n`` tensor of `Float32` and the labels as a vector of `Int` in ``0, \ldots, 9``.
Use [`mnist_data_loader`](@ref) to get them in the form a transformer consumes.

The data set is downloaded on first use. Set `ENV["DATADEPS_ALWAYS_ACCEPT"] = true` to skip the
download prompt in a non-interactive session.
"""
mnist(split::Symbol = :train) = MLDatasets.MNIST(split = split)[:]

@doc raw"""
    fashion_mnist(split = :train)

Return the Fashion-MNIST images and labels for `split`, which is either `:train` or `:test`.

Fashion-MNIST is drop-in compatible with MNIST — ``28\times28`` grayscale images in ten classes — so
everything that works on one works on the other. See [`mnist`](@ref).
"""
fashion_mnist(split::Symbol = :train) = MLDatasets.FashionMNIST(split = split)[:]

@doc raw"""
    mnist_data_loader(split = :train; patch_length = 7, transform = identity, suppress_info = false)

Load MNIST and wrap it in a `GeometricMachineLearning.DataLoader`.

This composes [`mnist`](@ref) with [`DataLoader`](@ref), i.e. it cuts each image into patches of
`patch_length` ``\times`` `patch_length`, flattens each patch into a column and one-hot encodes the
labels.

`transform` is applied to the images and to the labels before the `DataLoader` is built, which is
where the data are moved onto an accelerator:

```julia
using CUDA

dl = mnist_data_loader(:train; transform = cu)
```

The transform is applied before rather than after the `DataLoader` is constructed so that
[`split_and_flatten`](@ref) and [`onehotbatch`](@ref) run their kernels on the device.
"""
function mnist_data_loader(split::Symbol = :train; transform = identity, kwargs...)
    images, labels = mnist(split)
    DataLoader(transform(images), transform(labels); kwargs...)
end

@doc raw"""
    fashion_mnist_data_loader(split = :train; patch_length = 7, transform = identity, suppress_info = false)

Load Fashion-MNIST and wrap it in a `GeometricMachineLearning.DataLoader`.

Like [`mnist_data_loader`](@ref), but for [`fashion_mnist`](@ref).
"""
function fashion_mnist_data_loader(split::Symbol = :train; transform = identity, kwargs...)
    images, labels = fashion_mnist(split)
    DataLoader(transform(images), transform(labels); kwargs...)
end
