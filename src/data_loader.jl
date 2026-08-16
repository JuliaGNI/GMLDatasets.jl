@doc raw"""
    DataLoader(images::AbstractArray{T, 3}, labels::AbstractVector)

Make an instance of `GeometricMachineLearning.DataLoader` for an image classification problem.

`images` is a tensor whose first two axes are the image axes and whose third axis indexes the images;
`labels` is a vector of integer labels, one per image. Each image is cut into square patches by
[`split_and_flatten`](@ref) and each patch is flattened into a column, which casts the data into the
*time series* format a transformer consumes. The labels are encoded by [`onehotbatch`](@ref).

# Arguments

There are two keyword arguments:
- `patch_length = 7`. This is the length of the patch in the ``x`` and the ``y`` direction;
- `suppress_info = false`.

MNIST images are of size ``28\times28``. For `patch_length = 7` such an image is therefore split into
16 patches of ``7\times7``, i.e. it becomes a matrix in ``\mathbb{R}^{49\times16}``
[brantner2023generalizing](@cite).

# Implementation

This method lives in `GMLDatasets` rather than in `GeometricMachineLearning` because it is the piece
that is specific to labelled image data. It is also the only way to build a `DataLoader` whose input
is `Float32` and whose output is `Int`: the generic tensor–tensor constructor in
`GeometricMachineLearning` requires a single shared element type, whereas a one-hot target is
integer-valued whatever the images are.
"""
function DataLoader(images::AbstractArray{T,3}, labels::AbstractVector{T1};
    patch_length=7,
    suppress_info=false) where {T,T1}
    if !suppress_info
        @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) parameter dimension."
    end
    im_dim₁, im_dim₂, n_params = size(images)
    @assert length(labels) == n_params
    number_of_patches = (im_dim₁ ÷ patch_length) * (im_dim₂ ÷ patch_length)
    target = onehotbatch(labels)
    data_preprocessed = split_and_flatten(images, patch_length=patch_length, number_of_patches=number_of_patches)
    DataLoader{T,typeof(data_preprocessed),typeof(target),:TimeSeries}(
        data_preprocessed, target, patch_length^2, number_of_patches, n_params, 10, 1
    )
end
