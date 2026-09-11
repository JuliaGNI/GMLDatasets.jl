using Test
using GMLDatasets
using GMLDatasets: patch_index
using GMLDatasets: within_patch_index
using GMLDatasets: index_conversion
using AbstractNeuralNetworks: networkbackend
using GeometricMachineLearning
# A `Chain`'s parameters are a `NetworkParameters`; the `Tuple` arm is what `applychain` takes once the
# layers have been split out. `NamedTuple` stays in the union too: `NeuralNetworkParameters`'
# `ZygoteRules.pullback` seeds the reverse pass with the wrapped `NamedTuple`, so a function
# differentiated with respect to a parameter set is called with one.
using NeuralNetworkParameters: NetworkParameters
import Zygote, Random

Random.seed!(1234)

"""
This function tests is used to test if all the patch nubmers are assigned correctly with `index_conversion`, i.e. tests `patch_index` by inverting it.
"""
function reverse_index(i::Integer, j::Integer, patch_length = 7)
    opt_i = i%patch_length==0 ? 1 : 0
    within_patch_index = i%patch_length + opt_i*patch_length, (i÷patch_length - opt_i + 1)

    sqrt_number_patches = 28÷patch_length
    opt_j = j%sqrt_number_patches==0 ? 1 : 0
    patch_index = j%sqrt_number_patches + opt_j*sqrt_number_patches,
    (j÷sqrt_number_patches - opt_j + 1)
    (patch_index[1]-1)*patch_length + within_patch_index[1],
    (patch_index[2]-1)*patch_length + within_patch_index[2]
end

"""
This function uses `reverse_index` to test `index_conversion`, i.e. checks if the functions are invertible.
"""
function test_index_conversion(patch_lengths = (2, 4, 7, 14))
    for patch_length in patch_lengths
        number_of_patches = (28÷patch_length)^2
        for i in 1:28
            for j in 1:28
                @test reverse_index(index_conversion(i, j, patch_length, number_of_patches)..., patch_length) ==
                      (i, j)
            end
        end
    end
end

"""
This function tests if `onehotbatch` does what it should; i.e. convert a vector of integers to a one-hot-tensor.
"""
function test_onehotbatch(V::AbstractVector{T}) where {T <: Integer}
    V_encoded = onehotbatch(V)
    for (i, v) in zip(length(V), V)
        @test sum(V_encoded[:, 1, i]) == 1
        @test V_encoded[v, 1, i] == 1
    end
end

test_onehotbatch([1, 2, 5, 0])
@test onehotbatch([0]) == reshape([1, zeros(Int, 9)...], 10, 1, 1)

@testset "MNIST repetition-trainer preprocessing contract" begin
    patch_length = 7
    n = 28 ÷ patch_length
    images = reshape(Float32.(1:(28 * 28 * 3)), 28, 28, 3)

    # Reference implementation previously carried by `mnist_cuda_repetitions.jl`.
    trainer_patches = reshape(
        permutedims(reshape(images, patch_length, n, patch_length, n, size(images, 3)),
            (1, 3, 2, 4, 5)),
        patch_length^2, n^2, size(images, 3))
    package_patches = split_and_flatten(
        images; patch_length=patch_length, number_of_patches=n^2)

    @test size(package_patches) == (49, 16, 3)
    @test eltype(package_patches) === Float32
    @test package_patches == trainer_patches
    @test typeof(networkbackend(package_patches)) === typeof(networkbackend(images))

    labels = UInt8[0, 9, 3]
    encoded = onehotbatch(Float32, labels)
    trainer_targets = zeros(Float32, 10, length(labels))
    for (sample, label) in pairs(labels)
        trainer_targets[label + 1, sample] = 1.0f0
    end

    @test size(encoded) == (10, 1, 3)
    @test eltype(encoded) === Float32
    @test reshape(encoded, 10, length(labels)) == trainer_targets
    @test typeof(networkbackend(encoded)) === typeof(networkbackend(labels))

    # The established API keeps its rank, values, element type, and host backend.
    compatible = onehotbatch(labels)
    @test size(compatible) == (10, 1, 3)
    @test eltype(compatible) === UInt8
    @test compatible == UInt8.(encoded)
    @test typeof(networkbackend(compatible)) === typeof(networkbackend(labels))
end

@doc raw"""
Generates an MNIST-like dummy data set.

The tests must not download anything, so everything that would otherwise come from `MLDatasets` is
faked here: a tensor of random images and a vector of random labels in ``0, \ldots, 9``.
"""
function generate_dummy_mnist(dim₁ = 28, dim₂ = 28, number_images = 100, T = Float32)
    train_x = rand(T, dim₁, dim₂, number_images)
    train_y = Int.(ceil.(10 * rand(T, number_images))) .- 1
    train_x, train_y
end

function test_data_loader(;
        dim₁ = 28, dim₂ = 28, number_images = 100, patch_length = 7, T = Float32)
    dl = DataLoader(generate_dummy_mnist(dim₁, dim₂, number_images, T)...;
        patch_length = patch_length, suppress_info = true)

    @test eltype(dl) == T
    @test dl.input_dim == patch_length ^ 2
    @test dl.input_time_steps == (dim₁ ÷ patch_length) * (dim₂ ÷ patch_length)
    @test dl.n_params == number_images
    @test dl.output_dim == 10
    @test size(dl.input) == (patch_length ^ 2, dl.input_time_steps, number_images)
    @test size(dl.output) == (10, 1, number_images)
end

function test_optimizer_for_classification_layer(;
        dim₁ = 28, dim₂ = 28, number_images = 100, patch_length = 7, T = Float32)
    dl = DataLoader(generate_dummy_mnist(dim₁, dim₂, number_images, T)...;
        patch_length = patch_length, suppress_info = true)

    activation_function(x) = tanh.(x)
    model = Chain(ClassificationLayer(patch_length * patch_length, 10, activation_function))

    ps = NeuralNetwork(model, CPU(), T).params
    loss = FeedForwardLoss()
    loss_dl(
        model::GeometricMachineLearning.Chain, ps::Union{
            Tuple, NamedTuple, NetworkParameters}, dl::DataLoader) = loss(model, ps, dl.input, dl.output)
    loss₁ = loss_dl(model, ps, dl)

    opt = Optimizer(GradientOptimizer(), ps)
    dx = Zygote.gradient(ps -> loss_dl(model, ps, dl), ps)[1]
    λY = GlobalSection(ps)
    optimization_step!(opt, λY, ps, dx)
    loss₂ = loss_dl(model, ps, dl)

    @test loss₂ < loss₁
end

test_index_conversion()
test_data_loader()
test_optimizer_for_classification_layer()
