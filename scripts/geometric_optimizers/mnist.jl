# A port of `scripts/gml/transformer_mnist.jl` that only relies on `GeometricOptimizers`
# (`GeometricMachineLearning` depends on that package, so it cannot be used here — which is also why
# this script loads `MLDatasets` directly instead of going through `GMLDatasets`, whose data loaders
# hand back a `GeometricMachineLearning.DataLoader`). Everything that `GeometricMachineLearning`
# supplies through `DataLoader`, `ClassificationTransformer` and `NeuralNetwork` is written out
# explicitly below.
#
# The neural network is the same as `ClassificationTransformer(dl; n_heads = n_heads, L = L,
# add_connection = false, Stiefel = Stiefel)`, i.e. `L` blocks of
#
#   1. `MultiHeadAttention` (with the projections optionally on the `StiefelManifold`) and
#   2. `ResNetLayer` (with `tanh` activation and a bias),
#
# followed by a classification layer that picks the last column of the output and applies
# `softmax`. Unlike the original script this runs on the CPU only.
#
# The gradient is computed with `Zygote` and handed to the `Optimizer` via the `∇F!` keyword.
# The default `ForwardDiff` gradient is not an option here: its cost scales with the number
# of parameters (of which there are more than 150000 for the default configuration).

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!
# `GeometricOptimizers` 0.5.0 dropped `ParameterHandling` for `NeuralNetworkParameters`, which is the
# package that owns a parameter set and does its flattening. This script used to reach the old one
# through `GeometricOptimizers`, so it stopped loading at that release; these are the replacements.
# `freeparameters` is the leaf protocol -- it is `Y.A` for a `StiefelManifold`, by a method
# `GeometricOptimizers` registers, so this script no longer needs its own copy of that knowledge.
using NeuralNetworkParameters: NetworkParameters, flatten, freeparameters,
                               parameterlayout, parameterrange, flatlength
using SimpleSolvers: Static
using LinearAlgebra: norm, Adjoint, Transpose
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using Printf: @printf
import ForwardDiff, JLD2, MLDatasets, Random, Zygote

# ------------------------------------------------------------------- hyperparameters ---

const patch_length = 7      # MNIST images are 28×28, so the sequence length is (28÷7)² = 16
const n_heads = 7
const L = 16                # the number of transformer blocks
const batch_size = 2048
const n_epochs = 5          # the original script uses 500
const add_connection = false
const T = Float32
const learning_rate = T(1e-3)
const momentum_coefficient = T(0.5)
const seed = 1234

const dim = patch_length^2                      # the transformer dimension
const seq_length = (28 ÷ patch_length)^2        # the number of patches
const n_classes = 10
const Dₕ = dim ÷ n_heads                        # the dimension of a single head

@assert dim % n_heads == 0

# ----------------------------------------------------------------------------- data ---

@doc raw"""
    split_and_flatten(input, patch_length)

Rearrange a batch of images into *flattened patches*, i.e. turn an ``(N, N, k)`` array into
an ``(\mathtt{patch\_length}^2, (N \div \mathtt{patch\_length})^2, k)`` array.

This is the equivalent of `GeometricMachineLearning.split_and_flatten` and produces the same
ordering: the patches are numbered column-major over the image and the entries within a
patch are numbered column-major as well.
"""
function split_and_flatten(input::AbstractArray{<:Number,3}, patch_length::Integer)
    @assert size(input, 1) == size(input, 2)
    @assert size(input, 1) % patch_length == 0
    n = size(input, 1) ÷ patch_length
    # (i_red, patch_row, j_red, patch_column, k) → (i_red, j_red, patch_row, patch_column, k)
    output = permutedims(reshape(input, patch_length, n, patch_length, n, size(input, 3)), (1, 3, 2, 4, 5))
    reshape(output, patch_length^2, n^2, size(input, 3))
end

"""
    onehotbatch(target)

Turn a vector of labels (`0` to `9`) into a `10`×`length(target)` matrix of unit vectors.
"""
function onehotbatch(target::AbstractVector{<:Integer})
    output = zeros(T, n_classes, length(target))
    for (k, label) in pairs(target)
        output[label+1, k] = one(T)
    end
    output
end

# ----------------------------------------------------------------------- parameters ---

# The parameters are stored in a flat `NamedTuple`. Note that only the projections of the
# attention layers are put on the `StiefelManifold`; the parameters of the `ResNetLayer`s and
# of the classification layer are ordinary arrays.
const parameter_layout = [
    [Symbol(s, "_", l, "_", h) => (dim, Dₕ) for l in 1:L for s in ("PQ", "PK", "PV") for h in 1:n_heads]
    vcat([[Symbol("Wres_", l) => (dim, dim), Symbol("bres_", l) => (dim,)] for l in 1:L]...)
    [:Wclass => (n_classes, dim)]
]

const parameter_keys = Tuple(first.(parameter_layout))
const parameter_sizes = last.(parameter_layout)
const n_attention_parameters = 3 * n_heads * L

# the position of a parameter within `parameter_layout`
attention_index(kind::Integer, l::Integer, h::Integer) = (l - 1) * 3 * n_heads + (kind - 1) * n_heads + h
resnet_index(l::Integer) = n_attention_parameters + 2 * (l - 1) + 1      # the bias follows right after
const classification_index = lastindex(parameter_layout)

# `GlorotUniform` of `AbstractNeuralNetworks`, which is what the original script uses for all
# parameters that are not on a manifold.
function glorot_uniform(rng::Random.AbstractRNG, size::Tuple)
    x = rand(rng, T, size...)
    sqrt(T(24) / sum(size)) * (x .- T(0.5))
end

function initial_parameters(rng::Random.AbstractRNG, stiefel::Bool)
    values = map(enumerate(parameter_sizes)) do (i, size)
        if i ≤ n_attention_parameters
            stiefel ? rand(rng, StiefelManifold{T}, size...) : glorot_uniform(rng, size)
        elseif length(size) == 1
            zeros(T, size...)               # the biases are initialized with zeros
        else
            glorot_uniform(rng, size)
        end
    end
    # `NetworkParameters` and not a bare `NamedTuple`: `GeometricOptimizers` takes a whole set of
    # parameters only as a container, because a `NamedTuple` alias is a type nobody owns and a method
    # on one is a method on `Base.NamedTuple`. The wrap shares the leaf arrays rather than copying
    # them, and the container forwards `keys`, `values`, `ps[i]` and `ps.field`, so `regroup`, `F` and
    # `∇F!` below read it exactly as they read the `NamedTuple` underneath.
    NetworkParameters(NamedTuple{parameter_keys}(Tuple(values)))
end

# The layout of the parameter set, and the ranges this script indexes the flat vector by.
#
# Both are read *off the layout* rather than computed here from `parameter_sizes`. That is the whole of
# what the port from `ParameterHandling` to `NeuralNetworkParameters` had to get right: this script
# reads and writes the flat vector by hand — `regroup` slices it, `∇F!` writes the gradient into it —
# while the optimizer flattens with `NeuralNetworkParameters`, so a disagreement between the two
# orderings would permute every gradient and nothing would say so. Deriving the ranges from the same
# layout the flattening uses makes the two agree by construction, where a `cumsum` over
# `parameter_sizes` beside an assertion that it matched only made them agree by inspection.
#
# `parameterlayout` needs a parameter set to read, so this comes after `initial_parameters` rather than
# beside `parameter_sizes`. `flatten!` writes through the layout with one `copyto!` per leaf over a
# known range, which is what makes the per-iteration flattening allocation-free, and `freeparameters`
# is what takes a `StiefelManifold` down to its dense `A` — by a method `GeometricOptimizers`
# registers, so this script needs no copy of that knowledge.
const parameter_flat_layout = parameterlayout(initial_parameters(Random.Xoshiro(seed), true))
const parameter_ranges = [parameterrange(getfield(parameter_flat_layout.children, i))
                          for i in eachindex(parameter_sizes)]
const n_parameters = flatlength(parameter_flat_layout)

# For the forward pass the parameters are regrouped into vectors of concrete element type.
# Doing this outside of the differentiated function keeps both the forward pass and `Zygote`
# type stable. `freeparameters` is what takes a `StiefelManifold` down to its dense `A`.

function regroup(get_parameter::Base.Callable, ::Type{R}=T) where {R<:Number}
    (Q=Matrix{R}[get_parameter(attention_index(1, l, h)) for l in 1:L for h in 1:n_heads],
        K=Matrix{R}[get_parameter(attention_index(2, l, h)) for l in 1:L for h in 1:n_heads],
        V=Matrix{R}[get_parameter(attention_index(3, l, h)) for l in 1:L for h in 1:n_heads],
        Wres=Matrix{R}[get_parameter(resnet_index(l)) for l in 1:L],
        bres=Vector{R}[get_parameter(resnet_index(l) + 1) for l in 1:L],
        Wclass=get_parameter(classification_index))
end

regroup(ps::NetworkParameters) = regroup(let p = values(ps)
    i -> freeparameters(p[i])
end)

# the element type is kept general here so that `check_gradient` can differentiate through it
regroup(v::AbstractVector{R}) where {R<:Number} = regroup(i -> reshape(v[parameter_ranges[i]], parameter_sizes[i]...), R)

# ---------------------------------------------------------------------------- model ---

"""
    mat_tensor_mul(A, x)

Multiply `A` onto every matrix stored in `x`, i.e. parallelize over the third axis.
"""
function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number,3})
    reshape(A * reshape(x, size(x, 1), :), size(A, 1), size(x, 2), size(x, 3))
end

# `Zygote` differentiates `mat_tensor_mul` by pulling the cotangent through the `reshape`, and
# the cotangent of a `Q` that enters `batched_mul` through `batched_transpose` is a *lazy*
# `BatchedTranspose`. Reshaping that produces a `ReshapedArray`, for which the multiplication
# of the backward pass falls back to the generic (i.e. element by element) method of
# `LinearAlgebra` instead of `BLAS`. The rule below writes the backward pass by hand and
# materializes such cotangents first.
_dense(Δ::AbstractArray) = Δ
_dense(Δ::BatchedAdjOrTrans) = permutedims(parent(Δ), (2, 1, 3))    # all arrays here are real
_dense(Δ::Union{Adjoint,Transpose}) = permutedims(parent(Δ), (2, 1))

Zygote.@adjoint function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number,3})
    function mat_tensor_mul_pullback(Δ)
        Δ₂ = reshape(_dense(Δ), size(A, 1), :)
        x₂ = reshape(x, size(x, 1), :)
        Δ₂ * transpose(x₂), reshape(transpose(A) * Δ₂, size(x))
    end
    mat_tensor_mul(A, x), mat_tensor_mul_pullback
end

"""
    predict(ps, input)

Apply the classification transformer to `input`, a `(dim, seq_length, k)` array, and return
the `(n_classes, k)` matrix of predictions. Here `ps` are *regrouped* parameters.
"""
function predict(ps::NamedTuple, input::AbstractArray{<:Number,3})
    x = input
    for l in 1:L
        # the multi head attention layer
        heads = ntuple(n_heads) do h
            i = (l - 1) * n_heads + h
            Q = mat_tensor_mul(transpose(ps.Q[i]), x)
            K = mat_tensor_mul(transpose(ps.K[i]), x)
            V = mat_tensor_mul(transpose(ps.V[i]), x)
            batched_mul(V, softmax(batched_mul(batched_transpose(Q), K) ./ sqrt(T(dim)); dims=1))
        end
        y = add_connection ? x + reduce(vcat, heads) : reduce(vcat, heads)
        # the ResNet layer
        x = y + tanh.(mat_tensor_mul(ps.Wres[l], y) .+ ps.bres[l])
    end
    # the classification layer picks the last column and applies softmax
    softmax(ps.Wclass * x[:, end, :]; dims=1)
end

"""
    network_loss(ps, input, output)

The equivalent of `GeometricMachineLearning.FeedForwardLoss`.
"""
network_loss(ps::NamedTuple, input, output) = norm(predict(ps, input) - output) / norm(output)

"""
    accuracy(ps, input, output)

The ratio of correctly classified images, i.e. the equivalent of
`GeometricMachineLearning.accuracy`. Here `ps` are the parameters as stored by the optimizer.
"""
function accuracy(ps::NamedTuple, input, output; chunk_size=batch_size)
    regrouped = regroup(ps)
    correct = 0
    for k in Iterators.partition(axes(input, 3), chunk_size)
        prediction = predict(regrouped, input[:, :, k])
        for (j, jₖ) in pairs(k)
            correct += argmax(view(prediction, :, j)) == argmax(view(output, :, jₖ))
        end
    end
    correct / size(input, 3)
end

# --------------------------------------------------------------- objective & gradient ---

# The `Optimizer` calls the objective on the parameter container and `∇F!` on the
# *flattened* parameters. Both read the current batch from `current_batch`.
const current_batch = Ref{Tuple{Array{T,3},Matrix{T}}}()

F(ps::NetworkParameters) = network_loss(regroup(ps), current_batch[]...)

# the cotangent of a `transpose(ps.Q[i])` is a lazy `Transpose`, which `vec` turns into a
# wrapped array that the device cannot copy from — hence `_dense` here as well
_write_gradient!(g, i::Integer, ∂) = copyto!(view(g, parameter_ranges[i]), vec(_dense(∂)))
_write_gradient!(g, i::Integer, ::Nothing) = fill!(view(g, parameter_ranges[i]), zero(T))

function ∇F!(g::AbstractVector{T}, v::AbstractVector{T})
    ∂ps = Zygote.gradient(ps -> network_loss(ps, current_batch[]...), regroup(v))[1]
    for l in 1:L, h in 1:n_heads
        i = (l - 1) * n_heads + h
        _write_gradient!(g, attention_index(1, l, h), ∂ps.Q[i])
        _write_gradient!(g, attention_index(2, l, h), ∂ps.K[i])
        _write_gradient!(g, attention_index(3, l, h), ∂ps.V[i])
    end
    for l in 1:L
        _write_gradient!(g, resnet_index(l), ∂ps.Wres[l])
        _write_gradient!(g, resnet_index(l) + 1, ∂ps.bres[l])
    end
    _write_gradient!(g, classification_index, ∂ps.Wclass)

    g
end

"""
    check_gradient(ps)

Compare `∇F!` to the *directional derivative* along a random direction and return the
relative error. This is a cheap way of making sure that the hand written gradient is
consistent with the objective: the full `ForwardDiff` gradient would need one dual number
per parameter, but a single directional derivative needs only one.

Note that a central difference is not a useful comparison here: the objective is evaluated in
`Float32` and the directional derivative is of the order of ``\\|g\\|/\\sqrt{n}``, so the
cancellation error of the difference quotient is of the same order as the quantity itself.
"""
function check_gradient(ps::NetworkParameters)
    v, _ = flatten(ps)
    g = zeros(T, length(v))
    ∇F!(g, v)
    d = Random.randn(Random.Xoshiro(seed), T, length(v))
    d ./= norm(d)
    directional_derivative = ForwardDiff.derivative(zero(T)) do t
        network_loss(regroup(v + t * d), current_batch[]...)
    end
    abs(directional_derivative - sum(g .* d)) / abs(directional_derivative)
end

# -------------------------------------------------------------------------- training ---

function train(stiefel::Bool, algorithm::GeometricOptimizers.OptimizerMethod, input, output;
    n_epochs=n_epochs, learning_rate=learning_rate, verbose=true)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, stiefel)

    n_batches = size(input, 3) ÷ batch_size
    current_batch[] = (input[:, :, 1:batch_size], output[:, 1:batch_size])

    # Note that the learning rate is supplied through the line search: the *methods* only
    # determine the direction. `Static(learning_rate)` is what `Optimizer` defaults to for
    # these three methods anyway; it is written out so that the rate is visible right here.
    optimizer = Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(learning_rate))
    state = OptimizerState(algorithm, ps)
    initialize_state!(state)

    losses = T[]
    initial_time = time()
    for epoch in 1:n_epochs
        # `solve!` cannot be used here: it optimizes a *fixed* objective until it converges,
        # whereas the objective changes with every batch.
        batches = Iterators.take(Iterators.partition(Random.shuffle(rng, axes(input, 3)), batch_size), n_batches)
        epoch_loss = zero(T)
        for (i, batch) in pairs(collect(batches))
            current_batch[] = (input[:, :, batch], output[:, batch])
            increase_iteration_number!(state)
            solver_step!(ps, state, optimizer)
            GeometricOptimizers.update!(state, optimizer, ps)
            loss = F(ps)
            push!(losses, loss)
            epoch_loss += loss / n_batches
            verbose && @printf("\r  epoch %3i/%i, batch %3i/%i, loss %.5f", epoch, n_epochs, i, n_batches, loss)
        end
        verbose && @printf("\r  epoch %3i/%i, average loss %.5f%20s\n", epoch, n_epochs, epoch_loss, "")
    end
    total_time = time() - initial_time

    ps, losses, total_time
end

# ------------------------------------------------------------------------------- run ---

println("loading MNIST ...")
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

const train_input = split_and_flatten(T.(train_x), patch_length)
const train_output = onehotbatch(train_y)
const test_input = split_and_flatten(T.(test_x), patch_length)
const test_output = onehotbatch(test_y)

@assert size(train_input) == (dim, seq_length, size(train_x, 3))

println(n_parameters, " parameters, ", size(train_input, 3) ÷ batch_size, " batches per epoch")

current_batch[] = (train_input[:, :, 1:batch_size], train_output[:, 1:batch_size])
@printf("relative error of ∇F! compared to a finite difference: %.2e\n\n",
    check_gradient(initial_parameters(Random.Xoshiro(seed), true)))

# The same four trainings as in the original script. `Adam` takes the *element type* of the
# parameters, not a learning rate, and it is not converted the way `MomentumMethod` is, so
# `Adam(T)` is what dispatches to the `Float32` cache.
#
# `regular weights, Adam` is the *baseline* and is not expected to learn. It is the one
# configuration whose attention projections are unconstrained, and with 16 transformer blocks
# and no layer normalization, dropout or pre-training the gradient reaching the early blocks
# vanishes: the network collapses onto a constant prediction and stays there. This is the
# published result of B. Brantner, "Generalizing Adam To Manifolds For Efficiently Training
# Transformers" (§"Numerical Example: the Transformer"), where the loss is "stuck at around
# 1.34, which indicates a trivial prediction". The number is reproduced by `network_loss`
# here: `‖pred − out‖/‖out‖` for a constant prediction over one-hot targets is
# `√(2·(1 − 1/10)) = √1.8 ≈ 1.342`. Putting the projections on the Stiefel manifold is what
# removes the problem — `YᵀY = I`, so a block neither amplifies nor damps what passes through
# it — and that is why the other three configurations learn. A flat loss here is the
# experiment working, not a defect.
const runs = [
    (name="Stiefel weights, Adam    ", stiefel=true, algorithm=Adam(T)),
    (name="regular weights, Adam    ", stiefel=false, algorithm=Adam(T)),
    (name="Stiefel weights, gradient", stiefel=true, algorithm=GradientMethod()),
    (name="Stiefel weights, momentum", stiefel=true, algorithm=MomentumMethod(momentum_coefficient)),
]

results = []
for run in runs
    println(run.name, ":")
    ps, losses, total_time = train(run.stiefel, run.algorithm, train_input, train_output)
    score = accuracy(ps, test_input, test_output)
    @printf("  time %.1f s, test accuracy %.4f\n\n", total_time, score)
    push!(results, (name=run.name, parameters=map(freeparameters, ps), losses=losses, total_time=total_time, accuracy=score))
end

output = Dict{String,Any}("n_epochs" => n_epochs)
for (i, result) in pairs(results)
    output["parameters$i"] = result.parameters
    output["losses$i"] = result.losses
    output["total_time$i"] = result.total_time
    output["accuracy$i"] = result.accuracy
end
JLD2.save("mnist_parameters.jld2", output)

println("n_epochs: ", n_epochs)
for result in results
    @printf("%s: time: %8.1f s   classification accuracy: %.4f\n", result.name, result.total_time, result.accuracy)
end
