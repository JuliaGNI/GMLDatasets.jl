# The `Metal` version of `mnist.jl`, i.e. `mnist_cuda.jl` for the GPU of an Apple silicon Mac.
# Everything that concerns the network — the batches, the forward pass and the `Zygote`
# gradient — is evaluated on the GPU; the *parameters* and the `Optimizer` stay on the host.
#
# The reason for that split is the parameter interface of `GeometricOptimizers`: the
# optimizer flattens its parameter `NamedTuple` with `ParameterHandling.flatten` on every
# step (see `(grad::Gradient)(::ArrayNamedTuple)` in `src/optimizers/named_tuple_wrapper.jl`)
# and `ParameterHandling.flatten` has no method for GPU arrays — an `MtlVector` falls through
# to `ParameterHandling.flatten(::Type, ::AbstractVector)`, which maps over the *elements* of
# the vector. `_similar(::StiefelManifold)` allocates on the host as well. Neither of the two
# is a problem here, because the optimizer only ever touches the 154938 parameters (620 kB in
# `Float32`), whereas the forward and backward passes touch the whole batch (2048 images
# through 16 transformer blocks). Per optimizer step the parameters are copied to the device
# and the gradient is copied back once, i.e. about 1.2 MB of traffic; everything expensive
# happens on the device. Note that the memory of an Apple GPU is *unified*, so those copies
# do not cross a bus — but they are still copies, as an `MtlArray` does not share its storage
# with the `Array` it was built from.
#
# Run it with
#
#   julia --project=scripts -e 'using Pkg; Pkg.add("Metal")'   # once, on the Mac
#   julia --project=scripts scripts/geometric_optimizers/mnist_metal.jl
#
# `Metal` is deliberately *not* a dependency of `scripts/Project.toml`: it cannot be resolved
# on Linux, and a project that lists it leaves the whole environment unprecompilable there,
# which is what stopped the first attempt at `mnist_cuda.jl` on the RTX 4090. So the machine
# that has a Metal device adds it by hand. That `Pkg.add` writes `Metal` back into
# `scripts/Project.toml` — keep the line out of commits. Unlike `CUDA`, `Metal` does not
# *load* on platforms it does not support, so this script is for macOS on Apple silicon only
# — the other two scripts still work in the same environment everywhere. If no device is available this one runs on
# the host, so that it can be tested without a GPU — but at the 500 epochs of the original
# that is not a practical option.
#
# Apart from the device the setup is identical to `mnist.jl`: same network, same
# initialization (drawn on the host with the same seed) and the same four trainings, so the
# results of the three scripts are directly comparable. The network is `Float32` throughout,
# which is what `Metal` supports (`Float64` is not available on an Apple GPU).
#
# On device memory: `Zygote` keeps every intermediate of the forward pass alive for the
# backward pass, which at `L = 16` and a batch size of 2048 is roughly 80 MB per transformer
# block, i.e. about 3 GB including the backward pass itself. That is well within the working
# set size that the GPU recommends (printed below), but if it does not fit, reduce
# `batch_size` (the number of batches per epoch adapts automatically).
#
# Those intermediates are *not* released on their own, and on unified memory that takes the
# whole machine down rather than merely erroring — see the section on device memory below,
# which is the one substantial difference to `mnist_cuda.jl`. Everything the network does on
# the device therefore runs inside a `device_scope`, and a memory budget stops the script if
# it ever grows out of it again.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!, ParameterHandling
using SimpleSolvers: Static
using LinearAlgebra: norm, Adjoint, Transpose
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using Printf: @printf
import ForwardDiff, JLD2, Metal, MLDatasets, Random, Zygote

# --------------------------------------------------------------------------- device ---

const use_metal = Metal.functional()

# `to_device` is a `const` binding to a *function*, so the element types of the forward pass
# are still inferred.
const to_device = use_metal ? Metal.mtl : identity
to_host(x::AbstractArray) = Array(x)
synchronize_device() = use_metal ? Metal.synchronize() : nothing

if use_metal
    println("running on ", Metal.device().name, " (recommended working set size ",
        Metal.device().recommendedMaxWorkingSetSize ÷ 1024^2, " MB)")
else
    @warn "no Metal device available — the network is evaluated on the host instead"
end

# ------------------------------------------------------------------- device memory ---

# The memory of an `MtlArray` is allocated by Metal, not by Julia, and keeping it bounded
# needs *two* things per step — neither of which works on its own:
#
#   * `Metal.@autoreleasepool`. A kernel launch autoreleases its command buffer and a command
#     buffer retains every buffer it refers to, so without a pool that drains, the command
#     buffers of the forward and backward passes accumulate in the outermost pool — which in
#     a script never drains — and hold on to every intermediate. No amount of garbage
#     collection frees those, because the reference that keeps them alive is on the
#     Objective-C side.
#   * `GC.gc(false)`. Conversely the pool can only drop a buffer once the `MtlArray` owning it
#     has been finalized, and Julia's collector is driven by the size of the *Julia* heap: an
#     `MtlArray` is a few hundred bytes of Julia object in front of hundreds of megabytes of
#     Metal buffer, so it does not fire anywhere near often enough by itself.
#
# Measured over ten gradient steps at a batch size of 512 (M4 Max, `Metal` 1.10, `scripts/
# metal_memory_probe.jl`), as reported by `Metal.device().currentAllocatedSize`:
#
#     neither                     0.5 → 4.2 GB, climbing by 0.41 GB per step
#     `GC.gc(true)` per step      0.4 → 4.1 GB, i.e. a full collection changes nothing
#     pool per step               0.7 → 12.4 GB between the collections that happen to occur
#     pool *and* `GC.gc(false)`   0.31 GB, flat, at no measurable cost in time
#
# Without this the script grows without bound — at a batch size of 2048 by about 1.6 GB per
# step. That is worse than it sounds on an Apple GPU: the memory is *unified*, an allocation
# does not fail when it runs out, and so instead of an error one gets compression, then
# paging, and then a system that has to be power-cycled. Hence the budget below as well.

# a hard ceiling on what Metal may hold, as a fraction of the working set the device
# recommends — the point is not to be exact, but to fail loudly instead of dragging the whole
# machine into swap
const memory_budget = use_metal ? (Int(Metal.device().recommendedMaxWorkingSetSize) * 2) ÷ 3 : typemax(Int)

device_allocated() = use_metal ? Int(Metal.device().currentAllocatedSize) : 0

function check_device_memory(what::AbstractString)
    allocated = device_allocated()
    allocated ≤ memory_budget && return
    error("Metal has allocated $(allocated ÷ 1024^2) MB after $what, which is over the " *
          "budget of $(memory_budget ÷ 1024^2) MB. Stopping — an Apple GPU does not report " *
          "an out-of-memory error, it exhausts the unified memory of the whole system. " *
          "Either the batch size is too large or device memory is leaking again.")
end

"""
    device_scope(f)

Evaluate `f` inside an autorelease pool, release what it allocated on the device and check
that the result stays within `memory_budget`. Every batch is run inside one of these; see the
comment above for why both halves are needed.
"""
function device_scope(f::Base.Callable, what::AbstractString)
    use_metal || return f()
    result = Metal.@autoreleasepool f()
    GC.gc(false)
    check_device_memory(what)
    result
end

# ------------------------------------------------------------------- hyperparameters ---

const patch_length = 7      # MNIST images are 28×28, so the sequence length is (28÷7)² = 16
const n_heads = 7
const L = 16                # the number of transformer blocks
const batch_size = 2048
const n_epochs = 500
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

# The parameters are stored in a flat `NamedTuple` *on the host*. Note that only the
# projections of the attention layers are put on the `StiefelManifold`; the parameters of the
# `ResNetLayer`s and of the classification layer are ordinary arrays.
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

# the range that a parameter occupies in the flattened parameter vector, i.e. in
# `ParameterHandling.flatten(ps)[1]`
const parameter_ranges = let offsets = cumsum([0; prod.(parameter_sizes)])
    [(offsets[i]+1):offsets[i+1] for i in eachindex(parameter_sizes)]
end
const n_parameters = last(last(parameter_ranges))

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
    NamedTuple{parameter_keys}(Tuple(values))
end

_array(Y::StiefelManifold) = Y.A
_array(Y::AbstractArray) = Y

# For the forward pass the parameters are regrouped into vectors of concrete element type.
# Doing this outside of the differentiated function keeps both the forward pass and `Zygote`
# type stable.
function regroup(get_parameter::Base.Callable)
    (Q=[get_parameter(attention_index(1, l, h)) for l in 1:L for h in 1:n_heads],
        K=[get_parameter(attention_index(2, l, h)) for l in 1:L for h in 1:n_heads],
        V=[get_parameter(attention_index(3, l, h)) for l in 1:L for h in 1:n_heads],
        Wres=[get_parameter(resnet_index(l)) for l in 1:L],
        bres=[get_parameter(resnet_index(l) + 1) for l in 1:L],
        Wclass=get_parameter(classification_index))
end

"""
    regroup_host(v)

Regroup the *flattened* parameters without touching the device. The element type is kept
general so that `check_gradient` can differentiate through it — `ForwardDiff.Dual`s cannot be
handed to the GPU, so the reference derivative has to be computed on the host.
"""
regroup_host(v::AbstractVector{<:Number}) = regroup(i -> reshape(v[parameter_ranges[i]], parameter_sizes[i]...))

# The parameters are moved to the device in a single transfer and split up there; the 353
# individual parameters are far too small for a transfer each.
const host_parameters = zeros(T, n_parameters)
const device_parameters = to_device(zeros(T, n_parameters))
const device_gradient = to_device(zeros(T, n_parameters))

"""
    flatten_parameters!(v, ps)

Write the parameter `NamedTuple` into the flat vector `v`, in the same order in which
`ParameterHandling.flatten` writes it (which is the order of `parameter_ranges`).
"""
function flatten_parameters!(v::AbstractVector{T}, ps::NamedTuple)
    for (i, p) in pairs(values(ps))
        copyto!(view(v, parameter_ranges[i]), vec(_array(p)))
    end
    v
end

function regroup_device(v::AbstractVector{T})
    copyto!(device_parameters, v)
    regroup(i -> reshape(device_parameters[parameter_ranges[i]], parameter_sizes[i]...))
end

regroup_device(ps::NamedTuple) = regroup_device(flatten_parameters!(host_parameters, ps))

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
# `BatchedTranspose`. Reshaping that produces a `ReshapedArray` which the matrix
# multiplication of the GPU does not recognize, so the backward pass falls back to the generic
# method of `LinearAlgebra` — which indexes the arrays element by element and therefore errors
# on the device. The rule below writes the backward pass by hand and materializes such
# cotangents first; on the host it avoids the same (there merely slow) fallback.
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
the `(n_classes, k)` matrix of predictions. Here `ps` are *regrouped* parameters that live on
the same device as `input`.
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
`GeometricMachineLearning.accuracy`. Here `ps` are the parameters as stored by the optimizer
and `input`/`output` live on the host; only one chunk at a time is moved to the device. The
`argmax`es are taken on the host, as an `argmax` per column would be a scalar index on the
device.
"""
function accuracy(ps::NamedTuple, input, output; chunk_size=batch_size)
    regrouped = regroup_device(ps)
    correct = 0
    for k in Iterators.partition(axes(input, 3), chunk_size)
        prediction = device_scope("a chunk of the accuracy") do
            to_host(predict(regrouped, to_device(input[:, :, k])))
        end
        for (j, jₖ) in pairs(k)
            correct += argmax(view(prediction, :, j)) == argmax(view(output, :, jₖ))
        end
    end
    correct / size(input, 3)
end

# --------------------------------------------------------------- objective & gradient ---

# The `Optimizer` calls the objective on the parameter `NamedTuple` and `∇F!` on the
# *flattened* parameters, both of which live on the host. The current batch is on the device.
const current_batch = Ref{Tuple{AbstractArray{T,3},AbstractMatrix{T}}}()

function F(ps::NamedTuple)
    input, output = current_batch[]     # the function barrier keeps `network_loss` inferred
    network_loss(regroup_device(ps), input, output)
end

# the cotangent of a `transpose(ps.Q[i])` is a lazy `Transpose`, which `vec` turns into a
# wrapped array that the device cannot copy from — hence `_dense` here as well
_write_gradient!(g, i::Integer, ∂) = copyto!(view(g, parameter_ranges[i]), vec(_dense(∂)))
_write_gradient!(g, i::Integer, ::Nothing) = fill!(view(g, parameter_ranges[i]), zero(T))

function ∇F!(g::AbstractVector{T}, v::AbstractVector{T})
    input, output = current_batch[]
    ∂ps = Zygote.gradient(ps -> network_loss(ps, input, output), regroup_device(v))[1]
    # the gradient is assembled on the device and downloaded in a single transfer
    for l in 1:L, h in 1:n_heads
        i = (l - 1) * n_heads + h
        _write_gradient!(device_gradient, attention_index(1, l, h), ∂ps.Q[i])
        _write_gradient!(device_gradient, attention_index(2, l, h), ∂ps.K[i])
        _write_gradient!(device_gradient, attention_index(3, l, h), ∂ps.V[i])
    end
    for l in 1:L
        _write_gradient!(device_gradient, resnet_index(l), ∂ps.Wres[l])
        _write_gradient!(device_gradient, resnet_index(l) + 1, ∂ps.bres[l])
    end
    _write_gradient!(device_gradient, classification_index, ∂ps.Wclass)
    copyto!(g, device_gradient)

    # The intermediates of the backward pass are still alive here — the pool of the enclosing
    # `device_scope` has not drained yet — so this is the high water mark of the step and the
    # right place to notice that a batch does not fit.
    check_device_memory("a gradient")

    g
end

"""
    check_gradient(ps)

Compare `∇F!` — which is evaluated on the device — to the *directional derivative* along a
random direction, which is evaluated on the host, and return the relative error. This checks
the gradient and the device at the same time.

Note that a central difference is not a useful comparison here: the objective is evaluated in
`Float32` and the directional derivative is of the order of ``\\|g\\|/\\sqrt{n}``, so the
cancellation error of the difference quotient is of the same order as the quantity itself.
`ForwardDiff.derivative` needs a single dual number instead — and `ForwardDiff.Dual`s cannot
be multiplied on the GPU, so this part has to run on the host anyway.
"""
function check_gradient(ps::NamedTuple)
    v, _ = ParameterHandling.flatten(ps)
    g = zeros(T, length(v))
    ∇F!(g, v)
    d = Random.randn(Random.Xoshiro(seed), T, length(v))
    d ./= norm(d)
    input, output = current_batch[]
    host_input, host_output = to_host(input), to_host(output)
    directional_derivative = ForwardDiff.derivative(zero(T)) do t
        network_loss(regroup_host(v + t * d), host_input, host_output)
    end
    abs(directional_derivative - sum(g .* d)) / abs(directional_derivative)
end

# -------------------------------------------------------------------------- training ---

function train(stiefel::Bool, algorithm::GeometricOptimizers.OptimizerMethod, input, output;
    n_epochs=n_epochs, learning_rate=learning_rate, verbose=true)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, stiefel)

    n_batches = size(input, 3) ÷ batch_size
    current_batch[] = (to_device(input[:, :, 1:batch_size]), to_device(output[:, 1:batch_size]))

    # Note that the learning rate is supplied through the line search: the *methods* only
    # determine the direction. `Static(learning_rate)` is what `Optimizer` defaults to for
    # these three methods anyway; it is written out so that the rate is visible right here.
    optimizer = Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(learning_rate))
    state = OptimizerState(algorithm, ps)
    initialize_state!(state)

    losses = T[]
    synchronize_device()
    initial_time = time()
    for epoch in 1:n_epochs
        # `solve!` cannot be used here: it optimizes a *fixed* objective until it converges,
        # whereas the objective changes with every batch.
        batches = Iterators.take(Iterators.partition(Random.shuffle(rng, axes(input, 3)), batch_size), n_batches)
        epoch_loss = zero(T)
        for (i, batch) in pairs(collect(batches))
            # The whole step happens inside one `device_scope`, so that the intermediates of
            # the forward and backward passes are released before the next batch allocates
            # its own. This is what keeps the script from exhausting the unified memory.
            loss = device_scope("epoch $epoch, batch $i") do
                # the batch is gathered on the host and uploaded; at 6.4 MB per batch this is
                # negligible next to the forward and backward passes
                current_batch[] = (to_device(input[:, :, batch]), to_device(output[:, batch]))
                increase_iteration_number!(state)
                solver_step!(ps, state, optimizer)
                GeometricOptimizers.update!(state, optimizer, ps)
                F(ps)
            end
            push!(losses, loss)
            epoch_loss += loss / n_batches
            verbose && @printf("\r  epoch %3i/%i, batch %3i/%i, loss %.5f", epoch, n_epochs, i, n_batches, loss)
        end
        verbose && @printf("\r  epoch %3i/%i, average loss %.5f%20s\n", epoch, n_epochs, epoch_loss, "")
    end
    synchronize_device()
    total_time = time() - initial_time

    ps, losses, total_time
end

# ------------------------------------------------------------------------------- run ---

println("loading MNIST ...")
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# the data set stays on the host; the batches are uploaded one at a time
const train_input = split_and_flatten(T.(train_x), patch_length)
const train_output = onehotbatch(train_y)
const test_input = split_and_flatten(T.(test_x), patch_length)
const test_output = onehotbatch(test_y)

@assert size(train_input) == (dim, seq_length, size(train_x, 3))

println(n_parameters, " parameters, ", size(train_input, 3) ÷ batch_size, " batches per epoch")

current_batch[] = (to_device(train_input[:, :, 1:batch_size]), to_device(train_output[:, 1:batch_size]))
@printf("relative error of ∇F! compared to a directional derivative on the host: %.2e\n\n",
    device_scope(() -> check_gradient(initial_parameters(Random.Xoshiro(seed), true)), "the gradient check"))

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
    push!(results, (name=run.name, parameters=map(_array, ps), losses=losses, total_time=total_time, accuracy=score))
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
