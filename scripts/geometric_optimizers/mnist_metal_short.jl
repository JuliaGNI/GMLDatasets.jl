# A *two hour* version of `mnist_metal.jl`, for the Apple silicon GPU.
#
# `mnist_metal.jl` runs the four trainings of the original script at `n_epochs = 500`, which
# on an M4 Max is ≈36 h. That is the run that produces publishable numbers, and it should be
# started only once there is reason to believe it will not waste a day and a half. This script
# is that check: it establishes, inside a wall clock budget it is given (two hours by
# default), that the machinery is correct and that every configuration does what the paper
# says it should — which for three of the four means learning, and for the fourth means not
# learning (see below).
#
# Run it with
#
#   julia --project=scripts -e 'using Pkg; Pkg.add("Metal")'   # once, on the Mac; see mnist_metal.jl
#   julia --project=scripts scripts/geometric_optimizers/mnist_metal_short.jl
#
# and optionally override the budget or the batch size from the environment:
#
#   MNIST_TIME_BUDGET=1800 julia --project=scripts scripts/geometric_optimizers/mnist_metal_short.jl
#   MNIST_BATCH_SIZE=512   julia --project=scripts scripts/geometric_optimizers/mnist_metal_short.jl
#
# ## How the budget is spent
#
# The schedule is *derived*, not guessed. After the data is loaded and the gradient check has
# compiled the forward and backward passes, the script times a handful of real optimizer steps
# and divides the remaining budget by the number of configurations. So the run finishes inside
# the budget on any machine, and `n_epochs` is whatever that machine can afford — the number
# is reported before the training starts. A per-run deadline truncates a configuration that
# turns out to be slower than the calibration suggested, and says so in the table.
#
# All configurations get the *same* `n_epochs`, so their losses and accuracies stay
# comparable with each other. They are of course not comparable with a 500 epoch run.
#
# ## `regular weights, Adam` is not supposed to learn
#
# One of the four configurations — the one *without* the Stiefel constraint — sits at chance
# with a flat loss, and that is the point of the experiment rather than a defect. It is the
# published result of
#
#   B. Brantner, *Generalizing Adam To Manifolds For Efficiently Training Transformers*
#
# whose §"Numerical Example: the Transformer" reports that the vision transformer with
# unconstrained projections and no layer normalization, dropout or pre-training "is not able to
# learn much, as the error rate is stuck at around 1.34, which indicates a trivial prediction".
# With 16 transformer blocks and nothing normalizing between them, the gradient that reaches
# the early blocks vanishes; the network collapses onto a constant prediction eᵢ and stays
# there. Constraining the projections to the Stiefel manifold is what removes the problem —
# `YᵀY = I` means a block neither amplifies nor damps what passes through it — and that is why
# the other three configurations learn. Orthonormality is being used here in place of the
# heuristics (normalization, dropout, regularization) that a transformer usually needs.
#
# The `1.34` is reproduced exactly by this script's loss. `network_loss` is
# `‖pred − out‖/‖out‖`; over a batch of `k` one-hot targets a trivial prediction is wrong on
# 9 of 10 images and off by `√2` on each, so the loss is `√(2·0.9·k)/√k = √1.8 ≈ 1.342`. The
# three epoch run of 2026-08-07 measured `1.327 → 1.330` at an accuracy of `0.1049`, i.e. the
# plateau and chance.
#
# So this configuration is judged against the plateau instead of against the accuracy floor:
# it passes when it lands there, which is what reproduces the paper.
#
# ## What this establishes, and what it does not
#
# It establishes that
#
#   * `∇F!` agrees with a host side directional derivative, for Stiefel *and* for regular
#     weights — `mnist_metal.jl` only checks the Stiefel case;
#   * the device memory stays inside its budget for the whole run (the `device_scope` fix);
#   * every Stiefel configuration takes steps that reduce the loss;
#   * the Stiefel parameters are still on the manifold at the end, and everything is still
#     `Float32` and finite;
#   * every Stiefel configuration classifies well above the 10 % of chance, and the
#     unconstrained one sits at the trivial-prediction plateau.
#
# It does *not* establish the accuracy the full runs reach. A few hundred steps is a few per
# cent of the original schedule, so the accuracies below are a signal that learning happens,
# not a result. Comparing against the accuracies of the original script needs the full run on
# the RTX 4090.
#
# ## On `batch_size`
#
# The default is the `2048` of `mnist_metal.jl`, so that this script differs from it in
# exactly one dimension — the length of the schedule. Anything surprising here can then be
# attributed to the short schedule and not to a changed optimization problem.
#
# The work per *epoch* is independent of the batch size, so a smaller batch buys proportionally
# more optimizer steps for the same wall clock: at `MNIST_BATCH_SIZE=512` the same two hours
# are ≈4× the parameter updates and therefore a much stronger accuracy signal. It also changes
# the gradient noise, i.e. the optimization problem, which is why it is not the default.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!
# `GeometricOptimizers` 0.5.0 dropped `ParameterHandling` for `NeuralNetworkParameters`, which is the
# package that owns a parameter set and does its flattening. This script used to reach the old one
# through `GeometricOptimizers`, so it stopped loading at that release; these are the replacements.
# `freeparameters` is the leaf protocol -- it is `Y.A` for a `StiefelManifold`, by a method
# `GeometricOptimizers` registers, so this script no longer needs its own copy of that knowledge.
using NeuralNetworkParameters: flatten, flatten!, freeparameters, parameterlayout, flatlength
using SimpleSolvers: Static
using LinearAlgebra: norm, I, Adjoint, Transpose
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using Printf: @printf, @sprintf
import ForwardDiff, JLD2, Metal, MLDatasets, Random, Zygote

# --------------------------------------------------------------------------- budget ---

const script_start = time()

# the total wall clock this script may use, from the moment it started
const time_budget = parse(Float64, get(ENV, "MNIST_TIME_BUDGET", "7200"))

# what is held back from the budget for the accuracy evaluations, the final checks and saving
const time_reserve = 120.0

elapsed() = time() - script_start
remaining() = time_budget - elapsed()

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

# See the long comment in `mnist_metal.jl`: an `MtlArray` is a few hundred bytes of Julia
# object in front of hundreds of megabytes of Metal buffer, so bounding device memory needs
# both an autorelease pool that drains and a `GC.gc(false)` that finalizes the wrappers. On
# unified memory the failure mode of getting this wrong is not an out-of-memory error but a
# machine that has to be power-cycled, hence the budget as well.

const memory_budget = use_metal ? (Int(Metal.device().recommendedMaxWorkingSetSize) * 2) ÷ 3 : typemax(Int)

device_allocated() = use_metal ? Int(Metal.device().currentAllocatedSize) : 0

# the high water mark over the whole run, reported at the end
const peak_allocated = Ref(0)

function check_device_memory(what::AbstractString)
    allocated = device_allocated()
    peak_allocated[] = max(peak_allocated[], allocated)
    allocated ≤ memory_budget && return
    error("Metal has allocated $(allocated ÷ 1024^2) MB after $what, which is over the " *
          "budget of $(memory_budget ÷ 1024^2) MB. Stopping — an Apple GPU does not report " *
          "an out-of-memory error, it exhausts the unified memory of the whole system. " *
          "Either the batch size is too large or device memory is leaking again.")
end

"""
    device_scope(f, what)

Evaluate `f` inside an autorelease pool, release what it allocated on the device and check
that the result stays within `memory_budget`.
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
const batch_size = parse(Int, get(ENV, "MNIST_BATCH_SIZE", "2048"))
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

# The bars the runs are judged against. `chance_accuracy` is what a constant prediction gets.
# `accuracy_floor` is three times chance — low enough that a genuinely slow optimizer (plain
# gradient descent at `1e-3`) is not failed for being slow, high enough that it cannot be
# reached without learning something.
const chance_accuracy = T(1 / n_classes)
const accuracy_floor = T(0.30)

# a run that is expected to learn has to reduce its average epoch loss by at least this fraction
const loss_decrease_floor = T(0.02)

# The loss of a trivial prediction, i.e. of the plateau the unconstrained configuration is
# expected to sit at: `‖pred − out‖/‖out‖` with `pred` constant is `√(2·(1 − 1/n_classes))`.
# See the header — this is the `≈1.34` of the paper. The tolerance is wide because a run that
# has collapsed is at the plateau to three digits, while anything else is nowhere near it.
const trivial_loss = sqrt(T(2) * (one(T) - chance_accuracy))
const trivial_loss_tolerance = T(0.05)

# how far the Stiefel parameters may drift off the manifold over the run
const orthonormality_tolerance = T(1e-4)

# ----------------------------------------------------------------------------- data ---

@doc raw"""
    split_and_flatten(input, patch_length)

Rearrange a batch of images into *flattened patches*, i.e. turn an ``(N, N, k)`` array into
an ``(\mathtt{patch\_length}^2, (N \div \mathtt{patch\_length})^2, k)`` array.
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

# the range that a parameter occupies in the flattened parameter vector
const parameter_ranges = let offsets = cumsum([0; prod.(parameter_sizes)])
    [(offsets[i]+1):offsets[i+1] for i in eachindex(parameter_sizes)]
end
const n_parameters = last(last(parameter_ranges))

# `GlorotUniform` of `AbstractNeuralNetworks`
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


# For the forward pass the parameters are regrouped into vectors of concrete element type.
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

Regroup the *flattened* parameters without touching the device, with a general element type so
that `check_gradient` can push `ForwardDiff.Dual`s through it.
"""
regroup_host(v::AbstractVector{<:Number}) = regroup(i -> reshape(v[parameter_ranges[i]], parameter_sizes[i]...))

const host_parameters = zeros(T, n_parameters)
const device_parameters = to_device(zeros(T, n_parameters))
const device_gradient = to_device(zeros(T, n_parameters))

# The layout of the parameter set, built once from a prototype: `flatten!` writes through it with a
# `copyto!` per leaf over a known range, so the per-iteration flattening allocates nothing. This is what
# replaced a hand-written loop over `parameter_ranges`, and `freeparameters` is what takes a
# `StiefelManifold` down to its dense `A` -- by a method `GeometricOptimizers` registers, so the
# knowledge lives in the package that owns the type.
const parameter_flat_layout = parameterlayout(initial_parameters(Random.Xoshiro(seed), true))

# The hand-computed `parameter_ranges` above have to agree with the flattening. Asserted at load time
# and not in a comment: `regroup` reads the flat vector by those ranges and `∇F!` writes the gradient
# into it the same way, so if the two ever diverged every gradient would be wrong by a permutation and
# nothing would say so. This is the whole of what the port has to get right.
let ps = initial_parameters(Random.Xoshiro(seed), true)
    v, _ = flatten(ps)
    n_parameters == length(v) == flatlength(ps) ||
        error("parameter_ranges cover ", n_parameters, " numbers, the flattening has ", length(v))
    for (i, (key, p)) in enumerate(pairs(ps))
        v[parameter_ranges[i]] == vec(freeparameters(p)) ||
            error("`", key, "` is not at parameter_ranges[", i, "] in the flattening")
    end
end

"""
    flatten_parameters!(v, ps)

Write the parameter `NamedTuple` into the flat vector `v`, in the order of `parameter_ranges`
(which is the order `NeuralNetworkParameters.flatten` uses).
"""
function flatten_parameters!(v::AbstractVector{T}, ps::NamedTuple)
    flatten!(v, ps, parameter_flat_layout)
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

# The cotangent of a `Q` that enters `batched_mul` through `batched_transpose` is a *lazy*
# `BatchedTranspose`; reshaping it produces a `ReshapedArray` that the matrix multiplication of
# the GPU does not recognize, so the backward pass would fall back to the element-by-element
# generic method of `LinearAlgebra` and error on the device. Hence the hand written pullback.
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
the `(n_classes, k)` matrix of predictions. `ps` are *regrouped* parameters living on the same
device as `input`.
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

The ratio of correctly classified images. `ps` are the parameters as stored by the optimizer
and `input`/`output` live on the host; only one chunk at a time is moved to the device.
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

    # The intermediates of the backward pass are still alive here, so this is the high water
    # mark of the step and the right place to notice that a batch does not fit.
    check_device_memory("a gradient")

    g
end

"""
    check_gradient(ps)

Compare `∇F!` — evaluated on the device — to the *directional derivative* along a random
direction, evaluated on the host, and return the relative error. This checks the gradient and
the device at the same time.

A central difference is not a useful comparison here: the objective is `Float32` and the
directional derivative is of the order of ``\\|g\\|/\\sqrt{n}``, so the cancellation error of
the difference quotient is of the same order as the quantity itself. `ForwardDiff.derivative`
needs a single dual number instead — and `ForwardDiff.Dual`s cannot be multiplied on the GPU,
so the reference has to be computed on the host.
"""
function check_gradient(ps::NamedTuple)
    v, _ = flatten(ps)
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

# ------------------------------------------------------------------ parameter checks ---

"""
    orthonormality_error(ps)

The worst ``\\|Y^TY - I\\|`` over the attention projections, i.e. how far the parameters have
drifted off the Stiefel manifold. `NaN` if they are not on a manifold to begin with.
"""
function orthonormality_error(ps::NamedTuple)
    ps[1] isa StiefelManifold || return T(NaN)
    worst = zero(T)
    for i in 1:n_attention_parameters
        A = freeparameters(values(ps)[i])
        worst = max(worst, T(norm(A' * A - I)))
    end
    worst
end

"""
    parameters_are_sound(ps)

Every parameter is finite and still stored in `T`. A retraction that produced a `NaN`, or a
step that silently promoted the parameters to `Float64`, shows up here.
"""
function parameters_are_sound(ps::NamedTuple)
    all(values(ps)) do p
        A = freeparameters(p)
        eltype(A) === T && all(isfinite, A)
    end
end

# -------------------------------------------------------------------------- training ---

"""
    build_optimizer(ps, algorithm)

The optimizer and its state, exactly as `mnist_metal.jl` builds them. The learning rate is
supplied through the line search — the *methods* only determine the direction — and
`Static(learning_rate)` is what `Optimizer` defaults to for these three anyway; it is written
out so that the rate is visible.
"""
function build_optimizer(ps::NamedTuple, algorithm::GeometricOptimizers.OptimizerMethod)
    optimizer = Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(learning_rate))
    state = OptimizerState(algorithm, ps)
    initialize_state!(state)
    optimizer, state
end

"""
    optimizer_step!(ps, state, optimizer, what)

One optimizer step on the batch currently in `current_batch`, inside a `device_scope` so that
the intermediates of the forward and backward passes are released before the next batch
allocates its own. Returns the loss after the step.
"""
function optimizer_step!(ps, state, optimizer, what::AbstractString)
    device_scope(what) do
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        GeometricOptimizers.update!(state, optimizer, ps)
        F(ps)
    end
end

"""
    calibrate_step_time(input, output)

The wall clock of one optimizer step, measured on the most expensive configuration
(Stiefel weights with `Adam`) so that the derived schedule errs on the short side. The first
steps are discarded: they compile the optimizer path.
"""
function calibrate_step_time(input, output; n_warmup=2, n_timed=4)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, true)
    optimizer, state = build_optimizer(ps, Adam(T))

    times = Float64[]
    for i in 1:(n_warmup+n_timed)
        batch = ((i-1)*batch_size+1):(i*batch_size)
        current_batch[] = (to_device(input[:, :, batch]), to_device(output[:, batch]))
        synchronize_device()
        step_start = time()
        optimizer_step!(ps, state, optimizer, "calibration step $i")
        synchronize_device()
        i > n_warmup && push!(times, time() - step_start)
    end
    sum(times) / length(times)
end

"""
    train(stiefel, algorithm, input, output; n_epochs, deadline)

Train one configuration. Returns the parameters, the per-batch losses, the per-epoch average
losses, the wall clock, and whether `deadline` cut the run short.

`solve!` cannot be used here: it optimizes a *fixed* objective until it converges, whereas the
objective changes with every batch.
"""
function train(stiefel::Bool, algorithm::GeometricOptimizers.OptimizerMethod, input, output;
    n_epochs::Integer, deadline::Float64, verbose=true)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, stiefel)

    n_batches = size(input, 3) ÷ batch_size
    current_batch[] = (to_device(input[:, :, 1:batch_size]), to_device(output[:, 1:batch_size]))
    optimizer, state = build_optimizer(ps, algorithm)

    losses = T[]
    epoch_losses = T[]
    truncated = false
    synchronize_device()
    initial_time = time()
    for epoch in 1:n_epochs
        batches = Iterators.take(Iterators.partition(Random.shuffle(rng, axes(input, 3)), batch_size), n_batches)
        epoch_loss = zero(T)
        for (i, batch) in pairs(collect(batches))
            # the batch is gathered on the host and uploaded; at 6.4 MB per batch this is
            # negligible next to the forward and backward passes
            current_batch[] = (to_device(input[:, :, batch]), to_device(output[:, batch]))
            loss = optimizer_step!(ps, state, optimizer, "epoch $epoch, batch $i")
            push!(losses, loss)
            epoch_loss += loss / n_batches
            verbose && @printf("\r  epoch %3i/%i, batch %3i/%i, loss %.5f", epoch, n_epochs, i, n_batches, loss)
        end
        push!(epoch_losses, epoch_loss)
        verbose && @printf("\r  epoch %3i/%i, average loss %.5f%20s\n", epoch, n_epochs, epoch_loss, "")
        if time() > deadline && epoch < n_epochs
            truncated = true
            verbose && @printf("  out of time after %i of %i epochs — stopping this run here\n", epoch, n_epochs)
            break
        end
    end
    synchronize_device()
    total_time = time() - initial_time

    ps, losses, epoch_losses, total_time, truncated
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

const n_batches = size(train_input, 3) ÷ batch_size
@printf("%i parameters, batch size %i, %i batches per epoch, budget %.0f s\n\n",
    n_parameters, batch_size, n_batches, time_budget)

# --- the gradient, for both initializations -------------------------------------------

# `mnist_metal.jl` only checks the Stiefel case. The regular case is checked here as well: that
# configuration is expected to stall (see the header), and the check separates the stall the
# experiment is about — a vanishing gradient in the network — from a wrong `∇F!` on plain
# arrays, which would look the same from the outside.
current_batch[] = (to_device(train_input[:, :, 1:batch_size]), to_device(train_output[:, 1:batch_size]))

const gradient_errors = map((true, false)) do stiefel
    error = device_scope("the gradient check") do
        check_gradient(initial_parameters(Random.Xoshiro(seed), stiefel))
    end
    @printf("relative error of ∇F! vs a host directional derivative, %s weights: %.2e\n",
        stiefel ? "Stiefel" : "regular", error)
    error
end

# `Float32` and a directional derivative of order ‖g‖/√n leave a few digits of room; anything
# beyond this is a wrong derivative rather than round-off.
const gradient_tolerance = 1e-4
const gradient_ok = all(e -> isfinite(e) && e < gradient_tolerance, gradient_errors)
gradient_ok || @warn "∇F! disagrees with the host directional derivative — the trainings below are not meaningful"

# --- the schedule ---------------------------------------------------------------------

# The four trainings of the original. `learns` is what the configuration is expected to do:
# the three Stiefel ones should reduce the loss and classify above `accuracy_floor`, while the
# unconstrained one is the baseline of the paper and should sit at `trivial_loss` at chance.
#
# `Adam` takes the *element type* of the parameters, not a learning rate, and it is not
# converted the way `MomentumMethod` is, so `Adam(T)` is what dispatches to the `Float32` cache.
const runs = [
    (name="Stiefel weights, Adam", stiefel=true, learns=true, algorithm=Adam(T)),
    (name="regular weights, Adam", stiefel=false, learns=false, algorithm=Adam(T)),
    (name="Stiefel weights, gradient", stiefel=true, learns=true, algorithm=GradientMethod()),
    (name="Stiefel weights, momentum", stiefel=true, learns=true, algorithm=MomentumMethod(momentum_coefficient)),
]

println("\ncalibrating ...")
const seconds_per_step = calibrate_step_time(train_input, train_output)

# what is left after the calibration, split evenly over the configurations
const seconds_per_run = (remaining() - time_reserve) / length(runs)
const n_epochs = max(1, floor(Int, seconds_per_run / (seconds_per_step * n_batches)))

@printf("%.2f s per step → %i epochs (%i steps) per configuration, %.0f s each\n",
    seconds_per_step, n_epochs, n_epochs * n_batches, seconds_per_run)
n_epochs * n_batches < 50 && @warn "fewer than 50 steps per configuration — the accuracies below will say very little; raise MNIST_TIME_BUDGET"
println()

# --- the trainings --------------------------------------------------------------------

results = []
for (j, run) in pairs(runs)
    println(run.name, ":")
    # the deadline is recomputed per run, so that a run that overshoots its share is paid for
    # by the runs after it rather than by the budget as a whole
    deadline = time() + (remaining() - time_reserve) / (length(runs) - j + 1)
    ps, losses, epoch_losses, total_time, truncated =
        train(run.stiefel, run.algorithm, train_input, train_output; n_epochs=n_epochs, deadline=deadline)
    score = accuracy(ps, test_input, test_output)
    @printf("  time %.1f s, test accuracy %.4f\n\n", total_time, score)
    push!(results, (name=run.name, stiefel=run.stiefel, learns=run.learns, parameters=map(freeparameters, ps),
        losses=losses, epoch_losses=epoch_losses, total_time=total_time, accuracy=score,
        truncated=truncated, orthonormality=orthonormality_error(ps), sound=parameters_are_sound(ps)))
end

# --- the verdict ----------------------------------------------------------------------

"""
    verdict(result)

Whether a configuration did what it is supposed to do, and if not, why.

Two criteria apply to every run — the parameters survived and, on a manifold, stayed on it.
The remaining two depend on `learns`: a configuration that is expected to learn has to reduce
its loss and classify above `accuracy_floor`, while the unconstrained one is expected to
collapse onto the trivial prediction and is checked against that plateau instead. Note that
this makes *learning* a failure for the latter — not because learning would be bad, but
because it would mean this script is no longer running the experiment of the paper.
"""
function verdict(result)
    reasons = String[]
    result.sound || push!(reasons, "parameters not finite or no longer $T")
    if result.stiefel && !(result.orthonormality ≤ orthonormality_tolerance)
        push!(reasons, @sprintf("off the manifold (‖YᵀY-I‖ = %.1e)", result.orthonormality))
    end
    first_loss, last_loss = first(result.epoch_losses), last(result.epoch_losses)
    if result.learns
        if !(last_loss < (1 - loss_decrease_floor) * first_loss)
            push!(reasons, @sprintf("loss flat (%.3f → %.3f)", first_loss, last_loss))
        end
        if result.accuracy < accuracy_floor
            push!(reasons, @sprintf("accuracy %.4f below the floor of %.2f%s", result.accuracy,
                accuracy_floor, result.accuracy < 2 * chance_accuracy ? ", i.e. at chance" : ""))
        end
    elseif !(abs(last_loss - trivial_loss) ≤ trivial_loss_tolerance)
        push!(reasons, @sprintf("expected the trivial-prediction plateau of %.3f, got %.3f",
            trivial_loss, last_loss))
    end
    isempty(reasons) ? "ok" : "FAILED: " * join(reasons, "; ")
end

const verdicts = map(verdict, results)

println("=" ^ 100)
@printf("%i epochs (%i steps) per configuration, batch size %i, %.0f s total\n\n",
    n_epochs, n_epochs * n_batches, batch_size, elapsed())
@printf("%-26s %8s %8s %10s %10s   %s\n", "run", "loss 1", "loss N", "accuracy", "‖YᵀY-I‖", "verdict")
for (result, v) in zip(results, verdicts)
    @printf("%-26s %8.3f %8.3f %10.4f %10s   %s\n", result.name,
        first(result.epoch_losses), last(result.epoch_losses), result.accuracy,
        isnan(result.orthonormality) ? "—" : (@sprintf "%.1e" result.orthonormality),
        v * (result.truncated ? " [truncated: $(length(result.epoch_losses)) epochs]" : ""))
end

const n_passed = count(==("ok"), verdicts)
@printf("\n%i of %i configurations behave as expected. Gradient check %s. Peak device memory %i MB of a %i MB budget.\n",
    n_passed, length(runs), gradient_ok ? "ok" : "FAILED",
    peak_allocated[] ÷ 1024^2, memory_budget ÷ 1024^2)

println("\n`regular weights, Adam` is the baseline of the paper and is expected to stall at ",
    @sprintf("the trivial-prediction plateau of %.3f at an accuracy of %.2f", trivial_loss, chance_accuracy),
    "; without the Stiefel constraint the gradient vanishes through the 16 unnormalized ",
    "transformer blocks. It is a failure above only if it did something else.")

if !gradient_ok || n_passed < length(runs)
    println("\nDo not start the 500 epoch run yet.")
else
    @printf("\nEverything checks out. The full run is %i× this schedule.\n", 500 ÷ max(n_epochs, 1))
end

# --- saving -----------------------------------------------------------------------------

# a separate file, so that the `mnist_parameters.jld2` of the full runs is never overwritten
output = Dict{String,Any}("n_epochs" => n_epochs, "batch_size" => batch_size,
    "seconds_per_step" => seconds_per_step, "gradient_errors" => collect(gradient_errors))
for (i, result) in pairs(results)
    output["name$i"] = result.name
    output["parameters$i"] = result.parameters
    output["losses$i"] = result.losses
    output["epoch_losses$i"] = result.epoch_losses
    output["total_time$i"] = result.total_time
    output["accuracy$i"] = result.accuracy
end
JLD2.save("mnist_parameters_short.jld2", output)
