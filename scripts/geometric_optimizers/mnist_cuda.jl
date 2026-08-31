# The `CUDA` version of `mnist.jl`. Everything that concerns the network — the batches, the
# forward pass and the `Zygote` gradient — is evaluated on the GPU; the *parameters* and the
# `Optimizer` stay on the host.
#
# The reason for that split is the parameter interface of `GeometricOptimizers`: the
# optimizer flattens its parameter `NamedTuple` with `ParameterHandling.flatten` on every
# step (see `(grad::Gradient)(::ArrayNamedTuple)` in `src/optimizers/named_tuple_wrapper.jl`)
# and `ParameterHandling.flatten` has no method for GPU arrays — a `CuVector` falls through
# to `ParameterHandling.flatten(::Type, ::AbstractVector)`, which maps over the *elements* of
# the vector. `_similar(::StiefelManifold)` allocates on the host as well. Neither of the two
# is a problem here, because the optimizer only ever touches the 154938 parameters (620 kB in
# `Float32`), whereas the forward and backward passes touch the whole batch (2048 images
# through 16 transformer blocks). Per optimizer step the parameters are uploaded and the
# gradient is downloaded once, i.e. about 1.2 MB of traffic; everything expensive happens on
# the device.
#
# Run it inside `screen`, through the wrapper next to it:
#
#   screen -S mnist
#   scripts/geometric_optimizers/run_mnist_cuda.sh                       # detach with C-a d, come back with -r
#
# or start it detached in one go, which is what an ssh session is for:
#
#   screen -dmS mnist scripts/geometric_optimizers/run_mnist_cuda.sh
#
# The wrapper only sets the environment variables below and launches this script; running
# `julia --project=scripts scripts/geometric_optimizers/mnist_cuda.jl` by hand inside a `screen` is the same thing.
#
# `screen` because the four trainings at `n_epochs = 500` are 58000 optimizer steps and this
# script has never been timed on a CUDA device — the ≈36 h in `WORKPLAN.md` is extrapolated
# from three epochs on an M4 Max through `mnist_metal.jl`, and the RTX 4090 should be well
# under that, but by how much is not known. Either way it is longer than an ssh session should
# be expected to survive. `tail -f mnist_cuda_report.txt` follows it from a second window; the
# first epoch line gives the per-epoch time, so the total is `4 · n_epochs ·` that.
#
# `CUDA` is part of `scripts/Project.toml`; it installs (but is not functional) on platforms
# without a CUDA device, so `mnist.jl` still works in the same environment. If no device is
# available this script runs on the host as well, so that it can be tested without a GPU —
# but at the 500 epochs of the original that is not a practical option.
#
# Apart from the device the setup is identical to `mnist.jl`: same network, same
# initialization (drawn on the host with the same seed) and the same four trainings, so the
# results of the two scripts are directly comparable.
#
# On device memory: `Zygote` keeps every intermediate of the forward pass alive for the
# backward pass, which at `L = 16` and a batch size of 2048 is roughly 80 MB per transformer
# block, i.e. about 3 GB including the backward pass itself. If that does not fit, reduce
# `batch_size` (the number of batches per epoch adapts automatically).
#
# ## The report
#
# Nobody watches a run this long, and the terminal it was started in is not where its results
# will be read, so everything the script learns is also written to a plain text file:
#
#   mnist_cuda_report.txt     the environment, the gradient checks, one line per epoch, the
#                             per-run summaries and the verdict — flushed after every line
#   mnist_cuda_losses.csv     one row per optimizer step: run, epoch, batch, step, loss
#   mnist_parameters.jld2     the trained parameters, the per-batch and per-epoch losses, the
#                             timings and the accuracies — rewritten after every configuration
#
# All three are self-contained: they can be copied off the machine and read without it. The
# first two need nothing but a text editor, which the `.jld2` does — JLD2 writes a subset of
# HDF5, so `h5py` opens it, but the parameters come back as Julia types. Because the report is
# flushed line by line, the CSV after every epoch and the `.jld2` after each of the four
# configurations, a run that dies in configuration three still leaves configurations one and
# two complete, together with every epoch of three up to the point it died. A configuration
# that throws is caught, reported and does not take the remaining ones down with it; a loss
# that turns non-finite ends that configuration early rather than spending the rest of its
# schedule on a run that cannot recover.
#
# The environment variables below override the schedule, so that the machine can be tried out
# before it is committed to the whole run:
#
#   MNIST_N_EPOCHS        epochs per configuration        (default 500; use 2 for a smoke test)
#   MNIST_BATCH_SIZE      images per batch                (default 2048)
#   MNIST_ACCURACY_EVERY  epochs between test evaluations (default 25, `0` disables them)
#   MNIST_REPORT          path of the report              (default mnist_cuda_report.txt)
#   MNIST_LOSSES          path of the loss CSV            (default mnist_cuda_losses.csv)
#   MNIST_OUTPUT          path of the `.jld2`             (default mnist_parameters.jld2)
#   MNIST_PROGRESS        print the per-batch progress line (default: only on a terminal)
#
# A smoke test — two epochs of each configuration, writing both files exactly as the full run
# does, and its epoch lines are what the schedule of the full run should be extrapolated from:
#
#   scripts/geometric_optimizers/run_mnist_cuda.sh --smoke

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!
# `GeometricOptimizers` 0.5.0 dropped `ParameterHandling` for `NeuralNetworkParameters`, which is the
# package that owns a parameter set and does its flattening. This script used to reach the old one
# through `GeometricOptimizers`, so it stopped loading at that release; these are the replacements.
# `freeparameters` is the leaf protocol -- it is `Y.A` for a `StiefelManifold`, by a method
# `GeometricOptimizers` registers, so this script no longer needs its own copy of that knowledge.
using NeuralNetworkParameters: NetworkParameters, flatten, flatten!, freeparameters,
                               parameterlayout, parameterrange, flatlength
using SimpleSolvers: Static
using LinearAlgebra: norm, I, Adjoint, Transpose
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using Printf: @printf, @sprintf
import CUDA, ForwardDiff, JLD2, MLDatasets, Random, Zygote

# --------------------------------------------------------------------------- device ---

const use_cuda = CUDA.functional()

# `to_device` is a `const` binding to a *function*, so the element types of the forward pass
# are still inferred.
const to_device = use_cuda ? CUDA.cu : identity
to_host(x::AbstractArray) = Array(x)
synchronize_device() = use_cuda ? CUDA.synchronize() : nothing

# --------------------------------------------------------------------------- report ---

# See the header: the run outlives the terminal it was started in, so everything it learns
# goes into a file as well. `report` flushes after every line — the point of the file is that
# it is complete up to the moment the process died, whatever killed it.

const report_path = get(ENV, "MNIST_REPORT", "mnist_cuda_report.txt")
const output_path = get(ENV, "MNIST_OUTPUT", "mnist_parameters.jld2")
const losses_path = get(ENV, "MNIST_LOSSES", "mnist_cuda_losses.csv")
const report_io = open(report_path, "w")

# The per-batch losses go into a CSV as well as into the `.jld2`. The `.jld2` is the file the
# *parameters* have to live in — it is what the other three MNIST scripts write, so the four
# stay comparable — but reading it needs Julia and this package, whereas the loss curve is the
# part of the run somebody will want to look at from wherever they happen to be. A row is 40
# bytes and the full run is 58000 of them.
const losses_io = open(losses_path, "w")
println(losses_io, "run,configuration,epoch,batch,step,loss")
flush(losses_io)

"""
    report(line)

Append `line` to the report file and flush it.
"""
function report(line::AbstractString = "")
    println(report_io, line)
    flush(report_io)
    nothing
end

"""
    announce(line)

Write `line` to both the terminal and the report. `stdout` is flushed as well, so that the
redirected output of a `nohup`ed run is worth following.
"""
function announce(line::AbstractString = "")
    println(line)
    flush(stdout)
    report(line)
end

timestamp() = Libc.strftime("%Y-%m-%d %H:%M:%S", time())

"""
    duration(seconds)

`seconds` as `h:mm:ss`, which is the only readable unit for the numbers in this script.
"""
function duration(seconds::Real)
    total = round(Int, seconds)
    @sprintf("%i:%02i:%02i", total ÷ 3600, (total ÷ 60) % 60, total % 60)
end

# Whether the individual batches are worth printing: on a terminal the progress line is how
# you see that the first run is moving at all, but redirected into a file its carriage returns
# are 58000 lines of noise, so there it is epochs only. `MNIST_PROGRESS` forces either way.
const to_terminal = parse(Bool, get(ENV, "MNIST_PROGRESS", string(stdout isa Base.TTY)))

# clear the progress line before something permanent is printed over it
clear_progress() = to_terminal && print("\r", " "^78, "\r")

"""
    attempt(f, fallback)

`f()`, or `fallback` if it throws. Used for the environment section only: a `CUDA` accessor
that was renamed upstream should not be what stops a run of this length.
"""
attempt(f::Base.Callable, fallback = "unknown") =
    try
        string(f())
    catch
        fallback
    end

function report_environment()
    report("mnist_cuda.jl — " * (use_cuda ? "CUDA" : "host") * " run")
    report("=" ^ 100)
    report()
    report("started     " * timestamp())
    report("host        " * gethostname())
    report("julia       " * string(VERSION) * ", " * string(Threads.nthreads()) *
           " thread(s)")
    report("project     " * string(Base.active_project()))
    report("script      " * @__FILE__)
    report("revision    " *
           attempt(() -> readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)) *
           attempt(
               () -> isempty(readchomp(`git -C $(@__DIR__) status --porcelain`)) ? "" :
                     " (dirty)", ""))
    report("package     GeometricOptimizers " *
           attempt(() -> pkgversion(GeometricOptimizers)) *
           ", CUDA.jl " * attempt(() -> pkgversion(CUDA)))
    if use_cuda
        report("device      " * attempt(() -> CUDA.name(CUDA.device())) *
               ", " * attempt(() -> string(CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB")) *
               ", capability " * attempt(() -> CUDA.capability(CUDA.device())))
        report("cuda        driver " * attempt(CUDA.driver_version) *
               ", runtime " * attempt(CUDA.runtime_version))
    else
        report("device      none — the network is evaluated on the host")
    end
    report()
end

# The high water mark of device memory, reported at the end. `CUDA.jl` pools its allocations,
# so this is the memory the process holds, not what a single step needs — which is the number
# that matters when deciding whether a second run fits alongside this one.
const peak_used = Ref(0)

# `CUDA.available_memory` is gone: 5.8 renamed it to `free_memory`, and the 6.x split into
# `CUDACore`/`CUDATools` forwards the new name only. Resolve it once, and let the number be the
# thing that is lost if it is missing again — the rest of `attempt`'s reasoning applies here
# too, except that this one is called from inside the epoch loop, so it takes the run with it.
const free_device_memory = !use_cuda ? () -> nothing :
                           isdefined(CUDA, :free_memory) ? CUDA.free_memory :
                           isdefined(CUDA, :available_memory) ? CUDA.available_memory :
                           () -> nothing

function note_device_memory()
    free = free_device_memory()
    free === nothing && return nothing
    peak_used[] = max(peak_used[], Int(CUDA.total_memory() - free))
    nothing
end

if use_cuda
    println("running on ", CUDA.name(CUDA.device()), " (",
        CUDA.totalmem(CUDA.device()) ÷ 1024^2, " MB)")
else
    @warn "no CUDA device available — the network is evaluated on the host instead"
end
report_environment()
println("writing the report to ", report_path, " and the parameters to ", output_path)

# ------------------------------------------------------------------- hyperparameters ---

const patch_length = 7      # MNIST images are 28×28, so the sequence length is (28÷7)² = 16
const n_heads = 7
const L = 16                # the number of transformer blocks
const add_connection = false
const T = Float32
const learning_rate = T(1e-3)
const momentum_coefficient = T(0.5)
const seed = 1234

# The three the header documents. The defaults are the schedule of the original script, so a
# run without any of them set is the run `mnist.jl` performs.
const batch_size = parse(Int, get(ENV, "MNIST_BATCH_SIZE", "2048"))
const n_epochs = parse(Int, get(ENV, "MNIST_N_EPOCHS", "500"))

# How often the test accuracy is evaluated during a training. A single evaluation is a forward
# pass over the 10000 test images, i.e. about a sixth of an epoch, so at the default this
# costs well under a percent — cheap for a learning curve instead of one final number, which
# is the difference between seeing *that* a configuration ended up where it did and seeing
# *when* it got there.
const accuracy_every = parse(Int, get(ENV, "MNIST_ACCURACY_EVERY", "25"))

const dim = patch_length^2                      # the transformer dimension
const seq_length = (28 ÷ patch_length)^2        # the number of patches
const n_classes = 10
const Dₕ = dim ÷ n_heads                        # the dimension of a single head

@assert dim % n_heads == 0

# The bars the runs are judged against, as in `mnist_metal_short.jl` — a report nobody is
# watching being written is worth little if reading it means recomputing by hand what the
# numbers were supposed to be. `chance_accuracy` is what a constant prediction gets;
# `accuracy_floor` is three times chance, low enough not to fail a genuinely slow optimizer
# and high enough that it cannot be reached without learning something.
const chance_accuracy = T(1 / n_classes)
const accuracy_floor = T(0.30)

# a run that is expected to learn has to reduce its average epoch loss by at least this fraction
const loss_decrease_floor = T(0.02)

# The loss of a trivial prediction, i.e. the plateau the unconstrained configuration is
# expected to sit at: `‖pred − out‖/‖out‖` with `pred` constant is `√(2·(1 − 1/n_classes))`.
# See the comment above `runs` — this is the `≈1.34` of the paper.
const trivial_loss = sqrt(T(2) * (one(T) - chance_accuracy))
const trivial_loss_tolerance = T(0.05)

# How far the Stiefel parameters may drift off the manifold before the run is called broken.
# `mnist_metal_short.jl` uses `1e-4`, which is the right order for round-off accumulated over a
# few hundred steps — one epoch on the host gives `1.6e-6` for `GradientMethod` and `2.5e-6`
# for `MomentumMethod`. `Adam` gave `4.6e-4` over the same 29 steps, i.e. two orders of
# magnitude more, and that is not round-off: `Adam` rescales each coordinate of the update to
# about the learning rate, so where the raw gradient has vanished through the 16 blocks its
# steps are far larger than the other two methods', and a retraction departs from the manifold
# with the step it is given. The drift is therefore expected to be method- and step-dependent,
# and over 14500 steps it will be larger still. What the report is for is the *measurement*, so
# the bar here is set where the parameters have stopped being orthonormal in any useful sense
# rather than where round-off ends, and the trend is recorded alongside the accuracy below.
const orthonormality_tolerance = T(1e-2)

# The accuracy floor only means something once the schedule is long enough to reach it: three
# epochs on the M4 Max gave 0.2050 for `GradientMethod`, so a short run would fail it while
# behaving perfectly. Below this many epochs the accuracy is reported and not judged.
const accuracy_floor_min_epochs = 50

# `Float32` and a directional derivative of order ‖g‖/√n leave a few digits of room; anything
# beyond this is a wrong derivative rather than round-off.
const gradient_tolerance = 1e-4

# ----------------------------------------------------------------------------- data ---

@doc raw"""
    split_and_flatten(input, patch_length)

Rearrange a batch of images into *flattened patches*, i.e. turn an ``(N, N, k)`` array into
an ``(\mathtt{patch\_length}^2, (N \div \mathtt{patch\_length})^2, k)`` array.

This is the equivalent of `GeometricMachineLearning.split_and_flatten` and produces the same
ordering: the patches are numbered column-major over the image and the entries within a
patch are numbered column-major as well.
"""
function split_and_flatten(input::AbstractArray{<:Number, 3}, patch_length::Integer)
    @assert size(input, 1) == size(input, 2)
    @assert size(input, 1) % patch_length == 0
    n = size(input, 1) ÷ patch_length
    # (i_red, patch_row, j_red, patch_column, k) → (i_red, j_red, patch_row, patch_column, k)
    output = permutedims(reshape(input, patch_length, n, patch_length, n, size(input, 3)), (
        1, 3, 2, 4, 5))
    reshape(output, patch_length^2, n^2, size(input, 3))
end

"""
    onehotbatch(target)

Turn a vector of labels (`0` to `9`) into a `10`×`length(target)` matrix of unit vectors.
"""
function onehotbatch(target::AbstractVector{<:Integer})
    output = zeros(T, n_classes, length(target))
    for (k, label) in pairs(target)
        output[label + 1, k] = one(T)
    end
    output
end

# ----------------------------------------------------------------------- parameters ---

# The parameters are stored in a flat `NamedTuple` *on the host*. Note that only the
# projections of the attention layers are put on the `StiefelManifold`; the parameters of the
# `ResNetLayer`s and of the classification layer are ordinary arrays.
const parameter_layout = [[Symbol(s, "_", l, "_", h) => (dim, Dₕ) for l in 1:L
                           for s in ("PQ", "PK", "PV") for h in 1:n_heads]
                          vcat([[
                                    Symbol("Wres_", l) => (dim, dim), Symbol("bres_", l) =>
                                        (dim,)] for l in 1:L]...)
                          [:Wclass => (n_classes, dim)]]

const parameter_keys = Tuple(first.(parameter_layout))
const parameter_sizes = last.(parameter_layout)
const n_attention_parameters = 3 * n_heads * L

# the position of a parameter within `parameter_layout`
function attention_index(kind::Integer, l::Integer, h::Integer)
    (l - 1) * 3 * n_heads + (kind - 1) * n_heads + h
end
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

# For the forward pass the parameters are regrouped into vectors of concrete element type.
# Doing this outside of the differentiated function keeps both the forward pass and `Zygote`
# type stable.
function regroup(get_parameter::Base.Callable)
    (Q = [get_parameter(attention_index(1, l, h)) for l in 1:L for h in 1:n_heads],
        K = [get_parameter(attention_index(2, l, h)) for l in 1:L for h in 1:n_heads],
        V = [get_parameter(attention_index(3, l, h)) for l in 1:L for h in 1:n_heads],
        Wres = [get_parameter(resnet_index(l)) for l in 1:L],
        bres = [get_parameter(resnet_index(l) + 1) for l in 1:L],
        Wclass = get_parameter(classification_index))
end

"""
    regroup_host(v)

Regroup the *flattened* parameters without touching the device. The element type is kept
general so that `check_gradient` can differentiate through it — `ForwardDiff.Dual`s cannot be
handed to `cuBLAS`, so the reference derivative has to be computed on the host.
"""
function regroup_host(v::AbstractVector{<:Number})
    regroup(i -> reshape(v[parameter_ranges[i]], parameter_sizes[i]...))
end

# The parameters are moved to the device in a single transfer and split up there; the 353
# individual parameters are far too small for a transfer each.
const host_parameters = zeros(T, n_parameters)
const device_parameters = to_device(zeros(T, n_parameters))
const device_gradient = to_device(zeros(T, n_parameters))

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

"""
    flatten_parameters!(v, ps)

Write the parameter `NamedTuple` into the flat vector `v`, in the same order in which
`NeuralNetworkParameters.flatten` writes it (which is the order of `parameter_ranges`).
"""
function flatten_parameters!(v::AbstractVector{T}, ps::NetworkParameters)
    flatten!(v, ps, parameter_flat_layout)
end

function regroup_device(v::AbstractVector{T})
    copyto!(device_parameters, v)
    regroup(i -> reshape(device_parameters[parameter_ranges[i]], parameter_sizes[i]...))
end

function regroup_device(ps::NetworkParameters)
    regroup_device(flatten_parameters!(host_parameters, ps))
end

# ---------------------------------------------------------------------------- model ---

"""
    mat_tensor_mul(A, x)

Multiply `A` onto every matrix stored in `x`, i.e. parallelize over the third axis.
"""
function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number, 3})
    reshape(A * reshape(x, size(x, 1), :), size(A, 1), size(x, 2), size(x, 3))
end

# `Zygote` differentiates `mat_tensor_mul` by pulling the cotangent through the `reshape`, and
# the cotangent of a `Q` that enters `batched_mul` through `batched_transpose` is a *lazy*
# `BatchedTranspose`. Reshaping that produces a `ReshapedArray` which `cuBLAS` does not
# recognize, so the multiplication of the backward pass falls back to the generic method of
# `LinearAlgebra` — which indexes the arrays element by element and therefore errors on the
# device. The rule below writes the backward pass by hand and materializes such cotangents
# first; on the host it avoids the same (there merely slow) fallback.
_dense(Δ::AbstractArray) = Δ
_dense(Δ::BatchedAdjOrTrans) = permutedims(parent(Δ), (2, 1, 3))    # all arrays here are real
_dense(Δ::Union{Adjoint, Transpose}) = permutedims(parent(Δ), (2, 1))

Zygote.@adjoint function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number, 3})
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
function predict(ps::NamedTuple, input::AbstractArray{<:Number, 3})
    x = input
    for l in 1:L
        # the multi head attention layer
        heads = ntuple(n_heads) do h
            i = (l - 1) * n_heads + h
            Q = mat_tensor_mul(transpose(ps.Q[i]), x)
            K = mat_tensor_mul(transpose(ps.K[i]), x)
            V = mat_tensor_mul(transpose(ps.V[i]), x)
            batched_mul(V, softmax(batched_mul(batched_transpose(Q), K) ./ sqrt(T(dim)); dims = 1))
        end
        y = add_connection ? x + reduce(vcat, heads) : reduce(vcat, heads)
        # the ResNet layer
        x = y + tanh.(mat_tensor_mul(ps.Wres[l], y) .+ ps.bres[l])
    end
    # the classification layer picks the last column and applies softmax
    softmax(ps.Wclass * x[:, end, :]; dims = 1)
end

"""
    network_loss(ps, input, output)

The equivalent of `GeometricMachineLearning.FeedForwardLoss`.
"""
function network_loss(ps::NamedTuple, input, output)
    norm(predict(ps, input) - output) / norm(output)
end

"""
    accuracy(ps, input, output)

The ratio of correctly classified images, i.e. the equivalent of
`GeometricMachineLearning.accuracy`. Here `ps` are the parameters as stored by the optimizer
and `input`/`output` live on the host; only one chunk at a time is moved to the device. The
`argmax`es are taken on the host, as an `argmax` per column would be a scalar index on the
device.
"""
function accuracy(ps::NetworkParameters, input, output; chunk_size = batch_size)
    regrouped = regroup_device(ps)
    correct = 0
    for k in Iterators.partition(axes(input, 3), chunk_size)
        prediction = to_host(predict(regrouped, to_device(input[:, :, k])))
        for (j, jₖ) in pairs(k)
            correct += argmax(view(prediction, :, j)) == argmax(view(output, :, jₖ))
        end
    end
    correct / size(input, 3)
end

# --------------------------------------------------------------- objective & gradient ---

# The `Optimizer` calls the objective on the parameter `NamedTuple` and `∇F!` on the
# *flattened* parameters, both of which live on the host. The current batch is on the device.
const current_batch = Ref{Tuple{AbstractArray{T, 3}, AbstractMatrix{T}}}()

function F(ps::NetworkParameters)
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
be multiplied by `cuBLAS`, so this part has to run on the host anyway.
"""
function check_gradient(ps::NetworkParameters)
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

# ------------------------------------------------------------- checks on the outcome ---

@doc raw"""
    orthonormality_error(ps)

The worst ``\|Y^TY - I\|`` over the attention projections, i.e. how far the parameters have
drifted off the Stiefel manifold. `NaN` if they are not on a manifold to begin with.
"""
function orthonormality_error(ps::NetworkParameters)
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
function parameters_are_sound(ps::NetworkParameters)
    all(values(ps)) do p
        A = freeparameters(p)
        eltype(A) === T && all(isfinite, A)
    end
end

# -------------------------------------------------------------------------- training ---

"""
    train(stiefel, algorithm, input, output, test_input, test_output; label, n_epochs)

Train one configuration and return everything worth keeping about it: the parameters, the
per-batch and per-epoch losses, the per-epoch wall clock, the test accuracies evaluated along
the way and why the run ended.

One line per epoch goes into the report as the epoch finishes, so the file describes a run
that is still in progress just as well as a finished one. `label` is what those lines are
prefixed with.

A non-finite epoch loss ends the run: the parameters cannot come back from it, and the point
of noticing here is not to spend the remaining epochs proving that.
"""
function train(
        stiefel::Bool, algorithm::GeometricOptimizers.OptimizerMethod, input, output,
        test_input, test_output; label::AbstractString, run_index::Integer, run_name::AbstractString,
        n_epochs = n_epochs, learning_rate = learning_rate)
    rng = Random.Xoshiro(seed)
    ps = initial_parameters(rng, stiefel)

    n_batches = size(input, 3) ÷ batch_size
    current_batch[] = (
        to_device(input[:, :, 1:batch_size]), to_device(output[:, 1:batch_size]))

    # Note that the learning rate is supplied through the line search: the *methods* only
    # determine the direction. `Static(learning_rate)` is what `Optimizer` defaults to for
    # these three methods anyway; it is written out so that the rate is visible right here.
    optimizer = Optimizer(
        ps, F; (∇F!) = ∇F!, algorithm = algorithm, linesearch = Static(learning_rate))
    state = OptimizerState(algorithm, ps)
    initialize_state!(state)

    losses = T[]
    epoch_losses = T[]
    epoch_times = Float64[]
    accuracy_epochs = Int[]
    accuracies = T[]
    orthonormalities = T[]
    stopped = ""

    synchronize_device()
    initial_time = time()
    for epoch in 1:n_epochs
        epoch_start = time()
        # `solve!` cannot be used here: it optimizes a *fixed* objective until it converges,
        # whereas the objective changes with every batch.
        batches = Iterators.take(
            Iterators.partition(Random.shuffle(rng, axes(input, 3)), batch_size), n_batches)
        epoch_loss = zero(T)
        for (i, batch) in pairs(collect(batches))
            # the batch is gathered on the host and uploaded; at 6.4 MB per batch this is
            # negligible next to the forward and backward passes
            current_batch[] = (to_device(input[:, :, batch]), to_device(output[:, batch]))
            increase_iteration_number!(state)
            solver_step!(ps, state, optimizer)
            GeometricOptimizers.update!(state, optimizer, ps)
            loss = F(ps)
            push!(losses, loss)
            epoch_loss += loss / n_batches
            println(losses_io, run_index, ",\"", run_name, "\",",
                epoch, ",", i, ",", length(losses), ",", loss)
            to_terminal && @printf("\r  epoch %4i/%i, batch %3i/%i, loss %.5f",
                epoch, n_epochs, i, n_batches, loss)
        end
        synchronize_device()
        flush(losses_io)        # the CSV is complete to the last finished epoch, as the report is
        push!(epoch_losses, epoch_loss)
        push!(epoch_times, time() - epoch_start)
        note_device_memory()

        line = @sprintf("%s  epoch %4i/%-4i  avg loss %8.5f  last %8.5f  %6.1f s  elapsed %s",
            label, epoch, n_epochs, epoch_loss, last(losses),
            last(epoch_times),
            duration(time() - initial_time))
        # the last epoch is always evaluated, so that every run ends on a measured accuracy
        if accuracy_every > 0 && (epoch % accuracy_every == 0 || epoch == n_epochs)
            score = T(accuracy(ps, test_input, test_output))
            push!(accuracy_epochs, epoch)
            push!(accuracies, score)
            line *= @sprintf("  test accuracy %.4f", score)
            # alongside it, how far off the manifold the parameters have drifted by now. A
            # single number at the end cannot distinguish round-off that accumulates with the
            # step count from a retraction that gave up at some point; the series can.
            drift = orthonormality_error(ps)
            push!(orthonormalities, drift)
            isnan(drift) || (line *= @sprintf("  ‖YᵀY-I‖ %.1e", drift))
        end
        clear_progress()
        announce(line)

        if !isfinite(epoch_loss)
            stopped = "the loss is $epoch_loss after epoch $epoch"
            announce("  " * label * "  stopping this configuration: " * stopped)
            break
        end
    end
    synchronize_device()
    total_time = time() - initial_time

    (parameters = ps, losses = losses,
        epoch_losses = epoch_losses, epoch_times = epoch_times,
        accuracy_epochs = accuracy_epochs, accuracies = accuracies, orthonormalities = orthonormalities,
        total_time = total_time, stopped = stopped)
end

# ------------------------------------------------------------------------------ data ---

const script_start = time()

println("loading MNIST ...")
train_x, train_y = MLDatasets.MNIST(split = :train)[:]
test_x, test_y = MLDatasets.MNIST(split = :test)[:]

# the data set stays on the host; the batches are uploaded one at a time
const train_input = split_and_flatten(T.(train_x), patch_length)
const train_output = onehotbatch(train_y)
const test_input = split_and_flatten(T.(test_x), patch_length)
const test_output = onehotbatch(test_y)

@assert size(train_input) == (dim, seq_length, size(train_x, 3))

const n_batches = size(train_input, 3) ÷ batch_size

announce(@sprintf("%i parameters, batch size %i, %i batches per epoch, %i epochs per configuration (%i steps)",
    n_parameters, batch_size, n_batches, n_epochs, n_epochs * n_batches))
accuracy_every > 0 &&
    announce(@sprintf("the test accuracy is evaluated every %i epoch(s)", accuracy_every))
announce()

# -------------------------------------------------------------------- gradient check ---

# `mnist.jl` only checks the Stiefel case. The regular case is checked here as well: that
# configuration is expected to stall (see the comment above `runs`), and the check separates
# the stall the experiment is about — a vanishing gradient in the network — from a wrong `∇F!`
# on plain arrays, which would look the same from the outside. Reading that distinction off
# the report afterwards is not possible if only one of the two was ever measured.
current_batch[] = (
    to_device(train_input[:, :, 1:batch_size]), to_device(train_output[:, 1:batch_size]))

const gradient_errors = map((true, false)) do stiefel
    # not `error`, which would shadow `Base.error` for the rest of the closure
    relative_error = check_gradient(initial_parameters(Random.Xoshiro(seed), stiefel))
    announce(@sprintf("relative error of ∇F! vs a host directional derivative, %s weights: %.2e",
        stiefel ? "Stiefel" : "regular", relative_error))
    relative_error
end

const gradient_ok = all(e -> isfinite(e) && e < gradient_tolerance, gradient_errors)
if !gradient_ok
    announce(@sprintf("*** the gradient disagrees with the host by more than %.0e — the trainings below are not meaningful ***",
        gradient_tolerance))
    @warn "∇F! disagrees with the host directional derivative"
end
announce()

# ------------------------------------------------------------------------------- run ---

# The same four trainings as in the original script. `learns` is what the configuration is
# expected to do. `Adam` takes the *element type* of the parameters, not a learning rate, and
# it is not converted the way `MomentumMethod` is, so `Adam(T)` is what dispatches to the
# `Float32` cache.
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
    (name = "Stiefel weights, Adam", stiefel = true, learns = true, algorithm = Adam(T)),
    (name = "regular weights, Adam", stiefel = false, learns = false, algorithm = Adam(T)),
    (name = "Stiefel weights, gradient", stiefel = true,
        learns = true, algorithm = GradientMethod()),
    (name = "Stiefel weights, momentum", stiefel = true,
        learns = true, algorithm = MomentumMethod(momentum_coefficient))
]

"""
    save_results(results)

Write what has been trained so far to `output_path`. Called after every configuration rather
than once at the end: a file that only exists when all four have finished is worth nothing to
a run that dies in the third. The keys are those of `mnist.jl` — `parameters1`, `losses1`,
`total_time1`, `accuracy1`, … — plus the names, the per-epoch series and the accuracy history,
so anything that reads the output of the original script still reads this one.
"""
function save_results(results)
    output = Dict{String, Any}("n_epochs" => n_epochs, "batch_size" => batch_size,
        "n_batches" => n_batches, "gradient_errors" => collect(gradient_errors),
        "n_results" => length(results))
    for (i, result) in pairs(results)
        output["name$i"] = result.name
        output["parameters$i"] = result.parameters
        output["losses$i"] = result.losses
        output["epoch_losses$i"] = result.epoch_losses
        output["epoch_times$i"] = result.epoch_times
        output["accuracy_epochs$i"] = result.accuracy_epochs
        output["accuracies$i"] = result.accuracies
        output["orthonormalities$i"] = result.orthonormalities
        output["orthonormality$i"] = result.orthonormality
        output["total_time$i"] = result.total_time
        output["accuracy$i"] = result.accuracy
    end
    JLD2.save(output_path, output)
end

results = []
failures = Tuple{String, String}[]

for (j, run) in pairs(runs)
    label = @sprintf("[%i/%i %-25s]", j, length(runs), run.name)
    announce(@sprintf("%s starting at %s", label, timestamp()))
    try
        trained = train(
            run.stiefel, run.algorithm, train_input, train_output, test_input, test_output;
            label = label, run_index = j, run_name = run.name, n_epochs = n_epochs)
        score = T(accuracy(trained.parameters, test_input, test_output))
        push!(results,
            (name = run.name, stiefel = run.stiefel, learns = run.learns,
                parameters = map(freeparameters, trained.parameters), losses = trained.losses,
                epoch_losses = trained.epoch_losses, epoch_times = trained.epoch_times,
                accuracy_epochs = trained.accuracy_epochs, accuracies = trained.accuracies,
                orthonormalities = trained.orthonormalities,
                total_time = trained.total_time, accuracy = score, stopped = trained.stopped,
                orthonormality = orthonormality_error(trained.parameters),
                sound = parameters_are_sound(trained.parameters)))
        announce(@sprintf("%s done in %s (%.2f s/step), test accuracy %.4f", label,
            duration(trained.total_time),
            trained.total_time / max(1, length(trained.losses)), score))
        save_results(results)
        announce(@sprintf("%s written to %s", label, output_path))
    catch e
        # An `InterruptException` is the one exception that means *stop*, not *skip*; anything
        # else is this configuration's problem and the remaining ones still deserve the GPU.
        e isa InterruptException && rethrow()
        message = sprint(showerror, e)
        push!(failures, (run.name, message))
        clear_progress()
        announce(@sprintf("%s FAILED: %s", label, first(split(message, '\n'))))
        report(message)
        report(sprint(Base.show_backtrace, catch_backtrace()))
        @error "$(run.name) failed" exception = (e, catch_backtrace())
    end
    announce()
end

# --------------------------------------------------------------------------- verdict ---

"""
    verdict(result)

Whether a configuration did what it is supposed to do, and if not, why.

Two criteria apply to every run — the parameters survived and, on a manifold, stayed on it.
The remaining two depend on `learns`: a configuration that is expected to learn has to reduce
its loss and classify above `accuracy_floor`, while the unconstrained one is expected to
collapse onto the trivial prediction and is checked against that plateau instead. Note that
this makes *learning* a failure for the latter — not because learning would be bad, but
because it would mean this script is no longer running the experiment of the paper.

The bars are those of the *full* run. A short schedule is judged on what it can show: the loss
decrease is read off the per-batch series when there are not two epochs to compare, and the
accuracy is not judged at all below `accuracy_floor_min_epochs`, where a configuration that is
working still has no chance of clearing the floor.
"""
function verdict(result)
    isempty(result.epoch_losses) && return "FAILED: no epoch finished"
    reasons = String[]
    isempty(result.stopped) || push!(reasons, result.stopped)
    result.sound || push!(reasons, "parameters not finite or no longer $T")
    if result.stiefel && !(result.orthonormality ≤ orthonormality_tolerance)
        push!(reasons, @sprintf("off the manifold (‖YᵀY-I‖ = %.1e)", result.orthonormality))
    end
    # With a single epoch `first(epoch_losses)` *is* `last(epoch_losses)`, so the decrease has
    # to be read off the batches instead — otherwise every short run is "flat" by construction.
    first_loss = length(result.epoch_losses) ≥ 2 ? first(result.epoch_losses) :
                 first(result.losses)
    last_loss = last(result.epoch_losses)
    if result.learns
        if !(last_loss < (1 - loss_decrease_floor) * first_loss)
            push!(reasons, @sprintf("loss flat (%.3f → %.3f)", first_loss, last_loss))
        end
        if length(result.epoch_losses) ≥ accuracy_floor_min_epochs &&
           result.accuracy < accuracy_floor
            push!(reasons,
                @sprintf("accuracy %.4f below the floor of %.2f%s", result.accuracy,
                    accuracy_floor,
                    result.accuracy < 2 * chance_accuracy ? ", i.e. at chance" : ""))
        end
    elseif !(abs(last_loss - trivial_loss) ≤ trivial_loss_tolerance)
        push!(reasons,
            @sprintf("expected the trivial-prediction plateau of %.3f, got %.3f",
                trivial_loss, last_loss))
    end
    isempty(reasons) ? "ok" : "FAILED: " * join(reasons, "; ")
end

const verdicts = map(verdict, results)

announce("=" ^ 100)
announce(@sprintf("finished %s after %s — %i epochs (%i steps) per configuration, batch size %i",
    timestamp(), duration(time() - script_start), n_epochs, n_epochs * n_batches,
    batch_size))
announce()
announce(@sprintf("%-26s %8s %8s %10s %10s %10s   %s",
    "run", "loss 1", "loss N", "accuracy", "‖YᵀY-I‖", "time", "verdict"))
# `epoch_losses` is empty only if a run never finished an epoch; the verdict says so, and the
# table has to stay printable rather than throw at the end of a day of work
loss_cell(losses, which) = isempty(losses) ? "—" : @sprintf("%8.3f", which(losses))

for (result, v) in zip(results, verdicts)
    n_done = length(result.epoch_losses)
    announce(@sprintf("%-26s %8s %8s %10.4f %10s %10s   %s", result.name,
        loss_cell(result.epoch_losses, first), loss_cell(result.epoch_losses, last), result.accuracy,
        isnan(result.orthonormality) ? "—" : (@sprintf "%.1e" result.orthonormality),
        duration(result.total_time),
        v * (n_done < n_epochs ? " [$n_done of $n_epochs epochs]" : "")))
end
for (name, message) in failures
    announce(@sprintf("%-26s %8s %8s %10s %10s %10s   THREW: %s",
        name, "—", "—", "—", "—", "—",
        first(split(message, '\n'))))
end

# the learning curves, so that the shape of each run is in the report and not only in the
# `.jld2` — this is what says whether a configuration was still improving when it ran out
if accuracy_every > 0
    announce()
    announce("test accuracy over the run, as epoch:accuracy —")
    for result in results
        isempty(result.accuracies) && continue
        announce(@sprintf("%-26s %s", result.name,
            join(
                (@sprintf("%i:%.4f", e, a)
                for (e, a) in zip(result.accuracy_epochs, result.accuracies)),
                "  ")))
    end

    # The same series for the drift off the manifold. Whether this grows with the step count or
    # settles is the difference between accumulated round-off and a retraction that stopped
    # working, and it cannot be read off the single number in the table above.
    announce()
    announce("‖YᵀY-I‖ over the run, as epoch:drift —")
    for result in results
        (isempty(result.orthonormalities) || all(isnan, result.orthonormalities)) &&
            continue
        announce(@sprintf("%-26s %s", result.name,
            join(
                (@sprintf("%i:%.1e", e, d)
                for (e, d) in zip(result.accuracy_epochs, result.orthonormalities)),
                "  ")))
    end
end

const n_passed = count(==("ok"), verdicts)
const threw = isempty(failures) ? "" : @sprintf(", %i threw", length(failures))
const memory_note = use_cuda ?
                    @sprintf(" Peak device memory %i MB of %i MB.",
    peak_used[] ÷ 1024^2, Int(CUDA.totalmem(CUDA.device())) ÷ 1024^2) : ""

announce()
announce(@sprintf("%i of %i configurations behave as expected%s. Gradient check %s.%s",
    n_passed, length(runs), threw, gradient_ok ? "ok" : "FAILED", memory_note))
announce(@sprintf("bars applied: ‖YᵀY-I‖ ≤ %.0e, loss down by ≥ %.0f%%, %s.",
    orthonormality_tolerance, 100 * loss_decrease_floor,
    n_epochs ≥ accuracy_floor_min_epochs ?
    @sprintf("accuracy ≥ %.2f", accuracy_floor) :
    @sprintf("accuracy reported but not judged below %i epochs", accuracy_floor_min_epochs)))
announce()
announce("`regular weights, Adam` is the baseline of the paper and is expected to stall at the")
announce(@sprintf("trivial-prediction plateau of %.3f at an accuracy of %.2f; without the Stiefel constraint",
    trivial_loss, chance_accuracy))
announce("the gradient vanishes through the 16 unnormalized transformer blocks. It is a failure above")
announce("only if it did something else.")

save_results(results)
announce()
announce("parameters:   " * abspath(output_path))
announce("loss curves:  " * abspath(losses_path))
announce("this report:  " * abspath(report_path))
close(losses_io)
close(report_io)
