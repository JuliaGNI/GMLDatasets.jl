# The cost of one optimizer step of the MNIST transformer, without the network.
#
# The parameter set is the repetition trainer's: 16 layers × 7 heads × {Q, K, V} weights on
# St(7, 49), 16 ResNet weights and biases and the classifier, 154 938 numbers in `Float32`. The
# reverse pass is replaced by a copy of a fixed gradient, and the step is the trainer's own,
# `training_step!` of `training_step.jl`. So what is timed is the optimizer, in the schema-v4
# categories of `scripts/revision/README.md`: direction and retraction. In the trainer these run on
# the host (only the reverse pass is on the device), so the figures here are the Direction and
# Retraction columns of the paper's cost tables, up to the host they ran on.
#
#   julia --project=scripts/geometric_optimizers/optimizer_step_cost -e 'using Pkg; Pkg.instantiate()'
#   julia --project=scripts/geometric_optimizers/optimizer_step_cost scripts/geometric_optimizers/optimizer_step_cost/optimizer_step_cost.jl
#
# ENV: BENCH_THREADS (BLAS threads; default Julia's), BENCH_STEPS (default 200).
#
# The environment next to this file holds only what `training_step.jl` needs, so that it resolves
# while the scripts environment waits for GeometricOptimizers 0.9.
#
# Apple M4 Max, Julia 1.13.0, 10 BLAS threads, 200 steps (ms per step; allocation per step):
#
#   Geometric Adam        direction 2.02  retraction 3.12  (1 retraction)       8.9 MiB
#   Scalar-moment Adam    direction 1.61  retraction 2.93  (one per leaf, 369)  5.2 MiB
#   Riemannian gradient   direction 1.33  retraction 2.93                       5.6 MiB
#   Riemannian momentum   direction 1.31  retraction 2.94                       5.6 MiB
#   Standard Adam         direction 0.81  retraction 0.10                       4.5 MiB

using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters, parameterlayout, flatlength
using LinearAlgebra: BLAS
using Random, Printf

include(joinpath(@__DIR__, "..", "training_step.jl"))
include(joinpath(@__DIR__, "..", "..", "revision", "scalar_moment_adam.jl"))

haskey(ENV, "BENCH_THREADS") && BLAS.set_num_threads(parse(Int, ENV["BENCH_THREADS"]))
const STEPS = parse(Int, get(ENV, "BENCH_STEPS", "200"))
const T = Float32
const L, H, N, n = 16, 7, 49, 7

# The trainer's parameter set, with its initialisation: the attention weights on the Stiefel
# manifold or, for standard Adam, unconstrained.
function parameters(stiefel::Bool; seed = 1234)
    rng = Xoshiro(seed)
    Random.seed!(seed)      # the random completion of every global section
    leaves = Pair{Symbol, Any}[]
    for l in 1:L, h in 1:H, m in (:Q, :K, :V)
        push!(leaves, Symbol(m, l, "_", h) => stiefel ? rand(rng, StiefelManifold{T}, N, n) :
                                               randn(rng, T, N, n) ./ T(10))
    end
    for l in 1:L
        push!(leaves, Symbol(:W, l) => randn(rng, T, N, N) ./ T(10))
        push!(leaves, Symbol(:b, l) => zeros(T, N))
    end
    push!(leaves, :Wclass => randn(rng, T, 10, N) ./ T(10))
    NetworkParameters(NamedTuple(leaves))
end

# Two fixed gradient fields over the flat set, alternated as two minibatches would be.
const NFLAT = flatlength(parameterlayout(parameters(true)))
const G = (randn(Xoshiro(1), T, NFLAT) ./ 100, randn(Xoshiro(2), T, NFLAT) ./ 100)
const current = Ref(1)
∇F!(g, v) = copyto!(g, G[current[]])

const CONFIGURATIONS = (
    ("Geometric Adam", true, Adam()),
    ("Scalar-moment Adam", true, scalar_moment_adam_method()),
    ("Riemannian gradient", true, GradientMethod()),
    ("Riemannian momentum", true, MomentumMethod(; α = 0.5)),
    ("Standard Adam", false, Adam()))

println("GeometricOptimizers $(pkgversion(GeometricOptimizers)), Julia $VERSION, ",
    "$(BLAS.get_num_threads()) BLAS threads, $STEPS steps, $(Sys.cpu_info()[1].model)")
@printf("%-20s %10s %11s %12s %14s\n", "ms per step", "direction", "retraction",
    "retractions", "alloc (MiB)")
for (name, stiefel, method) in CONFIGURATIONS
    timer = PhaseTimer(; phases = (:gradient, :optimizer_state_direction, :retraction_application))
    ps = parameters(stiefel)
    step = TrainingStep(ps, ∇F!, method, T(1e-3); observer = timer)
    for _ in 1:5
        training_step!(ps, step)
    end
    empty!(timer)
    GC.gc()
    bytes = @allocated for k in 1:STEPS
        current[] = 1 + k % 2
        training_step!(ps, step)
    end
    ms(p) = 1.0e-6 * get(timer.exclusive, p, UInt64(0)) / STEPS
    @printf("%-20s %10.2f %11.2f %12.0f %14.2f\n", name, ms(:optimizer_state_direction),
        ms(:retraction_application), get(timer.calls, :retraction_application, 0) / STEPS,
        bytes / STEPS / 2^20)
end
