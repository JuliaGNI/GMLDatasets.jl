# Regression for `training_step.jl`, the optimizer step of the image trainer.
#
# What this file is *not* about: whether `optimization_step!` steps a method, a composite or a leaf
# correctly, or reports its retraction to the observer. Those are upstream's, and
# `GeometricOptimizers/test/integration/training_optimizer.jl` and
# `GeometricOptimizers/test/optimizers/composite_method.jl` pin them.
#
# What is local:
#
#   * the gradient buffer: the flat gradient `∇F!` writes is read back into the parameter-shaped
#     buffer in the order the parameters flatten in, the `StiefelManifold` leaf included;
#   * every step is built from the gradient of *its own* minibatch, on the configurations the trainer
#     runs: the step that consumed the previous minibatch's gradient is what this port removed;
#   * the three timing categories: one gradient, one direction and one retraction interval per step,
#     on the whole-tree and on the composite path.
#
#   julia --project=scripts scripts/geometric_optimizers/test_training_step.jl

using Test
using Random
using GeometricOptimizers
using GeometricOptimizers: EventLog
using NeuralNetworkParameters: NetworkParameters, flatten, flatlength, parameterlayout

include(joinpath(@__DIR__, "step_timing.jl"))
include(joinpath(@__DIR__, "training_step.jl"))
include(joinpath(@__DIR__, "..", "revision", "scalar_moment_adam.jl"))

# one leaf of each of the three shapes the trainer's set uses: a Stiefel attention projection, a bias
# vector, a ResNet weight matrix
const Y₀ = rand(Random.Xoshiro(1), StiefelManifold{Float64}, 5, 2)
const v₀ = rand(Random.Xoshiro(2), 3)
const W₀ = rand(Random.Xoshiro(3), 4, 3)
make_parameters() = NetworkParameters((Y = copy(Y₀), v = copy(v₀), W = copy(W₀)))
const n = flatlength(parameterlayout(make_parameters()))

# Two minibatches, as two ambient-gradient fields over the flat set that the trainer's `∇F!` reads
# from its `current_batch[]`; `calls` counts the reverse passes.
const G = (randn(Random.Xoshiro(4), n), randn(Random.Xoshiro(5), n))
const current = Ref(1)
const calls = Ref(0)
∇F!(g, v) = (calls[] += 1; copyto!(g, G[current[]]); g)

const METHODS = (geometric_adam = Adam(), scalar_moment_adam = scalar_moment_adam_method(),
    gradient = GradientMethod(), momentum = MomentumMethod())

@testset "the gradient buffer is the flat gradient in the shape of the parameters" begin
    ps = make_parameters()
    step = TrainingStep(ps, ∇F!, Adam(), 1.0e-3; observer = NoStepObserver())
    current[] = 1
    training_step!(ps, step)
    dp = step.euclidean_gradient
    @test dp.Y isa Matrix{Float64} && size(dp.Y) == (5, 2)
    @test flatten(dp)[1] == G[1]
    @test vec(dp.Y) == G[1][1:10]
end

@testset "each step is built from its own minibatch: $name" for (name, method) in pairs(METHODS)
    # Two steps through the batches (1, 2) against a first step on batch 1 followed by a step from
    # the same state on batch 2 under a fresh optimizer cannot be compared, the state differs; so the
    # second step is compared against the same two steps with batch 2 replaced by batch 1. If the
    # second step reused the first step's gradient, the two runs would end at the same point.
    function two_steps(batches)
        Random.seed!(7)         # the random completion of every global section
        ps = make_parameters()
        step = TrainingStep(ps, ∇F!, method, 1.0e-2; observer = NoStepObserver())
        calls[] = 0
        for b in batches
            current[] = b
            training_step!(ps, step)
        end
        ps
    end
    a = two_steps((1, 2))
    @test calls[] == 2
    b = two_steps((1, 1))
    @test flatten(a)[1] != flatten(b)[1]
    # and the first step is the same in both
    @test flatten(two_steps((1,)))[1] == flatten(two_steps((1,)))[1]
end

@testset "one interval of each category per step: $name" for (name, method) in pairs(METHODS)
    log = EventLog()
    ps = make_parameters()
    step = TrainingStep(ps, ∇F!, method, 1.0e-3; observer = log)
    training_step!(ps, step)
    retraction = [(:retraction_application, :enter), (:retraction_application, :exit)]
    # a composite retracts every leaf, each in a phase of its own; the whole-tree optimizer once
    retractions = method isa CompositeMethod ? 3 : 1
    @test log.events == [(:gradient, :enter); (:gradient, :exit);
                         (:optimizer_state_direction, :enter);
                         repeat(retraction, retractions);
                         (:optimizer_state_direction, :exit)]

    timer = ExclusiveStepTimer()
    ps = make_parameters()
    step = TrainingStep(ps, ∇F!, method, 1.0e-3; observer = timer)
    for _ in 1:3
        training_step!(ps, step)
    end
    timing = step_timing(timer, 3)
    @test timing.timed_steps == 3
    @test timer.calls[:gradient] == 3
    @test timer.calls[:retraction_application] == 3 * retractions
end
