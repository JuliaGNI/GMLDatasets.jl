# Two-minibatch regression for `leaf_composite.jl`.
#
# What this file is *not* about: which method each leaf gets, and whether a leaf stepped through a
# `GeometricOptimizers.CompositeMethod` is the leaf that method would have stepped. Both are
# upstream's, and `GeometricOptimizers/test/composite_method.jl` pins them — re-deriving `Adam`'s
# two-step recursion here would be a second copy of a formula this suite does not own, failing for
# upstream's reasons.
#
# What is local, and is the hazard this loop exists to close: `solver_step!` ends by refreshing the
# accepted-point gradient and marking it current, and `store_gradient!` reuses it on the next step
# when the point and the section still match. In a per-leaf composite the bare-array leaves compare
# by value after the in-place sync, so a step that follows a changed batch would, left to itself,
# consume the previous batch's cached gradient slice.
#
# This test drives two steps through two minibatches whose ambient gradients differ by construction
# and asserts that the second step was built from the *second* minibatch: on the array leaves by
# comparing the iterate against the same loop run with the batches in the other order, and on the
# Stiefel leaf by comparing the stored gradient against the lift of minibatch 2's slice in the
# state's section as it stood after step 1.
#
# Run it the way the scripts environment runs everything; a failed assertion exits 1:
#
#   julia --project=scripts scripts/geometric_optimizers/test_leaf_composite.jl

using Test
using Random
using LinearAlgebra: norm
using GeometricOptimizers
using GeometricOptimizers: cache, section
using NeuralNetworkParameters: NetworkParameters, flatlength, parameterlayout

include(joinpath(@__DIR__, "step_timing.jl"))
include(joinpath(@__DIR__, "leaf_composite.jl"))
include(joinpath(@__DIR__, "..", "revision", "scalar_moment_adam.jl"))

@testset "LeafComposite" begin
    rng = Random.Xoshiro(1234)

    # one leaf of each of the three shapes the trainer's set uses: a Stiefel attention projection, a
    # bias vector, a ResNet weight matrix — the last one pinning the `vec` view of a bare `Matrix`
    Y₀ = rand(rng, StiefelManifold{Float64}, 5, 2)
    v₀ = rand(rng, 3)
    W₀ = rand(rng, 4, 3)
    make_parameters() = NetworkParameters((Y = copy(Y₀), v = copy(v₀), W = copy(W₀)))

    # the two minibatches are two ambient-gradient fields over the whole flat set, as the trainer's
    # `current_batch[]` supplies them one at a time
    n = flatlength(parameterlayout(make_parameters()))
    G₁ = randn(rng, n)
    G₂ = G₁ + 2 * randn(rng, n)

    η = 1.0e-3
    method = scalar_moment_adam_method(Float64)

    """
        two_steps(batches)

    Two steps through `batches`, returning the parameters, the composite, the timer, and — captured
    immediately before the *last* step, while the section is still the one that step will read — the
    horizontal lift of that step's Stiefel gradient slice. That capture has to happen inside, because
    the step then moves the iterate and transports the section.
    """
    function two_steps(batches)
        current_batch = Ref{Vector{Float64}}()
        ∇F!(g, v) = (copyto!(g, current_batch[]); g)
        ps = make_parameters()
        timer = ExclusiveStepTimer()
        composite = LeafComposite(ps, ∇F!, method, η; observer = timer)
        range_Y = composite.ranges[1]
        expected_stored = nothing
        for (index, batch) in pairs(batches)
            if index == lastindex(batches)
                expected_stored = global_rep(section(composite.states[1]),
                    rgrad(ps.Y, reshape(batch[range_Y], size(ps.Y)...)))
            end
            current_batch[] = batch
            composite_step!(composite, ps)
        end
        (ps = ps, composite = composite, timer = timer, expected_stored = expected_stored)
    end

    second_batch_used = two_steps([G₁, G₂])
    second_batch_ignored = two_steps([G₁, G₁])

    ps = second_batch_used.ps
    @test check(ps.Y) ≤ 1.0e-10          # the exact `Cayley` retraction keeps ‖YᵀY − I‖ at round-off

    # Step two consumed minibatch two, on both array shapes. `G₂ ≠ G₁` puts the two cases O(η)
    # apart, far above round-off, which is what makes the comparison able to tell them apart at all.
    @test norm(ps.v - second_batch_ignored.ps.v) > 1.0e-4
    @test norm(vec(ps.W) - vec(second_batch_ignored.ps.W)) > 1.0e-4

    # ... and on the Stiefel leaf, where the staleness is not visible in the iterate: the gradient
    # stored after step 2 has to be the lift of minibatch 2's ambient slice, in the section as it
    # stood after step 1.
    stored = gradient(cache(second_batch_used.composite.optimizers[1]))
    @test stored.A.S ≈ second_batch_used.expected_stored.A.S
    @test stored.B ≈ second_batch_used.expected_stored.B
    # and it is not minibatch one's, which is the failure this closes
    @test !isapprox(stored.B, second_batch_ignored.expected_stored.B)

    # Only the shared whole-tree gradient is AD. The two per-leaf gradient callbacks in each solver
    # step merely copy slices and stay inside optimizer-state time.
    timer = second_batch_used.timer
    leaves = length(second_batch_used.composite.optimizers)
    @test timer.calls[:optimizer_state_direction] == 2
    @test timer.calls[:gradient] == 2
    # per leaf per step: two sentinel objectives, and three retraction/application operations (NaN
    # guard, accepted application, and state-section update)
    @test timer.calls[:objective] == 2 * leaves * 2
    @test timer.calls[:retraction_application] == 3 * leaves * 2
    @test step_timing(timer, 2).timed_steps == 2
end
