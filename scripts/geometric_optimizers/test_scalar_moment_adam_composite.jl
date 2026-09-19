# Two-minibatch regression for `scalar_moment_adam_composite.jl`.
#
# The hazard the composite exists to close: `solver_step!` ends by refreshing the
# accepted-point gradient and marking it current, and `store_gradient!` reuses it on the
# next step when the point and the section still match. In the per-leaf composite the
# bare-array leaves compare by value after the in-place sync, so a step that follows a
# changed batch would, left to itself, consume the previous batch's cached gradient slice.
#
# This test drives two steps through two minibatches whose ambient gradients differ by
# construction and asserts, on the array leaves, that the second step was built from the
# second minibatch — against a closed-form two-step `Adam` reference written here rather
# than read from `GeometricOptimizers` — and, on the Stiefel leaf, that the gradient
# stored after step 2 is the lift of minibatch 2's slice in the state's section as it
# stood after step 1.
#
# Run it the way the scripts environment runs everything; a failed assertion exits 1:
#
#   JULIA_DEPOT_PATH=/tmp/gmldatasets-pr12-depot \
#     <julia-1.12.6> --project=scripts \
#     scripts/geometric_optimizers/test_scalar_moment_adam_composite.jl

using Test
using Random
using LinearAlgebra: norm
using GeometricOptimizers
using GeometricOptimizers: cache, section
using NeuralNetworkParameters: NetworkParameters, flatlength, parameterlayout

include(joinpath(@__DIR__, "step_timing.jl"))
include(joinpath(@__DIR__, "scalar_moment_adam_composite.jl"))

@testset "ScalarMomentAdamComposite" begin
    rng = Random.Xoshiro(1234)

    # one leaf of each of the three shapes the trainer's set uses: a Stiefel attention
    # projection, a bias vector, a ResNet weight matrix — the last one pinning the
    # composite's distinct construction branch for a bare `Matrix` leaf
    Y₀ = rand(rng, StiefelManifold{Float64}, 5, 2)
    v₀ = rand(rng, 3)
    W₀ = rand(rng, 4, 3)
    ps = NetworkParameters((Y = Y₀, v = v₀, W = W₀))

    # the two minibatches are two ambient-gradient fields over the whole flat set, as the
    # trainer's `current_batch[]` supplies them one at a time
    n = flatlength(parameterlayout(ps))
    G₁ = randn(rng, n)
    G₂ = G₁ + 2 * randn(rng, n)
    current_batch = Ref{Vector{Float64}}()
    ∇F!(g, v) = (copyto!(g, current_batch[]); g)

    η = 1.0e-3
    config = ScalarMomentAdamConfig(η)
    β₁ = config.array_method.β₁
    β₂ = config.array_method.β₂
    δ = config.array_method.δ

    timer = ExclusiveStepTimer()
    composite = ScalarMomentAdamComposite(ps, ∇F!, config; observer=timer)
    (range_Y, range_v, range_W) = composite.ranges

    # -- the closed-form two-step `Adam` reference for the array leaves, written against
    #    `Adam`'s own update and not read from it: at t = 1 the bias-corrected moments
    #    collapse to the gradient itself, at t = 2 the factors are β/(1+β) and 1/(1+β)
    g₁ = G₁[range_v]
    dir₁ = -g₁ ./ (sqrt.(g₁ .^ 2) .+ δ)
    v₁ = v₀ .+ η * dir₁
    fac₁₁ = β₁ / (1 + β₁); fac₁₂ = 1 / (1 + β₁)
    fac₂₁ = β₂ / (1 + β₂); fac₂₂ = 1 / (1 + β₂)
    g₂ = G₂[range_v]
    m₁₂ = fac₁₁ * g₁ + fac₁₂ * g₂
    m₂₂ = fac₂₁ * g₁ .^ 2 + fac₂₂ * g₂ .^ 2
    v₂ = v₁ .+ η * (-m₁₂ ./ (sqrt.(m₂₂) .+ δ))
    # the same reference with step 2 reusing minibatch 1 — what the stale cached gradient
    # would give; `G₂ ≠ G₁` puts the two O(η) apart, far above round-off
    v₂stale = v₁ .+ η * dir₁

    # the matrix leaf, the same reference in the flattened representation the leaf's
    # optimizer works in
    w₀ = vec(W₀)
    wg₁ = G₁[range_W]
    wg₂ = G₂[range_W]
    w₁ = w₀ .+ η * (-wg₁ ./ (sqrt.(wg₁ .^ 2) .+ δ))
    wm₁₂ = fac₁₁ * wg₁ + fac₁₂ * wg₂
    wm₂₂ = fac₂₁ * wg₁ .^ 2 + fac₂₂ * wg₂ .^ 2
    w₂ = w₁ .+ η * (-wm₁₂ ./ (sqrt.(wm₂₂) .+ δ))

    @test norm(v₂ - v₂stale) > 1e-4     # the two cases the test has to tell apart

    # -- step one, on minibatch 1
    current_batch[] = G₁
    composite_step!(composite, ps)

    @test norm(ps.v - v₁) < 1e-12
    @test norm(vec(ps.W) - w₁) < 1e-12   # the `vec` write-back of the `Matrix` leaf
    @test check(ps.Y) ≤ 1e-10            # the exact `Cayley` retraction keeps ‖YᵀY − I‖ at round-off

    # the Stiefel leaf's stored gradient after step 2 must be the lift of minibatch 2's
    # ambient slice in the state's section as it stands after step 1; both are captured
    # now, before step 2 moves the iterate or transports the section
    leaf = composite.leaves[1]
    Y₁ = copy(ps.Y)
    expected_stored = global_rep(section(leaf.state), rgrad(Y₁, reshape(G₂[range_Y], size(Y₁)...)))

    # -- step two, on minibatch 2
    current_batch[] = G₂
    composite_step!(composite, ps)

    @test check(ps.Y) ≤ 1e-10
    stored = gradient(cache(leaf.optimizer))
    @test stored.A.S ≈ expected_stored.A.S
    @test stored.B ≈ expected_stored.B
    @test norm(ps.v - v₂) < 1e-12        # step two consumed minibatch two
    @test norm(ps.v - v₂stale) > 1e-4    # ... and not the cached minibatch one
    @test norm(vec(ps.W) - w₂) < 1e-12

    # Only the shared whole-tree gradient is AD. The two per-leaf gradient callbacks in each
    # solver step merely copy slices and remain inside optimizer-state time. With three leaves,
    # every step performs two sentinel objectives and three retraction/application operations per
    # leaf (NaN guard, accepted application, and state-section update).
    @test timer.calls[:optimizer_state_direction] == 2
    @test timer.calls[:gradient] == 2
    @test timer.calls[:objective] == 2 * length(composite.leaves) * 2
    @test timer.calls[:retraction_application] == 3 * length(composite.leaves) * 2
    @test step_timing(timer, 2).timed_steps == 2
end
