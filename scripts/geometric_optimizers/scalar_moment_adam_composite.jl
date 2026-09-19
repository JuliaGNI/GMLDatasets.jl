# The `ScalarMomentAdam` (Li et al. 2020) baseline as a per-leaf composite.
#
# `ScalarMomentAdam` — the Adam of `li2020efficient`, Algorithm 2, with its scalar second moment —
# accepts a single `StiefelManifold` and rejects arrays, parameter sets and mixed trees on
# purpose; `GeometricOptimizers` deliberately does not widen it. The transformer trained here is a
# mixed tree, `StiefelManifold` attention projections beside ordinary `Matrix` and `Vector`
# leaves, so the baseline is assembled around the released methods rather than passed to them: one
# `Optimizer`/`OptimizerState` per leaf, `ScalarMomentAdam` on each Stiefel leaf and ordinary
# `Adam` on each Euclidean one, all sharing the step's one minibatch gradient. The retraction is a
# parameter of the method; this composite runs `Cayley`, which the configuration records.
#
# `composite_step!` flattens the container into the shared buffer, evaluates `∇F!` once for the
# whole tree, invalidates every leaf cache, then steps the leaves in parameter-layout order with
# the trainer's `increase_iteration_number!` / `solver_step!` / `update!` triple. Each leaf has
# independent state and reads only its own slice of the same frozen pre-step gradient, so no leaf
# acts on information an earlier one has changed.
#
# The invalidation is the minibatch-correctness seam. `solver_step!` ends by refreshing the
# accepted-point gradient and marking it current, and `store_gradient!` reuses it on the next step
# while the point and the section still match — which for a bare-array leaf they do, by value,
# after the in-place sync. The step after a changed batch would then consume the previous batch's
# cached slice. The objective has changed under every cache, so every leaf is invalidated before
# any step: cheap, and it does not have to reason about which leaf the hazard bites on.
# `test_scalar_moment_adam_composite.jl` pins the seam on the array leaf, where the staleness is
# visible in the iterate.
#
# Each leaf's merit is a sentinel, `sum(abs2, freeparameters(x))`, not the network loss. `Static`
# never evaluates a merit inside the step, so the sentinel never enters the iterate; it is what
# the `solver_step!` NaN guard and the state's `f` read. It must stay O(leaf size) and finite
# because it runs once per leaf per step — 369 leaves, and a per-leaf network evaluation would
# cost a forward and a backward pass each. A retraction that corrupts its leaf still lands a
# non-finite sentinel, so the guard keeps working.
#
# The learning rate is a required argument of `ScalarMomentAdamConfig`, not a default: in
# `GeometricOptimizers` the methods give a direction and `linesearch = Static(η)` is the rate, and
# the scalar-moment direction has magnitude ≈ 1 in total where `Adam`'s has ≈ 1 per component, so
# the same number buys a step of a different length on the two leaf kinds.
#
# This file holds definitions only; `include` runs nothing.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!,
                           cache, invalidate_latest_gradient!, NoStepObserver,
                           observe_optimizer_phase
using NeuralNetworkParameters: NetworkParameters, ParameterLayout, flatten!, freeparameters,
                               parameterlayout, parameterrange, flatlength
using SimpleSolvers: Static

"""
    ScalarMomentAdamConfig{T}(learning_rate; β₁, β₂, δ, ambient_norm)

The configuration of the `scalar-moment-adam` baseline: `ScalarMomentAdam` on the Stiefel
leaves and ordinary `Adam` on the Euclidean ones, with `learning_rate` applied through the
`Static` line search of every leaf's optimizer.

The name is the method's, not the retraction's: `ScalarMomentAdam` takes any
`AbstractRetraction`, and this configuration runs it with `Cayley`, as the run records.

The learning rate is a required argument, not a default, for the reason in the file header:
the two leaf kinds scale it differently, so the baseline tunes it as its own number.

`ambient_norm` chooses which ‖·‖² the Stiefel leaves' scalar second moment accumulates —
`true` is the faithful `li2020efficient` Algorithm 2 port, `false` the `GeometricOptimizers`
default. It is not stored here: the trainer writes it into every run record's `second_moment`,
which is where a number is traced back to the mode that made it.
"""
struct ScalarMomentAdamConfig{T}
    learning_rate::T
    stiefel_method::ScalarMomentAdam{T}
    array_method::Adam{T}

    function ScalarMomentAdamConfig(learning_rate::T; β₁ = 9.0e-1, β₂ = 9.9e-1, δ = 1.0e-8,
            ambient_norm::Bool = false) where {T <: AbstractFloat}
        new{T}(T(learning_rate),
            ScalarMomentAdam(T; β₁ = β₁, β₂ = β₂, δ = δ, ambient_norm = ambient_norm),
            Adam(T; β₁ = β₁, β₂ = β₂, δ = δ))
    end
end

"""
    ScalarMomentAdamLeaf{T}

One leaf of a [`ScalarMomentAdamComposite`](@ref): its position in the parameter container,
whether it lives on the Stiefel, and the optimizer/state pair that steps it.
"""
struct ScalarMomentAdamLeaf{T}
    index::Int
    stiefel::Bool
    optimizer::GeometricOptimizers.Optimizer{T}
    state::GeometricOptimizers.OptimizerState{T}
end

"""
    ScalarMomentAdamComposite{T}

The per-repetition composite over a `NetworkParameters` container `ps`: one leaf per entry, the
shared flat buffers the whole-tree `∇F!` and the flattening read and write, and `∇F!` itself —
the trainer's, which reads the current batch from its own `current_batch[]`.
"""
struct ScalarMomentAdamComposite{T, OT}
    leaves::Vector{ScalarMomentAdamLeaf{T}}
    layout::ParameterLayout
    ranges::Vector{UnitRange{Int}}
    flat_parameters::Vector{T}
    flat_gradient::Vector{T}
    ∇F!::Any
    observer::OT
end

# A leaf gradient callback only copies one slice of the shared whole-tree gradient; it is
# optimizer bookkeeping, not another reverse pass. Suppress only that notification while
# forwarding every leaf objective and retraction/application boundary to the run observer.
struct ScalarMomentAdamLeafObserver{OT}
    observer::OT
end

@inline function (observer::ScalarMomentAdamLeafObserver)(phase, event)
    phase === :gradient || observer.observer(phase, event)
    nothing
end

"""
    leaf_solution(x, stiefel)

The point the leaf's optimizer works on: the leaf itself where it is already an
`OptimizerSolution` (a `StiefelManifold`, a `Vector`), the `vec` view where it is a bare
`Matrix`, which is not one — `OptimizerProblem` and `Optimizer` bind their point to
`OptimizerSolution`, and that union excludes `Matrix{T}`. The view writes through to the
leaf, so the in-place sync of `solver_step!` reaches the container without a copy.
"""
leaf_solution(x, stiefel::Bool) = (stiefel || x isa AbstractVector) ? x : vec(x)

function ScalarMomentAdamComposite(
        ps::NetworkParameters, ∇F!, config::ScalarMomentAdamConfig{T};
        observer = NoStepObserver()) where {T}
    layout = parameterlayout(ps)
    # A `NetworkParameters` adds a `ParametersLayout` around the wrapped `NamedTuple` layout.
    # Read that inner layout explicitly; `eachindex(ps)` yields the parameter keys, while the
    # integer indices stored below are what `ps[i]` accepts during the layout-order step.
    parameter_values = values(ps)
    ranges = [parameterrange(getfield(layout.inner.children, i))
              for i in eachindex(parameter_values)]
    flat_parameters = Vector{T}(undef, flatlength(layout))
    flat_gradient = Vector{T}(undef, flatlength(layout))
    leaf_observer = ScalarMomentAdamLeafObserver(observer)

    leaves = Vector{ScalarMomentAdamLeaf{T}}(undef, length(ps))
    for (i, x) in enumerate(parameter_values)
        stiefel = x isa StiefelManifold
        x₀ = leaf_solution(x, stiefel)

        # the sentinel merit: O(leaf size), finite, NaN-reporting; it never enters a `Static`
        # step, it is what the NaN guard and the state's `f` read (see the file header)
        F_leaf(y) = sum(abs2, freeparameters(y))
        # the leaf's slice of the one shared whole-tree gradient, re-read at every call: this
        # closure is the seam the uniform invalidation in `composite_step!` keeps honest
        ∇F_leaf!(g, y) = (copyto!(g, view(flat_gradient, ranges[i])); g)

        G = GradientFunction{T}(F_leaf, ∇F_leaf!, length(x₀))
        problem = OptimizerProblem(F_leaf, ∇F_leaf!, x₀)
        method = stiefel ? config.stiefel_method : config.array_method
        optimizer = Optimizer(x₀, problem; algorithm = method,
            linesearch = Static(T; α = config.learning_rate),
            gradient = G, retraction = Cayley(), observer = leaf_observer)
        state = OptimizerState(method, x₀)
        initialize_state!(state)
        leaves[i] = ScalarMomentAdamLeaf{T}(i, stiefel, optimizer, state)
    end

    ScalarMomentAdamComposite{T, typeof(observer)}(leaves, layout, ranges, flat_parameters,
        flat_gradient, ∇F!, observer)
end

"""
    composite_step!(composite, ps)

One step of the composite on the current batch, replacing the trainer's
`increase_iteration_number!` / `solver_step!` / `update!` triple: flatten, one whole-tree
`∇F!`, invalidate every leaf cache, then step the leaves in parameter-layout order.
"""
function composite_step!(composite::ScalarMomentAdamComposite{T}, ps::NetworkParameters) where {T}
    observe_optimizer_phase(composite.observer, :optimizer_state_direction) do
        flatten!(composite.flat_parameters, ps, composite.layout)
        observe_optimizer_phase(composite.observer, :gradient) do
            composite.∇F!(composite.flat_gradient, composite.flat_parameters)
        end
        for leaf in composite.leaves
            invalidate_latest_gradient!(cache(leaf.optimizer))
        end
        for leaf in composite.leaves
            x = leaf_solution(ps[leaf.index], leaf.stiefel)
            increase_iteration_number!(leaf.state)
            solver_step!(x, leaf.state, leaf.optimizer)
            GeometricOptimizers.update!(leaf.state, leaf.optimizer, x)
        end
    end

    ps
end
