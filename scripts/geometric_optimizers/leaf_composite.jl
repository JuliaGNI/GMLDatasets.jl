# The per-leaf step loop this trainer takes over its flat parameter container.
#
# Which method each leaf is stepped with is not decided here: it is
# `GeometricOptimizers.CompositeMethod`, built by `scripts/revision/scalar_moment_adam.jl`, which
# both trainers share. What is here is the *loop* — one `Optimizer`/`OptimizerState` per leaf of
# `ps`, one whole-tree `∇F!` per minibatch, and the cache invalidation that makes reusing that one
# gradient correct — and the loop is this trainer's own business, because this script drives
# `GeometricOptimizers` directly rather than through a training framework.
#
# `composite_step!` flattens the container into the shared buffer, evaluates `∇F!` once for the whole
# tree, invalidates every leaf cache, then steps the leaves in parameter-layout order with the
# trainer's `increase_iteration_number!` / `solver_step!` / `update!` triple. Each leaf has
# independent state and reads only its own slice of the same frozen pre-step gradient, so no leaf
# acts on information an earlier one has changed.
#
# The invalidation is the minibatch-correctness seam. `solver_step!` ends by refreshing the
# accepted-point gradient and marking it current, and `store_gradient!` reuses it on the next step
# while the point and the section still match — which for a bare-array leaf they do, by value, after
# the in-place sync. The step after a changed batch would then consume the previous batch's cached
# slice. The objective has changed under every cache, so every leaf is invalidated before any step:
# cheap, and it does not have to reason about which leaf the hazard bites on.
# `test_leaf_composite.jl` pins the seam on the array leaf, where the staleness is
# visible in the iterate.
#
# Each leaf's merit is a sentinel, `sum(abs2, freeparameters(x))`, not the network loss. `Static`
# never evaluates a merit inside the step, so the sentinel never enters the iterate; it is what the
# `solver_step!` NaN guard and the state's `f` read. It must stay O(leaf size) and finite because it
# runs once per leaf per step — 369 leaves, and a per-leaf network evaluation would cost a forward
# and a backward pass each. A retraction that corrupts its leaf still lands a non-finite sentinel, so
# the guard keeps working.
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
    LeafComposite{T, OT}

The per-repetition composite over a `NetworkParameters` container `ps`: one
`Optimizer`/`OptimizerState` per entry, the shared flat buffers the whole-tree `∇F!` and the
flattening read and write, and `∇F!` itself — the trainer's, which reads the current batch from its
own `current_batch[]`.
"""
struct LeafComposite{T, OT}
    optimizers::Vector{GeometricOptimizers.Optimizer{T}}
    states::Vector{GeometricOptimizers.OptimizerState{T}}
    layout::ParameterLayout
    ranges::Vector{UnitRange{Int}}
    flat_parameters::Vector{T}
    flat_gradient::Vector{T}
    ∇F!::Any
    observer::OT
end

# A leaf gradient callback only copies one slice of the shared whole-tree gradient; it is optimizer
# bookkeeping, not another reverse pass. Suppress only that notification while forwarding every leaf
# objective and retraction/application boundary to the run observer.
struct LeafObserver{OT}
    observer::OT
end

@inline function (observer::LeafObserver)(phase, event)
    phase === :gradient || observer.observer(phase, event)
    nothing
end

"""
    leaf_solution(x)

The point the leaf's optimizer works on: the leaf itself where it is already an `OptimizerSolution`
(a `StiefelManifold`, a `Vector`), the `vec` view where it is a bare `Matrix`, which is not one —
`OptimizerProblem` and `Optimizer` bind their point to `OptimizerSolution`, and that union excludes
`Matrix{T}`. The view writes through to the leaf, so the in-place sync of `solver_step!` reaches the
container without a copy.
"""
leaf_solution(x) = (x isa Manifold || x isa AbstractVector) ? x : vec(x)

function LeafComposite(ps::NetworkParameters, ∇F!, method::GeometricOptimizers.OptimizerMethod,
        learning_rate::T; observer = NoStepObserver()) where {T}
    layout = parameterlayout(ps)
    # A `NetworkParameters` adds a `ParametersLayout` around the wrapped `NamedTuple` layout. Read
    # that inner layout explicitly; `eachindex(ps)` yields the parameter keys, while the integer
    # indices `ps[i]` accepts are what the layout-order step below walks.
    parameter_values = values(ps)
    ranges = [parameterrange(getfield(layout.inner.children, i))
              for i in eachindex(parameter_values)]
    flat_parameters = Vector{T}(undef, flatlength(layout))
    flat_gradient = Vector{T}(undef, flatlength(layout))
    leaf_observer = LeafObserver(observer)

    optimizers = Vector{GeometricOptimizers.Optimizer{T}}(undef, length(ps))
    states = Vector{GeometricOptimizers.OptimizerState{T}}(undef, length(ps))
    for (i, x) in enumerate(parameter_values)
        x₀ = leaf_solution(x)

        # the sentinel merit: O(leaf size), finite, NaN-reporting; it never enters a `Static` step,
        # it is what the NaN guard and the state's `f` read (see the file header)
        F_leaf(y) = sum(abs2, freeparameters(y))
        # the leaf's slice of the one shared whole-tree gradient, re-read at every call: this closure
        # is the seam the uniform invalidation in `composite_step!` keeps honest
        ∇F_leaf!(g, y) = (copyto!(g, view(flat_gradient, ranges[i])); g)

        G = GradientFunction{T}(F_leaf, ∇F_leaf!, length(x₀))
        problem = OptimizerProblem(F_leaf, ∇F_leaf!, x₀)
        optimizers[i] = Optimizer(x₀, problem; algorithm = method,
            linesearch = Static(T; α = learning_rate),
            gradient = G, retraction = Cayley(), observer = leaf_observer)
        states[i] = OptimizerState(method, x₀)
        initialize_state!(states[i])
    end

    LeafComposite{T, typeof(observer)}(optimizers, states, layout, ranges, flat_parameters,
        flat_gradient, ∇F!, observer)
end

"""
    composite_step!(composite, ps)

One step of the composite on the current batch, replacing the trainer's
`increase_iteration_number!` / `solver_step!` / `update!` triple: flatten, one whole-tree `∇F!`,
invalidate every leaf cache, then step the leaves in parameter-layout order.
"""
function composite_step!(composite::LeafComposite{T}, ps::NetworkParameters) where {T}
    observe_optimizer_phase(composite.observer, :optimizer_state_direction) do
        flatten!(composite.flat_parameters, ps, composite.layout)
        observe_optimizer_phase(composite.observer, :gradient) do
            composite.∇F!(composite.flat_gradient, composite.flat_parameters)
        end
        for optimizer in composite.optimizers
            invalidate_latest_gradient!(cache(optimizer))
        end
        for (i, optimizer) in enumerate(composite.optimizers)
            x = leaf_solution(ps[i])
            increase_iteration_number!(composite.states[i])
            solver_step!(x, composite.states[i], optimizer)
            GeometricOptimizers.update!(composite.states[i], optimizer, x)
        end
    end

    ps
end
