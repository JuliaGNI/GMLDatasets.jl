# The optimizer step the image trainer takes, once per minibatch.
#
# The step is `GeometricOptimizers.optimization_step!` of a `TrainingOptimizer`: the trainer computes
# the Euclidean gradient of the loss on the current minibatch itself, and the step projects it, forms
# the direction of the method, scales it by the learning rate and retracts it. Every configuration —
# the `scalar-moment-adam` `CompositeMethod` included — is one `TrainingOptimizer` over the whole
# parameter set; a composite keeps a cache and a state per leaf inside it.
#
# This replaces the `increase_iteration_number!` / `solver_step!` / `update!` triple the trainer used
# to drive itself, and with it two things that step carried and a training step has no use for:
#
#   * an objective evaluation per step, the `NaN` guard of `solver_step!`, which cost a forward pass
#     and was charged to none of the three timing categories;
#   * the gradient `solver_step!` refreshes at the point it ended at. The next step reused it while
#     the point and the section still matched, which after the state update they did, so a step was
#     built from the gradient of the *previous* minibatch at the current point. The per-leaf
#     composite invalidated every cache before each step to avoid this; the whole-tree path did not.
#
# Here the gradient is evaluated once per step, on the minibatch of that step, and passed in.
#
# The three schema-v4 categories are the trainer's phases around the step: `:gradient` around the
# reverse pass and the copy of its flat result into the parameter-shaped buffer, and
# `:optimizer_state_direction` around `optimization_step!`, inside which the `TrainingOptimizer`
# reports `:retraction_application` itself; the timer subtracts the nested phase from the enclosing
# one, so the three are disjoint.
#
# This file holds definitions only; `include` runs nothing.

using GeometricOptimizers: TrainingOptimizer, optimization_step!, observe_optimizer_phase, Cayley
using NeuralNetworkParameters: NetworkParameters, ParameterLayout, flatten!, unflatten!,
                               mapparameters, parameterlayout
using SimpleSolvers: Static

"""
    TrainingStep(ps, ∇F!, algorithm, learning_rate; observer)

What one step of the trainer needs besides the parameters `ps`: the `TrainingOptimizer` of
`algorithm` at the fixed rate `learning_rate` with the Cayley retraction, the whole-tree gradient
`∇F!(g, v)` of the trainer — which reads the current minibatch from the trainer's own state and
writes the Euclidean gradient at the flat parameters `v` into the flat `g` — and the buffers that
connect the two: the flat parameters and gradient, and the gradient in the shape of `ps`.

The gradient buffer holds plain arrays, a `Matrix` where `ps` holds a `StiefelManifold`, because the
step takes the *Euclidean* gradient and projects it itself. Its layout is its own: a manifold leaf
flattens through `freeparameters`, which for a `StiefelManifold` is the `N × n` matrix the gradient
leaf is, so the two layouts agree entry for entry.
"""
struct TrainingStep{T, OT <: TrainingOptimizer, DT <: NetworkParameters, FT, VT}
    optimizer::OT
    parameter_layout::ParameterLayout
    gradient_layout::ParameterLayout
    flat_parameters::Vector{T}
    flat_gradient::Vector{T}
    euclidean_gradient::DT
    ∇F!::FT
    observer::VT
end

function TrainingStep(ps::NetworkParameters{T}, ∇F!, algorithm, learning_rate;
        observer) where {T}
    optimizer = TrainingOptimizer(ps; algorithm = algorithm,
        linesearch = Static(T(learning_rate)), retraction = Cayley(), observer = observer)
    parameter_layout = parameterlayout(ps)
    euclidean_gradient = mapparameters(leaf -> zeros(T, size(leaf)), ps)
    gradient_layout = parameterlayout(euclidean_gradient)
    length(gradient_layout) == length(parameter_layout) || throw(DimensionMismatch(
        "the gradient buffer flattens to $(length(gradient_layout)) numbers and the parameters " *
        "to $(length(parameter_layout))"))
    n = length(parameter_layout)
    TrainingStep(optimizer, parameter_layout, gradient_layout, zeros(T, n), zeros(T, n),
        euclidean_gradient, ∇F!, observer)
end

"""
    training_step!(ps, step)

Take one optimizer step on the current minibatch: the gradient of the loss at `ps`, then
`optimization_step!`, which writes the new parameters into `ps`.
"""
function training_step!(ps::NetworkParameters, step::TrainingStep)
    observe_optimizer_phase(step.observer, :gradient) do
        flatten!(step.flat_parameters, ps, step.parameter_layout)
        step.∇F!(step.flat_gradient, step.flat_parameters)
        unflatten!(step.euclidean_gradient, step.gradient_layout, step.flat_gradient)
    end
    observe_optimizer_phase(step.observer, :optimizer_state_direction) do
        optimization_step!(ps, step.optimizer, step.euclidean_gradient)
    end
    ps
end
