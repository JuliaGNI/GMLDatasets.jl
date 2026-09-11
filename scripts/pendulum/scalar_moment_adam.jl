# The scalar-second-moment Riemannian Adam baseline for the pendulum SAE.
#
# A symplectic autoencoder is a tree of *layers*: PSD layers have only
# Stiefel-manifold weights, while SympNet layers have only Euclidean arrays.
# `GeometricOptimizers.ScalarMomentAdam` intentionally accepts just one
# Stiefel solution, so this adapter selects it for each PSD layer and ordinary
# `Adam` for each Euclidean layer. Every layer still consumes the one gradient
# produced by GML's normal minibatch pullback.

import GeometricMachineLearning
import GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters, parameter_eltype

const GML = GeometricMachineLearning
const GO = GeometricOptimizers

"""Scalar-moment Adam on PSD layers and ordinary Adam on Euclidean SAE layers."""
struct SAEScalarMomentAdam{T<:AbstractFloat} <: GO.OptimizerMethod
    stiefel::GO.ScalarMomentAdam{T}
    euclidean::GO.Adam{T}
end

function SAEScalarMomentAdam(::Type{T}; beta1=0.9, beta2=0.99, epsilon=1e-8,
        ambient_norm::Bool=false) where {T<:AbstractFloat}
    SAEScalarMomentAdam{T}(
        GO.ScalarMomentAdam(T; β₁=T(beta1), β₂=T(beta2), δ=T(epsilon), ambient_norm),
        GO.Adam(T; β₁=T(beta1), β₂=T(beta2), δ=T(epsilon)),
    )
end

"""Select the genuine GO method for a flat SAE layer, rejecting mixed layers."""
function sae_layer_method(method::SAEScalarMomentAdam, layer::NamedTuple)
    values_layer = values(layer)
    !isempty(values_layer) || throw(ArgumentError("an SAE layer cannot be empty"))
    if all(value -> value isa GML.StiefelManifold, values_layer)
        method.stiefel
    elseif all(value -> value isa AbstractArray, values_layer)
        method.euclidean
    else
        throw(ArgumentError(
            "scalar-moment SAE layers must be entirely Stiefel or entirely Euclidean"))
    end
end

# GML normally makes one GO cache/state per flat layer.  Its generic route
# cannot choose two algorithms, so give this one method the same layer tree
# explicitly.  The `NetworkParameters` wrapper shares the underlying arrays,
# making each GO update visible to the network without a copy.
function GML._make_optimizer_cache(method::SAEScalarMomentAdam, ps::NetworkParameters)
    NamedTuple{keys(ps)}(Tuple(GML._make_optimizer_cache(method, ps[key]) for key in keys(ps)))
end

function GML._make_optimizer_cache(method::SAEScalarMomentAdam, layer::NamedTuple)
    algorithm = sae_layer_method(method, layer)
    solution = algorithm isa GO.ScalarMomentAdam ? only(values(layer)) : NetworkParameters(layer)
    GO.OptimizerCache(algorithm, solution)
end

function GML._make_optimizer_state(method::SAEScalarMomentAdam, ps::NetworkParameters)
    NamedTuple{keys(ps)}(Tuple(GML._make_optimizer_state(method, ps[key]) for key in keys(ps)))
end

function GML._make_optimizer_state(method::SAEScalarMomentAdam, layer::NamedTuple)
    algorithm = sae_layer_method(method, layer)
    solution = algorithm isa GO.ScalarMomentAdam ? only(values(layer)) : NetworkParameters(layer)
    GO.OptimizerState(algorithm, solution)
end

# The GML wrapper already knows how to take a GO step and apply its Cayley
# section update.  Dispatching through the actual scalar method lets GO form
# the correct scalar second moment; then copy the moments back into the state,
# exactly as GML's native Adam path does for its coordinate-wise moments.
function GML._leaf_optim_step!(cache::GO.ScalarMomentAdamCache,
        state::GO.ScalarMomentAdamState, dp_leaf, ps_leaf, section_leaf,
        method::SAEScalarMomentAdam, retraction, step_size)
    weight = only(values(ps_leaf))
    gradient = only(values(dp_leaf))
    local_section = only(values(section_leaf))
    T = parameter_eltype(weight)
    local_gradient = GML._GMLGradient{T,typeof(gradient)}(gradient)
    state.iterations += 1
    GO.update!(cache, state, local_gradient, method.stiefel, weight)
    GO._rmul!(GO.direction(cache), step_size)
    GO.update_section!(GO.section(cache), GO.section(state), GO.direction(cache), retraction)
    GO._copyto!(GO.solution(cache), GO.section(cache))
    GO._copyto!(weight, GO.solution(cache))
    GO._copyto!(local_section, GO.section(cache))
    GO._copyto!(GO.section(state), GO.section(cache))
    GO._copyto!(GO.first_moment(state), GO.first_moment(cache))
    setfield!(state, :m₂, GO.second_moment(cache))
    nothing
end

# The Euclidean layers use ordinary GO Adam, not a scalar moment.  Passing the
# underlying method also preserves GML's existing Adam-state synchronization.
function GML._leaf_optim_step!(cache::GO.AdamCache, state::GO.AdamState,
        dp_leaf, ps_leaf, section_leaf, method::SAEScalarMomentAdam, retraction, step_size)
    GML._leaf_optim_step!(cache, state, dp_leaf, ps_leaf, section_leaf,
        method.euclidean, retraction, step_size)
end
