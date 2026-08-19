using GeometricProblems.Pendulum: hodeproblem
using GeometricIntegrators: Gauss, integrate

"""One integrated mathematical-pendulum trajectory."""
struct PendulumTrajectory{T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    t::V
    angle::V
    momentum::V
    q::M
    p::M
end

Base.length(trajectory::PendulumTrajectory) = length(trajectory.t)
Base.size(trajectory::PendulumTrajectory) = size(trajectory.q)

"""A collection of independently generated pendulum trajectories."""
struct PendulumDataset{V<:AbstractVector{<:PendulumTrajectory}}
    trajectories::V
end

Base.length(dataset::PendulumDataset) = length(dataset.trajectories)
Base.getindex(dataset::PendulumDataset, index::Int) = dataset.trajectories[index]

function _validate_pendulum_parameters(length, mass, gravity, timestep, timespan)
    length > 0 || throw(ArgumentError("length must be positive"))
    mass > 0 || throw(ArgumentError("mass must be positive"))
    gravity > 0 || throw(ArgumentError("gravity must be positive"))
    timestep > 0 || throw(ArgumentError("timestep must be positive"))
    first(timespan) < last(timespan) || throw(ArgumentError("timespan must be increasing"))
    nothing
end

function _validate_coordinates(q, p)
    ndims(q) == 2 && ndims(p) == 2 || throw(ArgumentError("q and p must be two-dimensional matrices"))
    size(q, 1) == size(p, 1) == 2 || throw(DimensionMismatch("q and p must have two rows"))
    size(q, 2) == size(p, 2) || throw(DimensionMismatch("q and p must have the same number of samples"))
    nothing
end

"""
    angular_to_euclidean(angle, momentum; length=1)

Convert canonical angular coordinates to columns of Euclidean coordinates with
ordering `q = (length*sin(angle), length*cos(angle))` and
`p = (cos(angle)*momentum/length, -sin(angle)*momentum/length)`.
"""
function angular_to_euclidean(angle::AbstractVector, momentum::AbstractVector; length=1)
    length > 0 || throw(ArgumentError("length must be positive"))
    axes(angle, 1) == axes(momentum, 1) || throw(DimensionMismatch("angle and momentum must have equal lengths"))
    q = length .* permutedims(hcat(sin.(angle), cos.(angle)))
    p = (1 / length) .* permutedims(hcat(cos.(angle), -sin.(angle))) .* reshape(momentum, 1, :)
    q, p
end

"""Convert Euclidean pendulum coordinates to wrapped angular coordinates."""
function euclidean_to_angular(q::AbstractMatrix, p::AbstractMatrix; length=1)
    length > 0 || throw(ArgumentError("length must be positive"))
    _validate_coordinates(q, p)
    angle = atan.(q[1, :], q[2, :])
    momentum = length .* (cos.(angle) .* p[1, :] .- sin.(angle) .* p[2, :])
    angle, momentum
end

"""Return the pendulum Hamiltonian for canonical coordinates."""
function pendulum_energy(angle, momentum; length=1, mass=1, gravity=1)
    _validate_pendulum_parameters(length, mass, gravity, 1, (0, 1))
    momentum .^ 2 ./ (2 * mass * length^2) .+ mass * gravity * length .* cos.(angle)
end

"""Return the pendulum Hamiltonian for Euclidean coordinates."""
function pendulum_energy(q::AbstractMatrix, p::AbstractMatrix; length=1, mass=1, gravity=1)
    _validate_pendulum_parameters(length, mass, gravity, 1, (0, 1))
    _validate_coordinates(q, p)
    vec(sum(abs2, p; dims=1)) ./ (2 * mass) .+ mass * gravity .* vec(q[2, :])
end

function _solution_vectors(solution)
    t = collect(solution.t)
    indices = 0:(length(t) - 1)
    angle = [solution.q[i][1] for i in indices]
    momentum = [solution.p[i][1] for i in indices]
    t, angle, momentum
end

"""
    pendulum_trajectory(; kwargs...)

Generate a deterministic trajectory using `GeometricProblems.Pendulum` and a
Gauss collocation integrator. Keywords are `length`, `mass`, `gravity`,
`timestep`, `timespan`, `angle`, `momentum`, and `integrator`.
"""
function pendulum_trajectory(; length=1.0, mass=1.0, gravity=1.0,
    timestep=0.1, timespan=(0.0, 10.0), angle=acos(0.4), momentum=0.0,
    integrator=Gauss(2))
    _validate_pendulum_parameters(length, mass, gravity, timestep, timespan)
    solution = integrate(
        hodeproblem([angle], [momentum]; parameters=(l=length, m=mass, g=gravity), timespan, timestep),
        integrator,
    )
    t, angle_values, momentum_values = _solution_vectors(solution)
    q, p = angular_to_euclidean(angle_values, momentum_values; length)
    T = promote_type(eltype(t), eltype(q), eltype(p))
    PendulumTrajectory{T,Vector{T},Matrix{T}}(T.(t), T.(angle_values), T.(momentum_values), T.(q), T.(p))
end

function _initial_condition_values(initial_condition)
    if initial_condition isa NamedTuple
        hasproperty(initial_condition, :angle) && hasproperty(initial_condition, :momentum) ||
            throw(ArgumentError("named initial conditions need :angle and :momentum"))
        return initial_condition.angle, initial_condition.momentum
    elseif initial_condition isa Tuple && length(initial_condition) == 2
        return initial_condition
    end
    throw(ArgumentError("initial conditions must be (angle, momentum) tuples or named tuples"))
end

"""Generate one trajectory for each `(angle, momentum)` initial condition."""
function pendulum_dataset(initial_conditions; kwargs...)
    trajectories = [begin
        angle, momentum = _initial_condition_values(initial_condition)
        pendulum_trajectory(; kwargs..., angle, momentum)
    end for initial_condition in initial_conditions]
    isempty(trajectories) && throw(ArgumentError("initial_conditions cannot be empty"))
    PendulumDataset(trajectories)
end

"""
    pendulum_matrix(dataset; concatenate=true)

Return the four-dimensional Euclidean representation in `(q₁, q₂, p₁, p₂)`
row order. Concatenated columns are returned by default; pass `false` to get
one matrix per trajectory.
"""
function pendulum_matrix(dataset::PendulumDataset; concatenate=true)
    matrices = [vcat(trajectory.q, trajectory.p) for trajectory in dataset.trajectories]
    concatenate ? hcat(matrices...) : matrices
end

"""Create a `GeometricMachineLearning.DataLoader` from the concatenated data."""
function pendulum_data_loader(dataset::PendulumDataset; suppress_info=false)
    DataLoader(pendulum_matrix(dataset); autoencoder=true, suppress_info)
end
