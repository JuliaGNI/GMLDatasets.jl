import GeometricProblems.Pendulum as Pendulum

using GeometricIntegrators: Gauss, integrate
using GeometricSolutions: EnsembleSolution, GeometricSolution

# `parameters` is `GeometricBase.parameters`, reexported by `GeometricIntegrators`. It is imported
# under a different name because `parameters` is also the natural name for the keyword argument that
# carries `(l, m, g)` through to `GeometricProblems`, and one shadowing the other is a trap.
using GeometricIntegrators: parameters as problem_parameters

@doc raw"""
    pendulum(; qmin, qmax, pmin, pmax, qsamples, psamples, parameters, timespan, timestep, integrator)

Integrate an ensemble of mathematical pendula and return the `GeometricSolutions.EnsembleSolution`.

This is `GeometricProblems.Pendulum.hodeensemble` composed with `GeometricIntegrators.integrate` and
nothing more. The Hamiltonian is the one `GeometricProblems` defines,

```math
H(\theta, p_\theta) = \frac{p_\theta^2}{2m\ell^2} + mg\ell\cos(\theta),
```

so the potential is at its *minimum* at ``\theta = \pi``: the pendulum hangs down at ``\theta = \pi``
and stands upright at ``\theta = 0``. Trajectories with ``H < mg\ell`` librate about ``\theta = \pi``
and trajectories with ``H > mg\ell`` rotate.

The initial conditions are a Cartesian grid: `qsamples` angles spread evenly over `[qmin, qmax]`
times `psamples` momenta spread evenly over `[pmin, pmax]`, each given as a one-element vector
because the pendulum has one degree of freedom. The defaults are the grid `GeometricProblems` itself
uses — a hundred trajectories covering both libration and rotation — over a longer `timespan` than
its default, so that there is enough of each trajectory to learn from.

`parameters` is the ``(\ell, m, g)`` named tuple, `integrator` any `GeometricIntegrators` method.
The default is Gauss collocation with two stages, which is symplectic, so ``H`` is conserved to
rounding over the whole run rather than drifting.

The canonical coordinates the ensemble carries are two-dimensional. Use [`angular_to_euclidean`](@ref)
to lift them into ``\mathbb{R}^4``, where they trace out a two-dimensional submanifold — which is
what makes them a worthwhile test case for a `GeometricMachineLearning.SymplecticAutoencoder`:

```julia
using GeometricMachineLearning

dl = DataLoader(angular_to_euclidean(pendulum()); autoencoder = true)
```

See also [`pendulum_energy`](@ref).
"""
function pendulum(;
    qmin=[0.0],
    qmax=[2π],
    pmin=[-2.0],
    pmax=[2.0],
    qsamples=[10],
    psamples=[10],
    parameters=Pendulum.default_parameters(),
    timespan=(0.0, 10.0),
    timestep=0.1,
    integrator=Gauss(2))
    problem = Pendulum.hodeensemble(qmin, qmax, pmin, pmax, qsamples, psamples;
        parameters=parameters, timespan=timespan, timestep=timestep)
    integrate(problem, integrator)
end

@doc raw"""
    angular_to_euclidean(θ, pθ; l = 1)
    angular_to_euclidean(solution)

Lift the canonical pendulum coordinates into the Euclidean coordinates of the bob in the plane:

```math
q = (\ell\sin\theta,\; \ell\cos\theta), \qquad
p = (p_\theta\cos\theta/\ell,\; -p_\theta\sin\theta/\ell).
```

The lift is a symplectomorphism onto its image, so the Euclidean Hamiltonian
[`pendulum_energy`](@ref) agrees with `GeometricProblems.Pendulum.hamiltonian` and a symplectic
integrator stays symplectic under it. Its image is a two-dimensional submanifold of
``\mathbb{R}^4``: ``q`` lies on the circle of radius ``\ell`` and ``p`` is tangent to that circle,
i.e. ``\|q\| = \ell`` and ``q\cdot{}p = 0`` hold exactly.

Given two vectors of ``n`` samples this returns the pair of ``2\times{}n`` matrices ``q`` and ``p``.

Given a `GeometricSolutions.GeometricSolution` or `EnsembleSolution` — what [`pendulum`](@ref) hands
back — it returns the data set instead: one ``4\times{}n_t\times{}n`` array whose rows are
``(q_1, q_2, p_1, p_2)``, one column per time step and one slice per trajectory. That is the layout
every symplectic architecture in `GeometricMachineLearning` assumes (the first half of the rows is
``q``, the second half is ``p``) and the one `DataLoader` reads off a tensor, so nothing further is
needed:

```julia
dl = DataLoader(angular_to_euclidean(pendulum()); autoencoder = true)
```

``\ell`` is taken from the problem each solution was integrated from rather than assumed to be one.

[`euclidean_to_angular`](@ref) is the inverse, up to the ``2\pi``-periodicity of ``\theta``.
"""
function angular_to_euclidean(θ::AbstractVector, pθ::AbstractVector; l=1)
    l > 0 || throw(ArgumentError("the pendulum length must be positive, got l = $l"))
    axes(θ) == axes(pθ) ||
        throw(DimensionMismatch("θ has axes $(axes(θ)) but pθ has axes $(axes(pθ))"))
    q = permutedims(hcat(l .* sin.(θ), l .* cos.(θ)))
    p = permutedims(hcat(cos.(θ) .* pθ ./ l, -sin.(θ) .* pθ ./ l))
    q, p
end

# The pendulum has one degree of freedom, so `solution.q[:, 1]` is the angle over the whole time grid
# and `solution.p[:, 1]` its conjugate momentum. Both come back as `OffsetVector`s indexed from zero,
# because a `GeometricSolution` counts the initial condition as step 0; they are collected so that
# nothing downstream ever sees a zero-based axis.
_canonical(solution::GeometricSolution) = collect(solution.q[:, 1]), collect(solution.p[:, 1])

_lift(solution::GeometricSolution) = angular_to_euclidean(_canonical(solution)...;
    l=problem_parameters(solution.problem).l)

function angular_to_euclidean(solution::GeometricSolution)
    q, p = _lift(solution)
    reshape(vcat(q, p), 4, :, 1)
end

function angular_to_euclidean(solution::EnsembleSolution)
    lifted = [vcat(_lift(s)...) for s in solution.s]
    reshape(reduce(hcat, lifted), 4, :, length(lifted))
end

@doc raw"""
    euclidean_to_angular(q, p; l = 1)

Project the Euclidean coordinates of the bob back onto the canonical ``(\theta, p_\theta)``.

`q` and `p` are ``2\times{}n`` matrices as produced by [`angular_to_euclidean`](@ref), which this
inverts. The angle comes back wrapped into ``(-\pi, \pi]``, so the round trip is the identity only
for angles that were in that interval to begin with.
"""
function euclidean_to_angular(q::AbstractMatrix, p::AbstractMatrix; l=1)
    l > 0 || throw(ArgumentError("the pendulum length must be positive, got l = $l"))
    size(q, 1) == size(p, 1) == 2 ||
        throw(DimensionMismatch("q and p must have two rows, got $(size(q, 1)) and $(size(p, 1))"))
    axes(q, 2) == axes(p, 2) ||
        throw(DimensionMismatch("q has $(size(q, 2)) columns but p has $(size(p, 2))"))
    θ = atan.(q[1, :], q[2, :])
    pθ = l .* (cos.(θ) .* p[1, :] .- sin.(θ) .* p[2, :])
    θ, pθ
end

@doc raw"""
    pendulum_energy(θ, pθ, parameters = GeometricProblems.Pendulum.default_parameters())
    pendulum_energy(q, p, parameters = ...)
    pendulum_energy(data, parameters = ...)
    pendulum_energy(solution)

Evaluate the pendulum Hamiltonian, in canonical or in Euclidean coordinates.

Given two vectors this is `GeometricProblems.Pendulum.hamiltonian` broadcast over them. Given the
Euclidean coordinates of [`angular_to_euclidean`](@ref) — either as two arrays whose first axis is
the two Euclidean components, or as the single four-row array that function returns for a solution —
it is the same Hamiltonian written in those coordinates,

```math
H(q, p) = \frac{\|p\|^2}{2m} + mgq_2,
```

which is what the lift being a symplectomorphism buys: ``\|p\|^2 = p_\theta^2/\ell^2`` because ``p``
is tangent to the circle, and ``q_2 = \ell\cos\theta``. Given a solution of [`pendulum`](@ref) the
parameters are read off the problem it was integrated from.

The result keeps every axis but the first, so a ``2\times{}n_t\times{}n`` tensor gives an
``n_t\times{}n`` matrix of energies — one column per trajectory, which is how a symplectic
integrator is checked for drift.
"""
pendulum_energy(θ::AbstractVector, pθ::AbstractVector,
    parameters::NamedTuple=Pendulum.default_parameters()) =
    Pendulum.hamiltonian.(zero(eltype(θ)), θ, pθ, Ref(parameters))

function pendulum_energy(q::AbstractArray, p::AbstractArray,
    parameters::NamedTuple=Pendulum.default_parameters())
    size(q, 1) == size(p, 1) == 2 ||
        throw(DimensionMismatch("q and p must have two rows, got $(size(q, 1)) and $(size(p, 1))"))
    axes(q) == axes(p) ||
        throw(DimensionMismatch("q has axes $(axes(q)) but p has axes $(axes(p))"))
    dropdims(sum(abs2, p; dims=1); dims=1) ./ (2 * parameters.m) .+
    (parameters.m * parameters.g) .* selectdim(q, 1, 2)
end

function pendulum_energy(data::AbstractArray, parameters::NamedTuple=Pendulum.default_parameters())
    size(data, 1) == 4 ||
        throw(DimensionMismatch("the lifted data must have four rows, got $(size(data, 1))"))
    pendulum_energy(selectdim(data, 1, 1:2), selectdim(data, 1, 3:4), parameters)
end

pendulum_energy(solution::GeometricSolution) =
    pendulum_energy(_canonical(solution)..., problem_parameters(solution.problem))

pendulum_energy(solution::EnsembleSolution) =
    reduce(hcat, pendulum_energy(s) for s in solution.s)
