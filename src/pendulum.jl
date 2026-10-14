import GeometricProblems.Pendulum as Pendulum

using GeometricEquations: HODEEnsemble
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

`momentum_fractions` replaces that grid with a separatrix-relative one, which is what a data set
straddling the separatrix wants. Passing a vector of numbers `f` takes the same `qsamples` angles
and gives each of them the momenta ``p_\theta = f\,p_\mathrm{sep}(\theta)``, where
[`separatrix_momentum`](@ref) is the momentum that puts the pendulum exactly on the separatrix at
that angle. The energy of such an initial condition is

```math
H = mg\ell\left[f^2(1 - \cos\theta) + \cos\theta\right] ,
```

so ``|f| < 1`` librates, ``|f| = 1`` is the separatrix itself and ``|f| > 1`` rotates, *whatever the
angle*, and the sign of ``f`` picks the direction of rotation. A Cartesian ``(\theta, p_\theta)``
grid cannot express that: a fixed ``p_\theta`` crosses the separatrix as ``\theta`` varies, so a
rectangle in ``(\theta, p_\theta)`` always mixes the two regimes and never controls how close to
the separatrix the data comes. With fractions, how densely the crossing is covered is chosen
directly — fractions just above one in modulus fill the band of rotating orbits nearest the
separatrix, which is the region a single Cartesian grid leaves empty.

`pmin`, `pmax` and `psamples` are unused when `momentum_fractions` is given. The fraction is
undefined where ``p_\mathrm{sep}`` vanishes, that is at the unstable equilibrium ``\theta \equiv 0``
modulo ``2\pi``, and an angle grid that contains one of those points is rejected rather than
silently collapsed onto the fixed point.

`parameters` is the ``(\ell, m, g)`` named tuple, `integrator` any `GeometricIntegrators` method.
The default is Gauss collocation with two stages, which is symplectic, so the energy of each
trajectory oscillates within a bounded band around its initial value over the whole run rather than
drifting away from it. On the default grid that band is about ``2\cdot{}10^{-6}`` wide.

The canonical coordinates the ensemble carries are two-dimensional. Use [`angular_to_euclidean`](@ref)
to lift them into ``\mathbb{R}^4``, where they trace out a two-dimensional submanifold — which is
what makes them a worthwhile test case for a `GeometricMachineLearning.SymplecticAutoencoder`:

```julia
using GeometricMachineLearning

dl = DataLoader(angular_to_euclidean(pendulum()); autoencoder = true)
```

See also [`separatrix_momentum`](@ref) and [`pendulum_energy`](@ref).
"""
function pendulum(;
        qmin = [0.0],
        qmax = [2π],
        pmin = [-2.0],
        pmax = [2.0],
        qsamples = [10],
        psamples = [10],
        momentum_fractions = nothing,
        parameters = Pendulum.default_parameters(),
        timespan = (0.0, 10.0),
        timestep = 0.1,
        integrator = Gauss(2))
    problem = if isnothing(momentum_fractions)
        Pendulum.hodeensemble(qmin, qmax, pmin, pmax, qsamples, psamples;
            parameters = parameters, timespan = timespan, timestep = timestep)
    else
        _separatrix_relative_ensemble(qmin, qmax, qsamples, momentum_fractions,
            parameters, timespan, timestep)
    end
    integrate(problem, integrator)
end

@doc raw"""
    separatrix_momentum(θ, parameters = GeometricProblems.Pendulum.default_parameters())

The momentum that puts the pendulum exactly on the separatrix at angle ``\theta``:

```math
p_\mathrm{sep}(\theta) = m\ell\sqrt{2g\ell\,(1 - \cos\theta)} ,
```

the positive root of ``H(\theta, p_\theta) = mg\ell``. It vanishes at the unstable equilibrium
``\theta \equiv 0`` modulo ``2\pi``, where the separatrix meets itself, and is largest at
``\theta = \pi``.

This is the scale `pendulum`'s `momentum_fractions` measures against.
"""
function separatrix_momentum(θ, parameters::NamedTuple = Pendulum.default_parameters())
    parameters.m * parameters.l *
        sqrt(2 * parameters.g * parameters.l * (1 - cos(θ)))
end

# The curved counterpart of `GeometricProblems`' `_pode_samples`. `hodeensemble` only builds
# Cartesian grids, so the ensemble is assembled here from the same pieces it uses: the pendulum's
# vector fields and Hamiltonian, and one initial condition per (angle, fraction) pair.
function _separatrix_relative_ensemble(qmin, qmax, qsamples, fractions,
        parameters, timespan, timestep)
    length(qmin) == length(qmax) == length(qsamples) == 1 || throw(ArgumentError(
        "the pendulum has one degree of freedom, so qmin, qmax and qsamples must have one " *
        "element each, got $(length(qmin)), $(length(qmax)) and $(length(qsamples))"))
    isempty(fractions) &&
        throw(ArgumentError("momentum_fractions must contain at least one fraction"))
    all(isfinite, fractions) || throw(ArgumentError(
        "momentum_fractions must be finite, got $fractions"))

    θs = range(qmin[1], qmax[1]; length = qsamples[1])
    psep = separatrix_momentum.(θs, Ref(parameters))
    any(iszero, psep) && throw(ArgumentError(
        "the separatrix momentum vanishes at the unstable equilibrium, so a momentum fraction " *
        "is undefined there; the angle grid range($(qmin[1]), $(qmax[1]); length = " *
        "$(qsamples[1])) contains such an angle"))

    q₀ = [[θ] for θ in θs for _ in fractions]
    p₀ = [[f * p] for p in psep for f in fractions]
    HODEEnsemble(Pendulum.pendulum_pode_v, Pendulum.pendulum_pode_f, Pendulum.hamiltonian,
        timespan, timestep, q₀, p₀; parameters = parameters)
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
function angular_to_euclidean(θ::AbstractVector, pθ::AbstractVector; l = 1)
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
function _canonical(solution::GeometricSolution)
    collect(solution.q[:, 1]), collect(solution.p[:, 1])
end

function _lift(solution::GeometricSolution)
    angular_to_euclidean(_canonical(solution)...;
        l = problem_parameters(solution.problem).l)
end

function angular_to_euclidean(solution::GeometricSolution)
    q, p = _lift(solution)
    reshape(vcat(q, p), 4, :, 1)
end

function angular_to_euclidean(solution::EnsembleSolution)
    lifted = [vcat(_lift(s)...) for s in solution]
    reshape(reduce(hcat, lifted), 4, :, length(lifted))
end

@doc raw"""
    euclidean_to_angular(q, p; l = 1)

Project the Euclidean coordinates of the bob back onto the canonical ``(\theta, p_\theta)``.

`q` and `p` are ``2\times{}n`` matrices as produced by [`angular_to_euclidean`](@ref), which this
inverts. The angle comes back wrapped into ``(-\pi, \pi]``, so the round trip is the identity only
for angles that were in that interval to begin with.
"""
function euclidean_to_angular(q::AbstractMatrix, p::AbstractMatrix; l = 1)
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
    parameters::NamedTuple = Pendulum.default_parameters()) = Pendulum.hamiltonian.(
    zero(eltype(θ)), θ, pθ, Ref(parameters))

function pendulum_energy(q::AbstractArray, p::AbstractArray,
        parameters::NamedTuple = Pendulum.default_parameters())
    size(q, 1) == size(p, 1) == 2 ||
        throw(DimensionMismatch("q and p must have two rows, got $(size(q, 1)) and $(size(p, 1))"))
    axes(q) == axes(p) ||
        throw(DimensionMismatch("q has axes $(axes(q)) but p has axes $(axes(p))"))
    dropdims(sum(abs2, p; dims = 1); dims = 1) ./ (2 * parameters.m) .+
    (parameters.m * parameters.g) .* selectdim(q, 1, 2)
end

function pendulum_energy(data::AbstractArray, parameters::NamedTuple = Pendulum.default_parameters())
    size(data, 1) == 4 ||
        throw(DimensionMismatch("the lifted data must have four rows, got $(size(data, 1))"))
    pendulum_energy(selectdim(data, 1, 1:2), selectdim(data, 1, 3:4), parameters)
end

function pendulum_energy(solution::GeometricSolution)
    pendulum_energy(_canonical(solution)..., problem_parameters(solution.problem))
end

function pendulum_energy(solution::EnsembleSolution)
    reduce(hcat, [pendulum_energy(s) for s in solution])
end
