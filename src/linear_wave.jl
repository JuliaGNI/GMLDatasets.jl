import GeometricProblems.LinearWave as LinearWave

using GeometricIntegrators: ImplicitMidpoint

@doc raw"""
    linear_wave(; N, μ, timespan, n_time_steps, integrator)

Integrate the discretized linear wave equation for each wave speed in `μ` and return the
`GeometricSolutions.EnsembleSolution`.

This is `GeometricProblems.LinearWave.hodeensemble` composed with `GeometricIntegrators.integrate`
and nothing more. The problem is the one `GeometricProblems` defines: the wave equation
``\partial_{tt}u = \mu^2\partial_{\xi\xi}u`` on ``\xi \in (-1/2, 1/2)`` with homogeneous Dirichlet
boundary conditions, discretized on the `N` interior points ``\xi_i = i\Delta\xi - 1/2``,
``\Delta\xi = 1/(N+1)``, with the Hamiltonian

```math
H(q, p; \mu) = \frac{1}{2}p^Tp - \frac{\mu^2}{2}q^TD_{\xi\xi}q ,
```

where ``D_{\xi\xi}`` is the central-difference matrix. The state has ``2N`` components.

Every wave speed gets its own initial condition, `GeometricProblems.LinearWave.initial_condition(μ, N)`:
a cubic-spline pulse that starts at ``\xi = -\mu/2`` with width ``\propto 1/\mu`` and initial momentum
``p = -\mu\,\partial_\xi q``, so that it travels to the right at speed ``\mu``. Over the default
`timespan` it does not reach the boundary for any ``\mu \in [5/12, 2/3]``.

The defaults are the training set of the symplectic-autoencoder experiment: twenty equally spaced
values ``\mu \in [5/12, 2/3]``, ``N = 128``, `n_time_steps = 200` snapshots on ``t \in [0, 1]``
(so the time step is ``1/199``), integrated with the implicit midpoint rule.

Use [`linear_wave_snapshots`](@ref) to turn the solution into the data tensor a
`GeometricMachineLearning.DataLoader` takes:

```julia
using GeometricMachineLearning

dl = DataLoader(linear_wave_snapshots(linear_wave()); autoencoder = true)
```
"""
function linear_wave(;
        N::Integer = 128,
        μ = range(5 / 12, 2 / 3; length = 20),
        timespan = (0.0, 1.0),
        n_time_steps::Integer = 200,
        integrator = ImplicitMidpoint())
    μs = collect(float.(μ))
    isempty(μs) && throw(ArgumentError("μ must contain at least one wave speed"))
    n_time_steps ≥ 2 ||
        throw(ArgumentError("n_time_steps must be at least 2, got $n_time_steps"))
    ics = [LinearWave.initial_condition(μᵢ, N) for μᵢ in μs]
    timestep = (timespan[2] - timespan[1]) / (n_time_steps - 1)
    ensemble = LinearWave.hodeensemble(N, [ic.q for ic in ics], [ic.p for ic in ics];
        parameters = [(μ = μᵢ,) for μᵢ in μs], timespan = timespan, timestep = timestep)
    integrate(ensemble, integrator)
end

"""
    linear_wave_snapshots(solution)

The snapshots of a [`linear_wave`](@ref) solution as one ``2N × n_t × n_μ`` array: the rows are
``(q_1, …, q_N, p_1, …, p_N)``, one column per time step (the initial condition first) and one slice
per wave speed, in the order of `μ`.

That is the layout every symplectic architecture in `GeometricMachineLearning` assumes (the first half
of the rows is ``q``, the second half ``p``) and the one `DataLoader` reads off a tensor.

Given a single `GeometricSolution` it returns the ``2N × n_t`` matrix of that trajectory.
"""
function linear_wave_snapshots(solution::GeometricSolution)
    # `solution.q[n]` is the state at step n, counted from zero
    reduce(hcat, [vcat(solution.q[n], solution.p[n]) for n in 0:(length(solution.t) - 1)])
end

function linear_wave_snapshots(solution::EnsembleSolution)
    cat((linear_wave_snapshots(s) for s in solution)...; dims = 3)
end

"""
    linear_wave_parameters(solution)

The wave speeds ``\\mu`` of a [`linear_wave`](@ref) solution, one per trajectory and in the order of
the slices of [`linear_wave_snapshots`](@ref).
"""
function linear_wave_parameters(solution::EnsembleSolution)
    [problem_parameters(s.problem).μ
     for s in solution]
end
