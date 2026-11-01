```@meta
CurrentModule = GMLDatasets
```

# The Linear Wave Data Set

The one-dimensional linear wave equation with homogeneous Dirichlet boundary conditions, discretized
in space as a Hamiltonian system and integrated with the implicit midpoint rule, for a range of wave
speeds. Its solutions are thin pulses travelling at different speeds, a standard test for model
reduction because a linear subspace needs many modes to represent them.

[`linear_wave`](@ref) is one call to `GeometricProblems.LinearWave.hodeensemble` and one to
`GeometricIntegrators.integrate`; [`linear_wave_snapshots`](@ref) arranges the result as the tensor a
`GeometricMachineLearning.DataLoader` takes.

## The system

On ``\xi \in (-1/2, 1/2)`` the wave equation ``\partial_{tt}u = \mu^2\partial_{\xi\xi}u`` with
``u(t, \pm 1/2) = 0`` is discretized on the ``N`` interior points ``\xi_i = i\Delta\xi - 1/2``,
``i = 1, \ldots, N``, ``\Delta\xi = 1/(N+1)``. With ``q_i \approx u(t, \xi_i)`` and
``p_i = \dot q_i`` the semi-discrete system is canonical Hamiltonian with

```math
H(q, p; \mu) = \frac{1}{2}p^Tp - \frac{\mu^2}{2}q^TD_{\xi\xi}q,
\qquad D_{\xi\xi} = \frac{1}{\Delta\xi^2}\,\mathrm{tridiag}(1, -2, 1),
```

the full-order model of [buchfink2023symplectic](@cite). The state ``(q, p)`` has ``2N``
components.

Each wave speed ``\mu`` gets its own initial condition, `GeometricProblems.LinearWave.initial_condition`:
``q(0, \xi) = h(20\mu\,|\xi + \mu/2|)`` with the cubic spline ``h`` and ``p(0, \xi) = -\mu\,\partial_\xi q(0, \xi)``,
so the pulse starts at ``\xi = -\mu/2`` and travels to the right at speed ``\mu``. For
``\mu \in [5/12, 2/3]`` it stays away from the boundary for ``t \in [0, 1]``.

## Generating the data

The defaults are the training set of the symplectic-autoencoder experiment: twenty equally spaced
``\mu \in [5/12, 2/3]``, ``N = 128`` and 200 snapshots on ``[0, 1]``.

```julia
using GMLDatasets, GeometricMachineLearning

solution = linear_wave()
data = linear_wave_snapshots(solution)      # 256 × 200 × 20
dl = DataLoader(data; autoencoder = true)
```

The scripts of that experiment, which train symplectic autoencoders on these data and compare them with
the proper symplectic decomposition, are in `scripts/linear_wave/`; its README lists the commands.

## Library functions

```@docs
linear_wave
linear_wave_snapshots
linear_wave_parameters
```
