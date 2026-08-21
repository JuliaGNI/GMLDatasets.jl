```@meta
CurrentModule = GMLDatasets
```

# The Pendulum Data Set

The mathematical pendulum, integrated symplectically and lifted into ``\mathbb{R}^4``. It is small,
deterministic and needs no download, and it is the smallest system on which a
`GeometricMachineLearning.SymplecticAutoencoder` has something real to do: the data are
two-dimensional but they arrive in four dimensions, on a curved submanifold that no linear method
recovers exactly.

Everything here is [`pendulum`](@ref) and [`angular_to_euclidean`](@ref) — one call to
[`GeometricProblems`](https://github.com/JuliaGNI/GeometricProblems.jl) and
[`GeometricIntegrators`](https://github.com/JuliaGNI/GeometricIntegrators.jl), and one coordinate
change. The data loader is `GeometricMachineLearning`'s own; this package adds no adapter for it.

## The system

The Hamiltonian is the one `GeometricProblems.Pendulum` defines,

```math
H(\theta, p_\theta) = \frac{p_\theta^2}{2m\ell^2} + mg\ell\cos(\theta),
```

with the angle ``\theta``, its conjugate momentum ``p_\theta``, and the length, mass and
gravitational acceleration ``\ell``, ``m``, ``g``.

!!! note "The pendulum hangs down at θ = π"
    The potential is ``+mg\ell\cos\theta``, not ``-mg\ell\cos\theta``, so it is at its *minimum* at
    ``\theta = \pi`` and at its maximum at ``\theta = 0``. The stable equilibrium is therefore
    ``\theta = \pi`` and the inverted one is ``\theta = 0``. Trajectories with ``H < mg\ell``
    librate about ``\pi``, trajectories with ``H > mg\ell`` rotate, and ``H = mg\ell`` is the
    separatrix between them.

## Generating the data

[`pendulum`](@ref) integrates a grid of initial conditions and hands back the
[`GeometricSolutions.EnsembleSolution`](@extref):

```@example pendulum
using GMLDatasets

solution = pendulum()
(length(solution.s), length(solution.t))
```

That is a hundred trajectories of a hundred and one time steps each, spread over
``\theta \in [0, 2\pi]`` and ``p_\theta \in [-2, 2]`` — the grid `GeometricProblems` itself uses,
which covers both libration and rotation. The keyword arguments are the six grid bounds, the
physical `parameters`, the `timespan` and `timestep`, and the `integrator`. The default integrator
is two-stage Gauss collocation, which is symplectic.

## The lift into four dimensions

[`angular_to_euclidean`](@ref) replaces the angle by the position of the bob in the plane and
``p_\theta`` by the corresponding Euclidean momentum:

```math
q = (\ell\sin\theta,\; \ell\cos\theta), \qquad
p = (p_\theta\cos\theta/\ell,\; -p_\theta\sin\theta/\ell).
```

The map is a symplectomorphism onto its image, and its image is the tangent bundle of the circle of
radius ``\ell``: two dimensions of data inside four, held there by ``\|q\| = \ell`` and
``q\cdot{}p = 0``.

```@example pendulum
data = angular_to_euclidean(solution)
size(data)
```

The rows are ``(q_1, q_2, p_1, p_2)`` — the ``q``-components first, then the ``p``-components, which
is what every symplectic architecture in `GeometricMachineLearning` assumes. The two constraints
hold to rounding:

```@example pendulum
q, p = data[1:2, :, :], data[3:4, :, :]
(maximum(abs, sum(abs2, q; dims = 1) .- 1), maximum(abs, sum(q .* p; dims = 1)))
```

[`euclidean_to_angular`](@ref) inverts the lift, up to the ``2\pi``-periodicity of the angle.

## Handing it to a network

There is no `pendulum_data_loader` here: the array above is already in the shape
[`GeometricMachineLearning.DataLoader`](@extref) reads, so the constructor takes it directly.

```@example pendulum
using GeometricMachineLearning

dl = DataLoader(data; autoencoder = true, suppress_info = true)
(dl.input_dim, dl.input_time_steps, dl.n_params)
```

`autoencoder = true` is what marks the columns as independent samples rather than a time series,
which is what an autoencoder wants. From here on it is an ordinary `GeometricMachineLearning`
training run; `scripts/pendulum/train_sae.jl` is a complete one.

!!! tip "Train across the separatrix"
    The standard reduction ``(\theta, p_\theta)`` cannot be represented globally by two ordinary
    real-valued coordinates because ``\theta`` is periodic. That does **not** prevent an SAE from
    finding a different two-dimensional representation of a bounded part of the cylinder. For
    example, it can represent the angular direction by an annular geometry in its latent plane.
    `scripts/pendulum/train_sae.jl` therefore trains on the default bounded grid, which contains both
    librating and rotating trajectories and crosses the separatrix. The learned coordinates need not
    resemble ``(\theta, p_\theta)`` to represent those trajectories faithfully.

## Checking the integrator

[`pendulum_energy`](@ref) evaluates ``H`` in either set of coordinates — on the canonical solution,
on the lifted array, or on a pair of coordinate arrays. That the two agree is the statement that the
lift preserved the dynamics:

```@example pendulum
energy = pendulum_energy(data)
maximum(abs, energy .- pendulum_energy(solution))
```

And because the integrator is symplectic, the energy of each trajectory stays in a bounded band
around its initial value rather than drifting away from it over the run:

```@example pendulum
maximum(abs, energy .- energy[1:1, :])
```

`scripts/pendulum/plot_dataset.jl` draws both of these — the phase portrait in each set of
coordinates, and the energy error over time.

## Other parameters

`parameters` is the ``(\ell, m, g)`` named tuple that `GeometricProblems` takes, and it travels with
the solution: [`angular_to_euclidean`](@ref) reads ``\ell`` off the problem rather than assuming it
is one, and `pendulum_energy(solution)` reads all three.

```@example pendulum
heavy = pendulum(; parameters = (l = 2.0, m = 3.0, g = 9.81),
                   qsamples = [4], psamples = [3], timespan = (0.0, 2.0))
maximum(abs, sum(abs2, angular_to_euclidean(heavy)[1:2, :, :]; dims = 1))
```

For the Euclidean methods of [`pendulum_energy`](@ref) the parameters are an argument, since a bare
array does not carry them:

```@example pendulum
const heavy_parameters = (l = 2.0, m = 3.0, g = 9.81)
maximum(abs, pendulum_energy(angular_to_euclidean(heavy), heavy_parameters) .-
             pendulum_energy(heavy))
```
