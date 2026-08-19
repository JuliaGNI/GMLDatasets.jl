```@meta
CurrentModule = GMLDatasets
```

# Four-dimensional pendulum dataset

`GMLDatasets` provides short, reproducible mathematical-pendulum trajectories
for numerical experiments and symplectic autoencoders. The canonical
coordinates are an angle `θ` and its conjugate momentum `pθ`. The four
Euclidean rows are `(q₁, q₂, p₁, p₂)`:

```math
q = (\ell\sin θ,\; \ell\cos θ), \qquad
p = (p_θ\cos θ/\ell,\; -p_θ\sin θ/\ell).
```

Generate one trajectory:

```julia
using GMLDatasets

trajectory = pendulum_trajectory(
    timespan=(0.0, 10.0), timestep=0.1,
    angle=0.8, momentum=0.0,
)
trajectory.q  # 2 × n Euclidean positions
trajectory.p  # 2 × n Euclidean momenta
```

Generate an ensemble and concatenate it for autoencoder input:

```julia
dataset = pendulum_dataset([
    (angle=0.2, momentum=0.0),
    (angle=0.8, momentum=0.2),
]; timespan=(0.0, 5.0), timestep=0.1)

data = pendulum_matrix(dataset)  # 4 × total_samples
loader = pendulum_data_loader(dataset; suppress_info=true)
```

The integrator includes both `timespan` endpoints. Use
`pendulum_energy` to evaluate the Hamiltonian in either canonical or
Euclidean coordinates, and `angular_to_euclidean`/
`euclidean_to_angular` for explicit coordinate conversion.

```@autodocs
Modules = [GMLDatasets]
Pages = ["pendulum_implementation.md"]
Order = [:type, :function]
```
