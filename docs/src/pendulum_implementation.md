# Pendulum dataset implementation note

The canonical API is `pendulum_trajectory` for one initial condition and
`pendulum_dataset` for a collection. A `PendulumTrajectory` stores the
integrator time grid, canonical angular coordinates, and Euclidean coordinates;
`PendulumDataset` preserves trajectory boundaries through its `trajectories`
field.

The Euclidean coordinates are column-oriented and ordered as

```text
(q₁, q₂, p₁, p₂) =
(length*sin(θ), length*cos(θ), cos(θ)*pθ/length, -sin(θ)*pθ/length).
```

`timespan` endpoints are included. Thus a trajectory has one column per value
in the integrator time grid, and `pendulum_matrix(dataset)` concatenates those
columns across trajectories. Physical parameters (`length`, `mass`, and
`gravity`), integration controls (`timestep`, `timespan`, and `integrator`),
and each initial `(angle, momentum)` are explicit keyword arguments.

The numerical sanity checks are coordinate round trips, fixed-radius position,
tangency of Euclidean momentum, and bounded Hamiltonian drift for a short
Gauss-integrated trajectory. `pendulum_data_loader` is only an adapter for the
existing `GeometricMachineLearning.DataLoader`; generation and conversion do
not depend on a training session or random state.
