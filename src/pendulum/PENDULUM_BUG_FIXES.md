# Pendulum Bug Fixes

This file records bugs found and fixed while integrating the pendulum dataset
API into `GMLDatasets.jl`.

## 1. Offset-indexed solution data could not construct a trajectory

- **Symptom:** Constructing a `PendulumTrajectory` failed with a
  `DimensionMismatch` because the angle and momentum vectors had indices
  `0:n-1` instead of the ordinary Julia `1:n` axes.
- **Root cause:** `GeometricSolutions` stores its data series with an
  `OffsetArray`. Iterating over the stored data preserved those offset axes,
  while the trajectory type requires ordinary vectors.
- **Fix:** `_solution_vectors` now indexes the solution explicitly over
  `0:(nsteps - 1)` and builds fresh one-based vectors with comprehensions.
- **Regression check:** The smoke test verifies that a short trajectory can be
  constructed and has the expected time-grid and matrix dimensions.

## 2. `PendulumDataset` had an uninferable, unused type parameter

- **Symptom:** Calling `pendulum_dataset` failed with a `MethodError` when it
  attempted to construct `PendulumDataset(trajectories)`.
- **Root cause:** `PendulumDataset` declared a type parameter `T` that did not
  appear in any field. Julia therefore could not infer `T` from the vector of
  trajectories.
- **Fix:** Removed the unused `T` parameter. The dataset now has only the
  inferable vector type parameter used by its `trajectories` field.
- **Regression check:** The smoke test generates a two-trajectory dataset and
  verifies both per-trajectory and concatenated matrix shapes.

## 3. Regression test compared incompatible array shapes

- **Symptom:** The trajectory test compared a `2`-element column vector with
  the full `2×1` matrix returned by `angular_to_euclidean`.
- **Root cause:** The test extracted the tuple's first return value but did
  not flatten its single-sample column.
- **Fix:** The assertion now applies `vec` to the expected `2×1` matrix before
  comparing it with the trajectory column.
- **Regression check:** The focused trajectory test now checks the initial
  Euclidean coordinate without a shape mismatch.

## 4. Training script used the removed Adam learning-rate constructor

- **Symptom:** The one-epoch SAE smoke test failed with `MethodError: no method
  matching Adam(::Float32)`.
- **Root cause:** Current `GeometricOptimizers` stores the learning rate on the
  `GeometricMachineLearning.Optimizer`, while `Adam` accepts only its numeric
  type and algorithmic hyperparameters.
- **Fix:** Construct `Adam()` and pass `step_size=1f-3` to `Optimizer`.
- **Regression check:** The short CPU training smoke test must create an HDF5
  model artifact and the artifact must load in a clean Julia process.

## 5. SAE smoke used mismatched network and loader element types

- **Symptom:** After fixing the optimizer constructor, the smoke test failed in
  `mat_tensor_mul` because the network parameters were `Float32` while the
  `DataLoader` contained `Float64` data.
- **Root cause:** The dataset API preserves its native `Float64` trajectory
  values, while the script explicitly constructed a `Float32` network.
- **Fix:** Construct the CPU network with `Float64` parameters for this workflow.
- **Regression check:** The next smoke run reaches the model-save step.

## Validation context

The focused pendulum tests and dataset-generation smoke pass in the resolved
environment. The remaining training blocker is the HDF5 save-call API, not
dependency resolution. Full package tests, clean-process model loading, the
plotting smoke, and Documenter validation are still pending.
