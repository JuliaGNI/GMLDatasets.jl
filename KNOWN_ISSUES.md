# Known issues

### K1 · The `DataLoader` constructor for labelled images is type piracy

- **location:** `src/data_loader.jl:29`
- **evidence:** `Aqua.test_piracies(GMLDatasets)` fails with one result; `test/quality/aqua.jl` marks
  it `broken = true`. The method adds to `GeometricMachineLearning.DataLoader` for argument types
  that this package does not own.
- **kind:** defect
- **found:** issue #29, part M29 of the test-suite migration

## Scripts

### K4 · Four MNIST scripts outside the revision harness use an API that GeometricOptimizers 0.9 does not define

- **location:** `scripts/geometric_optimizers/mnist.jl:404`, `mnist_cuda.jl:828`, `mnist_metal.jl:569`,
  `mnist_metal_short.jl:576`
- **evidence:** the four scripts call `Adam(T)` and `MomentumMethod(momentum_coefficient)`;
  GeometricOptimizers 0.9 defines only `Adam(; β₁, β₂, δ)` and `MomentumMethod(; α)`. The scripts
  environment resolves GeometricOptimizers 0.9, so each script stops at its first such call. The
  harness scripts are ported.
- **kind:** defect
- **found:** 2026-10-23, in the port of the revision harness to GeometricOptimizers 0.9

### K5 · On a CUDA host the environment preflight stops, and the pendulum stage cannot run, because the gradient of a device-resident point arrives on the host

- **location:** `scripts/revision/check_environment.jl:62`
- **evidence:** the preflight calls `rgrad(Y, randn(Float32, 4, 2))` at a `StiefelManifold` on the
  device. GeometricOptimizers 0.9 checks the backends there (`_check_gradient_backend`) and raises
  `ArgumentError: mixed backends`, where 0.8 moved the gradient across with a temporary
  `_match_backend`. The pendulum trainer passes Zygote's gradient through GML's `train!` to the
  optimizer step, and that gradient is on the host for device parameters (GML #258, AbstractNeuralNetworks
  #39, both open). Shown with a fake GPU backend on GeometricOptimizers 0.9.0; not run on a CUDA
  device. The image stages keep their parameters on the host and are unaffected.
- **kind:** upstream
- **found:** 2026-10-23, in the port of the revision harness to GeometricOptimizers 0.9
