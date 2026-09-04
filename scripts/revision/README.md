# Revision experiment runner

Run these commands from any directory. The runner creates a timestamped directory, streams every
stage to durable logs, records stage status, captures the Julia/CUDA/NVIDIA and Git environments,
and always packages the partial or complete directory as a `.tar.gz` plus SHA-256 checksum.

## Setup

Instantiate the pinned scripts environment before disconnecting from the network:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The checked-in manifest records the reviewed revision-experiment stack:
`GeometricMachineLearning` v0.7.0, `NeuralNetworkParameters` v0.3.0, and temporarily the exact
`ae50ece` head of
[`GeometricOptimizers.jl` PR #79](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/79).
The Git revision supplies the observer and `PhaseTimer` used by the timing adapter below, from
[PR #78](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/78) which #79 is branched off, and
the two fixes the GPU runs required. `similar` of a horizontal lift allocated on the host, which
made the optimizer of a device-resident network a `MethodError` and stopped the pendulum stage of run
`20260903T125418Z_smoke`; and then, with that out of the way, `rgrad` receiving an ambient gradient
the pullback had left on the host beside a device-resident point, which is a CPU `gemm!` on a device
pointer and stopped the pendulum stage of runs `20260903T185459Z_smoke` and `20260903T191704Z_smoke`
at the first optimizer step.

That second one is a **temporary** shim in `GeometricOptimizers` for a defect in the packages that
produce the gradient: [`GeometricMachineLearning` #258](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/258)
and [`AbstractNeuralNetworks` #39](https://github.com/JuliaGNI/AbstractNeuralNetworks.jl/issues/39).
While it is in place, the pendulum stage pays a host-to-device transfer per manifold leaf per step
inside the region the phase timer attributes to the step, so **the pendulum stage's decomposed
timings are an upper bound rather than a measurement** and its `optimizer_state_direction_seconds`
in particular carry that transfer. The image stages are unaffected: they never take this path.
Replace the pin with the exact registry release before freezing or running the experiment head, and
expect the shim to be gone by then. The environment preflight checks the reviewed versions, the
presence of `PhaseTimer`, and that an optimizer cache and state can be built for a parameter set that
lives on the GPU.

The full run rejects a dirty tree and any CUDA device whose name does not contain `RTX 4090`.
Use `--allow-dirty` only deliberately; the patch and status are included in the bundle. Use
`--allow-any-gpu` only for local smoke testing.

## Smoke test

For an SSH-accessible run that survives disconnects, launch it in a detached GNU `screen` session:

```bash
scripts/revision/run_in_screen.sh --session gml-smoke --smoke
screen -r gml-smoke
```

Detach without stopping the run by pressing `Ctrl-A`, then `D`. Reconnect later with
`screen -r gml-smoke`; use `screen -ls` to list sessions. For local CPU-only testing, invoke
the complete smoke directly rather than using the workstation command above:

```bash
scripts/revision/run_experiments.sh --smoke --allow-dirty --allow-any-gpu --allow-no-cuda
```

Limit stages with `--stages mnist,pendulum` and configurations with
`--configurations geometric-adam-cayley`. `--configurations all` selects all five rows, including
the stable key `scalar-moment-adam`, reported as
`Scalar Moment Adam (Stiefel, Cayley retraction)`. The legacy keys `adam-stiefel` and
`adam-regular` remain accepted aliases. Smoke mode uses one seed and two epochs.
For the CPU control-flow smoke only, each image data set is limited to its first 32 training and
32 test samples with one 32-sample batch. Override that bound with `MNIST_SMOKE_SAMPLES`, or set
`MNIST_TRAINING_SAMPLES`, `MNIST_TEST_SAMPLES`, and `MNIST_BATCH_SIZE` explicitly. Full mode never
applies this subset and retains the complete data sets.

## Optimizer-step timing records

The MNIST and Fashion-MNIST repetition trainer writes schema-version-4 run records to its
`*-runs.csv` file and stores the same timings in its JLD2 checkpoint. The observer measures three
exclusive categories. It synchronizes the active CUDA device immediately before every enter and
exit timestamp; the same boundary calls are no-ops on CPU. When phases nest, the enclosing timer is
paused, so the categories never overlap:

- **Gradient/AD** begins immediately before the flat gradient callback and ends when it returns. It
  includes the reverse pass and its output write, but not the subsequent Riemannian tangent
  projection. The mixed-tree composite observes its one shared whole-tree gradient and suppresses
  gradient events for leaf callbacks that only copy slices of that frozen gradient.
- **Optimizer-state/direction** begins before `increase_iteration_number!` and ends after
  `solver_step!` and the following state `update!`. Its exclusive intervals include state, cache,
  moment, direction, line-search-control, and tangent-projection work. For the mixed-tree composite,
  they also include parameter flattening, cache invalidation, leaf gradient-slice copies, and the
  sequential leaf bookkeeping. Nested gradient, objective, and retraction/application intervals
  are excluded.
- **Retraction/application** begins immediately before and ends immediately after each observed
  trial or accepted retraction/application operation. It includes the associated section updates
  and solution/parameter application copies, but excludes any nested gradient interval.

Objective and merit evaluations inside an optimizer step are observed only to pause an enclosing
category. Their time is deliberately not emitted. Batch selection and upload, the post-step loss,
accuracy and drift evaluation, reporting, and other orchestration also lie outside the three
categories. Consequently, their totals are disjoint optimizer-step components and are not expected
to add up to `total_seconds`.

Before each measured repetition, the trainer executes the exact optimizer path for one discarded
warm-up step, synchronizes the device, resets every timing accumulator, and discards the warmed
state. It then repeats the same seed setup to reconstruct the task-local and data-order RNGs,
parameters, randomized global sections, optimizer states, and caches. The measured run therefore
starts from the state it would have had without the compilation warm-up.

Schema version 4 adds these seven columns, in this order:

```text
timed_steps
gradient_ad_seconds_total
gradient_ad_seconds_per_step
optimizer_state_direction_seconds_total
optimizer_state_direction_seconds_per_step
retraction_application_seconds_total
retraction_application_seconds_per_step
```

The CSV uses those names directly. JLD2 appends the one-based result number, for example
`timed_steps1` and `gradient_ad_seconds_total1`. For every successful result, `timed_steps` equals
the number of completed minibatch optimizer steps recorded in the loss series; all six seconds
fields are finite and nonnegative, and each per-step value is its total divided by `timed_steps`.
A successful zero-step result has zero in all seven fields. An `exception` CSV row instead has
`timed_steps = 0` and `NaN` in all six seconds fields: no successful result or partial timing
snapshot was retained, so these values mean “unavailable,” not a measured zero.

The local synthetic CPU exercise used to validate this control flow and schema is test evidence
only. It is neither a scientific experiment nor evidence about CPU or GPU performance; use only a
gated experiment bundle for reported timings.

## Retraction benchmark records

The retraction stage writes schema-version-1 CSV to `retraction-runs.csv`. It is a thin
machine-readable wrapper around the algorithms and seeded lift sweep in the checked-out
GeometricOptimizers `scripts/retraction_accuracy.jl`; it never parses that script's prose output.
Each row records the algorithm, backend/device and precision, full square lift shape and Stiefel
column count, lift norm, agreement with host `AugmentedPade`, forward error against a host
`Float64` matrix exponential, manifold-constraint error, synchronized runtime, and allocated
bytes. CPU rows report Julia host allocation; CUDA rows report CUDA device allocation.

Every algorithm/shape/scale path emits repetition `0` with `warmup=true` before repetitions `1` and
later. Treat only successful `warmup=false` rows as steady-state measurements. An exception emits
an explicit `success=false` row with its type and message, `NaN` error fields, and zero memory bytes;
the benchmark and orchestration still fail so a partial result cannot pass the gate.

CPU mode runs `ScaledSquaring`, `NativePade`, and `AugmentedPade` on the host. CUDA mode runs the
first two on the physical GPU, synchronizes around timing, transfers each result to the host for
comparison, and runs `AugmentedPade` only on the host because it uses dense LAPACK. Each row carries
the exact GeometricOptimizers commit and dirty flag. `geometricoptimizers-retraction.patch` is the
corresponding SHA-256-addressed binary patch; unlike plain `git diff`, it also contains untracked
source files.

For a small CPU exercise and an independent validation:

```bash
julia --project=/path/to/GeometricOptimizers scripts/revision/retraction_records.jl \
  --go-repo /path/to/GeometricOptimizers --output /tmp/retraction-runs.csv \
  --patch-output /tmp/geometricoptimizers-retraction.patch --backend cpu \
  --rows 6 --columns 2 --scales 0.1 --repetitions 1
julia --project=scripts scripts/revision/validate_retraction_records.jl \
  --input /tmp/retraction-runs.csv --go-repo /path/to/GeometricOptimizers \
  --require ScaledSquaring:CPU --require NativePade:CPU --require AugmentedPade:CPU
```

The runner uses CUDA/`Float32` unless `--allow-no-cuda` explicitly selects the CPU/`Float64` smoke
path. Its default full sweep uses the upstream eight scales and 20 measured repetitions; smoke uses
one scale and one measured repetition. `RETRACTION_PRECISION`, `RETRACTION_ROWS`,
`RETRACTION_COLUMNS`, `RETRACTION_SCALES`, `RETRACTION_REPETITIONS`, and `RETRACTION_SEED` provide
explicit overrides. The physical-GPU run remains behind the final release gate.

## Full RTX 4090 run

The orchestration matrix is complete: `--full` needs no development bypass, and its default `all`
selection contains all five configurations. The experiment-local mixed-tree composite applies an
independent `ScalarMomentAdam` to each Stiefel leaf and ordinary `Adam` to each Euclidean leaf. It
computes one shared whole-tree gradient, then updates leaves sequentially in parameter-layout order.
The proposed `geometric-adam-cayley` row instead uses coordinate-wise moments, while
`standard-adam` is the unconstrained non-geometric ablation. Do not launch the workstation run until
the separately documented release gate is satisfied.

```bash
scripts/revision/run_in_screen.sh --session gml-revision --full
screen -r gml-revision
```

Full mode requires exactly ten seeds. Override them explicitly with
`--seeds 1234,1235,1236,1237,1238,1239,1240,1241,1242,1243`. The default stage list is
`mnist,fashion-mnist,pendulum,retraction`.

Give `geometric-adam-cayley` and `scalar-moment-adam` the same number of learning-rate candidates
and the same validation protocol. Set the baseline's independently selected value with
`MNIST_SCALAR_MOMENT_ADAM_LEARNING_RATE`; every run record stores the stable configuration key and
the applied learning rate.

At the existing estimate of about 95 minutes for one 500-epoch image-data configuration, the two
data sets × five configurations × ten seeds require about 158 GPU-hours (6.6 days) before the
pendulum and retraction stages. Use the RTX 4090 smoke logs to refine that estimate. Reserve at least
25 GiB until its five-configuration checkpoints establish the actual compressed archive size.

## Resume and monitor

The runner prints a normalized restart command into `restart-command.txt`. It includes
`--resume-dir`, all resolved command-line choices, and every recognized environment override that
changes an experiment. A completed image-data stage is skipped only after its run and loss CSVs
revalidate with exact configuration/seed coverage. Pendulum skips only checkpoint/record pairs from
a schema-valid partial record file. Retraction output is skipped only after its schema, required
algorithm/backend paths, source identity, and patch checksum revalidate. An interrupted image-data
matrix is rerun as a unit because the trainer does not resume within that matrix.

```bash
tail -f results/revision/<stamp>/run.log
cat results/revision/<stamp>/stages.csv
```

To restart inside a new detached session, copy the command from `restart-command.txt` and run:

```bash
screen -DmS gml-revision-resume bash -lc 'cd /path/to/GMLDatasets && exec <restart-command>'
```

## Validation and archive contract

The runner validates every CSV it emits: the exact schema-version-4 image run header and timing
invariants, image loss rows and their step counts, schema-version-1 pendulum records and checkpoint
coverage, schema-version-1 retraction rows and source patch, and the stage table. No JSON files are
currently emitted. Smoke permits a scientifically inconclusive two-epoch `failed_validation` row,
but never an exception or missing configuration; full mode requires every row to be `ok`.

Before packaging, `archive-required-members.txt` is generated from the selected mode, stages, seeds,
and configurations. At minimum every archive contains `run.log`, `stages.csv`, `environment.txt`,
`nvidia-smi.txt`, `run-configuration.txt`, `restart-command.txt`, the root `Project.toml`, the
scripts `Project.toml` and resolved `Manifest.toml`, both repository SHAs/statuses/exact dirty-tree
patches when available, `artifact-validation.txt`, and the member list itself. Selected stages add
their raw records, reports, stdout/stderr logs, checkpoints, and validation logs. The tar member list
is checked against that manifest before the SHA-256 file is written, and the checksum is immediately
verified. Failed runs still produce a partial archive, but do not claim artifact-validation success.

## Transfer and verify

Copy both generated files off the workstation:

```bash
scp results/revision/<stamp>.tar.gz* HOST:/path/to/OptimizerPaper/results/
sha256sum -c <stamp>.tar.gz.sha256
```

On macOS, verify with `shasum -a 256 -c <stamp>.tar.gz.sha256`.
