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
`GeometricMachineLearning` v0.7.0, `GeometricOptimizers` v0.7.0, and
`NeuralNetworkParameters` v0.3.0. The scripts project pins those exact releases and the environment
preflight rejects substitutions. Instantiate the committed manifest from a clean checkout before
producing paper results.

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
`run_experiments.sh` directly with `--allow-dirty --allow-any-gpu --allow-no-cuda` rather than
using the workstation command above.

Limit stages with `--stages mnist,pendulum` and configurations with
`--configurations geometric-adam-cayley`. `--configurations all` selects all five rows, including
the stable key `scalar-moment-adam`, reported as
`Scalar Moment Adam (Stiefel, Cayley retraction)`. The legacy keys `adam-stiefel` and
`adam-regular` remain accepted aliases. Smoke mode uses one seed and two epochs.

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

## Full RTX 4090 run

Full paper runs remain intentionally blocked while the retraction-record schema, smoke, and release
gates are completed. Decomposed timing and its schema-version-4 documentation are complete. The
experiment-local mixed-tree composite is present:
`ScalarMomentAdam` acts on Stiefel leaves and ordinary `Adam` on Euclidean leaves, and `all` includes
that row. `GML_ALLOW_INCOMPLETE_MATRIX=1` temporarily bypasses only this development gate; output
produced with it is not paper-ready.

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

Budget several days for the complete two-dataset, five-configuration, ten-seed matrix. The existing
RTX 4090 measurement is about 95 minutes for one 500-epoch MNIST Adam configuration; use the smoke
logs to refine the estimate before launch. Keep at least 20 GiB free until checkpoint sizes from the
smoke run are known.

## Resume and monitor

The runner prints an exact restart command into `restart-command.txt`. It includes `--resume-dir`
so the existing run directory, logs, stage history, and completed pendulum checkpoints are reused.
Pendulum checkpoints that already exist and are non-empty are skipped; image runs retain their
partial JLD2, CSV, report, and stdout files for inspection but are rerun because the legacy trainer
does not yet resume within an epoch matrix.

```bash
tail -f results/revision/<stamp>/run.log
cat results/revision/<stamp>/stages.csv
```

To restart inside a new detached session, copy the command from `restart-command.txt` and run:

```bash
screen -DmS gml-revision-resume bash -lc 'cd /path/to/GMLDatasets && exec <restart-command>'
```

## Transfer and verify

Copy both generated files off the workstation:

```bash
scp results/revision/<stamp>.tar.gz* HOST:/path/to/OptimizerPaper/results/
sha256sum -c <stamp>.tar.gz.sha256
```

On macOS, verify with `shasum -a 256 -c <stamp>.tar.gz.sha256`.
