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
`GeometricMachineLearning` v0.6.1, `GeometricOptimizers` v0.6.0, and
`NeuralNetworkParameters` v0.2.5. The scripts project pins those exact releases and the environment
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

## Full RTX 4090 run

Full paper runs remain intentionally blocked while the remaining timing and retraction-record
schema, smoke, and release gates are completed. The experiment-local mixed-tree composite is present:
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
