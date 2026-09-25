# The CSV headers and schema versions of the harness, in one place.
#
# The trainers write these tables and the validators read them, so a second copy of a column list
# is a table whose two halves disagree without anyone noticing. `step_timing.jl` includes this
# file for the timing columns, and `records.jl` includes it for all of them; the trainers reach it
# through `step_timing.jl`.
#
# This file holds definitions only.

const MNIST_RUN_SCHEMA_VERSION = 4
const PENDULUM_RUN_SCHEMA_VERSION = 2
const RETRACTION_SCHEMA_VERSION = 1

"""The seven schema-v4 timing columns, totals and per-step values of the three phases."""
const STEP_TIMING_COLUMNS = [
    "timed_steps",
    "gradient_ad_seconds_total",
    "gradient_ad_seconds_per_step",
    "optimizer_state_direction_seconds_total",
    "optimizer_state_direction_seconds_per_step",
    "retraction_application_seconds_total",
    "retraction_application_seconds_per_step"
]

"""The columns every run record has, whichever trainer wrote it."""
const RUN_RECORD_PREFIX = [
    "schema_version", "dataset", "configuration_key", "configuration", "optimizer_role",
    "learning_rate", "retraction", "second_moment", "transport", "repetition", "seed",
    "status", "epochs_completed", "final_loss", "best_loss"
]

const IMAGE_RECORD_HEADER = [RUN_RECORD_PREFIX...,
    "test_accuracy", "total_seconds", "seconds_per_epoch", STEP_TIMING_COLUMNS...,
    "peak_device_bytes", "backend", "message"]

const PENDULUM_RECORD_HEADER = [RUN_RECORD_PREFIX...,
    "total_seconds", "seconds_per_epoch", "host_allocated_bytes", "gc_seconds", "backend",
    "checkpoint", "message"]

const IMAGE_LOSS_HEADER = [
    "run", "configuration", "repetition", "epoch", "batch", "step", "loss"]

const PENDULUM_LOSS_HEADER = [
    "configuration_key", "configuration", "repetition", "seed", "epoch", "loss"]

const STAGE_HEADER = ["stage", "status", "started_utc", "finished_utc", "command"]

const RETRACTION_HEADER = [
    "schema_version", "algorithm", "backend", "device", "precision",
    "matrix_rows", "matrix_columns", "stiefel_columns", "lift_norm",
    "agreement_error", "forward_error", "manifold_constraint_error",
    "runtime_seconds", "memory_bytes", "memory_metric",
    "repetition", "warmup", "success", "error_type", "error_message",
    "seed", "scale", "go_sha", "go_dirty", "go_patch_file", "go_patch_sha256"
]
