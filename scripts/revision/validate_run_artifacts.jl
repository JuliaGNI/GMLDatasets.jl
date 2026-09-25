#!/usr/bin/env julia

include("arguments.jl")
include("records.jl")
using .RunRecords: PENDULUM_RECORD_HEADER, normalize_configurations,
                   normalize_pendulum_configurations, read_table, split_list,
                   validate_image_artifacts, validate_pendulum_artifacts,
                   validate_run_artifacts

const USAGE = """usage: validate_run_artifacts.jl --run-dir DIR [options]

Validate one complete runner directory:
  --mode smoke|full
  --stages LIST
  --seeds LIST
  --configurations LIST
  --image-epochs N
  --pendulum-epochs N
  --backend cpu|cuda
  --retraction-repo DIR
  --allow-validation-failures

Focused validation used by restart logic:
  --image DATASET [--artifact-prefix PREFIX]
  --pendulum [--allow-partial] [--list-complete FILE]

--list-complete writes one `configuration_key,repetition,seed` line per validated pendulum
record, which is what the runner skips. The runner cannot read those three columns out of the
CSV itself: the display name beside them is a quoted field containing a comma.
"""

function main(args = ARGS)
    options = parse_arguments(args,
        (run_dir = "", mode = "smoke",
            stages = "mnist,fashion-mnist,pendulum,retraction",
            seeds = "1234", configurations = "all", image_epochs = 2, pendulum_epochs = 2,
            backend = "cpu", retraction_repo = "", image = "", artifact_prefix = "",
            list_complete = "", pendulum = false, allow_partial = false,
            allow_validation_failures = false);
        usage = USAGE)
    options === nothing && return 0

    isempty(options.run_dir) && throw(ArgumentError("--run-dir is required"))
    run_dir = abspath(options.run_dir)
    isdir(run_dir) || throw(ArgumentError("run directory does not exist: $run_dir"))
    options.image_epochs >= 1 && options.pendulum_epochs >= 1 ||
        throw(ArgumentError("--image-epochs and --pendulum-epochs must be at least 1"))
    seeds = map(value -> parse_seed(value), split_list(options.seeds))
    configurations = normalize_configurations(options.configurations)
    backend = lowercase(options.backend)
    backend in ("cpu", "cuda") || throw(ArgumentError("--backend must be cpu or cuda"))
    image = lowercase(options.image)
    !isempty(image) && options.pendulum &&
        throw(ArgumentError("select at most one focused validation"))
    allow_validation_failures = options.allow_validation_failures

    if !isempty(image)
        prefix = isempty(options.artifact_prefix) ? image : options.artifact_prefix
        summary = validate_image_artifacts(joinpath(run_dir, "$prefix-runs.csv"),
            joinpath(run_dir, "$prefix-losses.csv"); dataset = image, seeds, configurations,
            expected_epochs = options.image_epochs, expected_backend = backend,
            allow_validation_failures)
        println("validated $image artifacts: $(summary.records) records, " *
                "$(summary.losses) loss rows")
    elseif options.pendulum
        summary = validate_pendulum_artifacts(joinpath(run_dir, "pendulum-runs.csv"),
            joinpath(run_dir, "pendulum-losses.csv"), run_dir; seeds,
            configurations = normalize_pendulum_configurations(options.configurations),
            expected_epochs = options.pendulum_epochs, expected_backend = backend,
            allow_partial = options.allow_partial, allow_validation_failures)
        isempty(options.list_complete) ||
            write_complete_jobs(options.list_complete, joinpath(run_dir, "pendulum-runs.csv"))
        println("validated pendulum artifacts: $(summary.records) records, " *
                "$(summary.losses) loss rows")
    else
        stages = lowercase.(split_list(options.stages))
        pendulum_configurations = "pendulum" in stages ?
                                  normalize_pendulum_configurations(options.configurations) :
                                  String[]
        if "retraction" in stages
            isempty(options.retraction_repo) && throw(ArgumentError(
                "--retraction-repo is required when the retraction stage is selected"))
            isdir(options.retraction_repo) || throw(ArgumentError(
                "retraction repository does not exist: $(options.retraction_repo)"))
        end
        summaries = validate_run_artifacts(run_dir; mode = lowercase(options.mode), stages,
            seeds, configurations, expected_image_epochs = options.image_epochs,
            expected_pendulum_epochs = options.pendulum_epochs, expected_backend = backend,
            retraction_repo = options.retraction_repo, pendulum_configurations,
            allow_validation_failures)
        println("validated run artifacts: ", join(summaries, "; "))
    end
    0
end

"""Write the `configuration_key,repetition,seed` of every `ok` record in `records_path`."""
function write_complete_jobs(path::AbstractString, records_path::AbstractString)
    records = read_table(records_path, PENDULUM_RECORD_HEADER; allow_empty = true)
    open(path, "w") do io
        for record in records
            record["status"] == "ok" || continue
            println(io, join(
                (record["configuration_key"], record["repetition"],
                    record["seed"]), ','))
        end
    end
end

function parse_seed(value)
    seed = tryparse(Int, value)
    seed === nothing && throw(ArgumentError("--seeds requires an integer, got $value"))
    seed >= 0 || throw(ArgumentError("--seeds must be nonnegative, got $seed"))
    seed
end

try
    exit(main())
catch exception
    println(stderr, "run artifact validation failed: ", sprint(showerror, exception))
    exit(1)
end
