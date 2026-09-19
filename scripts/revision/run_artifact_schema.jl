# Validation of one runner output directory: the image and pendulum run records, their loss
# curves, the retraction records and the stage table.
#
# The image and pendulum record headers share their first fifteen columns — everything that
# identifies a job and reports its outcome — so `validate_run_row` checks that prefix once and
# each caller adds the columns that are its own.
#
# This file holds definitions only; `records.jl` is the module that includes it.

const IMAGE_DATASETS = Set(("mnist", "fashion-mnist"))
const RUN_STAGES = Set(("mnist", "fashion-mnist", "pendulum", "retraction", "none"))

countmap(values) = Dict(value => count(==(value), values) for value in unique(values))

"""
    require_exactly(observed, expected, what, where)

Reject a set of identities that is not exactly `expected`. Coverage is the one property of a
results matrix no individual row can carry.
"""
function require_exactly(observed, expected, what, where)
    observed == expected ||
        throw(ArgumentError("$where does not have exact $what coverage"))
    nothing
end

"""
    validate_run_row(record, where; schema_version, dataset, configurations, statuses,
                     epochs_minimum, allow_validation_failures)

Check the fifteen columns every run record shares and return the identifying triple and the
epoch count. `statuses` is the set this record kind admits; anything in it other than `ok` is a
scientific failure and is rejected unless `allow_validation_failures`.
"""
function validate_run_row(record, where; schema_version, dataset, configurations, statuses,
        epochs_minimum, allow_validation_failures)
    record["schema_version"] == schema_version || throw(ArgumentError(
        "$where has unsupported schema version $(record["schema_version"])"))
    record["dataset"] == dataset || throw(ArgumentError(
        "$where has dataset $(record["dataset"]), expected $dataset"))
    key = record["configuration_key"]
    key in configurations ||
        throw(ArgumentError("$where has unexpected configuration key $key"))
    record["configuration"] == CONFIGURATION_NAMES[key] ||
        throw(ArgumentError("$where has the wrong display name for $key"))
    isempty(record["optimizer_role"]) &&
        throw(ArgumentError("$where has an empty optimizer role"))
    parse_float(record, "learning_rate", where; nonnegative = true)

    status = record["status"]
    status in statuses || throw(ArgumentError("$where has unknown status $status"))
    status == "exception" && throw(ArgumentError("$where records an exception for $key"))
    allow_validation_failures || status == "ok" ||
        throw(ArgumentError("$where records failed scientific validation for $key"))

    repetition = parse_integer(record, "repetition", where; minimum = 1)
    seed = parse_integer(record, "seed", where; minimum = 0)
    epochs = parse_integer(record, "epochs_completed", where; minimum = epochs_minimum)
    parse_float(record, "final_loss", where)
    parse_float(record, "best_loss", where)
    ((key, repetition, seed), epochs)
end

"""Collect one row's identity into `observed`, rejecting an unexpected or repeated job."""
function record_job!(observed, expected, identity, where)
    identity in expected ||
        throw(ArgumentError("$where has unexpected job $(join(identity, '/'))"))
    identity in observed &&
        throw(ArgumentError("$where duplicates job $(join(identity, '/'))"))
    push!(observed, identity)
    nothing
end

function expected_jobs(configurations, seeds)
    Set((key, repetition, seed)
    for key in configurations for (repetition, seed) in enumerate(seeds))
end

"""Record `index` under `key`, rejecting a repeat; used for loss steps and loss epochs."""
function record_index!(indices, key, index, what, where)
    seen = get!(() -> Set{Int}(), indices, key)
    index in seen &&
        throw(ArgumentError("$where duplicates $what $index for $(join(key, '/'))"))
    push!(seen, index)
    nothing
end

function validate_image_artifacts(
        records_path::AbstractString, losses_path::AbstractString;
        dataset::AbstractString, seeds::Vector{Int}, configurations::Vector{String},
        expected_epochs::Int, expected_backend::AbstractString,
        allow_validation_failures::Bool = false)
    dataset in IMAGE_DATASETS || throw(ArgumentError("unknown image dataset: $dataset"))
    expected_backend in ("cpu", "cuda") ||
        throw(ArgumentError("unknown image backend: $expected_backend"))
    records = read_table(records_path, IMAGE_RECORD_HEADER)
    expected = expected_jobs(configurations, seeds)
    observed = Set{Tuple{String, Int, Int}}()
    timed_steps = Dict{Tuple{String, Int}, Int}()

    for (offset, record) in enumerate(records)
        where = "$records_path:$(offset + 1)"
        identity, epochs = validate_run_row(
            record, where; schema_version = string(MNIST_RUN_SCHEMA_VERSION), dataset,
            configurations, statuses = ("ok", "failed_validation", "exception"),
            epochs_minimum = 0, allow_validation_failures)
        record_job!(observed, expected, identity, where)
        epochs == expected_epochs ||
            throw(ArgumentError("$where completed $epochs epochs; expected $expected_epochs"))
        record["backend"] == expected_backend || throw(ArgumentError(
            "$where has backend $(record["backend"]), expected $expected_backend"))
        parse_float(record, "test_accuracy", where)
        parse_float(record, "total_seconds", where; nonnegative = true)
        parse_float(record, "seconds_per_epoch", where; nonnegative = true)
        parse_integer(record, "peak_device_bytes", where; minimum = 0)

        steps = parse_integer(record, "timed_steps", where; minimum = 0)
        totals, per_steps = timing_pairs(record, where)
        if steps == 0
            all(iszero, totals) && all(iszero, per_steps) || throw(ArgumentError(
                "$where has nonzero timing for a zero-step result"))
        else
            all(isapprox(per_step, total / steps; rtol = 1.0e-10, atol = 1.0e-12)
            for (total, per_step) in zip(totals, per_steps)) ||
                throw(ArgumentError("$where has inconsistent total/per-step timing"))
        end
        timed_steps[(record["configuration"], identity[2])] = steps
    end
    require_exactly(observed, expected, "configuration/repetition/seed", records_path)

    losses = read_table(losses_path, IMAGE_LOSS_HEADER)
    loss_steps = Dict{Tuple{String, Int}, Set{Int}}()
    expected_names = Set(CONFIGURATION_NAMES[key] for key in configurations)
    for (offset, record) in enumerate(losses)
        where = "$losses_path:$(offset + 1)"
        parse_integer(record, "run", where; minimum = 1)
        record["configuration"] in expected_names ||
            throw(ArgumentError("$where has an unexpected configuration"))
        repetition = parse_integer(record, "repetition", where; minimum = 1)
        parse_integer(record, "epoch", where; minimum = 1)
        parse_integer(record, "batch", where; minimum = 1)
        step = parse_integer(record, "step", where; minimum = 1)
        parse_float(record, "loss", where)
        key = (record["configuration"], repetition)
        haskey(timed_steps, key) ||
            throw(ArgumentError("$where has no matching run record"))
        record_index!(loss_steps, key, step, "step", where)
    end
    for (key, step_count) in timed_steps
        get(loss_steps, key, Set{Int}()) == Set(1:step_count) || throw(ArgumentError(
            "$losses_path does not have exact steps 1:$step_count for $(key[1]) " *
            "repetition $(key[2])"))
    end
    (records = length(records), losses = length(losses),
        statuses = countmap([record["status"] for record in records]))
end

"""The three timing totals and the three per-step values of one image record."""
function timing_pairs(record, where)
    totals = [parse_float(record, field, where; nonnegative = true)
              for field in STEP_TIMING_COLUMNS if endswith(field, "_total")]
    per_steps = [parse_float(record, field, where; nonnegative = true)
                 for field in STEP_TIMING_COLUMNS if endswith(field, "_per_step")]
    (totals, per_steps)
end

function validate_pendulum_artifacts(
        records_path::AbstractString, losses_path::AbstractString,
        run_dir::AbstractString; seeds::Vector{Int}, configurations::Vector{String},
        expected_epochs::Int, expected_backend::AbstractString, allow_partial::Bool = false,
        allow_validation_failures::Bool = false)
    all(key -> key in PENDULUM_CONFIGURATION_ORDER, configurations) || throw(ArgumentError(
        "pendulum configurations must be intrinsic SAE configurations"))
    records = read_table(records_path, PENDULUM_RECORD_HEADER; allow_empty = allow_partial)
    expected = expected_jobs(configurations, seeds)
    observed = Set{Tuple{String, Int, Int}}()

    for (offset, record) in enumerate(records)
        where = "$records_path:$(offset + 1)"
        identity, epochs = validate_run_row(
            record, where; schema_version = string(PENDULUM_RUN_SCHEMA_VERSION),
            dataset = "pendulum", configurations, statuses = ("ok", "failed_validation"),
            epochs_minimum = 1, allow_validation_failures)
        record_job!(observed, expected, identity, where)
        epochs == expected_epochs ||
            throw(ArgumentError("$where completed $epochs epochs; expected $expected_epochs"))
        record["retraction"] == "cayley" || throw(ArgumentError(
            "$where has retraction $(record["retraction"]), expected cayley"))
        parse_float(record, "total_seconds", where; nonnegative = true)
        parse_float(record, "seconds_per_epoch", where; nonnegative = true)
        parse_float(record, "gc_seconds", where; nonnegative = true)
        parse_integer(record, "host_allocated_bytes", where; minimum = 0)
        occursin(expected_backend, lowercase(record["backend"])) || throw(ArgumentError(
            "$where has backend $(record["backend"]), expected $expected_backend"))

        key, _, seed = identity
        checkpoint = joinpath(run_dir, basename(record["checkpoint"]))
        isfile(checkpoint) && filesize(checkpoint) > 0 || throw(ArgumentError(
            "$where refers to a missing or empty checkpoint: $checkpoint"))
        basename(checkpoint) == "pendulum-$key-seed-$seed.h5" || throw(ArgumentError(
            "$where has unexpected checkpoint name $(basename(checkpoint))"))
    end
    if allow_partial
        observed ⊆ expected ||
            throw(ArgumentError("pendulum records exceed expected coverage"))
    else
        require_exactly(observed, expected, "configuration/repetition/seed", records_path)
    end

    losses = read_table(losses_path, PENDULUM_LOSS_HEADER; allow_empty = allow_partial)
    epochs_by_run = Dict{Tuple{String, Int, Int}, Set{Int}}()
    for (offset, record) in enumerate(losses)
        where = "$losses_path:$(offset + 1)"
        key = record["configuration_key"]
        key in configurations ||
            throw(ArgumentError("$where has unexpected configuration key $key"))
        record["configuration"] == CONFIGURATION_NAMES[key] ||
            throw(ArgumentError("$where has the wrong display name for $key"))
        identity = (key, parse_integer(record, "repetition", where; minimum = 1),
            parse_integer(record, "seed", where; minimum = 0))
        identity in observed || throw(ArgumentError("$where has no matching run record"))
        parse_float(record, "loss", where)
        record_index!(epochs_by_run, identity,
            parse_integer(record, "epoch", where; minimum = 1), "epoch", where)
    end
    for identity in observed
        get(epochs_by_run, identity, Set{Int}()) == Set(1:expected_epochs) ||
            throw(ArgumentError("$losses_path does not have exact epochs 1:$expected_epochs " *
                                "for $(join(identity, '/'))"))
    end
    (records = length(records), losses = length(losses),
        statuses = countmap([record["status"] for record in records]))
end

function validate_stage_table(path::AbstractString, expected_stages::Vector{String})
    records = read_table(path, STAGE_HEADER; allow_empty = isempty(expected_stages))
    latest = Dict{String, String}()
    for (offset, record) in enumerate(records)
        where = "$path:$(offset + 1)"
        for field in ("stage", "started_utc", "finished_utc", "command")
            isempty(record[field]) && throw(ArgumentError("$where has an empty $field"))
        end
        record["status"] == "ok" || occursin(r"^failed:[1-9][0-9]*$", record["status"]) ||
            throw(ArgumentError("$where has invalid status $(record["status"])"))
        latest[record["stage"]] = record["status"]
    end
    for stage in expected_stages
        get(latest, stage, "missing") == "ok" || throw(ArgumentError(
            "$path has no latest successful row for required stage $stage"))
    end
    (rows = length(records), passed = count(==("ok"), values(latest)),
        failed = count(!=("ok"), values(latest)))
end

"""The stage names a run of this shape must have completed, in the order the runner writes them."""
function expected_stage_names(
        mode::AbstractString, stages::Vector{String}, seeds::Vector{Int},
        pendulum_configurations::Vector{String} = PENDULUM_CONFIGURATION_ORDER)
    names = String[]
    for stage in stages
        if stage == "none"
            continue
        elseif stage in IMAGE_DATASETS
            mode == "full" &&
                push!(names, "$stage-warmup", "$stage-warmup-record-validation")
            push!(names, stage, "$stage-record-validation")
        elseif stage == "pendulum"
            for key in pendulum_configurations
                mode == "full" && push!(names, "pendulum-$key-warmup")
                append!(names, ["pendulum-$key-seed-$seed" for seed in seeds])
            end
            push!(names, "pendulum-record-validation")
        elseif stage == "retraction"
            push!(names, "retraction", "retraction-record-validation")
        end
    end
    names
end

function require_file(path::AbstractString)
    isfile(path) || throw(ArgumentError("missing required artifact: $path"))
    filesize(path) == 0 && throw(ArgumentError("required artifact is empty: $path"))
    path
end

"""Require and validate the four artifacts of one image stage, and describe what it held."""
function validate_image_stage(
        run_dir, dataset, prefix; seeds, configurations, expected_epochs,
        expected_backend, allow_validation_failures)
    for name in ("$prefix-report.txt", "$prefix-losses.csv", "$prefix-runs.csv", "$prefix.jld2")
        require_file(joinpath(run_dir, name))
    end
    summary = validate_image_artifacts(joinpath(run_dir, "$prefix-runs.csv"),
        joinpath(run_dir, "$prefix-losses.csv"); dataset, seeds, configurations,
        expected_epochs, expected_backend, allow_validation_failures)
    "$prefix=$(summary.records) records/$(summary.losses) losses"
end

function validate_run_artifacts(run_dir::AbstractString; mode::AbstractString,
        stages::Vector{String}, seeds::Vector{Int}, configurations::Vector{String},
        expected_image_epochs::Int, expected_pendulum_epochs::Int,
        expected_backend::AbstractString, retraction_repo::AbstractString,
        pendulum_configurations::Vector{String} = PENDULUM_CONFIGURATION_ORDER,
        allow_validation_failures::Bool = false)
    mode in ("smoke", "full") || throw(ArgumentError("unknown mode: $mode"))
    isempty(stages) && throw(ArgumentError("stage list is empty"))
    all(stage -> stage in RUN_STAGES, stages) ||
        throw(ArgumentError("unknown stage in $(join(stages, ','))"))
    "none" in stages && length(stages) != 1 &&
        throw(ArgumentError(
            "the test-only `none` stage cannot be combined with experiment stages"))
    isempty(seeds) && throw(ArgumentError("seed list is empty"))

    require_file(joinpath(run_dir, "environment.txt"))
    require_file(joinpath(run_dir, "nvidia-smi.txt"))
    summaries = String[]
    for dataset in ("mnist", "fashion-mnist")
        dataset in stages || continue
        push!(summaries,
            validate_image_stage(run_dir, dataset, dataset; seeds, configurations,
                expected_epochs = expected_image_epochs, expected_backend,
                allow_validation_failures))
        # `full` also runs a one-epoch warm-up of the whole matrix before the measured run. It
        # is one seed and one epoch, and is allowed to be scientifically inconclusive.
        mode == "full" && push!(summaries,
            validate_image_stage(
                run_dir, dataset, "$dataset-warmup"; seeds = [first(seeds)],
                configurations, expected_epochs = 1, expected_backend,
                allow_validation_failures = true))
    end
    if "pendulum" in stages
        require_file(joinpath(run_dir, "pendulum-runs.csv"))
        require_file(joinpath(run_dir, "pendulum-losses.csv"))
        summary = validate_pendulum_artifacts(joinpath(run_dir, "pendulum-runs.csv"),
            joinpath(run_dir, "pendulum-losses.csv"), run_dir; seeds,
            configurations = pendulum_configurations,
            expected_epochs = expected_pendulum_epochs, expected_backend,
            allow_validation_failures)
        push!(summaries, "pendulum=$(summary.records) records/$(summary.losses) losses")
    end
    if "retraction" in stages
        device_backend = expected_backend == "cuda" ? "CUDA" : "CPU"
        required_paths = [("AugmentedPade", "CPU"), ("ScaledSquaring", device_backend),
            ("NativePade", device_backend)]
        summary = validate_records(joinpath(run_dir, "retraction-runs.csv");
            required_paths, go_repo = retraction_repo)
        push!(summaries, "retraction=$(summary.rows) records")
    end
    stage_summary = validate_stage_table(joinpath(run_dir, "stages.csv"),
        expected_stage_names(mode, stages, seeds, pendulum_configurations))
    push!(summaries, "stages=$(stage_summary.rows) rows")
    summaries
end
