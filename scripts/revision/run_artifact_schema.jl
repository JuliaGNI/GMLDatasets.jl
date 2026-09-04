module RunArtifactSchema

include("retraction_record_schema.jl")
using .RetractionRecordSchema: validate_records

export CONFIGURATION_ORDER, IMAGE_LOSS_HEADER, IMAGE_RECORD_HEADER, PENDULUM_RECORD_HEADER,
       STAGE_HEADER, parse_csv_line, read_table, validate_image_artifacts,
       validate_pendulum_artifacts, validate_run_artifacts, validate_stage_table

const CONFIGURATION_ORDER = [
    "geometric-adam-cayley",
    "scalar-moment-adam",
    "standard-adam",
    "gradient",
    "momentum",
]

const CONFIGURATION_ALIASES = Dict(
    "adam-stiefel" => "geometric-adam-cayley",
    "adam-regular" => "standard-adam",
)

const CONFIGURATION_NAMES = Dict(
    "geometric-adam-cayley" => "Geometric Adam (Stiefel, Cayley retraction)",
    "scalar-moment-adam" => "Scalar Moment Adam (Stiefel, Cayley retraction)",
    "standard-adam" => "Standard Adam (unconstrained)",
    "gradient" => "Riemannian gradient (Stiefel, Cayley retraction)",
    "momentum" => "Riemannian momentum (Stiefel, Cayley retraction)",
)

const IMAGE_LOSS_HEADER = [
    "run", "configuration", "repetition", "epoch", "batch", "step", "loss",
]

const IMAGE_RECORD_HEADER = [
    "schema_version", "dataset", "configuration_key", "configuration", "optimizer_role",
    "learning_rate", "retraction", "second_moment", "transport", "repetition", "seed",
    "status", "epochs_completed", "final_loss", "best_loss", "test_accuracy",
    "total_seconds", "seconds_per_epoch", "timed_steps", "gradient_ad_seconds_total",
    "gradient_ad_seconds_per_step", "optimizer_state_direction_seconds_total",
    "optimizer_state_direction_seconds_per_step", "retraction_application_seconds_total",
    "retraction_application_seconds_per_step", "peak_device_bytes", "backend", "message",
]

const PENDULUM_RECORD_HEADER = [
    "schema_version", "dataset", "configuration", "repetition", "seed", "status",
    "epochs_completed", "final_loss", "best_loss", "total_seconds", "seconds_per_epoch",
    "host_allocated_bytes", "gc_seconds", "backend", "checkpoint",
]

const STAGE_HEADER = ["stage", "status", "started_utc", "finished_utc", "command"]
const IMAGE_DATASETS = Set(("mnist", "fashion-mnist"))
const RUN_STAGES = Set(("mnist", "fashion-mnist", "pendulum", "retraction", "none"))

function parse_csv_line(line::AbstractString)
    fields = String[]
    field = IOBuffer()
    quoted = false
    just_closed_quote = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if quoted
            if character == '"'
                next_index = nextind(line, index)
                if next_index <= lastindex(line) && line[next_index] == '"'
                    write(field, '"')
                    index = next_index
                else
                    quoted = false
                    just_closed_quote = true
                end
            else
                write(field, character)
            end
        elseif just_closed_quote
            character == ',' || throw(ArgumentError(
                "unexpected character after a closing CSV quote"))
            push!(fields, String(take!(field)))
            just_closed_quote = false
        elseif character == ','
            push!(fields, String(take!(field)))
        elseif character == '"'
            position(field) == 0 || throw(ArgumentError("CSV quote must begin a field"))
            quoted = true
        else
            write(field, character)
        end
        index = nextind(line, index)
    end
    quoted && throw(ArgumentError("unterminated quoted CSV field"))
    push!(fields, String(take!(field)))
    fields
end

function read_table(path::AbstractString, expected_header; allow_empty::Bool=false)
    isfile(path) || throw(ArgumentError("missing CSV file: $path"))
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("CSV file is empty: $path"))
    header = parse_csv_line(first(lines))
    header == expected_header || throw(ArgumentError(
        "unexpected header in $path; expected $(join(expected_header, ','))"))

    records = Dict{String,String}[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        line_number = offset + 1
        isempty(line) && throw(ArgumentError("blank CSV row at $path:$line_number"))
        fields = parse_csv_line(line)
        length(fields) == length(expected_header) || throw(ArgumentError(
            "$path:$line_number has $(length(fields)) fields; expected $(length(expected_header))"))
        push!(records, Dict(zip(expected_header, fields)))
    end
    !allow_empty && isempty(records) && throw(ArgumentError("CSV file has no data rows: $path"))
    records
end

function parse_integer(record, field, path, line; minimum=nothing)
    value = tryparse(Int, record[field])
    value === nothing && throw(ArgumentError(
        "$path:$line has invalid integer $field: $(record[field])"))
    minimum !== nothing && value < minimum && throw(ArgumentError(
        "$path:$line has $field below $minimum: $value"))
    value
end

function parse_float(record, field, path, line; finite::Bool=true, nonnegative::Bool=false)
    value = tryparse(Float64, record[field])
    value === nothing && throw(ArgumentError(
        "$path:$line has invalid float $field: $(record[field])"))
    finite && !isfinite(value) && throw(ArgumentError(
        "$path:$line has non-finite $field: $value"))
    nonnegative && value < 0 && throw(ArgumentError(
        "$path:$line has negative $field: $value"))
    value
end

function split_list(value::AbstractString)
    entries = strip.(split(value, ','; keepempty=false))
    filter!(!isempty, entries)
    entries
end

function normalize_configurations(value::AbstractString)
    requested = [get(CONFIGURATION_ALIASES, lowercase(entry), lowercase(entry))
                 for entry in split_list(value)]
    "all" in requested && return copy(CONFIGURATION_ORDER)
    isempty(requested) && throw(ArgumentError("configuration list is empty"))
    unknown = filter(key -> key ∉ CONFIGURATION_ORDER, requested)
    isempty(unknown) || throw(ArgumentError("unknown configurations: $(join(unknown, ", "))"))
    unique(filter(key -> key in requested, CONFIGURATION_ORDER))
end

function validate_image_artifacts(records_path::AbstractString, losses_path::AbstractString;
        dataset::AbstractString, seeds::Vector{Int}, configurations::Vector{String},
        expected_epochs::Int, expected_backend::AbstractString,
        allow_validation_failures::Bool=false)
    dataset in IMAGE_DATASETS || throw(ArgumentError("unknown image dataset: $dataset"))
    expected_backend in ("cpu", "cuda") || throw(ArgumentError(
        "unknown image backend: $expected_backend"))
    records = read_table(records_path, IMAGE_RECORD_HEADER)
    expected = Set((key, repetition, seed)
        for key in configurations for (repetition, seed) in enumerate(seeds))
    observed = Set{Tuple{String,Int,Int}}()
    timed_steps = Dict{Tuple{String,Int},Int}()

    total_columns = (
        "gradient_ad_seconds_total",
        "optimizer_state_direction_seconds_total",
        "retraction_application_seconds_total",
    )
    per_step_columns = (
        "gradient_ad_seconds_per_step",
        "optimizer_state_direction_seconds_per_step",
        "retraction_application_seconds_per_step",
    )
    for (offset, record) in enumerate(records)
        line = offset + 1
        record["schema_version"] == "4" || throw(ArgumentError(
            "$records_path:$line has unsupported schema version $(record["schema_version"])"))
        record["dataset"] == dataset || throw(ArgumentError(
            "$records_path:$line has dataset $(record["dataset"]), expected $dataset"))
        key = record["configuration_key"]
        key in configurations || throw(ArgumentError(
            "$records_path:$line has unexpected configuration key $key"))
        record["configuration"] == CONFIGURATION_NAMES[key] || throw(ArgumentError(
            "$records_path:$line has the wrong display name for $key"))
        repetition = parse_integer(record, "repetition", records_path, line; minimum=1)
        seed = parse_integer(record, "seed", records_path, line; minimum=0)
        identity = (key, repetition, seed)
        identity in expected || throw(ArgumentError(
            "$records_path:$line has unexpected job $(join(identity, '/'))"))
        identity in observed && throw(ArgumentError(
            "$records_path:$line duplicates job $(join(identity, '/'))"))
        push!(observed, identity)

        status = record["status"]
        status in ("ok", "failed_validation", "exception") || throw(ArgumentError(
            "$records_path:$line has unknown status $status"))
        status == "exception" && throw(ArgumentError(
            "$records_path:$line records an exception for $key repetition $repetition"))
        !allow_validation_failures && status != "ok" && throw(ArgumentError(
            "$records_path:$line records failed scientific validation for $key repetition $repetition"))
        isempty(record["optimizer_role"]) && throw(ArgumentError(
            "$records_path:$line has an empty optimizer role"))
        parse_float(record, "learning_rate", records_path, line; nonnegative=true)
        record["backend"] == expected_backend || throw(ArgumentError(
            "$records_path:$line has backend $(record["backend"]), expected $expected_backend"))
        epochs = parse_integer(record, "epochs_completed", records_path, line; minimum=0)
        epochs == expected_epochs || throw(ArgumentError(
            "$records_path:$line completed $epochs epochs; expected $expected_epochs"))
        for field in ("final_loss", "best_loss", "test_accuracy", "total_seconds",
                "seconds_per_epoch")
            parse_float(record, field, records_path, line;
                nonnegative=field in ("total_seconds", "seconds_per_epoch"))
        end
        steps = parse_integer(record, "timed_steps", records_path, line; minimum=0)
        totals = [parse_float(record, field, records_path, line; nonnegative=true)
                  for field in total_columns]
        per_steps = [parse_float(record, field, records_path, line; nonnegative=true)
                     for field in per_step_columns]
        if steps == 0
            all(iszero, totals) && all(iszero, per_steps) || throw(ArgumentError(
                "$records_path:$line has nonzero timing for a zero-step result"))
        else
            all(isapprox(per_step, total / steps; rtol=1e-10, atol=1e-12)
                for (total, per_step) in zip(totals, per_steps)) || throw(ArgumentError(
                "$records_path:$line has inconsistent total/per-step timing"))
        end
        parse_integer(record, "peak_device_bytes", records_path, line; minimum=0)
        timed_steps[(record["configuration"], repetition)] = steps
    end
    observed == expected || throw(ArgumentError(
        "$records_path does not have exact configuration/repetition/seed coverage"))

    losses = read_table(losses_path, IMAGE_LOSS_HEADER)
    loss_steps = Dict{Tuple{String,Int},Set{Int}}()
    expected_names = Set(CONFIGURATION_NAMES[key] for key in configurations)
    for (offset, record) in enumerate(losses)
        line = offset + 1
        parse_integer(record, "run", losses_path, line; minimum=1)
        record["configuration"] in expected_names || throw(ArgumentError(
            "$losses_path:$line has an unexpected configuration"))
        repetition = parse_integer(record, "repetition", losses_path, line; minimum=1)
        parse_integer(record, "epoch", losses_path, line; minimum=1)
        parse_integer(record, "batch", losses_path, line; minimum=1)
        step = parse_integer(record, "step", losses_path, line; minimum=1)
        parse_float(record, "loss", losses_path, line)
        key = (record["configuration"], repetition)
        haskey(timed_steps, key) || throw(ArgumentError(
            "$losses_path:$line has no matching run record"))
        steps = get!(() -> Set{Int}(), loss_steps, key)
        step in steps && throw(ArgumentError(
            "$losses_path:$line duplicates step $step for $(key[1]) repetition $(key[2])"))
        push!(steps, step)
    end
    for (key, timed_step_count) in timed_steps
        observed_steps = get(loss_steps, key, Set{Int}())
        observed_steps == Set(1:timed_step_count) || throw(ArgumentError(
            "$losses_path does not have exact steps 1:$timed_step_count for $(key[1]) " *
            "repetition $(key[2])"))
    end
    (records=length(records), losses=length(losses), statuses=countmap(record["status"] for record in records))
end

function countmap(values)
    counts = Dict{String,Int}()
    for value in values
        counts[value] = get(counts, value, 0) + 1
    end
    counts
end

function validate_pendulum_artifacts(records_path::AbstractString, run_dir::AbstractString;
        seeds::Vector{Int}, expected_epochs::Int, expected_backend::AbstractString,
        allow_partial::Bool=false)
    records = read_table(records_path, PENDULUM_RECORD_HEADER; allow_empty=allow_partial)
    expected = Set((repetition, seed) for (repetition, seed) in enumerate(seeds))
    observed = Set{Tuple{Int,Int}}()
    for (offset, record) in enumerate(records)
        line = offset + 1
        record["schema_version"] == "1" || throw(ArgumentError(
            "$records_path:$line has unsupported schema version $(record["schema_version"])"))
        record["dataset"] == "pendulum" || throw(ArgumentError(
            "$records_path:$line has dataset $(record["dataset"])"))
        record["configuration"] == "geometric-adam" || throw(ArgumentError(
            "$records_path:$line has configuration $(record["configuration"])"))
        repetition = parse_integer(record, "repetition", records_path, line; minimum=1)
        seed = parse_integer(record, "seed", records_path, line; minimum=0)
        identity = (repetition, seed)
        identity in expected || throw(ArgumentError(
            "$records_path:$line has unexpected repetition/seed $(join(identity, '/'))"))
        identity in observed && throw(ArgumentError(
            "$records_path:$line duplicates repetition/seed $(join(identity, '/'))"))
        push!(observed, identity)
        record["status"] == "ok" || throw(ArgumentError(
            "$records_path:$line has status $(record["status"])"))
        epochs = parse_integer(record, "epochs_completed", records_path, line; minimum=1)
        epochs == expected_epochs || throw(ArgumentError(
            "$records_path:$line completed $epochs epochs; expected $expected_epochs"))
        for field in ("final_loss", "best_loss", "total_seconds", "seconds_per_epoch",
                "gc_seconds")
            parse_float(record, field, records_path, line;
                nonnegative=field in ("total_seconds", "seconds_per_epoch", "gc_seconds"))
        end
        parse_integer(record, "host_allocated_bytes", records_path, line; minimum=0)
        occursin(expected_backend, lowercase(record["backend"])) || throw(ArgumentError(
            "$records_path:$line has backend $(record["backend"]), expected $expected_backend"))
        checkpoint = joinpath(run_dir, basename(record["checkpoint"]))
        isfile(checkpoint) && filesize(checkpoint) > 0 || throw(ArgumentError(
            "$records_path:$line refers to a missing or empty checkpoint: $checkpoint"))
        basename(checkpoint) == "pendulum-seed-$seed.h5" || throw(ArgumentError(
            "$records_path:$line has unexpected checkpoint name $(basename(checkpoint))"))
    end
    if allow_partial
        observed ⊆ expected || throw(ArgumentError("pendulum records exceed expected coverage"))
    else
        observed == expected || throw(ArgumentError(
            "$records_path does not have exact repetition/seed coverage"))
    end
    (records=length(records), statuses=countmap(record["status"] for record in records))
end

function validate_stage_table(path::AbstractString, expected_stages::Vector{String})
    records = read_table(path, STAGE_HEADER; allow_empty=isempty(expected_stages))
    latest = Dict{String,String}()
    for (offset, record) in enumerate(records)
        line = offset + 1
        isempty(record["stage"]) && throw(ArgumentError("$path:$line has an empty stage"))
        (record["status"] == "ok" || occursin(r"^failed:[1-9][0-9]*$", record["status"])) ||
            throw(ArgumentError("$path:$line has invalid status $(record["status"])"))
        isempty(record["started_utc"]) && throw(ArgumentError(
            "$path:$line has an empty start timestamp"))
        isempty(record["finished_utc"]) && throw(ArgumentError(
            "$path:$line has an empty finish timestamp"))
        isempty(record["command"]) && throw(ArgumentError("$path:$line has an empty command"))
        latest[record["stage"]] = record["status"]
    end
    for stage in expected_stages
        get(latest, stage, "missing") == "ok" || throw(ArgumentError(
            "$path has no latest successful row for required stage $stage"))
    end
    (rows=length(records), passed=count(==("ok"), values(latest)), failed=count(!=("ok"), values(latest)))
end

function expected_stage_names(mode::AbstractString, stages::Vector{String}, seeds::Vector{Int})
    names = String[]
    for stage in stages
        stage == "none" && continue
        if stage in IMAGE_DATASETS
            mode == "full" && push!(names, "$stage-warmup", "$stage-warmup-record-validation")
            push!(names, stage, "$stage-record-validation")
        elseif stage == "pendulum"
            mode == "full" && push!(names, "pendulum-warmup")
            for seed in seeds
                push!(names, "pendulum-seed-$seed")
            end
            push!(names, "pendulum-record-validation")
        elseif stage == "retraction"
            push!(names, "retraction", "retraction-record-validation")
        end
    end
    names
end

function require_file(path::AbstractString; nonempty::Bool=true)
    isfile(path) || throw(ArgumentError("missing required artifact: $path"))
    nonempty && filesize(path) == 0 && throw(ArgumentError("required artifact is empty: $path"))
    path
end

function validate_run_artifacts(run_dir::AbstractString; mode::AbstractString,
        stages::Vector{String}, seeds::Vector{Int}, configurations::Vector{String},
        expected_image_epochs::Int, expected_pendulum_epochs::Int,
        expected_backend::AbstractString, retraction_repo::AbstractString,
        allow_validation_failures::Bool=false)
    mode in ("smoke", "full") || throw(ArgumentError("unknown mode: $mode"))
    isempty(stages) && throw(ArgumentError("stage list is empty"))
    all(stage -> stage in RUN_STAGES, stages) || throw(ArgumentError(
        "unknown stage in $(join(stages, ','))"))
    "none" in stages && length(stages) != 1 && throw(ArgumentError(
        "the test-only `none` stage cannot be combined with experiment stages"))
    isempty(seeds) && throw(ArgumentError("seed list is empty"))

    require_file(joinpath(run_dir, "environment.txt"))
    require_file(joinpath(run_dir, "nvidia-smi.txt"))
    summaries = String[]
    for dataset in ("mnist", "fashion-mnist")
        dataset in stages || continue
        for filename in ("$dataset-report.txt", "$dataset-losses.csv", "$dataset-runs.csv",
                "$dataset.jld2")
            require_file(joinpath(run_dir, filename))
        end
        summary = validate_image_artifacts(
            joinpath(run_dir, "$dataset-runs.csv"),
            joinpath(run_dir, "$dataset-losses.csv"); dataset, seeds, configurations,
            expected_epochs=expected_image_epochs, expected_backend,
            allow_validation_failures)
        push!(summaries, "$dataset=$(summary.records) records/$(summary.losses) losses")
        if mode == "full"
            for filename in ("$dataset-warmup-report.txt", "$dataset-warmup-losses.csv",
                    "$dataset-warmup-runs.csv", "$dataset-warmup.jld2")
                require_file(joinpath(run_dir, filename))
            end
            warmup = validate_image_artifacts(
                joinpath(run_dir, "$dataset-warmup-runs.csv"),
                joinpath(run_dir, "$dataset-warmup-losses.csv"); dataset,
                seeds=[first(seeds)], configurations, expected_epochs=1, expected_backend,
                allow_validation_failures=true)
            push!(summaries,
                "$dataset-warmup=$(warmup.records) records/$(warmup.losses) losses")
        end
    end
    if "pendulum" in stages
        require_file(joinpath(run_dir, "pendulum-runs.csv"))
        summary = validate_pendulum_artifacts(joinpath(run_dir, "pendulum-runs.csv"), run_dir;
            seeds, expected_epochs=expected_pendulum_epochs, expected_backend)
        push!(summaries, "pendulum=$(summary.records) records")
    end
    if "retraction" in stages
        required_paths = expected_backend == "cuda" ?
            [("AugmentedPade", "CPU"), ("ScaledSquaring", "CUDA"), ("NativePade", "CUDA")] :
            [("AugmentedPade", "CPU"), ("ScaledSquaring", "CPU"), ("NativePade", "CPU")]
        summary = validate_records(joinpath(run_dir, "retraction-runs.csv");
            required_paths, go_repo=retraction_repo)
        push!(summaries, "retraction=$(summary.rows) records")
    end
    expected_stages = expected_stage_names(mode, stages, seeds)
    stage_summary = validate_stage_table(joinpath(run_dir, "stages.csv"), expected_stages)
    push!(summaries, "stages=$(stage_summary.rows) rows")
    summaries
end

end
