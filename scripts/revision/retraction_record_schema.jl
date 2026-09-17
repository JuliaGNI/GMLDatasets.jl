module RetractionRecordSchema

using SHA: sha256

export HEADER, capture_source, read_records, validate_records, write_records

const HEADER = [
    "schema_version",
    "algorithm",
    "backend",
    "device",
    "precision",
    "matrix_rows",
    "matrix_columns",
    "stiefel_columns",
    "lift_norm",
    "agreement_error",
    "forward_error",
    "manifold_constraint_error",
    "runtime_seconds",
    "memory_bytes",
    "memory_metric",
    "repetition",
    "warmup",
    "success",
    "error_type",
    "error_message",
    "seed",
    "scale",
    "go_sha",
    "go_dirty",
    "go_patch_file",
    "go_patch_sha256",
]

const ALGORITHMS = Set(("ScaledSquaring", "NativePade", "AugmentedPade"))
const BACKENDS = Set(("CPU", "CUDA"))
const PRECISIONS = Set(("Float32", "Float64"))
const MEMORY_METRICS = Dict("CPU" => "host_allocated_bytes", "CUDA" => "device_allocated_bytes")

csv_field(value) = begin
    text = string(value)
    occursin(r"[\",\r\n]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
end

function write_records(path::AbstractString, records)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(HEADER, ','))
        for record in records
            Set(keys(record)) == Set(HEADER) ||
                throw(ArgumentError("record fields do not match the retraction schema"))
            println(io, join((csv_field(record[field]) for field in HEADER), ','))
        end
    end
    path
end

function parse_csv_line(line::AbstractString)
    fields = String[]
    field = IOBuffer()
    quoted = false
    just_closed_quote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if quoted
            if c == '\"'
                next_i = nextind(line, i)
                if next_i <= lastindex(line) && line[next_i] == '\"'
                    write(field, '\"')
                    i = next_i
                else
                    quoted = false
                    just_closed_quote = true
                end
            else
                write(field, c)
            end
        elseif just_closed_quote
            c == ',' || throw(ArgumentError("unexpected character after a closing CSV quote"))
            push!(fields, String(take!(field)))
            just_closed_quote = false
        elseif c == ','
            push!(fields, String(take!(field)))
        elseif c == '\"'
            position(field) == 0 || throw(ArgumentError("CSV quote must begin a field"))
            quoted = true
        else
            write(field, c)
        end
        i = nextind(line, i)
    end
    quoted && throw(ArgumentError("unterminated quoted CSV field"))
    push!(fields, String(take!(field)))
    fields
end

function read_records(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("retraction record file is empty: $path"))
    header = parse_csv_line(first(lines))
    header == HEADER || throw(ArgumentError(
        "unexpected retraction record header; expected $(join(HEADER, ','))"))

    records = Dict{String,String}[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        line_number = offset + 1
        isempty(line) && throw(ArgumentError("blank CSV row at line $line_number"))
        fields = parse_csv_line(line)
        length(fields) == length(HEADER) || throw(ArgumentError(
            "line $line_number has $(length(fields)) fields; expected $(length(HEADER))"))
        push!(records, Dict(zip(HEADER, fields)))
    end
    isempty(records) && throw(ArgumentError("retraction record file has no data rows: $path"))
    records
end

git(root::AbstractString, args...) = Cmd(vcat(["git", "-C", root], collect(args)))

function repository_root(path::AbstractString)
    root = readchomp(git(path, "rev-parse", "--show-toplevel"))
    realpath(root)
end

function source_patch(root::AbstractString)
    tracked = read(git(root, "diff", "--binary", "HEAD", "--"))
    untracked_output = read(git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    untracked = filter(path -> !isempty(path), split(String(untracked_output), '\0'))
    patch = IOBuffer()
    write(patch, tracked)
    for path in sort(untracked)
        command = git(root, "diff", "--binary", "--no-index", "--", "/dev/null", path)
        write(patch, read(ignorestatus(command)))
    end
    take!(patch)
end

"""Capture the exact checked-out GO commit plus a reproducible patch, including untracked files."""
function capture_source(repo::AbstractString, patch_path::AbstractString; write_patch::Bool=true)
    root = repository_root(repo)
    sha = readchomp(git(root, "rev-parse", "HEAD"))
    status = read(git(root, "status", "--porcelain=v1", "-z"))
    patch = source_patch(root)
    dirty = !isempty(status)
    dirty == !isempty(patch) || throw(ArgumentError(
        "GO status and captured patch disagree; ignored or unusual worktree entries may be present"))

    if write_patch
        mkpath(dirname(abspath(patch_path)))
        write(patch_path, patch)
    end
    (root=root, sha=sha, dirty=dirty, patch=patch, patch_sha256=bytes2hex(sha256(patch)))
end

parse_bool(value, field, line) = value == "true" ? true : value == "false" ? false :
    throw(ArgumentError("line $line has invalid $field boolean: $value"))

function parse_integer(record, field, line; minimum=nothing)
    value = tryparse(Int, record[field])
    value === nothing && throw(ArgumentError("line $line has invalid integer $field: $(record[field])"))
    minimum !== nothing && value < minimum &&
        throw(ArgumentError("line $line has $field below $minimum: $value"))
    value
end

function parse_float(record, field, line; finite::Bool=true, nonnegative::Bool=false)
    value = tryparse(Float64, record[field])
    value === nothing && throw(ArgumentError("line $line has invalid float $field: $(record[field])"))
    finite && !isfinite(value) && throw(ArgumentError("line $line has non-finite $field: $value"))
    nonnegative && value < 0 && throw(ArgumentError("line $line has negative $field: $value"))
    value
end

function valid_hex(value::AbstractString, length_required::Integer)
    ncodeunits(value) == length_required && all(c -> isdigit(c) || c in 'a':'f', value)
end

function validate_records(path::AbstractString; required_paths=Tuple{String,String}[],
    allow_failures::Bool=false, require_failure::Bool=false, go_repo::Union{Nothing,String}=nothing)
    records = read_records(path)
    seen = Set{Tuple}()
    warmups = Dict{Tuple,Int}()
    successes = Set{Tuple{String,String}}()
    failure_count = 0

    identity = (records[1]["go_sha"], records[1]["go_dirty"], records[1]["go_patch_file"],
        records[1]["go_patch_sha256"])
    for (index, record) in enumerate(records)
        line = index + 1
        record["schema_version"] == "1" ||
            throw(ArgumentError("line $line has unsupported schema version $(record["schema_version"])"))
        record["algorithm"] in ALGORITHMS ||
            throw(ArgumentError("line $line has unknown algorithm $(record["algorithm"])"))
        record["backend"] in BACKENDS ||
            throw(ArgumentError("line $line has unknown backend $(record["backend"])"))
        record["algorithm"] == "AugmentedPade" && record["backend"] != "CPU" &&
            throw(ArgumentError("line $line runs host-only AugmentedPade on $(record["backend"])"))
        record["precision"] in PRECISIONS ||
            throw(ArgumentError("line $line has unknown precision $(record["precision"])"))
        record["memory_metric"] == MEMORY_METRICS[record["backend"]] || throw(ArgumentError(
            "line $line has memory metric $(record["memory_metric"]) for backend $(record["backend"])"))
        isempty(record["device"]) && throw(ArgumentError("line $line has an empty device"))

        matrix_rows = parse_integer(record, "matrix_rows", line; minimum=1)
        matrix_columns = parse_integer(record, "matrix_columns", line; minimum=1)
        stiefel_columns = parse_integer(record, "stiefel_columns", line; minimum=1)
        matrix_rows == matrix_columns ||
            throw(ArgumentError("line $line does not describe a square lift matrix"))
        stiefel_columns <= matrix_rows ||
            throw(ArgumentError("line $line has more Stiefel columns than matrix rows"))
        parse_float(record, "lift_norm", line; nonnegative=true)
        parse_float(record, "scale", line; nonnegative=true)
        parse_integer(record, "seed", line; minimum=0)
        repetition = parse_integer(record, "repetition", line; minimum=0)
        warmup = parse_bool(record["warmup"], "warmup", line)
        success = parse_bool(record["success"], "success", line)
        parse_bool(record["go_dirty"], "go_dirty", line)
        warmup == (repetition == 0) || throw(ArgumentError(
            "line $line must use repetition 0 exactly for warm-up rows"))

        valid_hex(record["go_sha"], 40) ||
            throw(ArgumentError("line $line has an invalid GO SHA"))
        valid_hex(record["go_patch_sha256"], 64) ||
            throw(ArgumentError("line $line has an invalid GO patch SHA-256"))
        isempty(record["go_patch_file"]) &&
            throw(ArgumentError("line $line has an empty GO patch file"))
        current_identity = (record["go_sha"], record["go_dirty"], record["go_patch_file"],
            record["go_patch_sha256"])
        current_identity == identity ||
            throw(ArgumentError("line $line has inconsistent GO source identity"))

        key = (record["algorithm"], record["backend"], record["precision"], matrix_rows,
            stiefel_columns, record["seed"], record["scale"], repetition)
        key in seen && throw(ArgumentError("line $line duplicates a benchmark invocation"))
        push!(seen, key)
        path_key = (record["algorithm"], record["backend"], record["precision"], matrix_rows,
            stiefel_columns, record["seed"], record["scale"])
        if warmup
            warmups[path_key] = line
        else
            haskey(warmups, path_key) || throw(ArgumentError(
                "line $line is a steady-state row without a preceding warm-up row"))
        end

        parse_float(record, "runtime_seconds", line; nonnegative=true)
        if success
            for field in ("agreement_error", "forward_error", "manifold_constraint_error")
                parse_float(record, field, line; nonnegative=true)
            end
            parse_integer(record, "memory_bytes", line; minimum=0)
            isempty(record["error_type"]) ||
                throw(ArgumentError("line $line is successful but has an error type"))
            isempty(record["error_message"]) ||
                throw(ArgumentError("line $line is successful but has an error message"))
            warmup || push!(successes, (record["algorithm"], record["backend"]))
        else
            failure_count += 1
            allow_failures || throw(ArgumentError("line $line records a benchmark failure"))
            for field in ("agreement_error", "forward_error", "manifold_constraint_error")
                isnan(parse_float(record, field, line; finite=false)) || throw(ArgumentError(
                    "line $line failure must use NaN for $field"))
            end
            parse_integer(record, "memory_bytes", line; minimum=0) == 0 ||
                throw(ArgumentError("line $line failure must use zero memory bytes"))
            isempty(record["error_type"]) &&
                throw(ArgumentError("line $line failure has no error type"))
            isempty(record["error_message"]) &&
                throw(ArgumentError("line $line failure has no error message"))
        end
    end

    for required in required_paths
        required in successes || throw(ArgumentError(
            "no successful steady-state row for $(required[1]) on $(required[2])"))
    end
    require_failure && failure_count == 0 &&
        throw(ArgumentError("no explicit failure row was found"))

    patch_path = normpath(joinpath(dirname(abspath(path)), identity[3]))
    isfile(patch_path) || throw(ArgumentError("GO patch file is missing: $patch_path"))
    patch = read(patch_path)
    bytes2hex(sha256(patch)) == identity[4] ||
        throw(ArgumentError("GO patch SHA-256 does not match $patch_path"))
    (identity[2] == "true") == !isempty(patch) || throw(ArgumentError(
        "GO dirty flag does not agree with the captured patch"))

    if go_repo !== nothing
        current = capture_source(go_repo, patch_path; write_patch=false)
        current.sha == identity[1] || throw(ArgumentError(
            "GO HEAD changed after the benchmark: $(identity[1]) != $(current.sha)"))
        current.dirty == (identity[2] == "true") ||
            throw(ArgumentError("GO dirty state changed after the benchmark"))
        current.patch == patch ||
            throw(ArgumentError("GO patch changed after the benchmark"))
    end

    (rows=length(records), warmups=count(r -> r["warmup"] == "true", records),
        steady_state=count(r -> r["warmup"] == "false", records), failures=failure_count)
end

end
