# Schema, provenance capture and validation for the retraction benchmark records.
#
# The header itself is in `headers.jl`, next to the ones the trainers write.
#
# This file holds definitions only; `records.jl` is the module that includes it.

const RETRACTION_ALGORITHMS = Set(("ScaledSquaring", "NativePade", "AugmentedPade"))
const RETRACTION_BACKENDS = Set(("CPU", "CUDA"))
const RETRACTION_PRECISIONS = Set(("Float32", "Float64"))
const RETRACTION_MEMORY_METRICS = Dict(
    "CPU" => "host_allocated_bytes", "CUDA" => "device_allocated_bytes")
const RETRACTION_ERROR_FIELDS = (
    "agreement_error", "forward_error", "manifold_constraint_error")

# ---------------------------------------------------------------- source provenance ---

git(root::AbstractString, args...) = Cmd(vcat(["git", "-C", root], collect(args)))

function source_patch(root::AbstractString)
    patch = IOBuffer()
    write(patch, read(git(root, "diff", "--binary", "HEAD", "--")))
    untracked = read(git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    for path in sort(filter(!isempty, split(String(untracked), '\0')))
        # `--no-index` exits 1 when the files differ, which is every untracked file
        write(patch, read(ignorestatus(git(root, "diff", "--binary", "--no-index", "--",
            "/dev/null", path))))
    end
    take!(patch)
end

"""
    capture_source(repo, patch_path; write_patch = true)

The exact checked-out commit of `repo` plus a reproducible patch of everything not in it, including
untracked files.

This is the harness's one provenance capture. `capture_source.jl` is its command line and
`run_experiments.sh` calls that for every repository a bundle records — the package's own and the
`GeometricOptimizers` checkout the retraction stage measures — so there is one implementation of
this and not one per repository or per language.

The `dirty` flag is cross-checked against the patch rather than merely reported beside it: a tree
that `git status` calls dirty and that produces an empty patch has entries the capture did not see,
which would make the bundle claim to record a state it does not.
"""
function capture_source(repo::AbstractString, patch_path::AbstractString;
        write_patch::Bool = true)
    root = realpath(readchomp(git(repo, "rev-parse", "--show-toplevel")))
    sha = readchomp(git(root, "rev-parse", "HEAD"))
    status = read(git(root, "status", "--porcelain=v1"), String)
    dirty = !isempty(strip(status))
    patch = source_patch(root)
    dirty == !isempty(patch) || throw(ArgumentError(
        "$(basename(root)) status and captured patch disagree; ignored or unusual worktree " *
        "entries may be present"))

    if write_patch
        mkpath(dirname(abspath(patch_path)))
        write(patch_path, patch)
    end
    (root = root, sha = sha, dirty = dirty, status = status, patch = patch,
        patch_sha256 = bytes2hex(sha256(patch)))
end

# --------------------------------------------------------------------- validation ---

function is_hex(value::AbstractString, digits::Integer)
    ncodeunits(value) == digits && all(c -> isdigit(c) || c in 'a':'f', value)
end

# The per-row checks that read one column and need nothing else. `parse_integer` and
# `parse_float` carry their own bounds, so the table is (field, checker) and the loop below is
# the whole of the column validation; the cross-column rules follow it as named lines.
const RETRACTION_COLUMN_CHECKS = (
    ("matrix_rows", (r, w) -> parse_integer(r, "matrix_rows", w; minimum = 1)),
    ("matrix_columns", (r, w) -> parse_integer(r, "matrix_columns", w; minimum = 1)),
    ("stiefel_columns", (r, w) -> parse_integer(r, "stiefel_columns", w; minimum = 1)),
    ("lift_norm", (r, w) -> parse_float(r, "lift_norm", w; nonnegative = true)),
    ("scale", (r, w) -> parse_float(r, "scale", w; nonnegative = true)),
    ("seed", (r, w) -> parse_integer(r, "seed", w; minimum = 0)),
    ("repetition", (r, w) -> parse_integer(r, "repetition", w; minimum = 0)),
    ("runtime_seconds", (r, w) -> parse_float(r, "runtime_seconds", w; nonnegative = true)),
    ("warmup", (r, w) -> parse_bool(r, "warmup", w)),
    ("success", (r, w) -> parse_bool(r, "success", w)),
    ("go_dirty", (r, w) -> parse_bool(r, "go_dirty", w))
)

function validate_retraction_row(record, where)
    values = Dict{String, Any}(field => check(record, where)
    for (field, check) in RETRACTION_COLUMN_CHECKS)

    record["schema_version"] == string(RETRACTION_SCHEMA_VERSION) || throw(ArgumentError(
        "$where has unsupported schema version $(record["schema_version"])"))
    record["algorithm"] in RETRACTION_ALGORITHMS ||
        throw(ArgumentError("$where has unknown algorithm $(record["algorithm"])"))
    record["backend"] in RETRACTION_BACKENDS ||
        throw(ArgumentError("$where has unknown backend $(record["backend"])"))
    record["algorithm"] == "AugmentedPade" && record["backend"] != "CPU" &&
        throw(ArgumentError(
            "$where runs host-only AugmentedPade on $(record["backend"])"))
    record["precision"] in RETRACTION_PRECISIONS ||
        throw(ArgumentError("$where has unknown precision $(record["precision"])"))
    record["memory_metric"] == RETRACTION_MEMORY_METRICS[record["backend"]] ||
        throw(ArgumentError(
            "$where has memory metric $(record["memory_metric"]) for backend $(record["backend"])"))
    isempty(record["device"]) && throw(ArgumentError("$where has an empty device"))
    values["matrix_rows"] == values["matrix_columns"] ||
        throw(ArgumentError("$where does not describe a square lift matrix"))
    values["stiefel_columns"] <= values["matrix_rows"] ||
        throw(ArgumentError("$where has more Stiefel columns than matrix rows"))
    values["warmup"] == (values["repetition"] == 0) ||
        throw(ArgumentError("$where must use repetition 0 exactly for warm-up rows"))
    is_hex(record["go_sha"], 40) || throw(ArgumentError("$where has an invalid GO SHA"))
    is_hex(record["go_patch_sha256"], 64) ||
        throw(ArgumentError("$where has an invalid GO patch SHA-256"))
    isempty(record["go_patch_file"]) &&
        throw(ArgumentError("$where has an empty GO patch file"))

    if values["success"]
        for field in RETRACTION_ERROR_FIELDS
            parse_float(record, field, where; nonnegative = true)
        end
        parse_integer(record, "memory_bytes", where; minimum = 0)
        isempty(record["error_type"]) ||
            throw(ArgumentError("$where is successful but has an error type"))
        isempty(record["error_message"]) ||
            throw(ArgumentError("$where is successful but has an error message"))
    else
        for field in RETRACTION_ERROR_FIELDS
            isnan(parse_float(record, field, where; finite = false)) ||
                throw(ArgumentError("$where failure must use NaN for $field"))
        end
        parse_integer(record, "memory_bytes", where; minimum = 0) == 0 ||
            throw(ArgumentError("$where failure must use zero memory bytes"))
        isempty(record["error_type"]) &&
            throw(ArgumentError("$where failure has no error type"))
        isempty(record["error_message"]) &&
            throw(ArgumentError("$where failure has no error message"))
    end
    values
end

"""
    validate_records(path; required_paths, allow_failures, require_failure, go_repo)

Validate a retraction record file and return its row counts. Every row is checked on its own,
the rows are checked against each other for duplicate invocations, warm-up structure and a
single GO source identity, and the identity is checked against the patch file beside the CSV —
and, when `go_repo` is given, against that checkout as it stands now.
"""
function validate_records(path::AbstractString; required_paths = Tuple{String, String}[],
        allow_failures::Bool = false, require_failure::Bool = false,
        go_repo::Union{Nothing, String} = nothing)
    records = read_table(path, RETRACTION_HEADER)
    seen = Set{Tuple}()
    warmups = Set{Tuple}()
    successes = Set{Tuple{String, String}}()
    failure_count = 0
    identity = Tuple(record_identity(first(records)))

    for (offset, record) in enumerate(records)
        where = "$path:$(offset + 1)"
        values = validate_retraction_row(record, where)
        Tuple(record_identity(record)) == identity ||
            throw(ArgumentError("$where has inconsistent GO source identity"))

        path_key = (record["algorithm"], record["backend"], record["precision"],
            values["matrix_rows"], values["stiefel_columns"], record["seed"], record["scale"])
        (path_key, values["repetition"]) in seen &&
            throw(ArgumentError("$where duplicates a benchmark invocation"))
        push!(seen, (path_key, values["repetition"]))
        if values["warmup"]
            push!(warmups, path_key)
        else
            path_key in warmups || throw(ArgumentError(
                "$where is a steady-state row without a preceding warm-up row"))
            values["success"] && push!(successes, (record["algorithm"], record["backend"]))
        end
        values["success"] || (failure_count += 1)
        values["success"] || allow_failures ||
            throw(ArgumentError("$where records a benchmark failure"))
    end

    for required in required_paths
        required in successes || throw(ArgumentError(
            "no successful steady-state row for $(required[1]) on $(required[2])"))
    end
    require_failure && failure_count == 0 &&
        throw(ArgumentError("no explicit failure row was found"))

    validate_source_identity(path, identity, go_repo)
    warmup_count = count(record -> record["warmup"] == "true", records)
    (rows = length(records), warmups = warmup_count,
        steady_state = length(records) - warmup_count, failures = failure_count)
end

function record_identity(record)
    (record["go_sha"], record["go_dirty"], record["go_patch_file"],
        record["go_patch_sha256"])
end

"""Check the recorded GO identity against the patch beside the CSV, and against `go_repo`."""
function validate_source_identity(path, identity, go_repo)
    sha, dirty, patch_file, patch_sha256 = identity
    patch_path = normpath(joinpath(dirname(abspath(path)), patch_file))
    isfile(patch_path) || throw(ArgumentError("GO patch file is missing: $patch_path"))
    patch = read(patch_path)
    bytes2hex(sha256(patch)) == patch_sha256 ||
        throw(ArgumentError("GO patch SHA-256 does not match $patch_path"))
    (dirty == "true") == !isempty(patch) ||
        throw(ArgumentError("GO dirty flag does not agree with the captured patch"))

    go_repo === nothing && return nothing
    current = capture_source(go_repo, patch_path; write_patch = false)
    current.sha == sha ||
        throw(ArgumentError("GO HEAD changed after the benchmark: $sha != $(current.sha)"))
    current.dirty == (dirty == "true") ||
        throw(ArgumentError("GO dirty state changed after the benchmark"))
    current.patch == patch || throw(ArgumentError("GO patch changed after the benchmark"))
    nothing
end
