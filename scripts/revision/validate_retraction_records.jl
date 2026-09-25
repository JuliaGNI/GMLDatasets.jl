#!/usr/bin/env julia

include("arguments.jl")
include("records.jl")
using .RunRecords: validate_records

const USAGE = """usage: validate_retraction_records.jl --input FILE [options]

  --go-repo DIR                 compare record identity and patch with this checkout
  --require ALGORITHM:BACKEND   require a successful non-warm-up row (repeatable)
  --allow-failures              accept well-formed explicit failure rows
  --require-failure             require at least one failure row (schema regression only)
"""

function main(args = ARGS)
    options = parse_arguments(args,
        (input = "", go_repo = "", require = String[], allow_failures = false,
            require_failure = false); usage = USAGE)
    options === nothing && return 0

    isempty(options.input) && throw(ArgumentError("--input is required"))
    options.require_failure && !options.allow_failures &&
        throw(ArgumentError("--require-failure also requires --allow-failures"))
    required_paths = map(options.require) do value
        parts = split(value, ':'; limit = 2)
        length(parts) == 2 || throw(ArgumentError("--require must be ALGORITHM:BACKEND"))
        (String(parts[1]), String(parts[2]))
    end

    summary = validate_records(options.input; required_paths,
        allow_failures = options.allow_failures, require_failure = options.require_failure,
        go_repo = isempty(options.go_repo) ? nothing : options.go_repo)
    println("validated $(summary.rows) retraction records ($(summary.warmups) warm-up, " *
            "$(summary.steady_state) steady-state, $(summary.failures) failure)")
    0
end

try
    exit(main())
catch error
    println(stderr, "retraction record validation failed: ", sprint(showerror, error))
    exit(1)
end
