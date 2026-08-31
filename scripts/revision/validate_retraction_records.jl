#!/usr/bin/env julia

include("retraction_record_schema.jl")
using .RetractionRecordSchema: validate_records

function usage(io::IO=stdout)
    println(io, """usage: validate_retraction_records.jl --input FILE [options]

  --go-repo DIR                 compare record identity and patch with this checkout
  --require ALGORITHM:BACKEND   require a successful non-warm-up row (repeatable)
  --allow-failures              accept well-formed explicit failure rows
  --require-failure             require at least one failure row (schema regression only)
""")
end

function main(args=ARGS)
    input = ""
    go_repo = nothing
    required = Tuple{String,String}[]
    allow_failures = false
    require_failure = false
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            usage()
            return 0
        elseif arg in ("--input", "--go-repo", "--require")
            index == length(args) && throw(ArgumentError("missing value for $arg"))
            value = args[index + 1]
            index += 2
            if arg == "--input"
                input = value
            elseif arg == "--go-repo"
                go_repo = value
            else
                parts = split(value, ':'; limit=2)
                length(parts) == 2 ||
                    throw(ArgumentError("--require must be ALGORITHM:BACKEND"))
                push!(required, (parts[1], parts[2]))
            end
        elseif arg == "--allow-failures"
            allow_failures = true
            index += 1
        elseif arg == "--require-failure"
            require_failure = true
            index += 1
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end
    isempty(input) && throw(ArgumentError("--input is required"))
    require_failure && !allow_failures &&
        throw(ArgumentError("--require-failure also requires --allow-failures"))
    summary = validate_records(input; required_paths=required, allow_failures,
        require_failure, go_repo)
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
