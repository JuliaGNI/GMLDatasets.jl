# A command-line parser for the three scripts in this directory that take one.
#
# Each of them used to carry its own `while index <= length(args)` chain with one branch per
# option — around fifty lines apiece, all of it the same shape. Here the option set *is* the
# table of defaults: a `Bool` default makes a flag that takes no value, and anything else takes
# one and is parsed to the type of its default. Keys are `:snake_case`; the option spelling is
# the same word in `--kebab-case`.
#
# Domain constraints stay with the caller. This file only turns `ARGS` into a `NamedTuple`.

"""
    parse_arguments(args, defaults::NamedTuple; usage)

Parse `args` against `defaults` and return a `NamedTuple` with the same keys. `--help` prints
`usage` and returns `nothing`. An unknown option, a missing value or an unparsable value is an
`ArgumentError`.

A `Vector` default collects a repeated option, so `--require a --require b` gives `["a", "b"]`.
"""
function parse_arguments(args, defaults::NamedTuple; usage::AbstractString)
    options = Dict(String(key) => key for key in keys(defaults))
    values = Dict{Symbol, Any}(key => copy_default(getfield(defaults, key))
    for key in keys(defaults))

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            println(usage)
            return nothing
        end
        startswith(argument, "--") ||
            throw(ArgumentError("unexpected argument: $argument"))
        name = replace(argument[3:end], '-' => '_')
        haskey(options, name) || throw(ArgumentError("unknown argument: $argument"))
        key = options[name]
        default = getfield(defaults, key)

        if default isa Bool
            values[key] = true
            index += 1
        else
            index == length(args) &&
                throw(ArgumentError("missing value for $argument"))
            values[key] = parse_argument(default, args[index + 1], argument, values[key])
            index += 2
        end
    end
    NamedTuple{keys(defaults)}(Tuple(values[key] for key in keys(defaults)))
end

copy_default(default) = default isa Vector ? copy(default) : default

parse_argument(::AbstractString, value, _, _) = value
parse_argument(default::Vector, value, _, collected) = push!(collected, value)
function parse_argument(default::Number, value, option, _)
    parsed = tryparse(typeof(default), value)
    parsed === nothing &&
        throw(ArgumentError("$option requires a $(typeof(default)), got $value"))
    parsed
end
