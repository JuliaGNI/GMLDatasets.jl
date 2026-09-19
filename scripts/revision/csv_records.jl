# CSV reading, writing and field parsing, shared by the record schemas of this directory.
#
# The records are written by the trainers and the retraction benchmark and read back by the
# validators, so writer and reader have to agree on quoting. RFC 4180 quoting is what both do, and
# `parse_csv_line` is its inverse: a field may be quoted, a quoted field may contain commas,
# newlines and doubled quotes, and nothing else is accepted. A hand-rolled parser rather than
# `CSV.jl`: these tables are written by this directory and read by it, and a dependency whose own
# `[compat]` would enter the experiment manifest costs more than sixty lines here.
#
# The `parse_*` helpers take a location string rather than a path and a line number, so one caller
# can say `"retraction-runs.csv:12"` and another `"line 12"` without a second set of methods.
#
# This file holds definitions only; `records.jl` is the module that includes it.

"""Quote `value` for CSV output, but only where a bare field would be ambiguous."""
function csv_field(value)
    text = string(value)
    occursin(r"[\",\r\n]", text) ? "\"" * replace(text, '"' => "\"\"") * "\"" : text
end

"""Split one CSV line into its fields, honouring quoted fields and doubled quotes."""
function parse_csv_line(line::AbstractString)
    fields = String[]
    field = IOBuffer()
    quoted = false
    just_closed_quote = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if quoted
            next_index = nextind(line, index)
            if character != '"'
                write(field, character)
            elseif next_index <= lastindex(line) && line[next_index] == '"'
                write(field, '"')
                index = next_index
            else
                quoted = false
                just_closed_quote = true
            end
        elseif just_closed_quote
            character == ',' ||
                throw(ArgumentError("unexpected character after a closing CSV quote"))
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

"""
    write_records(path, header, records)

Write `records` — dictionaries keyed by the entries of `header` — to `path`, in header order.
A record whose keys are not exactly `header` is a programming error and is rejected rather than
written with a silently missing column.
"""
function write_records(path::AbstractString, header, records)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(header, ','))
        for record in records
            Set(keys(record)) == Set(header) ||
                throw(ArgumentError("record fields do not match the schema of $path"))
            println(io, join((csv_field(record[field]) for field in header), ','))
        end
    end
    path
end

"""
    append_records(path, header, records)

Append rows to `path`, writing `header` first if the file is new. The pendulum matrix runs one
process per configuration and seed, each appending its own rows to the shared tables, so those
cannot be written as one `write_records` call.
"""
function append_records(path::AbstractString, header, records)
    mkpath(dirname(abspath(path)))
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, join(header, ','))
        for record in records
            Set(keys(record)) == Set(header) ||
                throw(ArgumentError("record fields do not match the schema of $path"))
            println(io, join((csv_field(record[field]) for field in header), ','))
        end
    end
    path
end

function append_record(path::AbstractString, header, record)
    append_records(path, header, (record,))
end

"""
    read_table(path, expected_header; allow_empty = false)

Read `path` as a CSV table with exactly `expected_header`, returning one `Dict` per data row.
"""
function read_table(path::AbstractString, expected_header; allow_empty::Bool = false)
    isfile(path) || throw(ArgumentError("missing CSV file: $path"))
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("CSV file is empty: $path"))
    header = parse_csv_line(first(lines))
    header == expected_header || throw(ArgumentError(
        "unexpected header in $path; expected $(join(expected_header, ','))"))

    records = Dict{String, String}[]
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        location = "$path:$(offset + 1)"
        isempty(line) && throw(ArgumentError("blank CSV row at $location"))
        fields = parse_csv_line(line)
        length(fields) == length(expected_header) || throw(ArgumentError(
            "$location has $(length(fields)) fields; expected $(length(expected_header))"))
        push!(records, Dict(zip(expected_header, fields)))
    end
    allow_empty || !isempty(records) ||
        throw(ArgumentError("CSV file has no data rows: $path"))
    records
end

function parse_bool(record, field, location)
    value = record[field]
    value == "true" && return true
    value == "false" && return false
    throw(ArgumentError("$location has invalid $field boolean: $value"))
end

function parse_integer(record, field, location; minimum = nothing)
    value = tryparse(Int, record[field])
    value === nothing &&
        throw(ArgumentError("$location has invalid integer $field: $(record[field])"))
    minimum === nothing || value >= minimum ||
        throw(ArgumentError("$location has $field below $minimum: $value"))
    value
end

function parse_float(
        record, field, location; finite::Bool = true, nonnegative::Bool = false)
    value = tryparse(Float64, record[field])
    value === nothing &&
        throw(ArgumentError("$location has invalid float $field: $(record[field])"))
    !finite || isfinite(value) ||
        throw(ArgumentError("$location has non-finite $field: $value"))
    !nonnegative || value >= 0 ||
        throw(ArgumentError("$location has negative $field: $value"))
    value
end
