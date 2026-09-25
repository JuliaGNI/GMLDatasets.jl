"""
The record schemas of the revision harness: the CSV primitives, the optimizer configuration
table, the retraction benchmark schema and the run-directory validators.

One module over five files rather than two modules with two copies of the CSV layer between
them. The artifact validator calls the retraction validator, both parse the same CSV dialect and
both name the same configurations, so splitting them into separate modules meant duplicating
everything they share.

Consumers include this file and say `using .RunRecords`.
"""
module RunRecords

using SHA: sha256

include("headers.jl")
include("csv_records.jl")
include("configurations.jl")
include("retraction_record_schema.jl")
include("run_artifact_schema.jl")

# Everything the trainers, the validators and the runner reach for. The two trainers used to
# `include` `headers.jl`, `configurations.jl` and `csv_records.jl` at top level while this module
# included the same three into itself, which put two copies of every definition in one session,
# reachable by two different paths. They go through here now, as the validators always did.
export CONFIGURATIONS, CONFIGURATION_NAMES, CONFIGURATION_ORDER,
       PENDULUM_CONFIGURATION_ORDER,
       IMAGE_LOSS_HEADER, IMAGE_RECORD_HEADER, PENDULUM_LOSS_HEADER,
       PENDULUM_RECORD_HEADER, RETRACTION_HEADER, STAGE_HEADER, STEP_TIMING_COLUMNS,
       MNIST_RUN_SCHEMA_VERSION, PENDULUM_RUN_SCHEMA_VERSION, RETRACTION_SCHEMA_VERSION,
       append_record, append_records,
       capture_source, csv_field, normalize_configurations,
       normalize_pendulum_configurations,
       parse_csv_line, read_table, split_list, validate_image_artifacts,
       validate_pendulum_artifacts, validate_records, validate_run_artifacts,
       validate_stage_table, write_records, write_retraction_records

"""Write retraction benchmark rows in the order of [`RETRACTION_HEADER`](@ref)."""
function write_retraction_records(path::AbstractString, records)
    write_records(path, RETRACTION_HEADER, records)
end

end
