# Distil one `mnist_cuda.jl` run into the three small series the documentation plots.
#
# The run itself leaves four files in `results/`, all of them gitignored:
# a report, a per-batch loss CSV of ~58000 rows, the trained parameters as a `.jld2` and the
# raw stdout. None of those belong in the repository — the CSV is 2.9 MB and the `.jld2` 2.9 MB,
# and the figures need neither the individual batches nor the parameters. What they do need is
# three series of at most 500 rows, which is what this script writes to `docs/src/data/`:
#
#   mnist_training_loss.csv    the per-epoch mean loss of all four configurations, 500 rows
#   mnist_test_accuracy.csv    the test accuracy, every 25 epochs, 20 rows
#   mnist_manifold_drift.csv   ‖YᵀY-I‖ for the three constrained runs, every 25 epochs, 20 rows
#
# Those three are checked in, so building the documentation needs neither a GPU nor the run.
# Rerun this after a new run to refresh them:
#
#   julia scripts/geometric_optimizers/distill_mnist_results.jl results/20260808_212753_full
#
# The argument is the prefix the run shares; `_losses.csv` and `_report.txt` are appended. Only
# the standard library is used, so no project needs to be instantiated.

using Printf: @printf, @sprintf
using Statistics: mean

const CONFIGURATIONS = ["Stiefel weights, Adam", "regular weights, Adam",
                        "Stiefel weights, gradient", "Stiefel weights, momentum"]

# The column names the plotting code in `docs/src/manifold_optimizers.md` looks for. They are
# the configuration names of `mnist_cuda.jl` written so that they survive a CSV header.
const COLUMNS = Dict(CONFIGURATIONS .=>
                     ["adam_stiefel", "adam_regular", "gradient_stiefel", "momentum_stiefel"])

# The constrained runs, in the order the columns are written. `regular weights, Adam` has no
# manifold to drift off, so it is absent from the drift file rather than empty in it.
const CONSTRAINED = ["Stiefel weights, Adam", "Stiefel weights, gradient",
                     "Stiefel weights, momentum"]

"""
    epoch_losses(path)

Mean loss per epoch and configuration, from the per-batch CSV a run writes.

The CSV is `run,configuration,epoch,batch,step,loss` with the configuration quoted because it
contains a comma, which is why this is a regular expression and not a `split` on `,`. A line
that does not match is a corrupted tail of an interrupted run and is skipped rather than fatal.
"""
function epoch_losses(path::AbstractString)
    losses = Dict{String,Dict{Int,Vector{Float64}}}()
    line_pattern = r"^\d+,\"([^\"]*)\",(\d+),\d+,\d+,([-\d.eE+]+)$"
    open(path) do io
        readline(io)  # header
        for line in eachline(io)
            m = match(line_pattern, line)
            m === nothing && continue
            configuration, epoch, loss = m[1], parse(Int, m[2]), parse(Float64, m[3])
            per_epoch = get!(() -> Dict{Int,Vector{Float64}}(), losses, configuration)
            push!(get!(() -> Float64[], per_epoch, epoch), loss)
        end
    end
    Dict(configuration => Dict(epoch => mean(values) for (epoch, values) in per_epoch)
         for (configuration, per_epoch) in losses)
end

"""
    report_series(text, heading)

The `epoch:value` series a run's report prints under `heading`, per configuration.

The report ends with two such blocks — one for the test accuracy, one for the drift — each a
heading line followed by one line per configuration and terminated by a blank line. Reading them
here rather than recomputing anything keeps the checked-in files traceable to a line of the
report. The blank line is what separates the two blocks, so `.` must not match a newline here:
with the `s` flag the accuracy block would run on into the drift block and the drift numbers
would overwrite the accuracies.
"""
function report_series(text::AbstractString, heading::AbstractString)
    block = match(Regex("^$(heading)[^\\n]*\\n((?:[^\\n]+\\n)+)", "m"), text)
    block === nothing && error("no `$(heading)` block in the report")
    series = Dict{String,Vector{Pair{Int,Float64}}}()
    for line in eachline(IOBuffer(block[1]))
        index = findfirst(c -> startswith(line, c), CONFIGURATIONS)
        index === nothing && continue
        configuration = CONFIGURATIONS[index]
        pairs = [parse(Int, first(p)) => parse(Float64, last(p))
                 for p in split.(split(line[(ncodeunits(configuration) + 1):end]), ':')]
        series[configuration] = sort(pairs, by=first)
    end
    series
end

"""
    write_series(path, epochs, columns, values, format)

One CSV with an `epoch` column and one column per entry of `columns`.
"""
function write_series(path, epochs, columns, values, format)
    open(path, "w") do io
        println(io, "epoch,", join(columns, ','))
        for epoch in epochs
            println(io, epoch, ',', join((format(v[epoch]) for v in values), ','))
        end
    end
    @printf("%-40s %4i rows\n", path, length(epochs))
end

function main(prefix::AbstractString)
    text = read(prefix * "_report.txt", String)
    losses = epoch_losses(prefix * "_losses.csv")
    accuracy = report_series(text, "test accuracy over the run")
    drift = report_series(text, "‖YᵀY-I‖ over the run")

    for configuration in CONFIGURATIONS
        haskey(losses, configuration) || error("no `$(configuration)` in the loss CSV")
        haskey(accuracy, configuration) || error("no `$(configuration)` accuracy in the report")
    end

    directory = joinpath(dirname(dirname(@__DIR__)), "docs", "src", "data")
    mkpath(directory)

    epochs = sort(collect(keys(losses[first(CONFIGURATIONS)])))
    write_series(joinpath(directory, "mnist_training_loss.csv"), epochs,
                 [COLUMNS[c] for c in CONFIGURATIONS],
                 [losses[c] for c in CONFIGURATIONS], v -> @sprintf("%.6f", v))

    evaluated = first.(accuracy[first(CONFIGURATIONS)])
    write_series(joinpath(directory, "mnist_test_accuracy.csv"), evaluated,
                 [COLUMNS[c] for c in CONFIGURATIONS],
                 [Dict(accuracy[c]) for c in CONFIGURATIONS], v -> @sprintf("%.4f", v))

    write_series(joinpath(directory, "mnist_manifold_drift.csv"), evaluated,
                 [COLUMNS[c] for c in CONSTRAINED],
                 [Dict(drift[c]) for c in CONSTRAINED], v -> @sprintf("%.2e", v))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        error("usage: julia scripts/geometric_optimizers/distill_mnist_results.jl results/<timestamp>_full")
    main(only(ARGS))
end
