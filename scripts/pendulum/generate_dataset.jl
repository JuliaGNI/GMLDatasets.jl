using GMLDatasets
using Serialization

const output = get(ENV, "PENDULUM_DATASET_OUTPUT", "pendulum_dataset.jls")
const seed = parse(Int, get(ENV, "PENDULUM_SEED", "123"))

mkpath(dirname(abspath(output)))
isfile(output) && error("refusing to overwrite existing output: $output")

initial_conditions = [(angle, 0.0) for angle in range(0.2, 2.8; length=8)]
dataset = pendulum_dataset(
    initial_conditions;
    timespan=(0.0, 10.0),
    timestep=0.1,
)

open(output, "w") do io
    serialize(io, (; seed, dataset, coordinate_order=("q₁", "q₂", "p₁", "p₂")))
end
println("Wrote $(length(dataset)) trajectories to $output (seed=$seed)")
