# Train one configuration of the four-way symplectic-autoencoder comparison on
# the four-dimensional pendulum data set.
#
# The PSD layers in `SymplecticAutoencoder` have Stiefel weights. All four
# configurations below keep those constraints, because removing them would make
# the model non-symplectic rather than provide a meaningful optimizer baseline:
#
#   geometric-adam-cayley   proposed coordinate-wise geometric Adam
#   scalar-moment-adam      Li et al.-style scalar-second-moment baseline
#   gradient                Riemannian gradient descent
#   momentum                Riemannian momentum
#
# `scripts/revision/run_experiments.sh` supplies the configuration, seed,
# checkpoint, and CSV paths for the full paired matrix. The script remains
# useful on its own: it defaults to geometric Adam and writes one HDF5 file.

using CUDA
using GeometricMachineLearning
using Random
using Printf

import AbstractNeuralNetworks: save
import GMLDatasets: angular_to_euclidean, pendulum
import HDF5
import GeometricOptimizers

# The record layer, through its one entry point: the headers, the configuration table and the CSV
# dialect the validators read, reached by the route they reach them by. Including its parts
# individually puts two copies of every definition in one session.
include(joinpath(@__DIR__, "..", "revision", "records.jl"))
using .RunRecords

# The `scalar-moment-adam` method, which the image trainer builds from the same call.
include(joinpath(@__DIR__, "..", "revision", "scalar_moment_adam.jl"))

const REDUCED_DIM = parse(Int, get(ENV, "SAE_REDUCED_DIM", "2"))
const N_EPOCHS = parse(Int, get(ENV, "SAE_N_EPOCHS", "1000"))
const BATCH_SIZE = parse(Int, get(ENV, "SAE_BATCH_SIZE", "256"))
const LEARNING_RATE = parse(Float32, get(ENV, "SAE_STEP_SIZE", "1e-4"))
const SCALAR_MOMENT_LEARNING_RATE = parse(Float32,
    get(ENV, "SAE_SCALAR_MOMENT_ADAM_LEARNING_RATE", string(LEARNING_RATE)))
const MOMENTUM_COEFFICIENT = parse(Float32, get(ENV, "SAE_MOMENTUM_COEFFICIENT", "0.5"))
const ADAM_BETA1 = parse(Float32, get(ENV, "SAE_ADAM_BETA1", "0.9"))
const ADAM_BETA2 = parse(Float32, get(ENV, "SAE_ADAM_BETA2", "0.99"))
const ADAM_EPSILON = parse(Float32, get(ENV, "SAE_ADAM_EPSILON", "1e-8"))
const SEED = parse(Int, get(ENV, "SAE_SEED", "123"))
const REPETITION = parse(Int, get(ENV, "SAE_REPETITION", "1"))
const CONFIGURATION_KEY = lowercase(get(ENV, "SAE_CONFIGURATION", "geometric-adam-cayley"))
const OUTPUT = get(ENV, "SAE_OUTPUT", "pendulum_sae.h5")
const RECORD_PATH = get(ENV, "SAE_RECORD", "")
const LOSSES_PATH = get(ENV, "SAE_LOSSES", "")
const REQUIRE_CUDA = parse(Bool, get(ENV, "SAE_REQUIRE_CUDA", "0"))

"""
    configuration(T)

The configuration `SAE_CONFIGURATION` selects, as its shared metadata plus the optimizer method
and the learning rate this script builds for it. `PENDULUM_CONFIGURATION_ORDER` is the four
intrinsic configurations: an SAE cannot have an unconstrained Adam row and stay symplectic.
"""
function configuration(T::Type{<:AbstractFloat})
    selected = normalize_pendulum_configurations(CONFIGURATION_KEY)
    length(selected) == 1 || error(
        "SAE_CONFIGURATION names one configuration, got `$CONFIGURATION_KEY`")
    key = only(selected)
    methods = Dict(
        "geometric-adam-cayley" => (learning_rate = T(LEARNING_RATE),
            method = GeometricOptimizers.Adam(T;
                β₁ = T(ADAM_BETA1), β₂ = T(ADAM_BETA2), δ = T(ADAM_EPSILON))),
        "scalar-moment-adam" => (learning_rate = T(SCALAR_MOMENT_LEARNING_RATE),
            method = scalar_moment_adam_method(T;
                beta1 = ADAM_BETA1, beta2 = ADAM_BETA2, epsilon = ADAM_EPSILON)),
        "gradient" => (learning_rate = T(LEARNING_RATE),
            method = GeometricOptimizers.GradientMethod()),
        "momentum" => (learning_rate = T(LEARNING_RATE),
            method = GeometricOptimizers.MomentumMethod(T(MOMENTUM_COEFFICIENT)))
    )
    merge(CONFIGURATIONS[key], methods[key])
end

function write_record(
        configuration, losses, elapsed_seconds, allocated_bytes, gc_seconds, backend)
    isempty(RECORD_PATH) && return nothing
    finite_losses = all(isfinite, losses)
    append_record(RECORD_PATH,
        PENDULUM_RECORD_HEADER,
        Dict{String, Any}(
            "schema_version" => PENDULUM_RUN_SCHEMA_VERSION,
            "dataset" => "pendulum",
            "configuration_key" => CONFIGURATION_KEY,
            "configuration" => configuration.name,
            "optimizer_role" => configuration.role,
            "learning_rate" => configuration.learning_rate,
            "retraction" => configuration.retraction,
            "second_moment" => configuration.second_moment,
            "transport" => configuration.transport,
            "repetition" => REPETITION,
            "seed" => SEED,
            "status" => finite_losses ? "ok" : "failed_validation",
            "epochs_completed" => length(losses),
            "final_loss" => last(losses),
            "best_loss" => minimum(losses),
            "total_seconds" => elapsed_seconds,
            "seconds_per_epoch" => elapsed_seconds / max(length(losses), 1),
            "host_allocated_bytes" => allocated_bytes,
            "gc_seconds" => gc_seconds,
            "backend" => backend,
            "checkpoint" => abspath(OUTPUT),
            "message" => finite_losses ? "ok" : "non-finite reconstruction loss"
        ))
    nothing
end

function write_losses(configuration, losses)
    isempty(LOSSES_PATH) && return nothing
    append_records(LOSSES_PATH, PENDULUM_LOSS_HEADER,
        (Dict{String, Any}(
             "configuration_key" => CONFIGURATION_KEY,
             "configuration" => configuration.name,
             "repetition" => REPETITION,
             "seed" => SEED,
             "epoch" => epoch,
             "loss" => loss
         ) for (epoch, loss) in enumerate(losses)))
    nothing
end

# The initial weights are random; the generated pendulum data are not. Seeding
# before both the network and its optimizer makes a repetition paired across the
# four configurations.
Random.seed!(SEED)

REQUIRE_CUDA && !CUDA.functional() &&
    error("SAE_REQUIRE_CUDA=1 but CUDA.functional() is false")
backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)

solution = pendulum()
data = to_device(Float32.(angular_to_euclidean(solution)))
dl = DataLoader(data; autoencoder = true, suppress_info = true)

# `SymplecticAutoencoder` caps the number of blocks at `full_dim - reduced_dim`,
# so depth comes from the layers inside each block.
architecture = SymplecticAutoencoder(dl.input_dim, REDUCED_DIM;
    n_encoder_blocks = 2,
    n_decoder_blocks = 2,
    n_encoder_layers = 10,
    n_decoder_layers = 20,
    n_decoder_output_layers = 10,
    sympnet_upscale = 20)
network = NeuralNetwork(architecture, backend, eltype(dl))
selected = configuration(eltype(dl))
# `step_size` and not the default. `Optimizer`'s default is a property of the *method* -- 1e-3 for
# `Adam`, 1e-2 for the rest -- so without this the rate every run record quotes was a number the run
# did not use, and the four configurations were not compared at the rates the comparison says they
# were.
optimizer = Optimizer(selected.method, network;
    retraction = cayley, step_size = selected.learning_rate)

CUDA.functional() && CUDA.synchronize()
timed = @timed optimizer(network, dl, Batch(BATCH_SIZE), N_EPOCHS; show_progress = false)
CUDA.functional() && CUDA.synchronize()
losses = timed.value
backend_name = CUDA.functional() ? "cuda" : "cpu"

HDF5.h5open(OUTPUT, "w") do file
    save(file, GeometricMachineLearning.map_to_cpu(network))
    file["loss"] = collect(losses)
    attributes = HDF5.attributes(file)
    attributes["configuration_key"] = CONFIGURATION_KEY
    attributes["configuration"] = selected.name
    attributes["optimizer_role"] = selected.role
    attributes["learning_rate"] = selected.learning_rate
    attributes["retraction"] = selected.retraction
    attributes["second_moment"] = selected.second_moment
    attributes["transport"] = selected.transport
    attributes["seed"] = SEED
    attributes["repetition"] = REPETITION
    attributes["n_epochs"] = N_EPOCHS
    attributes["reduced_dim"] = REDUCED_DIM
    attributes["backend"] = backend_name
    attributes["elapsed_seconds"] = timed.time
    attributes["host_allocated_bytes"] = timed.bytes
    attributes["gc_seconds"] = timed.gctime
end

write_losses(selected, losses)
write_record(selected, losses, timed.time, timed.bytes, timed.gctime, backend_name)

@printf("%s: wrote %s after %d epochs in %.2f s: reconstruction error %g → %g\n",
    selected.name, OUTPUT, N_EPOCHS, timed.time, first(losses), last(losses))
