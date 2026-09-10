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

include("scalar_moment_adam.jl")

const REDUCED_DIM = parse(Int, get(ENV, "SAE_REDUCED_DIM", "2"))
const N_EPOCHS = parse(Int, get(ENV, "SAE_N_EPOCHS", "12000"))
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

const CONFIGURATION_ORDER = (
    "geometric-adam-cayley",
    "scalar-moment-adam",
    "gradient",
    "momentum",
)

function configuration(T::Type{<:AbstractFloat})
    proposed = (
        name = "Geometric Adam (Stiefel, Cayley retraction)",
        role = "proposed",
        learning_rate = T(LEARNING_RATE),
        retraction = "cayley",
        second_moment = "coordinate-wise",
        transport = "global-section",
        method = GeometricOptimizers.Adam(T;
            β₁=T(ADAM_BETA1), β₂=T(ADAM_BETA2), δ=T(ADAM_EPSILON)),
    )
    scalar = (
        name = "Scalar Moment Adam (Stiefel, Cayley retraction)",
        role = "riemannian-adam-baseline",
        learning_rate = T(SCALAR_MOMENT_LEARNING_RATE),
        retraction = "cayley",
        second_moment = "scalar (quotient norm)",
        transport = "global-section",
        method = SAEScalarMomentAdam(T;
            beta1=T(ADAM_BETA1), beta2=T(ADAM_BETA2), epsilon=T(ADAM_EPSILON)),
    )
    gradient = (
        name = "Riemannian gradient (Stiefel, Cayley retraction)",
        role = "diagnostic",
        learning_rate = T(LEARNING_RATE),
        retraction = "cayley",
        second_moment = "none",
        transport = "none",
        method = GeometricOptimizers.GradientMethod(),
    )
    momentum = (
        name = "Riemannian momentum (Stiefel, Cayley retraction)",
        role = "diagnostic",
        learning_rate = T(LEARNING_RATE),
        retraction = "cayley",
        second_moment = "none",
        transport = "global-section",
        method = GeometricOptimizers.MomentumMethod(T(MOMENTUM_COEFFICIENT)),
    )
    configurations = Dict(
        "geometric-adam-cayley" => proposed,
        "scalar-moment-adam" => scalar,
        "gradient" => gradient,
        "momentum" => momentum,
    )
    haskey(configurations, CONFIGURATION_KEY) || error(
        "unknown SAE_CONFIGURATION=$CONFIGURATION_KEY; choose $(join(CONFIGURATION_ORDER, ", "))")
    configurations[CONFIGURATION_KEY]
end

csv_field(value) = "\"" * replace(string(value), '"' => "\"\"") * "\""

function write_record(configuration, losses, elapsed_seconds, allocated_bytes, gc_seconds, backend)
    isempty(RECORD_PATH) && return nothing
    mkpath(dirname(RECORD_PATH))
    new_file = !isfile(RECORD_PATH)
    finite_losses = all(isfinite, losses)
    status = finite_losses ? "ok" : "failed_validation"
    message = finite_losses ? "ok" : "non-finite reconstruction loss"
    open(RECORD_PATH, "a") do io
        new_file && println(io, join((
            "schema_version", "dataset", "configuration_key", "configuration", "optimizer_role",
            "learning_rate", "retraction", "second_moment", "transport", "repetition", "seed",
            "status", "epochs_completed", "final_loss", "best_loss", "total_seconds",
            "seconds_per_epoch", "host_allocated_bytes", "gc_seconds", "backend", "checkpoint",
            "message",
        ), ','))
        println(io, join((
            2, "pendulum", CONFIGURATION_KEY, csv_field(configuration.name),
            configuration.role, configuration.learning_rate, configuration.retraction,
            configuration.second_moment, configuration.transport, REPETITION, SEED, status,
            length(losses), last(losses), minimum(losses), elapsed_seconds,
            elapsed_seconds / max(length(losses), 1), allocated_bytes, gc_seconds,
            backend, abspath(OUTPUT), csv_field(message),
        ), ','))
    end
    nothing
end

function write_losses(configuration, losses)
    isempty(LOSSES_PATH) && return nothing
    mkpath(dirname(LOSSES_PATH))
    new_file = !isfile(LOSSES_PATH)
    open(LOSSES_PATH, "a") do io
        new_file && println(io, "configuration_key,configuration,repetition,seed,epoch,loss")
        for (epoch, loss) in enumerate(losses)
            println(io, join((CONFIGURATION_KEY, csv_field(configuration.name), REPETITION, SEED,
                epoch, loss), ','))
        end
    end
    nothing
end

# The initial weights are random; the generated pendulum data are not. Seeding
# before both the network and its optimizer makes a repetition paired across the
# four configurations.
Random.seed!(SEED)

REQUIRE_CUDA && !CUDA.functional() && error("SAE_REQUIRE_CUDA=1 but CUDA.functional() is false")
backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)

solution = pendulum()
data = to_device(Float32.(angular_to_euclidean(solution)))
dl = DataLoader(data; autoencoder=true, suppress_info=true)

# `SymplecticAutoencoder` caps the number of blocks at `full_dim - reduced_dim`,
# so depth comes from the layers inside each block.
architecture = SymplecticAutoencoder(dl.input_dim, REDUCED_DIM;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=10,
    n_decoder_layers=20,
    n_decoder_output_layers=10,
    sympnet_upscale=20)
network = NeuralNetwork(architecture, backend, eltype(dl))
selected = configuration(eltype(dl))
optimizer = Optimizer(selected.method, network; retraction=cayley)

CUDA.functional() && CUDA.synchronize()
timed = @timed optimizer(network, dl, Batch(BATCH_SIZE), N_EPOCHS; show_progress=false)
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
