# Train a symplectic autoencoder on the four-dimensional pendulum data set.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/train_sae.jl
#
# `pendulum` integrates a grid of initial conditions and `angular_to_euclidean` lifts them into ℝ⁴,
# where the bob traces the tangent bundle of a circle — a two-dimensional submanifold sitting in four
# dimensions. Recovering that submanifold is what a `SymplecticAutoencoder` is for, so the reduced
# dimension is 2.
#
# The default grid is deliberately retained: it contains librating and rotating trajectories and
# crosses the separatrix at H = mgl. The standard coordinates (θ, pθ) cannot be represented globally
# by two ordinary real-valued coordinates because θ is periodic, but that does not rule out a
# different two-dimensional representation. A bounded cylinder can, for example, be represented as
# an annulus in the latent plane. The experiment asks the SAE to learn such a representation rather
# than reproduce the standard angular coordinate.
#
# The deeper network and long run are intended for a GPU. CUDA is used when available and the script
# falls back to the host so that the setup remains inspectable on any machine. The output HDF5 file
# holds the weights and loss curve, allowing plots and further analysis without retraining.

using CUDA
using GeometricMachineLearning
using Random

import AbstractNeuralNetworks: save
import GMLDatasets: angular_to_euclidean, pendulum
import HDF5

const reduced_dim = 2
const n_epochs = 12000
const batch_size = 256
const step_size = 1.0f-4
const seed = 123
const output = "pendulum_sae.h5"

# The initial weights are random; the data are not, so this is the only thing that needs seeding.
Random.seed!(seed)

backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)

solution = pendulum()
data = to_device(Float32.(angular_to_euclidean(solution)))
dl = DataLoader(data; autoencoder = true, suppress_info = true)

# `SymplecticAutoencoder` caps the number of blocks at `full_dim - reduced_dim`, which is 2 here, so
# the depth has to come from the layers inside each block rather than from more blocks.
architecture = SymplecticAutoencoder(dl.input_dim, reduced_dim;
    n_encoder_blocks = 2,
    n_decoder_blocks = 2,
    n_encoder_layers = 10,
    n_decoder_layers = 20,
    n_decoder_output_layers = 10,
    sympnet_upscale = 20)
network = NeuralNetwork(architecture, backend, eltype(dl))

optimizer = Optimizer(Adam(), network; step_size = step_size)
losses = optimizer(network, dl, Batch(batch_size), n_epochs)

HDF5.h5open(output, "w") do file
    save(file, GeometricMachineLearning.map_to_cpu(network))
    file["loss"] = collect(losses)
    HDF5.attributes(file)["seed"] = seed
    HDF5.attributes(file)["n_epochs"] = n_epochs
    HDF5.attributes(file)["reduced_dim"] = reduced_dim
    HDF5.attributes(file)["backend"] = string(typeof(backend))
end

println("wrote $output after $n_epochs epochs: ",
    "reconstruction error $(first(losses)) → $(last(losses))")
