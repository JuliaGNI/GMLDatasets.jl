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
# The training set is *not* `pendulum()`'s default grid. That grid covers the whole phase space,
# angles from 0 to 2π and trajectories that go over the top, and the resulting data set is a
# cylinder: a reduced coordinate that covers it has to be an angle, and no continuous map from ℝ² can
# be one. The grid below is restricted to trajectories that swing back and forth about θ = π without
# reaching the separatrix at H = mgl, which is a topological disc and can be charted. On the full
# grid this same run plateaus around three times the reconstruction error.
# `scripts/pendulum/plot_dataset.jl` draws the whole phase space, separatrix included.
#
# The run is on the host — four rows by ten thousand columns is small enough that a GPU would spend
# more time being fed than computing — and takes a couple of minutes. It is a demonstration, not a
# converged result: the reconstruction error settles around 0.1 and pushing it lower is a matter of
# many more epochs, which is what `GeometricMachineLearning`'s own Toda-lattice tutorial does on a
# GPU. The output file holds the weights and the loss curve, so the run can be inspected without
# being repeated.

using GeometricMachineLearning
using Random

import AbstractNeuralNetworks: save
import GMLDatasets: angular_to_euclidean, pendulum
import HDF5

const reduced_dim = 2
const n_epochs = 2048
const batch_size = 256
const step_size = 1f-2
const seed = 123
const output = "pendulum_sae.h5"

# The initial weights are random; the data are not, so this is the only thing that needs seeding.
Random.seed!(seed)

solution = pendulum(; qmin = [π / 2], qmax = [3π / 2], pmin = [-1.0], pmax = [1.0],
                      qsamples = [10], psamples = [10])
dl = DataLoader(angular_to_euclidean(solution); autoencoder = true, suppress_info = true)

# `SymplecticAutoencoder` caps the number of blocks at `full_dim - reduced_dim`, which is 2 here, so
# the depth has to come from the layers inside each block rather than from more blocks.
architecture = SymplecticAutoencoder(dl.input_dim, reduced_dim;
                                     n_encoder_blocks = 2,
                                     n_decoder_blocks = 2,
                                     n_encoder_layers = 4,
                                     n_decoder_layers = 4,
                                     n_decoder_output_layers = 2)
network = NeuralNetwork(architecture, CPU(), eltype(dl))

optimizer = Optimizer(Adam(), network; step_size = step_size)
losses = optimizer(network, dl, Batch(batch_size), n_epochs)

HDF5.h5open(output, "w") do file
    save(file, network)
    file["loss"] = collect(losses)
    HDF5.attributes(file)["seed"] = seed
    HDF5.attributes(file)["n_epochs"] = n_epochs
    HDF5.attributes(file)["reduced_dim"] = reduced_dim
end

println("wrote $output after $n_epochs epochs: ",
        "reconstruction error $(first(losses)) → $(last(losses))")
