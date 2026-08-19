using GMLDatasets
using GeometricMachineLearning
using HDF5
using Random
import AbstractNeuralNetworks: save

const seed = parse(Int, get(ENV, "PENDULUM_SEED", "123"))
const epochs = parse(Int, get(ENV, "PENDULUM_N_EPOCHS", "10"))
const output = get(ENV, "PENDULUM_SAE_OUTPUT", "pendulum_sae.h5")
epochs > 0 || error("PENDULUM_N_EPOCHS must be positive")
isfile(output) && error("refusing to overwrite existing output: $output")

Random.seed!(seed)
initial_conditions = [
    (angle, fraction * sqrt(2 * (1 - cos(angle))))
    for angle in range(0.6, 2.8; length=8)
    for fraction in (0.0, 0.5, -0.5)
]
dataset = pendulum_dataset(
    initial_conditions;
    timespan=(0.0, 10.0),
    timestep=0.1,
)
loader = pendulum_data_loader(dataset; suppress_info=true)

architecture = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=2,
    n_decoder_layers=2,
    n_decoder_output_layers=2,
    sympnet_upscale=2,
)
network = NeuralNetwork(architecture, CPU(), Float64)
optimizer = Optimizer(Adam(), network; step_size=1f-3)
loss = optimizer(network, loader, Batch(128), epochs)
save(output, network)
println("Saved SAE to $output (seed=$seed, epochs=$epochs, final_loss=$(loss[end]))")
