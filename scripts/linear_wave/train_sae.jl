# Train the symplectic autoencoders of the linear-wave experiment: one network per reduced dimension
# 2n, n ∈ LW_HALF_REDUCED_DIMS, and per seed in LW_SEEDS, on the snapshots of LW_N_PARAMS wave speeds
# (see `common.jl` for every setting and its default).
#
#     julia --project=scripts scripts/linear_wave/train_sae.jl
#
# With LW_WARM_START=1 (the default) every network starts at the PSD of its dimension
# (`warm_start!`), so its reconstruction error on the training data starts at the PSD's.
#
# Each network is written to `$(LW_OUTPUT_DIR)/weights/sae_nXX_seedS_eEEEE[_warm].h5` after
# LW_N_EPOCHS epochs, and after every epoch count in LW_CHECKPOINTS, together with its loss history,
# the settings and the package versions. A network whose final file exists is skipped, so an
# interrupted run resumes where it stopped and several processes can share the work by splitting
# LW_HALF_REDUCED_DIMS or LW_SEEDS between them. The seed is set before each network is initialized,
# so a file depends on (n, seed) alone and not on what was trained before it in the same process.
#
# The PSD baseline needs no training; `errors.jl` fits it.

using CUDA

include(joinpath(@__DIR__, "common.jl"))

backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)

data = training_data()
@printf("training data: %d × %d × %d (2N × time steps × wave speeds)\n", size(data)...)
dl = DataLoader(to_device(data); autoencoder = true, suppress_info = true)

mkpath(dirname(sae_path(first(HALF_REDUCED_DIMS), first(SEEDS))))

function save_network(path, network, losses, n, seed, elapsed)
    # Written to a temporary name and moved, so that an interrupted write never leaves a file that
    # the next run would take for a finished network.
    HDF5.h5open(path * ".partial", "w") do file
        save(file, GeometricMachineLearning.map_to_cpu(network))
        file["loss"] = collect(losses)
        write_metadata(file)
        attributes = HDF5.attributes(file)
        attributes["half_reduced_dim"] = n
        attributes["seed"] = seed
        attributes["epochs"] = length(losses)
        attributes["backend"] = CUDA.functional() ? "cuda" : "cpu"
        attributes["elapsed_seconds"] = elapsed
    end
    mv(path * ".partial", path; force = true)
end

for n in HALF_REDUCED_DIMS, seed in SEEDS

    if isfile(sae_path(n, seed))
        println("$(sae_path(n, seed)) exists, skipped")
        continue
    end

    Random.seed!(seed)
    network = NeuralNetwork(architecture(n), backend, Float64)
    WARM_START && warm_start!(network, fit_psd(data, n))
    optimizer = TrainingOptimizer(network.params;
        algorithm = Adam(; β₁ = ADAM_BETA1, β₂ = ADAM_BETA2, δ = ADAM_DELTA),
        linesearch = STEP_SIZE, retraction = Cayley())

    # `train!` resumes with the same optimizer, so training in segments between the checkpoints is
    # the same as training once for N_EPOCHS.
    losses = Float64[]
    elapsed = 0.0
    for epochs in CHECKPOINTS
        timed = @timed train!(network, optimizer, dl, Batch(BATCH_SIZE),
            epochs - length(losses); show_progress = false)
        append!(losses, timed.value)
        elapsed += timed.time
        save_network(sae_path(n, seed, epochs), network, losses, n, seed, elapsed)
        @printf("n = %2d, seed %d, %4d epochs: loss %.3e → %.3e in %.1f s\n",
            n, seed, epochs, first(losses), last(losses), elapsed)
        flush(stdout)
    end
end
