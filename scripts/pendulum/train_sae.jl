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
# The grid is separatrix-relative rather than Cartesian: ten initial angles times momentum fractions
# f of the separatrix momentum. |f| < 1 librates and |f| > 1 rotates at every angle, so a fraction
# controls how close a trajectory comes to the separatrix, which a fixed pθ cannot.
#
# Two grids, chosen by SAE_GRID:
#
#   paper       (default) the fractions {0, ±2/5, ±3/4, -1, -2, -3} of the paper's figure: the
#               outermost librating level is H_L = 0.913 and the innermost rotating one H_R = 1.597.
#   separatrix  JuliaGNI/GMLDatasets.jl#28's grid, which adds ±9/10, ±19/20 and -1.02 ... -1.6, -5/2
#               and so puts training levels at H_L = 0.981 and H_R = 1.008.
#
# The second grid is not an improvement of the first, and that is why it is not the default. By the
# complete-orbit error threshold, an encoder
# that embeds the librating region, the separatrix and one rotating family misses the action on the
# orbit at H_L or the one at H_R by at least (J_L - J_R)/(J_L + J_R): 15% on the paper's grid, 32% on
# the separatrix grid, at any capacity and any training length. A run on the separatrix grid
# (12000 epochs, JuliaGNI/GMLDatasets.jl#31) fitted the near-separatrix rotating orbits to 3% and
# stopped being an embedding on the librating side; its weights fail every librating check of
# branch_report.jl. The paper's grid leaves a 15% floor that the current weights, at 55%, are far
# from, so that is the grid on which better training can still show.
#
# Both grids integrate over (0, 40). A librating orbit's period is 4K(√((1+H)/2)), which diverges at
# the separatrix: 11.9 at H = 0.913, longer than the (0, 10) the paper's run used, so its outermost
# librating trajectories never closed. Over (0, 40) every trajectory closes at least twice, apart from
# f = -1, which is the separatrix itself and has no period at all. That changes the arcs sampled, not
# the energies, so it leaves the threshold alone.
#
# Only one direction of rotation is sampled. That is deliberate: a symmetric grid puts both
# directions in the data, and a planar latent space cannot nest both of them outside the separatrix
# image, so one direction has to be folded inside it. See JuliaGNI/GMLDatasets.jl#26.
#
# The standard coordinates (θ, pθ) cannot be represented globally by two ordinary real-valued
# coordinates because θ is periodic, but that does not rule out a different two-dimensional
# representation. A bounded cylinder can, for example, be represented as an annulus in the latent
# plane. The experiment asks the SAE to learn such a representation rather than reproduce the
# standard angular coordinate.
#
# The deeper network and long run are intended for a GPU. CUDA is used when available and the script
# falls back to the host so that the setup remains inspectable on any machine. The output HDF5 file
# holds the weights and loss curve, allowing plots and further analysis without retraining.

using CUDA
include(joinpath(@__DIR__, "cuda_compat.jl"))
using GeometricMachineLearning
using Random

import AbstractNeuralNetworks: save
import GMLDatasets: angular_to_euclidean, pendulum, pendulum_energy
import HDF5
import LinearAlgebra
using Printf

# CUDA 5.11 and GPUArrays 11.5 both define a triangular solve for device matrices, and for a solve
# with a `CuMatrix` on both sides neither method is more specific: a `MethodError` for an ambiguous
# call. `GlobalSection` of a `StiefelManifold` weight is where it is met, orthonormalizing with
# `A / UpperTriangular(R)` when the optimizer is built, so the run fails before its first epoch.
# Metal has no CUBLAS and the host no GPUArrays, which is why no run but a CUDA one ever saw it.
# These two methods are more specific than both and forward to CUBLAS's `trsm!`, which is what
# either of them would have done. Delete them once CUDA.jl resolves the ambiguity itself.
LinearAlgebra.generic_mattridiv!(C::CuMatrix{T}, uploc, isunitc, tfun::Function, A::CuMatrix{T},
        B::CuMatrix{T}) where {T <: CUDA.CUBLAS.CublasFloat} =
    invoke(LinearAlgebra.generic_mattridiv!,
        Tuple{CUDA.StridedCuMatrix{T}, Any, Any, Function, AbstractMatrix{T}, CUDA.StridedCuMatrix{T}},
        C, uploc, isunitc, tfun, A, B)
LinearAlgebra.generic_trimatdiv!(C::CuMatrix{T}, uploc, isunitc, tfun::Function, A::CuMatrix{T},
        B::CuMatrix{T}) where {T <: CUDA.CUBLAS.CublasFloat} =
    invoke(LinearAlgebra.generic_trimatdiv!,
        Tuple{CUDA.StridedCuMatrix{T}, Any, Any, Function, CUDA.StridedCuMatrix{T}, AbstractMatrix{T}},
        C, uploc, isunitc, tfun, A, B)

# The defaults are the configuration this experiment is specified at; the environment overrides
# exist so that a sweep on a remote machine needs no file edits there. `SAE_FRACS=both` adds the
# positive rotating fractions, which is the one knob whose result must not be read as an ordinary
# accuracy number -- see the note on rotation directions above, and run `branch_report.jl`.
#
# SAE_EPOCHS is an upper limit; the run stops earlier once the loss has flattened (see the training
# loop). In the 12000-epoch run on the separatrix grid, at batch 256 and step 1e-4, the loss averaged
# over 100 epochs fell from 0.160 at epoch 3000 to 0.135 near epoch 7350 and rose to 0.155 by the
# last, which is the epoch whose weights that run kept. Batch 2048 at step 1e-3 reaches a lower loss per epoch and
# runs about three times faster per epoch on the CPU of an M4 Max. A test run on the paper's grid with
# -5/2 in place of -1 passed the embedding checks after 1100 epochs.
const reduced_dim = 2
const n_epochs   = parse(Int,     get(ENV, "SAE_EPOCHS",  "3000"))
const batch_size = parse(Int,     get(ENV, "SAE_BATCH",   "2048"))
const step_size  = parse(Float32, get(ENV, "SAE_ETA",     "1e-3"))
const seed       = parse(Int,     get(ENV, "SAE_SEED",    "123"))
const upscale    = parse(Int,     get(ENV, "SAE_UPSCALE", "20"))
const outdir     = mkpath(get(ENV, "GML_OUTDIR", pwd()))
const output     = joinpath(outdir, get(ENV, "SAE_OUT", "pendulum_sae.h5"))
const check_every     = parse(Int,     get(ENV, "SAE_CHECK_EVERY", "100"))
const check_nsamp     = parse(Int,     get(ENV, "SAE_NSAMP",       "1600"))
const patience        = parse(Int,     get(ENV, "SAE_PATIENCE",    "500"))
const min_improvement = parse(Float64, get(ENV, "SAE_MIN_GAIN",    "0.01"))

# Ten angles between the stable and the unstable equilibrium, times the momentum fractions of the
# chosen grid. Negative fractions of modulus above one are the rotating trajectories; |f| = 1 is the
# separatrix.
const angle_range = ([π - 5 / 2], [π - 3 / 20])
const angle_samples = [10]
const grid = get(ENV, "SAE_GRID", "paper")
const one_direction = grid == "paper" ?
    [0, 2 / 5, -2 / 5, 3 / 4, -3 / 4, -1, -2, -3] :
    grid == "separatrix" ?
    [0, 2 / 5, -2 / 5, 3 / 4, -3 / 4,             # the paper's librating fractions
     9 / 10, -9 / 10, 19 / 20, -19 / 20,          # librating, up against the separatrix
     -1,                                          # the separatrix itself
     -1.02, -1.05, -1.1, -1.2, -1.4, -1.6,        # rotating, up against the separatrix
     -2, -5 / 2, -3] :
    error("SAE_GRID must be `paper` or `separatrix`, not `$grid`")
const momentum_fractions = get(ENV, "SAE_FRACS", "one") == "both" ?
    vcat(one_direction, -one_direction[one_direction .< -1]) :
    one_direction

# Long enough for every librating orbit of either grid to close at least twice (see the header).
const timespan = (0.0, parse(Float64, get(ENV, "SAE_TSPAN", "40")))
const timestep = 0.1

# The initial weights are random; the data are not, so this is the only thing that needs seeding.
Random.seed!(seed)

backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)
# Said here rather than at the end. A silent fallback to the host is a wasted day at this grid size,
# and the point of saying it is to be able to kill the run in the first second instead of the last.
println("Backend: ", CUDA.functional() ? "CUDA" : "CPU")

solution = pendulum(; qmin = angle_range[1], qmax = angle_range[2], qsamples = angle_samples,
    momentum_fractions = momentum_fractions, timespan = timespan, timestep = timestep)

energies = pendulum_energy(solution)[1, :]
println("grid $grid: $(length(solution)) trajectories, H ∈ [",
    round(minimum(energies); digits = 3), ", ", round(maximum(energies); digits = 3), "]")
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
    sympnet_upscale = upscale)
# Initialized on the host and then moved, rather than initialized on the device. Initialization is
# the one place a host RNG meets device arrays, and `PSDLayer`'s orthonormal starting weight is a
# Cholesky factorization of a device matrix there — code no CPU run ever exercises. The CUDA run
# failed inside this initialization. On the host it is ordinary LAPACK, and the starting weights for
# a given seed are the same whichever backend trains them. `mapstorage` rebuilds each leaf around
# the moved storage, so the `StiefelManifold` weight stays one: `map_to_cpu` in the other direction.
host_network = NeuralNetwork(architecture, CPU(), eltype(dl))
network = NeuralNetwork(architecture, host_network.model,
    GeometricMachineLearning.mapstorage(to_device, host_network.params), backend)

# Training runs in chunks of `check_every` epochs, and after each chunk the encoder is put through
# the checks of `branch_report.jl` (structure_checks.jl): γ± of opposite sign, and the librating and
# p_θ < 0 families sign-constant, nested and simple. The weights written are the checkpoint with the
# lowest loss among those that pass, not the last epoch's.
#
# This is a selection, and it has to be read as one: the checks pick the checkpoints that are
# charts, and the loss picks among those. Every checkpoint also prints the action errors on the two
# training orbits nearest the separatrix, against the threshold (J_L - J_R)/(J_L + J_R) that an
# embedding cannot get both below (see the header). A run in which no checkpoint passes writes its
# last weights anyway, for the report, and exits with an error so that the pipeline stops before
# reduced-network training.
#
# A run stops early once a checkpoint has passed and the smoothed loss has not improved by
# `min_improvement` for `patience` epochs.
optimizer = Optimizer(Adam(), network; step_size = step_size)
include(joinpath(@__DIR__, "structure_checks.jl"))

host_copy() = GeometricMachineLearning.map_to_cpu(network)
function write_weights(path, nn, losses, history, selected)
    HDF5.h5open(path, "w") do file
        save(file, nn)
        file["loss"] = losses
        file["check_epoch"] = [h.epoch for h in history]
        file["check_passed"] = [h.ok for h in history]
        file["check_loss"] = [h.loss for h in history]
        file["check_area_gamma_minus"] = [h.Am for h in history]
        file["check_area_gamma_plus"] = [h.Ap for h in history]
        file["check_action_error_H_L"] = [h.eL for h in history]
        file["check_action_error_H_R"] = [h.eR for h in history]
        attrs = HDF5.attributes(file)
        attrs["seed"] = seed
        attrs["n_epochs"] = length(losses)
        attrs["selected_epoch"] = selected
        attrs["checks_passed"] = selected > 0 ? 1 : 0
        attrs["reduced_dim"] = reduced_dim
        attrs["sympnet_upscale"] = upscale
        attrs["batch_size"] = batch_size
        attrs["step_size"] = step_size
        attrs["grid"] = grid
        attrs["momentum_fractions"] = collect(Float64, momentum_fractions)
        attrs["timespan"] = timespan[2]
        attrs["H_L"] = H_L
        attrs["H_R"] = H_R
        attrs["backend"] = string(typeof(backend))
    end
end

# The two training levels nearest the separatrix, which set the threshold.
const H_L = maximum(energies[energies .< 1 - 1e-9])
const H_R = minimum(energies[energies .> 1 + 1e-9])
@printf("H_L = %.4f, H_R = %.4f\n", H_L, H_R)

losses = Float64[]
history = []
best = (loss = Inf, epoch = 0, nn = nothing)
smoothed(l) = sum(l[max(1, end - check_every + 1):end]) / min(check_every, length(l))
best_smoothed, last_gain = Inf, 0
for epoch in check_every:check_every:n_epochs
    append!(losses, optimizer(network, dl, Batch(batch_size), check_every; show_progress = false))
    m = smoothed(losses)
    cpu = host_copy()
    enc = encoder(cpu)
    r = embedding_checks(enc; nsamp = check_nsamp)
    t = threshold_errors(enc, H_L, H_R; nsamp = check_nsamp)
    push!(history, (; epoch, r.ok, loss = m, r.Am, r.Ap, t.eL, t.eR))
    @printf("epoch %6d  loss %.5f  %s\n               %s\n", epoch, m, checks_line(r), threshold_line(t))
    flush(stdout)
    if r.ok && m < best.loss
        global best = (loss = m, epoch = epoch, nn = cpu)
    end
    if m < best_smoothed * (1 - min_improvement)
        global best_smoothed, last_gain = m, epoch
    end
    if best.epoch > 0 && epoch - last_gain >= patience
        println("stopping at epoch $epoch: no loss improvement of $(100min_improvement)% in $patience epochs")
        break
    end
end

if best.epoch > 0
    write_weights(output, best.nn, losses, history, best.epoch)
    println("wrote $output: the checkpoint at epoch $(best.epoch) of $(length(losses)), loss $(best.loss), ",
        "passes the embedding checks")
else
    write_weights(output, host_copy(), losses, history, 0)
    println("ERROR: no checkpoint in $(length(losses)) epochs passed the embedding checks; ",
        "wrote the last weights to $output for branch_report.jl. Try another SAE_SEED.")
    exit(1)
end
