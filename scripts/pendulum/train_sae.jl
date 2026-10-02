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
# The grid is separatrix-relative rather than Cartesian, because what this experiment is about is
# the crossing, and a rectangle in (θ, pθ) cannot control how close to the separatrix the data
# comes: a fixed pθ crosses it as θ varies. A momentum fraction f does control it — |f| < 1
# librates and |f| > 1 rotates at every angle — so the crossing is covered as densely as one likes.
#
# The fractions below are the ones used for the paper's figure, with two gaps closed.
#
# Rotating: the paper's set jumped from f = -1 straight to f = -2, which leaves H ∈ (1, 1.60)
# empty — every rotating orbit nearest the separatrix, and so every rotating number the figure
# reports, was an extrapolation across that band. Six fractions between 1 and 2 close it.
#
# Librating: f = 3/4 was the closest approach to the separatrix from below, reaching H = 0.913,
# so the band H ∈ (0.913, 1) was empty too. `timespan` is the reason it has to be widened at the
# same time. A librating orbit's period is 4K(√((1+H)/2)), which diverges at the separatrix, and at
# f = 3/4 it is already 11.9 — longer than the (0, 10) the script used to integrate over. The
# outermost librating trajectories never completed one oscillation, so the level sets nearest the
# separatrix were only ever partially traced, whatever the network size. Over (0, 40) every
# trajectory here closes at least twice, apart from f = -1, which is the separatrix itself and has
# no period at all.
#
# Only one direction of rotation is sampled, as before. That is deliberate and not an oversight: a
# symmetric grid puts both directions in the data, and a planar latent space cannot nest both of
# them outside the separatrix image, so one direction has to be folded inside it. See
# JuliaGNI/GMLDatasets.jl#26.
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
using GeometricMachineLearning
using Random

import AbstractNeuralNetworks: save
import GMLDatasets: angular_to_euclidean, pendulum, pendulum_energy
import HDF5
import LinearAlgebra

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
const reduced_dim = 2
const n_epochs   = parse(Int,     get(ENV, "SAE_EPOCHS",  "12000"))
const batch_size = parse(Int,     get(ENV, "SAE_BATCH",   "256"))
const step_size  = parse(Float32, get(ENV, "SAE_ETA",     "1e-4"))
const seed       = parse(Int,     get(ENV, "SAE_SEED",    "123"))
const upscale    = parse(Int,     get(ENV, "SAE_UPSCALE", "20"))
const outdir     = mkpath(get(ENV, "GML_OUTDIR", pwd()))
const output     = joinpath(outdir, get(ENV, "SAE_OUT", "pendulum_sae.h5"))

# Ten angles between the stable and the unstable equilibrium, times fifteen momentum fractions.
# Negative fractions of modulus above one are the rotating trajectories; |f| = 1 is the separatrix.
const angle_range = ([π - 5 / 2], [π - 3 / 20])
const angle_samples = [10]
const one_direction =
    [0, 2 / 5, -2 / 5, 3 / 4, -3 / 4,             # librating, as before
     9 / 10, -9 / 10, 19 / 20, -19 / 20,          # librating, up against the separatrix
     -1,                                          # the separatrix itself
     -1.02, -1.05, -1.1, -1.2, -1.4, -1.6,        # the band the paper's grid left empty
     -2, -5 / 2, -3]                              # the rotating trajectories it did have
const momentum_fractions = get(ENV, "SAE_FRACS", "one") == "both" ?
    vcat(one_direction, [1, 1.02, 1.05, 1.1, 1.2, 1.4, 1.6, 2, 5 / 2, 3]) :
    one_direction

# Long enough for the slowest orbit in the grid — f = 19/20, period 14.8 — to close more than twice.
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
println("$(length(solution)) trajectories, H ∈ [",
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
