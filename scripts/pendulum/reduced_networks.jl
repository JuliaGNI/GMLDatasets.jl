# Train SympNet and ResNet on the 2D reduced dynamics, and plot the comparison
#
# Step 2 of the pipeline: load the trained SAE, encode the trajectories, learn the reduced dynamics
# in the latent plane with a SympNet and with a ResNet, and produce the solution and energy-error
# plots that compare them.
#
# Needs the weights from `train_sae.jl`.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/reduced_networks.jl
#
# Produces:
#   plots/pendulum_sae_loss_comparison.png
#   plots/pendulum_sae_{ref,sympnet,resnet}_<label>.png
#   plots/pendulum_sae_energy_error_<label>.png
#   sympnet_weights.h5, resnet_weights.h5
#
# Moved here from the symplectic-autoencoder talk's working directory
# (its `pendulum_4d_train_reduced_and_plot.jl`), where it sat beside the figure it makes.
# A figure is a claim and a generator nobody's CI runs rots against the API it was written
# for -- which this one had: see the note at the end of this header.
#
# This trains on the librating-only grid (fractions {0, +-2/5, +-3/4}), unlike
# `train_sae.jl`, which includes the rotating trajectories. That is a real difference
# between the two steps of the pipeline and not an oversight to tidy away silently -- but it does
# mean the reduced-dynamics comparison says nothing about the separatrix.

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(parse(Int, get(ENV, "SAE_SEED", "123")))

# CUDA where there is a working GPU, the host where there is not, so the same file runs on the
# workstation and here. The `using` and the `functional()` call are separate top-level statements on
# purpose: calling into a module in the statement that loads it is a world-age error.
const want_gpu = get(ENV, "SAE_BACKEND", "auto") != "cpu"
if want_gpu
    try
        @eval using CUDA
    catch err
        @info "CUDA.jl is not available in this environment; training on the CPU" err
    end
end
const gpu_ok  = want_gpu && isdefined(@__MODULE__, :CUDA) && CUDA.functional()
gpu_ok && include(joinpath(@__DIR__, "cuda_compat.jl"))
const backend = gpu_ok ? CUDA.CUDABackend() : CPU()
to_device(A)  = gpu_ok ? CUDA.CuArray(A) : A
println("Backend: ", gpu_ok ? "CUDA" : "CPU")

const timestep      = 0.1
const tspan         = (0.0, 50.0)
const train_tspan   = (0.0, 100.0)
# Outputs go to GML_OUTDIR, or to the working directory. Never beside the script: a figure is a
# build product and the generator is the artifact, which is the whole reason these live here now.
const outdir  = get(ENV, "GML_OUTDIR", pwd())
const plotdir = mkpath(joinpath(outdir, "plots"))
const sae_path      = get(ENV, "SAE_WEIGHTS", joinpath(outdir, "pendulum_sae.h5"))
const sympnet_path  = joinpath(outdir, "sympnet_weights.h5")
const resnet_path   = joinpath(outdir, "resnet_weights.h5")

const params = (l=1.0, m=1.0, g=1.0)

# ---- Helpers ----------------------------------------------------------------

function angular_to_euclidean(θ::AbstractVector, p_ang::AbstractVector)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

function euclidean_to_angular(q_e::AbstractMatrix, p_e::AbstractMatrix)
    x, y   = q_e[1, :], q_e[2, :]
    px, py = p_e[1, :], p_e[2, :]
    θ  = atan.(x, y)
    θ̇  = (y .* px .- x .* py) ./ (x .^ 2 .+ y .^ 2)
    θ, θ̇
end

hamiltonian_euclidean(q::AbstractMatrix, p::AbstractMatrix) =
    0.5f0 .* (p[1, :] .^ 2 .+ p[2, :] .^ 2) .+ q[2, :]

compute_energy_error(h) = (h .- h[1]) ./ abs.(h[1])

# ---- Training trajectories (needed for encoding) ----------------------------

const train_angles = range(π - 2.5, π - 0.15; length=10)
const mom_fracs    = [0.0, 0.40, 0.75, -0.40, -0.75]

train_ics = [(θ₀, frac) for θ₀ in train_angles for frac in mom_fracs]
println("Regenerating $(length(train_ics)) training trajectories for encoding ...")

train_ref = map(train_ics) do (θ₀, frac)
    p_sep  = sqrt(max(2.0 * (1.0 - cos(θ₀)), 0.0))
    p₀_ang = p_sep * frac
    sol = integrate(
        hodeproblem([θ₀], [p₀_ang]; parameters=params,
            timespan=train_tspan, timestep=timestep),
        Gauss(2))
    ns_   = length(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang  = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, p₀=p₀_ang, nsteps=ns_,
        t=collect(sol.t), θ=θ_vec, θ̇=pang ./ params.l^2,
        q_eucl=q_e, p_eucl=p_e)
end

# ---- Evaluation trajectories ------------------------------------------------

const eval_angles = range(acos(0.9), acos(-0.9); length=6)

eval_ref = map(eval_angles) do θ₀
    sol = integrate(
        hodeproblem([θ₀], [0.0]; parameters=params,
            timespan=tspan, timestep=timestep),
        Gauss(2))
    ns_   = length(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang  = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, nsteps=ns_,
        t=collect(sol.t), θ=θ_vec, θ̇=pang ./ params.l^2,
        q_eucl=q_e, p_eucl=p_e)
end

println("Evaluation: $(length(eval_ref)) trajectories (p₀=0, tspan=$(tspan[2])).")

# ---- Load SAE from disk (CPU) -----------------------------------------------

const sae_arch = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=10,
    n_decoder_layers=20,
    n_decoder_output_layers=10,
    sympnet_upscale=20)

println("Loading SAE weights from $sae_path ...")
sae_nn = load(NeuralNetwork, sae_path, sae_arch)
enc = encoder(sae_nn)
dec = decoder(sae_nn)
println("SAE loaded ($(parameterlength(sae_nn)) params, Float32).")

# ---- Encode training trajectories into 2D reduced coordinates (CPU) ---------

const ns_train = train_ref[1].nsteps
q_red = Array{Float32}(undef, 1, ns_train, length(train_ref))
p_red = Array{Float32}(undef, 1, ns_train, length(train_ref))

for (i, d) in enumerate(train_ref)
    z = enc(Float32.(vcat(d.q_eucl, d.p_eucl)))
    q_red[:, :, i] = z[1:1, :]
    p_red[:, :, i] = z[2:2, :]
end

# Encoding runs on the CPU; training data must share the reduced networks' backend.
dl_reduced = DataLoader((q=to_device(q_red), p=to_device(p_red)))
println("Reduced DataLoader: input_dim=$(dl_reduced.input_dim) [$(gpu_ok ? "CUDA" : "CPU"), Float32]")

# ---- Train SympNet and ResNet on the selected backend -----------------------

const n_epochs = 11000
const batch    = Batch(128)

println("Training SympNet on $(gpu_ok ? "CUDA" : "CPU") ...")
const sympnet_arch = GSympNet(2; n_layers=10, upscaling_dimension=32)
nn1  = NeuralNetwork(sympnet_arch, backend, Float32)
o1   = Optimizer(AdamOptimizer(), nn1)
loss1 = o1(nn1, dl_reduced, batch, n_epochs)
println("SympNet: $(parameterlength(nn1)) params, final loss=$(round(loss1[end]; sigdigits=4))")

println("Training ResNet on $(gpu_ok ? "CUDA" : "CPU") ...")
const resnet_arch = ResNet(2, 79, tanh)
nn2  = NeuralNetwork(resnet_arch, backend, Float32)
o2   = Optimizer(AdamOptimizer(), nn2)
loss2 = o2(nn2, dl_reduced, batch, n_epochs)
println("ResNet: $(parameterlength(nn2)) params, final loss=$(round(loss2[end]; sigdigits=4))")

# ---- Save reduced network weights -------------------------------------------

GeometricMachineLearning.save(sympnet_path, GeometricMachineLearning.map_to_cpu(nn1))
GeometricMachineLearning.save(resnet_path,  GeometricMachineLearning.map_to_cpu(nn2))
println("SympNet weights saved → $sympnet_path")
println("ResNet  weights saved → $resnet_path")

# ---- Predict: encode IC → iterate → decode ----------------------------------

# The SAE and plotting data live on the CPU. Reload with CPU storage and backend
# before iteration, and verify that the saved weights can be used for inference.
nn1_cpu = load(NeuralNetwork, sympnet_path, sympnet_arch)
nn2_cpu = load(NeuralNetwork, resnet_path, resnet_arch)

function predict_reduced(nn_red, d)
    z₀   = enc(Float32.(vcat(d.q_eucl[:, 1], d.p_eucl[:, 1])))
    pred = iterate(nn_red, (q=z₀[1:1], p=z₀[2:2]); n_points=d.nsteps)
    full = dec(vcat(pred.q, pred.p))
    q4d  = full[1:2, :]
    p4d  = full[3:4, :]
    θ, θ̇ = euclidean_to_angular(q4d, p4d)
    h    = hamiltonian_euclidean(q4d, p4d)
    θ, θ̇, h
end

# ---- Loss plots -------------------------------------------------------------

let fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss",
        title="Reduced Network Training (2D, $(gpu_ok ? "CUDA" : "CPU"))")
    lines!(ax, loss1;
        label=L"SympNet; $n_\mathrm{params}$=%$(parameterlength(nn1))",
        color=:orange, linewidth=2)
    lines!(ax, loss2;
        label=L"ResNet; $n_\mathrm{params}$=%$(parameterlength(nn2))",
        color=:purple, linewidth=2)
    axislegend(ax, position=:rt)
    CairoMakie.save(joinpath(plotdir, "pendulum_sae_loss_comparison.png"), fig)
end

# ---- Per-IC solution and energy-error plots ---------------------------------

for d in eval_ref
    label = "θ0_$(round(d.θ₀; digits=2))"
    t_vec = d.t
    h_ref = hamiltonian_euclidean(d.q_eucl, d.p_eucl)

    θ1, θ̇1, h1 = predict_reduced(nn1_cpu, d)
    θ2, θ̇2, h2 = predict_reduced(nn2_cpu, d)

    println("Saving plots for $label ...")

    CairoMakie.save(joinpath(plotdir, "pendulum_sae_ref_$(label).png"),
        plot_solution(d.nsteps, t_vec, d.θ, d.θ̇, h_ref, labels_hamiltonian))
    CairoMakie.save(joinpath(plotdir, "pendulum_sae_sympnet_$(label).png"),
        plot_solution(d.nsteps, t_vec, θ1, θ̇1, h1, labels_hamiltonian))
    CairoMakie.save(joinpath(plotdir, "pendulum_sae_resnet_$(label).png"),
        plot_solution(d.nsteps, t_vec, θ2, θ̇2, h2, labels_hamiltonian))

    let fig = Figure(size=(800, 400))
        ax = Axis(fig[1, 1], xlabel="t",
            ylabel=L"[H(t) - H(0)] / |H(0)|",
            title="Energy Error — $label (SAE + reduced dynamics, CPU)")
        lines!(ax, t_vec, compute_energy_error(h_ref);
            label="Reference", color=:black, linewidth=2)
        lines!(ax, t_vec, compute_energy_error(h1);
            label="SympNet", color=:orange, linewidth=2)
        lines!(ax, t_vec, compute_energy_error(h2);
            label="ResNet", color=:purple, linewidth=2)
        axislegend(ax, position=:lt)
        CairoMakie.save(joinpath(plotdir, "pendulum_sae_energy_error_$(label).png"), fig)
    end
end

println("All plots saved to $outdir")
