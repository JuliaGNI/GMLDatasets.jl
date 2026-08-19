# pendulum_4d_train_reduced_and_plot.jl
#
# Step 2 of 2: load the trained SAE, encode trajectories, train SympNet and
# ResNet on the 2D reduced dynamics (CPU), then produce all solution and
# energy-error plots.
#
# Requires sae_weights.h5 produced by pendulum_4d_train_sae.jl.
#
# Run locally:
#   julia --project=~/.julia/environments/v1.13 pendulum_4d_train_reduced_and_plot.jl

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(123)

const cpu_backend = CPU()

const timestep      = 0.1
const tspan         = (0.0, 50.0)
const train_tspan   = (0.0, 100.0)
const outdir        = @__DIR__
const sae_path      = joinpath(outdir, "sae_weights.h5")
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

dl_reduced = DataLoader((q=q_red, p=p_red))
println("Reduced DataLoader: input_dim=$(dl_reduced.input_dim) [CPU, Float32]")

# ---- Train SympNet and ResNet on CPU ----------------------------------------

const n_epochs = 11000
const batch    = Batch(128)

println("Training SympNet on CPU ...")
const sympnet_arch = GSympNet(2; n_layers=10, upscaling_dimension=32)
nn1  = NeuralNetwork(sympnet_arch, cpu_backend, Float32)
o1   = Optimizer(AdamOptimizer(), nn1)
loss1 = o1(nn1, dl_reduced, batch, n_epochs)
println("SympNet: $(parameterlength(nn1)) params, final loss=$(round(loss1[end]; sigdigits=4))")

println("Training ResNet on CPU ...")
const resnet_arch = ResNet(2, 79, tanh)
nn2  = NeuralNetwork(resnet_arch, cpu_backend, Float32)
o2   = Optimizer(AdamOptimizer(), nn2)
loss2 = o2(nn2, dl_reduced, batch, n_epochs)
println("ResNet: $(parameterlength(nn2)) params, final loss=$(round(loss2[end]; sigdigits=4))")

# ---- Save reduced network weights -------------------------------------------

GeometricMachineLearning.save(sympnet_path, nn1)
GeometricMachineLearning.save(resnet_path,  nn2)
println("SympNet weights saved → $sympnet_path")
println("ResNet  weights saved → $resnet_path")

# ---- Predict: encode IC → iterate → decode ----------------------------------

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
        title="Reduced Network Training (2D, CPU)")
    lines!(ax, loss1;
        label=L"SympNet; $n_\mathrm{params}$=%$(parameterlength(nn1))",
        color=:orange, linewidth=2)
    lines!(ax, loss2;
        label=L"ResNet; $n_\mathrm{params}$=%$(parameterlength(nn2))",
        color=:purple, linewidth=2)
    axislegend(ax, position=:rt)
    CairoMakie.save(joinpath(outdir, "pendulum_sae_loss_comparison.png"), fig)
end

# ---- Per-IC solution and energy-error plots ---------------------------------

for d in eval_ref
    label = "θ0_$(round(d.θ₀; digits=2))"
    t_vec = d.t
    h_ref = hamiltonian_euclidean(d.q_eucl, d.p_eucl)

    θ1, θ̇1, h1 = predict_reduced(nn1, d)
    θ2, θ̇2, h2 = predict_reduced(nn2, d)

    println("Saving plots for $label ...")

    CairoMakie.save(joinpath(outdir, "pendulum_sae_ref_$(label).png"),
        plot_solution(d.nsteps, t_vec, d.θ, d.θ̇, h_ref, labels_hamiltonian))
    CairoMakie.save(joinpath(outdir, "pendulum_sae_sympnet_$(label).png"),
        plot_solution(d.nsteps, t_vec, θ1, θ̇1, h1, labels_hamiltonian))
    CairoMakie.save(joinpath(outdir, "pendulum_sae_resnet_$(label).png"),
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
        CairoMakie.save(joinpath(outdir, "pendulum_sae_energy_error_$(label).png"), fig)
    end
end

println("All plots saved to $outdir")
