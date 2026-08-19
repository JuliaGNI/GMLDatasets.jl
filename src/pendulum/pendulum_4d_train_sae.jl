# pendulum_4d_train_sae.jl
#
# Step 1 of 2: train the Symplectic Autoencoder (SAE) on CPU and save weights.
#
# Run locally:
#   julia --project=~/.julia/environments/v1.13 pendulum_4d_train_sae.jl

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(123)

const cpu_backend = CPU()

const timestep = 0.1
const outdir   = @__DIR__
const plotdir  = joinpath(outdir, "plots")
const sae_path = joinpath(outdir, "sae_weights.h5")

const params = (l=1.0, m=1.0, g=1.0)

# ---- Helper -----------------------------------------------------------------

function angular_to_euclidean(θ::AbstractVector, p_ang::AbstractVector)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

# ---- Training data ----------------------------------------------------------

const train_angles = range(π - 2.5, π - 0.15; length=10)
const mom_fracs    = [0.0, 0.40, 0.75, -0.40, -0.75, -1., -2., -3.]
const train_tspan  = (0.0, 10.0)

train_ics = [(θ₀, frac) for θ₀ in train_angles for frac in mom_fracs]
println("Generating $(length(train_ics)) training trajectories ...")

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
    (; θ₀, p₀=p₀_ang, nsteps=ns_, q_eucl=q_e, p_eucl=p_e)
end

println("Done. $(train_ref[1].nsteps) steps per trajectory.")

# ---- Move SAE training data to GPU (Float32) --------------------------------

q_sae_all  = hcat([d.q_eucl for d in train_ref]...)
p_sae_all  = hcat([d.p_eucl for d in train_ref]...)
sae_matrix = vcat(q_sae_all, p_sae_all)

dl_sae = DataLoader(Float32.(sae_matrix); autoencoder=true)
println("SAE data: $(size(sae_matrix, 2)) snapshots [CPU, Float32]")

# ---- Train SAE --------------------------------------------------------------

const sae_arch = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=10,
    n_decoder_layers=20,
    n_decoder_output_layers=10,
    sympnet_upscale=20)

sae_nn  = NeuralNetwork(sae_arch, cpu_backend, Float32)
sae_opt = Optimizer(AdamOptimizer(1f-4), sae_nn)

const n_sae_epochs = 12000

println("Training SAE ($(parameterlength(sae_nn)) params, $n_sae_epochs epochs) [CPU] ...")
sae_loss = sae_opt(sae_nn, dl_sae, Batch(256), n_sae_epochs)
println("SAE final loss: $(round(sae_loss[end]; sigdigits=4))")

# ---- Save weights to disk ---------------------------------------------------

GeometricMachineLearning.save(sae_path, sae_nn)
println("SAE weights saved → $sae_path")

let fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss",
        title="Symplectic Autoencoder Training (4D → 2D)")
    lines!(ax, sae_loss; color=:teal, linewidth=2, label="SAE")
    axislegend(ax, position=:rt)
    CairoMakie.save(joinpath(plotdir, "pendulum_sae_ae_loss.png"), fig)
end
println("SAE loss plot saved → pendulum_sae_ae_loss.png")
