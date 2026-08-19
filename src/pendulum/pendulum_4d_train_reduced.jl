# pendulum_4d_train_reduced.jl
#
# Step 2a of 3: load the trained SAE, encode trajectories, train SympNet and
# ResNet on the 2D reduced dynamics (CPU), and save all weights + loss histories.
#
# Requires sae_weights.h5 produced by pendulum_4d_train_sae.jl.
# Run pendulum_4d_plot.jl afterwards to generate all figures.
#
# Run locally:
#   julia --project=~/.julia/environments/v1.13 pendulum_4d_train_reduced.jl

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using DelimitedFiles
using Random: seed!

seed!(123)

const cpu_backend = CPU()

const timestep = 0.1
const train_tspan = (0.0, 100.0)
const outdir = @__DIR__
const sae_path = joinpath(outdir, "sae_weights.h5")
const sympnet_path = joinpath(outdir, "sympnet_weights.h5")
const resnet_path = joinpath(outdir, "resnet_weights.h5")

const params = (l=1.0, m=1.0, g=1.0)

# ---- Helpers ----------------------------------------------------------------

function angular_to_euclidean(θ::AbstractVector, p_ang::AbstractVector)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

# ---- Training trajectories (needed for encoding) ----------------------------

const train_angles = range(π - 2.5, π - 0.15; length=10)
const mom_fracs = [0.0, 0.40, 0.75, -0.40, -0.75]

train_ics = [(θ₀, frac) for θ₀ in train_angles for frac in mom_fracs]
println("Regenerating $(length(train_ics)) training trajectories for encoding ...")

train_ref = map(train_ics) do (θ₀, frac)
    p_sep = sqrt(max(2.0 * (1.0 - cos(θ₀)), 0.0))
    p₀_ang = p_sep * frac
    sol = integrate(
        hodeproblem([θ₀], [p₀_ang]; parameters=params,
            timespan=train_tspan, timestep=timestep),
        Gauss(2))
    ns_ = length(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, p₀=p₀_ang, nsteps=ns_, q_eucl=q_e, p_eucl=p_e)
end

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
println("SAE loaded ($(parameterlength(sae_nn)) params).")

# ---- Encode training trajectories into 2D reduced coordinates ---------------

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

const n_epochs = 200
const batch = Batch(1024)

println("Training SympNet on CPU ($n_epochs epochs) ...")
const sympnet_arch = GSympNet(2; n_layers=10, upscaling_dimension=32)
nn1 = NeuralNetwork(sympnet_arch, cpu_backend, Float32)
o1 = Optimizer(AdamOptimizer(), nn1)
loss1 = o1(nn1, dl_reduced, batch, n_epochs)
println("SympNet: $(parameterlength(nn1)) params, final loss=$(round(loss1[end]; sigdigits=4))")

println("Training ResNet on CPU ($n_epochs epochs) ...")
const resnet_arch = ResNet(2, 79, tanh)
nn2 = NeuralNetwork(resnet_arch, cpu_backend, Float32)
o2 = Optimizer(AdamOptimizer(), nn2)
loss2 = o2(nn2, dl_reduced, batch, n_epochs)
println("ResNet: $(parameterlength(nn2)) params, final loss=$(round(loss2[end]; sigdigits=4))")

# ---- Save weights and loss histories to disk --------------------------------

GeometricMachineLearning.save(sympnet_path, nn1)
GeometricMachineLearning.save(resnet_path, nn2)
println("SympNet weights saved → $sympnet_path")
println("ResNet  weights saved → $resnet_path")

writedlm(joinpath(outdir, "sympnet_loss.csv"), loss1)
writedlm(joinpath(outdir, "resnet_loss.csv"), loss2)
println("Loss histories saved → sympnet_loss.csv, resnet_loss.csv")
