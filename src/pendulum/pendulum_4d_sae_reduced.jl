# pendulum_4d_sae_reduced.jl
#
# Strategy:
#   (i)  Sample a 2-D IC grid (θ, p_angular) with fixed system params
#        l = m = g = 1 and train a Symplectic Autoencoder (SAE) to find a 2D
#        reduced representation of the 4D Euclidean solution manifold.
#   (ii) Encode all trajectories with the trained SAE encoder, then train a
#        SympNet and a ResNet on the resulting 2D reduced time-series.
#        Predictions are decoded back to 4D Euclidean → angular coordinates
#        for plotting with the existing GeometricProblems routines.

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!

seed!(123)

const timestep = 0.1
const tspan = (0.0, 50.0)
const outdir = @__DIR__

# ---- Fixed system parameters — ONLY initial conditions vary ----------------
const params = (l=1.0, m=1.0, g=1.0)

# ---- Helper functions -------------------------------------------------------

# Angular (θ, p_angular) → Euclidean 4D (q_eucl, p_eucl) for l = m = 1.
# p_angular = l² · θ̇  with m = 1.
function angular_to_euclidean(θ::AbstractVector, p_ang::AbstractVector)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

# Euclidean 4D → (θ, θ̇) — angular velocity equals canonical p for l = m = 1.
function euclidean_to_angular(q_e::AbstractMatrix, p_e::AbstractMatrix)
    x, y = q_e[1, :], q_e[2, :]
    px, py = p_e[1, :], p_e[2, :]
    θ = atan.(x, y)
    θ̇ = (y .* px .- x .* py) ./ (x .^ 2 .+ y .^ 2)
    θ, θ̇
end

# Euclidean Hamiltonian H = ½‖p‖² + q₂  (m = g = 1, independent of l = 1).
hamiltonian_euclidean(q::AbstractMatrix, p::AbstractMatrix) =
    0.5 .* (p[1, :] .^ 2 .+ p[2, :] .^ 2) .+ q[2, :]

compute_energy_error(h) = (h .- h[1]) ./ abs.(h[1])

# ---- Training IC grid (10 angles × 5 momenta = 50 ICs) --------------------
# A 2-D grid in (θ, p_angular) covering the libration region uniformly — not
# just the zero-velocity curve — mirrors the fix from pendulum_4d_bigger_network.jl.
const train_angles = range(π - 2.5, π - 0.15; length=10)
const mom_fracs = [0.0, 0.40, 0.75, -0.40, -0.75]
const train_tspan = (0.0, 100.0)

train_ics = [(θ₀, frac) for θ₀ in train_angles for frac in mom_fracs]
println("Generating $(length(train_ics)) training trajectories (2-D IC grid) ...")

train_ref = map(train_ics) do (θ₀, frac)
    p_sep = sqrt(max(2.0 * (1.0 - cos(θ₀)), 0.0))
    p₀_ang = p_sep * frac
    sol = integrate(
        hodeproblem([θ₀], [p₀_ang]; parameters=params,
            timespan=train_tspan, timestep=timestep),
        Gauss(2))
    ns_ = length(sol.t)
    t_vec = collect(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, p₀=p₀_ang, nsteps=ns_,
        t=t_vec, θ=θ_vec,
        θ̇=pang ./ params.l^2,
        q_eucl=q_e, p_eucl=p_e)
end

println("Done. $(train_ref[1].nsteps) steps per trajectory ($(length(train_ref)) total).")

# ---- Evaluation ICs (6 representative librating orbits, p₀ = 0) ------------
# Used only for final solution / energy-error plots.
const eval_angles = range(acos(0.9), acos(-0.9); length=6)

eval_ref = map(eval_angles) do θ₀
    sol = integrate(
        hodeproblem([θ₀], [0.0]; parameters=params,
            timespan=tspan, timestep=timestep),
        Gauss(2))
    ns_ = length(sol.t)
    t_vec = collect(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, nsteps=ns_,
        t=t_vec, θ=θ_vec,
        θ̇=pang ./ params.l^2,
        q_eucl=q_e, p_eucl=p_e)
end

println("Evaluation: $(length(eval_ref)) trajectories (p₀=0, tspan=$(tspan[2])).")

# ---- Build SAE training data -----------------------------------------------
# Concatenate every snapshot from all trajectories into one big matrix.
# Each column is a point on the 2D solution manifold embedded in ℝ⁴.

q_sae_all = hcat([d.q_eucl for d in train_ref]...)
p_sae_all = hcat([d.p_eucl for d in train_ref]...)
# SAE DataLoader requires a plain (full_dim, n_samples) matrix — not a NamedTuple.
sae_matrix = vcat(q_sae_all, p_sae_all)              # (4, n_train_ics × ns_train)
dl_sae = DataLoader(sae_matrix; autoencoder=true)
println("SAE training data: $(size(sae_matrix, 2)) snapshots, input_dim = $(dl_sae.input_dim)")

# ---- Train Symplectic Autoencoder (4D → 2D) --------------------------------
# Constraint: n_encoder_blocks ≤ full_dim − reduced_dim = 2  (GML requirement).
# The 4D Euclidean pendulum lives on a 2D symplectic submanifold (one DoF +
# circle constraint), so 2D is the natural reduced dimension.

sae_arch = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=10,
    n_decoder_layers=20,
    n_decoder_output_layers=10,
    sympnet_upscale=20)

sae_nn = NeuralNetwork(sae_arch)
sae_opt = Optimizer(AdamOptimizer(1f-4), sae_nn)

const n_sae_epochs = 15000

println("Training SAE ($(parameterlength(sae_nn)) params, $n_sae_epochs epochs) ...")
sae_loss = sae_opt(sae_nn, dl_sae, Batch(256), n_sae_epochs)
println("SAE final loss: $(round(sae_loss[end]; sigdigits = 4))")

enc = encoder(sae_nn)
dec = decoder(sae_nn)

# ---- Encode all trajectories into 2D reduced coordinates ------------------
# enc maps (4, T) → (2, T) where the first row is z_q and the second z_p.
# We keep the time-series structure as a 3-tensor (1, ns, n_ics).

const ns_train = train_ref[1].nsteps
q_red = Array{Float64}(undef, 1, ns_train, length(train_ref))
p_red = Array{Float64}(undef, 1, ns_train, length(train_ref))

for (i, d) in enumerate(train_ref)
    z = enc(vcat(d.q_eucl, d.p_eucl))   # (2, ns_train) — convention: [z_q; z_p]
    q_red[:, :, i] = z[1:1, :]
    p_red[:, :, i] = z[2:2, :]
end

dl_reduced = DataLoader((q=q_red, p=p_red))
println("Reduced DataLoader: input_dim = $(dl_reduced.input_dim), n_params = $(dl_reduced.n_params)")

# ---- Train SympNet and ResNet on 2D reduced dynamics ----------------------

const n_epochs = 100000
const batch = Batch(128)

println("Training SympNet on 2D reduced dynamics ...")
nn1 = NeuralNetwork(GSympNet(2; n_layers=10, upscaling_dimension=32))
o1 = Optimizer(AdamOptimizer(), nn1)
loss1 = o1(nn1, dl_reduced, batch, n_epochs)
println("SympNet: $(parameterlength(nn1)) params, final loss = $(round(loss1[end]; sigdigits = 4))")

println("Training ResNet on 2D reduced dynamics ...")
nn2 = NeuralNetwork(ResNet(2, 79, tanh))
o2 = Optimizer(AdamOptimizer(), nn2)
loss2 = o2(nn2, dl_reduced, batch, n_epochs)
println("ResNet: $(parameterlength(nn2)) params, final loss = $(round(loss2[end]; sigdigits = 4))")

# ---- Helper: encode IC → iterate reduced network → decode to Euclidean -----
function predict_reduced(nn_red, d)
    # Initial condition in reduced 2D space
    z₀ = enc(vcat(d.q_eucl[:, 1], d.p_eucl[:, 1]))   # length-2 vector
    pred = iterate(nn_red, (q=z₀[1:1], p=z₀[2:2]); n_points=d.nsteps)
    # Decode from 2D back to 4D Euclidean
    full = dec(vcat(pred.q, pred.p))                    # (4, nsteps)
    q4d = full[1:2, :]
    p4d = full[3:4, :]
    θ, θ̇ = euclidean_to_angular(q4d, p4d)
    h = hamiltonian_euclidean(q4d, p4d)
    θ, θ̇, h
end

# ---- Loss plots ------------------------------------------------------------
let fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss",
        title="Symplectic Autoencoder Training (4D → 2D)")
    lines!(ax, sae_loss; color=:teal, linewidth=2, label="SAE")
    axislegend(ax, position=:rt)
    save(joinpath(outdir, "pendulum_sae_ae_loss.png"), fig)
end

let fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss",
        title="Reduced Network Training (2D)")
    lines!(ax, loss1;
        label=L"SympNet; $n_\mathrm{params}$=%$(parameterlength(nn1))",
        color=:orange, linewidth=2)
    lines!(ax, loss2;
        label=L"ResNet; $n_\mathrm{params}$=%$(parameterlength(nn2))",
        color=:purple, linewidth=2)
    axislegend(ax, position=:rt)
    save(joinpath(outdir, "pendulum_sae_loss_comparison.png"), fig)
end

# ---- Per-IC solution and energy-error plots --------------------------------
for d in eval_ref
    label = "θ0_$(round(d.θ₀; digits = 2))"
    t_vec = d.t
    h_ref = hamiltonian_euclidean(d.q_eucl, d.p_eucl)

    θ1, θ̇1, h1 = predict_reduced(nn1, d)
    θ2, θ̇2, h2 = predict_reduced(nn2, d)

    println("Saving plots for $label ...")

    save(joinpath(outdir, "pendulum_sae_ref_$(label).png"),
        plot_solution(d.nsteps, t_vec, d.θ, d.θ̇, h_ref, labels_hamiltonian))
    save(joinpath(outdir, "pendulum_sae_sympnet_$(label).png"),
        plot_solution(d.nsteps, t_vec, θ1, θ̇1, h1, labels_hamiltonian))
    save(joinpath(outdir, "pendulum_sae_resnet_$(label).png"),
        plot_solution(d.nsteps, t_vec, θ2, θ̇2, h2, labels_hamiltonian))

    let fig = Figure(size=(800, 400))
        ax = Axis(fig[1, 1], xlabel="t",
            ylabel=L"[H(t) - H(0)] / |H(0)|",
            title="Energy Error — $label (SAE + reduced dynamics)")
        lines!(ax, t_vec, compute_energy_error(h_ref);
            label="Reference", color=:black, linewidth=2)
        lines!(ax, t_vec, compute_energy_error(h1);
            label="SympNet", color=:orange, linewidth=2)
        lines!(ax, t_vec, compute_energy_error(h2);
            label="ResNet", color=:purple, linewidth=2)
        axislegend(ax, position=:lt)
        save(joinpath(outdir, "pendulum_sae_energy_error_$(label).png"), fig)
    end
end

println("All plots saved to $outdir")
