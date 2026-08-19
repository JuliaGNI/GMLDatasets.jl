# pendulum_sae_latent_plot.jl
#
# Two-panel figure comparing the reference angular phase portrait with the SAE
# 2D latent space, side by side.  The left panel gives viewers an immediate
# reference frame (familiar pendulum phase portrait); the right panel shows what
# the SAE learned from purely 4D Euclidean data.  Matching colours connect each
# energy level across both panels.
#
# Requires:  sae_weights.h5  (from pendulum_4d_train_sae.jl)
#
# Run:  julia --project=~/.julia/environments/v1.13 pendulum_sae_latent_plot.jl

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(123)

const outdir   = @__DIR__
const plotdir  = joinpath(outdir, "plots")
const sae_path = joinpath(outdir, "sae_weights.h5")

const params   = (l=1.0, m=1.0, g=1.0)
const timestep = 0.1
const tspan    = (0.0, 20.0)

# Energy-uniform initial conditions: H = p²/2 - cos(θ) = -cos(θ₀) for p₀=0.
# Five levels evenly spaced in H give better visual separation than angle-uniform
# spacing, which bunches orbits near the separatrix.
const eval_H      = range(-.9, .9; length=15)
const eval_angles = [acos(-h) for h in eval_H]   # θ₀ = acos(-H₀)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

function angular_to_euclidean(θ, p_ang)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

# ────────────────────────────────────────────────────────────────────────────────
# Load SAE
# ────────────────────────────────────────────────────────────────────────────────

const sae_arch = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2,
    n_decoder_blocks=2,
    n_encoder_layers=10,
    n_decoder_layers=20,
    n_decoder_output_layers=10,
    sympnet_upscale=20)

println("Loading SAE from $(sae_path) ...")
sae_nn = load(NeuralNetwork, sae_path, sae_arch)
enc    = encoder(sae_nn)

# ────────────────────────────────────────────────────────────────────────────────
# Reference trajectories
# ────────────────────────────────────────────────────────────────────────────────

eval_data = map(eval_angles) do θ₀
    sol   = integrate(
        hodeproblem([θ₀], [1.]; parameters=params,
            timespan=tspan, timestep=timestep),
        Gauss(2))
    ns_   = length(sol.t)
    θ_vec = [sol.q[i][1] for i in 0:ns_-1]
    pang  = [sol.p[i][1] for i in 0:ns_-1]
    q_e, p_e = angular_to_euclidean(θ_vec, pang)
    (; θ₀, nsteps=ns_, q_eucl=q_e, p_eucl=p_e, θ=θ_vec, pθ=pang)
end

println("Generated $(length(eval_data)) trajectories.")

# ────────────────────────────────────────────────────────────────────────────────
# Encode to 2D latent space
# ────────────────────────────────────────────────────────────────────────────────

latent_data = map(eval_data) do d
    z_traj = [enc(Float32.(vcat(d.q_eucl[:, i], d.p_eucl[:, i]))) for i in 1:d.nsteps]
    zq = Float64[z[1] for z in z_traj]
    zp = Float64[z[2] for z in z_traj]
    (; d.θ₀, zq, zp)
end

# ────────────────────────────────────────────────────────────────────────────────
# Two-panel figure
# ────────────────────────────────────────────────────────────────────────────────

# palette = [:steelblue4, :teal, :olivedrab, :goldenrod3, :tomato]

let fig = Figure(size=(1100, 460))
    ax1 = Axis(fig[1, 1];
        xlabel = L"\theta \; \mathrm{(rad)}",
        ylabel = L"p_\theta",
        title  = "Reference phase portrait (angular coordinates)")

    ax2 = Axis(fig[1, 2];
        xlabel = L"z_q",
        ylabel = L"z_p",
        title  = "SAE latent space (learned 2D embedding)")

    for (i, (d, ld)) in enumerate(zip(eval_data, latent_data))
        h0    = round(-cos(d.θ₀); digits=2)
        label = "H₀ = $h0"
        # c     = palette[i]
        lines!(ax1, (x -> x % 2π).(d.θ), d.pθ; linewidth=2, label=label)
        lines!(ax2, ld.zq, ld.zp; linewidth=2)
    end

    Legend(fig[1, 3], ax1;
        labelsize=12, framevisible=false, title="Energy level",
        titlesize=12)

    CairoMakie.save(joinpath(plotdir, "pendulum_sae_latent.png"), fig)
    println("Saved → pendulum_sae_latent.png")
end

println("Done.")
