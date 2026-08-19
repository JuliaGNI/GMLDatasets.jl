# pendulum_energy_conservation_preview.jl
#
# Quick check: generates a single mid-run frame (t ≈ 25) of the energy-
# conservation animation to verify the layout before running the full
# 501-frame job.
#
# Output: plots/energy_conservation_preview.png
#
# Requires: sae_weights.h5, sympnet_weights.h5, resnet_weights.h5
# Run:  julia --project=~/.julia/environments/v1.13 pendulum_energy_conservation_preview.jl
#
# See pendulum_energy_conservation_anim.jl for the Euclidean rod rendering rationale.

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(123)

const outdir       = @__DIR__
const plotdir      = joinpath(outdir, "plots")
const sae_path     = joinpath(outdir, "sae_weights.h5")
const sympnet_path = joinpath(outdir, "sympnet_weights.h5")
const resnet_path  = joinpath(outdir, "resnet_weights.h5")

const params   = (l=1.0, m=1.0, g=1.0)
const timestep = 0.1
const tspan    = (0.0, 50.0)
const θ₀_val   = 1.35   # moderate amplitude

# ────────────────────────────────────────────────────────────────────────────────
# Helpers (identical to pendulum_energy_conservation_anim.jl)
# ────────────────────────────────────────────────────────────────────────────────

function angular_to_euclidean(θ, p_ang)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

hamiltonian_eucl(q, p) = 0.5 .* (p[1,:].^2 .+ p[2,:].^2) .+ q[2,:]

# Reference ghost: exact angular integration, always on unit circle.
function draw_pendulum!(ax, θ; color, alpha=1.0, lw=3)
    bx, by = sin(θ) * params.l, cos(θ) * params.l   # q = l·(sinθ, cosθ) convention
    lines!(ax, [0.0, bx], [0.0, by]; color=(color, alpha), linewidth=lw)
    scatter!(ax, [0.0], [0.0]; color=(:gray40, alpha), markersize=10)
    scatter!(ax, [bx], [by]; color=(color, alpha), markersize=26, marker=:circle)
end

# SAE prediction: draw directly from decoded Euclidean position.
# ‖(bx,by)‖ ≠ l is possible and deliberately shown as a varying rod length.
function draw_pendulum_eucl!(ax, bx, by; color, alpha=1.0, lw=3)
    lines!(ax, [0.0, bx], [0.0, by]; color=(color, alpha), linewidth=lw)
    scatter!(ax, [0.0], [0.0]; color=(:gray40, alpha), markersize=10)
    scatter!(ax, [bx], [by]; color=(color, alpha), markersize=26, marker=:circle)
end

# ────────────────────────────────────────────────────────────────────────────────
# Load networks
# ────────────────────────────────────────────────────────────────────────────────

const sae_arch = SymplecticAutoencoder(4, 2;
    n_encoder_blocks=2, n_decoder_blocks=2,
    n_encoder_layers=4, n_decoder_layers=10,
    n_decoder_output_layers=10, sympnet_upscale=20)
const sympnet_arch = GSympNet(2; n_layers=10, upscaling_dimension=32)
const resnet_arch  = ResNet(2, 79, tanh)

println("Loading networks...")
sae_nn = load(NeuralNetwork, sae_path, sae_arch)
nn_sp  = load(NeuralNetwork, sympnet_path, sympnet_arch)
nn_rn  = load(NeuralNetwork, resnet_path, resnet_arch)
enc    = encoder(sae_nn)
dec    = decoder(sae_nn)

# ────────────────────────────────────────────────────────────────────────────────
# Reference trajectory + predictions
# ────────────────────────────────────────────────────────────────────────────────

println("Integrating reference (θ₀ = $θ₀_val rad)...")
sol = integrate(
    hodeproblem([θ₀_val], [0.0]; parameters=params,
        timespan=tspan, timestep=timestep),
    Gauss(2))
const ns = length(sol.t)

θ_ref  = [sol.q[i][1] for i in 0:ns-1]
pθ_ref = [sol.p[i][1] for i in 0:ns-1]
q_ref, p_ref = angular_to_euclidean(θ_ref, pθ_ref)
h_ref  = Float64.(hamiltonian_eucl(q_ref, p_ref))
t_vec  = collect(range(tspan...; length=ns))

function predict_euclidean(nn_red)
    z₀   = enc(Float32.(vcat(q_ref[:, 1], p_ref[:, 1])))
    pred = iterate(nn_red, (q=z₀[1:1], p=z₀[2:2]); n_points=ns)
    full = dec(vcat(pred.q, pred.p))
    q4d  = Float64.(full[1:2, :])
    p4d  = Float64.(full[3:4, :])
    h    = Float64.(hamiltonian_eucl(q4d, p4d))
    q4d, h
end

println("SympNet prediction...")
q_sp, h_sp = predict_euclidean(nn_sp)
println("ResNet prediction...")
q_rn, h_rn = predict_euclidean(nn_rn)

h0    = h_ref[1]
ee_sp = clamp.((h_sp .- h0) ./ abs(h0), -5.0, 5.0)
ee_rn = clamp.((h_rn .- h0) ./ abs(h0), -5.0, 5.0)

# ────────────────────────────────────────────────────────────────────────────────
# Single preview frame at t ≈ 25 (frame index 250)
# ────────────────────────────────────────────────────────────────────────────────

const i = 250   # change this to inspect different time points

fig = Figure(size=(860, 520); backgroundcolor=:white)
lims = (-1.4, 1.4, -1.5, 0.5)  # slightly wider than l=1 for off-circle SAE decoded positions

ax_sp = Axis(fig[1, 1];
    title="SAE + SympNet", titlecolor=:darkorange, titlesize=14,
    limits=lims, aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    leftspinevisible=false, rightspinevisible=false,
    topspinevisible=false, bottomspinevisible=false)

ax_rn = Axis(fig[1, 2];
    title="SAE + ResNet", titlecolor=:purple, titlesize=14,
    limits=lims, aspect=DataAspect(),
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false,
    leftspinevisible=false, rightspinevisible=false,
    topspinevisible=false, bottomspinevisible=false)

draw_pendulum!(ax_sp, θ_ref[i]; color=:gray70, alpha=0.45, lw=2)
draw_pendulum_eucl!(ax_sp, q_sp[1, i], q_sp[2, i]; color=:darkorange)

draw_pendulum!(ax_rn, θ_ref[i]; color=:gray70, alpha=0.45, lw=2)
draw_pendulum_eucl!(ax_rn, q_rn[1, i], q_rn[2, i]; color=:purple)

ax_e = Axis(fig[2, 1:2];
    xlabel=L"t",
    ylabel=L"[H(t)-H(0)]\,/\,|H(0)|",
    limits=(0.0, tspan[2], -5.5, 5.5))

hlines!(ax_e, [0.0]; color=(:black, 0.25), linewidth=1)
lines!(ax_e, t_vec, ee_sp; color=(:darkorange, 0.12), linewidth=1.5)
lines!(ax_e, t_vec, ee_rn; color=(:purple,     0.12), linewidth=1.5)
lines!(ax_e, t_vec[1:i], ee_sp[1:i]; color=:darkorange, linewidth=2, label="SympNet")
lines!(ax_e, t_vec[1:i], ee_rn[1:i]; color=:purple,     linewidth=2, label="ResNet")
vlines!(ax_e, [t_vec[i]]; color=(:black, 0.4), linewidth=1, linestyle=:dot)
axislegend(ax_e; position=:lt, framevisible=false, labelsize=12)

rowsize!(fig.layout, 1, Relative(0.44))
rowsize!(fig.layout, 2, Relative(0.56))

out = joinpath(plotdir, "energy_conservation_preview.png")
CairoMakie.save(out, fig)
println("Preview saved → $out")
