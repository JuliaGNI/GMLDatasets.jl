# pendulum_energy_conservation_anim.jl
#
# Generates 501 animation frames for the "Pendulum: Energy Conservation" slide.
# Each frame shows:
#   Top-left  — physical pendulum cartoon for SAE + SympNet (orange)
#   Top-right — physical pendulum cartoon for SAE + ResNet  (purple)
#               both with a grey ghost of the reference pendulum
#   Bottom    — relative energy error growing in real time for both networks
#
# IMPORTANT — Euclidean rod rendering:
#   The SAE pipeline works in the 4D Euclidean space (q₁=l·sinθ, q₂=l·cosθ).
#   The decoder maps the 2D latent state back to 4D Euclidean coords, but it
#   does NOT guarantee ‖q‖ = l exactly; the circle constraint is only
#   approximately satisfied.  To make this visible, the model pendulums are
#   drawn directly from the decoded Euclidean position (q₁, q₂) rather than
#   from the recovered angle.  Deviations from l produce a visibly changing
#   rod length that the purely angular representation would hide.
#   The grey reference ghost always stays on the unit circle (it is drawn from
#   the exact angular integration).
#
# Requires: sae_weights.h5, sympnet_weights.h5, resnet_weights.h5
# Run:  julia --project=~/.julia/environments/v1.13 pendulum_energy_conservation_anim.jl

using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using HDF5
using CairoMakie
using Random: seed!

seed!(123)

const outdir       = @__DIR__
const animdir      = joinpath(outdir, "Animations", "energy_conservation_pendulum")
const sae_path     = joinpath(outdir, "sae_weights.h5")
const sympnet_path = joinpath(outdir, "sympnet_weights.h5")
const resnet_path  = joinpath(outdir, "resnet_weights.h5")

isdir(animdir) || mkpath(animdir)

const params   = (l=1.0, m=1.0, g=1.0)
const timestep = 0.1
const tspan    = (0.0, 50.0)
const θ₀_val   = 1.35   # moderate amplitude — matches the left panel of the static slide

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

function angular_to_euclidean(θ, p_ang)
    q_e = params.l .* hcat(sin.(θ), cos.(θ))'
    p_e = (1.0 / params.l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    q_e, p_e
end

hamiltonian_eucl(q, p) = 0.5 .* (p[1,:].^2 .+ p[2,:].^2) .+ q[2,:]

# ────────────────────────────────────────────────────────────────────────────────
# Load networks
# ────────────────────────────────────────────────────────────────────────────────

const sae_arch     = SymplecticAutoencoder(4, 2;
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
# Reference trajectory
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

# ────────────────────────────────────────────────────────────────────────────────
# Network predictions: encode IC → iterate → decode → angular
# ────────────────────────────────────────────────────────────────────────────────

# Returns the decoded Euclidean bob positions (2×ns) and the Euclidean Hamiltonian
# values.  We deliberately do NOT convert back to angular coords: the SAE
# decoder does not preserve ‖q‖ = l exactly, so going through atan and then
# re-projecting onto the circle would silently hide the reconstruction error.
function predict_euclidean(nn_red)
    z₀   = enc(Float32.(vcat(q_ref[:, 1], p_ref[:, 1])))
    pred = iterate(nn_red, (q=z₀[1:1], p=z₀[2:2]); n_points=ns)
    full = dec(vcat(pred.q, pred.p))
    q4d  = Float64.(full[1:2, :])   # decoded Euclidean bob position — may have ‖q4d‖ ≠ l
    p4d  = Float64.(full[3:4, :])
    h    = Float64.(hamiltonian_eucl(q4d, p4d))
    q4d, h
end

println("SympNet prediction...")
q_sp, h_sp = predict_euclidean(nn_sp)
println("ResNet prediction...")
q_rn, h_rn = predict_euclidean(nn_rn)

# Relative energy error (clamped for readability)
const h0   = h_ref[1]
ee_sp = clamp.((h_sp .- h0) ./ abs(h0), -5.0, 5.0)
ee_rn = clamp.((h_rn .- h0) ./ abs(h0), -5.0, 5.0)

# ────────────────────────────────────────────────────────────────────────────────
# Physical pendulum drawing helper
# ────────────────────────────────────────────────────────────────────────────────

const L = params.l

# Draw using an exact angle — used for the reference ghost (always on unit circle).
function draw_pendulum!(ax, θ; color, alpha=1.0, lw=3)
    bx, by = sin(θ) * L, cos(θ) * L
    lines!(ax, [0.0, bx], [0.0, by]; color=(color, alpha), linewidth=lw)
    scatter!(ax, [0.0], [0.0]; color=(:gray40, alpha), markersize=10)
    scatter!(ax, [bx], [by]; color=(color, alpha), markersize=26, marker=:circle)
end

# Draw using decoded Euclidean coordinates — used for SAE model predictions.
# (bx, by) = (q₁, q₂) from the SAE decoder; if ‖(bx,by)‖ ≠ l the rod length
# visibly deviates from the reference, revealing the decoder reconstruction error.
function draw_pendulum_eucl!(ax, bx, by; color, alpha=1.0, lw=3)
    lines!(ax, [0.0, bx], [0.0, by]; color=(color, alpha), linewidth=lw)
    scatter!(ax, [0.0], [0.0]; color=(:gray40, alpha), markersize=10)
    scatter!(ax, [bx], [by]; color=(color, alpha), markersize=26, marker=:circle)
end

# ────────────────────────────────────────────────────────────────────────────────
# Frame generation
# ────────────────────────────────────────────────────────────────────────────────

println("Generating $ns frames in $animdir ...")

for i in 1:ns
    fig = Figure(size=(860, 520); backgroundcolor=:white)

    # Slightly wider limits than l=1 to accommodate off-circle SAE decoded positions.
    lims = (-1.4, 1.4, -1.5, 0.5)

    # ── SympNet pendulum ──────────────────────────────────────────────────────
    ax_sp = Axis(fig[1, 1];
        title = "SAE + SympNet", titlecolor = :darkorange, titlesize = 14,
        limits = lims, aspect = DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false)

    draw_pendulum!(ax_sp, θ_ref[i]; color=:gray70, alpha=0.45, lw=2)
    draw_pendulum_eucl!(ax_sp, q_sp[1, i], q_sp[2, i]; color=:darkorange)

    # ── ResNet pendulum ───────────────────────────────────────────────────────
    ax_rn = Axis(fig[1, 2];
        title = "SAE + ResNet", titlecolor = :purple, titlesize = 14,
        limits = lims, aspect = DataAspect(),
        xticksvisible=false, yticksvisible=false,
        xticklabelsvisible=false, yticklabelsvisible=false,
        leftspinevisible=false, rightspinevisible=false,
        topspinevisible=false, bottomspinevisible=false)

    draw_pendulum!(ax_rn, θ_ref[i]; color=:gray70, alpha=0.45, lw=2)
    draw_pendulum_eucl!(ax_rn, q_rn[1, i], q_rn[2, i]; color=:purple)

    # ── Energy error ──────────────────────────────────────────────────────────
    ax_e = Axis(fig[2, 1:2];
        xlabel = L"t",
        ylabel = L"[H(t)-H(0)]\,/\,|H(0)|",
        limits = (0.0, tspan[2], -5.5, 5.5))

    hlines!(ax_e, [0.0]; color=(:black, 0.25), linewidth=1)

    # Faded preview of full trajectory
    lines!(ax_e, t_vec, ee_sp; color=(:darkorange, 0.12), linewidth=1.5)
    lines!(ax_e, t_vec, ee_rn; color=(:purple,     0.12), linewidth=1.5)

    # Active portion up to current frame
    lines!(ax_e, t_vec[1:i], ee_sp[1:i]; color=:darkorange, linewidth=2,
        label="SympNet")
    lines!(ax_e, t_vec[1:i], ee_rn[1:i]; color=:purple,     linewidth=2,
        label="ResNet")

    # Moving time cursor
    vlines!(ax_e, [t_vec[i]]; color=(:black, 0.4), linewidth=1, linestyle=:dot)

    axislegend(ax_e; position=:lt, framevisible=false, labelsize=12)

    rowsize!(fig.layout, 1, Relative(0.44))
    rowsize!(fig.layout, 2, Relative(0.56))

    CairoMakie.save(joinpath(animdir, "pendulum_$(lpad(i, 4, '0')).png"), fig)
    i % 100 == 0 && println("  Frame $i / $ns")
end

println("Done. Frames in: $animdir")
