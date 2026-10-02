# Energy error of the learned harmonic oscillator, on training and test initial conditions
#
# The pair of figures that show a SympNet holding the energy where a ResNet drifts, on the initial
# conditions it was trained on and on ones it was not.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/harmonic_oscillator/energy_error.jl
#
# Produces:
#   plots/ho_energy_error_training_ic.png
#   plots/ho_energy_error_test_ic.png
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/harmonic_oscillator_plots.jl), where it sat beside the figure it makes.
# A figure is a claim and a generator nobody's CI runs rots against the API it was written
# for -- which this one had: see the note at the end of this header.

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: q₀, p₀, default_parameters, hamiltonian
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!

seed!(123)

# Outputs go to GML_OUTDIR, or to the working directory. Never beside the script: a figure is a
# build product and the generator is the artefact, which is the whole reason these live here now.
const outdir  = get(ENV, "GML_OUTDIR", pwd())
const plotdir = mkpath(joinpath(outdir, "plots"))
const prms    = default_parameters()

# ────────────────────────────────────────────────────────────────────────────────
# Training data: 50 data points  (t ∈ [0, 10], Δt = 0.2)
# ────────────────────────────────────────────────────────────────────────────────

const train_Δt    = 0.2
const train_tspan = (0.0, 10.0)
const n_train_pts = round(Int, train_tspan[2] / train_Δt)   # = 50

sol_train = integrate(
    hodeproblem(q₀, p₀; timespan=train_tspan, timestep=train_Δt),
    ImplicitMidpoint())
dl = DataLoader(sol_train)
println("Training: $(n_train_pts) data points (t ∈ $(train_tspan), Δt = $(train_Δt))")

# ────────────────────────────────────────────────────────────────────────────────
# Train networks
# ────────────────────────────────────────────────────────────────────────────────

const n_epochs = 3000

arch1 = GSympNet(2; n_layers=3)
nn1   = NeuralNetwork(arch1)
loss1 = Optimizer(AdamOptimizer(), nn1)(nn1, dl, Batch(1), n_epochs)
println("SympNet  $(parameterlength(nn1)) params  final loss = $(round(loss1[end]; sigdigits=4))")

arch2 = ResNet(2, 2, tanh)
nn2   = NeuralNetwork(arch2)
loss2 = Optimizer(AdamOptimizer(), nn2)(nn2, dl, Batch(1), n_epochs)
println("ResNet   $(parameterlength(nn2)) params  final loss = $(round(loss2[end]; sigdigits=4))")

# ────────────────────────────────────────────────────────────────────────────────
# Long-time evaluation: t ∈ [0, 100]
# ────────────────────────────────────────────────────────────────────────────────

const eval_tspan = (0.0, 100.0)
const eval_Δt    = 0.2
const n_eval     = round(Int, eval_tspan[2] / eval_Δt) + 1
const t_vec      = collect(range(eval_tspan...; length=n_eval))

function run_and_error(nn, q_ic, p_ic)
    pred = iterate(nn, (q=q_ic, p=p_ic); n_points=n_eval)
    h0   = hamiltonian(0.0, q_ic, p_ic, prms)
    err  = [(hamiltonian(0.0, pred.q[:, i], pred.p[:, i], prms) - h0) / abs(h0)
            for i in 1:n_eval]
    clamp.(err, -30.0, 30.0)   # cap extreme divergence for readability
end

# Training IC  (q₀ = [0.5], p₀ = [0.0])
ee_sp_train = run_and_error(nn1, q₀, p₀)
ee_rn_train = run_and_error(nn2, q₀, p₀)

# Test IC  (q₀/2, p₀/2 — never seen during training)
q_test = q₀ ./ 2
p_test = p₀ ./ 2
ee_sp_test = run_and_error(nn1, q_test, p_test)
ee_rn_test = run_and_error(nn2, q_test, p_test)

# ────────────────────────────────────────────────────────────────────────────────
# Plot helper
# ────────────────────────────────────────────────────────────────────────────────

function energy_error_figure(t, ee_sp, ee_rn, title_str;
        nsp=parameterlength(nn1), nrn=parameterlength(nn2))
    fig = Figure(size=(800, 360))
    ax  = Axis(fig[1, 1];
        xlabel = L"t",
        ylabel = L"[H(t) - H(0)]\,/\,|H(0)|",
        title  = title_str)
    vlines!(ax, [train_tspan[2]];
        color=(:gray, 0.7), linestyle=:dash, linewidth=1.5,
        label="Training horizon (t = $(Int(train_tspan[2])))")
    lines!(ax, t, ee_sp;
        color=:orange, linewidth=2,
        label="SympNet  ($(nsp) params)")
    lines!(ax, t, ee_rn;
        color=:purple, linewidth=2,
        label="ResNet   ($(nrn) params)")
    axislegend(ax; position=:lt)
    fig
end

# ────────────────────────────────────────────────────────────────────────────────
# Save plots
# ────────────────────────────────────────────────────────────────────────────────

let fig = energy_error_figure(t_vec, ee_sp_train, ee_rn_train,
        "Energy Error — Training IC  ($(n_train_pts) data points, t ∈ [0, $(Int(train_tspan[2]))])")
    CairoMakie.save(joinpath(plotdir, "ho_energy_error_training_ic.png"), fig)
    println("Saved → ho_energy_error_training_ic.png")
end

let fig = energy_error_figure(t_vec, ee_sp_test, ee_rn_test,
        "Energy Error — Test IC  (q₀/2, p₀/2 — unseen during training)")
    CairoMakie.save(joinpath(plotdir, "ho_energy_error_test_ic.png"), fig)
    println("Saved → ho_energy_error_test_ic.png")
end

println("Done.")
