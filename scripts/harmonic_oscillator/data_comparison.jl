# SympNet against ResNet on harmonic-oscillator data of varying quality
#
# How the two architectures degrade as the training data does.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/harmonic_oscillator/data_comparison.jl
#
# Produces:
#   plots/ho_data_comparison.png
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/harmonic_oscillator_data_comparison.jl), where it sat beside the figure it makes.
# A figure is a claim and a generator nobody's CI runs rots against the API it was written
# for -- which this one had: see the note at the end of this header.
#
# The optimizer calls were migrated: GeometricOptimizers' Adam no longer carries an `eta` field,
# because it was never applied to the direction. The rate is `step_size` now and it is applied, so
# a rerun is not bit-for-bit the run that made the tracked figure.

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: q₀, p₀, default_parameters, hamiltonian
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!

# Outputs go to GML_OUTDIR, or to the working directory. Never beside the script: a figure is a
# build product and the generator is the artefact, which is the whole reason these live here now.
const outdir  = get(ENV, "GML_OUTDIR", pwd())
const plotdir = mkpath(joinpath(outdir, "plots"))
mkpath(plotdir)

const prms = default_parameters()

# ── Evaluation grid (same for both panels) ────────────────────────────────────
const eval_Δt = 0.2
const eval_tspan = (0.0, 100.0)
const n_eval = round(Int, eval_tspan[2] / eval_Δt) + 1
const t_eval = collect(range(eval_tspan...; length=n_eval))

# ── Helper: relative energy error (clamped to avoid chart overflow) ───────────
function energy_errors(nn, q_ic, p_ic, n_pts)
    pred = iterate(nn, (q=q_ic, p=p_ic); n_points=n_pts)
    h0 = hamiltonian(0.0, q_ic, p_ic, prms)
    err = [(hamiltonian(0.0, pred.q[:, i], pred.p[:, i], prms) - h0) / abs(h0)
           for i in 1:n_pts]
    clamp.(err, -30.0, 30.0)
end

# ── Helper: train a SympNet / ResNet pair on a given solution ─────────────────
function train_pair(sol; n_epochs=50_000, seed_val=1234)
    dl = DataLoader(sol)

    seed!(seed_val)
    arch1 = GSympNet(2; n_layers=2)
    nn1 = NeuralNetwork(arch1)
    # SympNets needs less training iterations
    n_epochs_sympnet = n_epochs ÷ 500
    # The learning rate used to be `AdamOptimizer(1e-3)`, a field that was never applied to the
    # direction -- Adam produces a direction of magnitude ~1 per component and the rate is the line
    # search's alpha, so that call trained identically to `AdamOptimizer(1e2)`. It is `step_size`
    # now, and it is applied, so this run is not bit-for-bit the one that made the tracked figure.
    losses1 = Optimizer(AdamOptimizer(), nn1; step_size = 1e-3)(nn1, dl, Batch(1), n_epochs_sympnet)

    seed!(seed_val)
    arch2 = ResNet(2, 1, tanh)
    nn2 = NeuralNetwork(arch2)
    losses2 = Optimizer(nn2; AdamOptimizerWithDecay(n_epochs; η₁ = 1e-2, η₂ = 1e-6)...)(
        nn2, dl, Batch(1), n_epochs)

    println("  SympNet  $(parameterlength(nn1)) params, " *
            "final loss = $(round(losses1[end]; sigdigits=4))")
    println("  ResNet   $(parameterlength(nn2)) params, " *
            "final loss = $(round(losses2[end]; sigdigits=4))")

    nn1, nn2
end

# ── MANY data: t_train ∈ [0, 100], Δt = 0.2 (500 training steps) ─────────────
const many_tspan = (0.0, 100.0)
const many_Δt = 0.2
const n_many = round(Int, many_tspan[2] / many_Δt)

println("=== MANY data  ($(n_many) pts, t_train ∈ [0, $(Int(many_tspan[2]))]) ===")
sol_many = integrate(
    hodeproblem(q₀, p₀; timespan=many_tspan, timestep=many_Δt),
    ImplicitMidpoint())
nn1_many, nn2_many = train_pair(sol_many)
ee_sp_many = energy_errors(nn1_many, q₀, p₀, n_eval)
ee_rn_many = energy_errors(nn2_many, q₀, p₀, n_eval)

# ── FEW data: t_train ∈ [0, 10], Δt = 0.2 (50 training steps) ───────────────
const few_tspan = (0.0, 10.0)
const few_Δt = 0.2
const n_few = round(Int, few_tspan[2] / few_Δt)

println("=== FEW data  ($(n_few) pts, t_train ∈ [0, $(Int(few_tspan[2]))]) ===")
sol_few = integrate(
    hodeproblem(q₀, p₀; timespan=few_tspan, timestep=few_Δt),
    ImplicitMidpoint())
nn1_few, nn2_few = train_pair(sol_few)
ee_sp_few = energy_errors(nn1_few, q₀, p₀, n_eval)
ee_rn_few = energy_errors(nn2_few, q₀, p₀, n_eval)

# ── Figure ────────────────────────────────────────────────────────────────────
#   1 × 2 layout; shared y-range so panels are directly comparable.

nsp = parameterlength(nn1_few)  # same architecture, so same param count
nrn = parameterlength(nn2_few)

fig = Figure(size=(1200, 380), figure_padding=(10, 20, 10, 10))

function add_panel!(pos, t, ee_sp, ee_rn, train_end, title_str)
    ax = Axis(fig[pos...];
        title=title_str,
        xlabel=L"t",
        ylabel=L"[H(t) - H(0)]\;/\;|H(0)|")

    # Dashed vertical line at the training horizon
    vlines!(ax, [train_end];
        color=(:gray, 0.65),
        linestyle=:dash,
        linewidth=1.5,
        label="training horizon (t = $(Int(train_end)))")

    lines!(ax, t, ee_sp;
        color=:orange,
        linewidth=2.0,
        label=L"SympNet  ($%$(nsp)$ params)")
    lines!(ax, t, ee_rn;
        color=:purple,
        linewidth=2.0,
        label=L"ResNet  ($%$(nrn)$ params)")

    axislegend(ax; position=:lt, framevisible=true)
end

add_panel!((1, 1), t_eval, ee_sp_many, ee_rn_many,
    many_tspan[2],
    L"Many data (%$(n_many) pts,  $t_\mathrm{train}$ ∈ [0, %$(Int(many_tspan[2]))])")

add_panel!((1, 2), t_eval, ee_sp_few, ee_rn_few,
    few_tspan[2],
    L"Few data (%$(n_many) pts,  $t_\mathrm{train}$ ∈ [0, %$(Int(many_tspan[2]))])")

# ── Save ──────────────────────────────────────────────────────────────────────
outfile = joinpath(plotdir, "ho_data_comparison.png")
CairoMakie.save(outfile, fig; px_per_unit=2)
println("Saved → $(outfile)")
