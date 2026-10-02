# How the harmonic-oscillator energy error scales with network size
#
# Sweeps the network size and records the energy error, for the scaling figure.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/harmonic_oscillator/scaling.jl
#
# Produces:
#   plots/ho_scaling_energy_error.png
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/harmonic_oscillator_scaling.jl), where it sat beside the figure it makes.
# A figure is a claim and a generator nobody's CI runs rots against the API it was written
# for -- which this one had: see the note at the end of this header.

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
const prms = default_parameters()

# ── Training data (50 steps, one orbit) ──────────────────────────────────────
const train_Δt = 0.2
const train_tspan = (0.0, 10.0)

seed!(123)
sol_train = integrate(
    hodeproblem(q₀, p₀; timespan=train_tspan, timestep=train_Δt),
    ImplicitMidpoint())
dl = DataLoader(sol_train)

# ── Long-time evaluation on unseen test IC ────────────────────────────────────
const eval_tspan = (0.0, 100.0)
const eval_Δt = 0.2
const n_eval = round(Int, eval_tspan[2] / eval_Δt) + 1
const q_test = q₀ ./ 2
const p_test = p₀ ./ 2

const n_train = round(Int, train_tspan[2] / train_Δt) + 1

function peak_energy_error(nn)
    pred = iterate(nn, (q=q_test, p=p_test); n_points=n_eval)
    h0 = hamiltonian(0.0, q_test, p_test, prms)
    errs = [(hamiltonian(0.0, pred.q[:, i], pred.p[:, i], prms) - h0) / abs(h0)
            for i in 1:n_eval]
    maximum(abs, errs)
end

function peak_train_energy_error(nn)
    pred = iterate(nn, (q=q₀, p=p₀); n_points=n_train)
    h0 = hamiltonian(0.0, q₀, p₀, prms)
    errs = [(hamiltonian(0.0, pred.q[:, i], pred.p[:, i], prms) - h0) / abs(h0)
            for i in 1:n_train]
    maximum(abs, errs)
end

const n_epochs = 100000

# ── SympNet sweep: vary n_layers (default upscaling) ─────────────────────────
const symp_n_layers = [1, 2, 4, 8, 16]

symp_n_params = Int[]
symp_errors = Float64[]
symp_train_errors = Float64[]

for nl in symp_n_layers
    seed!(123)
    arch = GSympNet(2; n_layers=nl)
    nn = NeuralNetwork(arch)
    Optimizer(AdamOptimizer(), nn)(nn, dl, Batch(1), n_epochs)
    np = parameterlength(nn)
    err = peak_energy_error(nn)
    train_err = peak_train_energy_error(nn)
    push!(symp_n_params, np)
    push!(symp_errors, err)
    push!(symp_train_errors, train_err)
    println("SympNet n_layers=$nl  params=$np  peak_err=$(round(err; sigdigits=3))  train_err=$(round(train_err; sigdigits=3))")
end

# ── ResNet sweep: vary hidden width ───────────────────────────────────────────
const resnet_widths = [0, 1, 2, 4, 8, 16]

resnet_n_params = Int[]
resnet_errors = Float64[]
resnet_train_errors = Float64[]

for w in resnet_widths
    seed!(123)
    arch = ResNet(2, w, tanh)
    nn = NeuralNetwork(arch)
    Optimizer(AdamOptimizer(), nn)(nn, dl, Batch(1), n_epochs)
    np = parameterlength(nn)
    err = peak_energy_error(nn)
    train_err = peak_train_energy_error(nn)
    push!(resnet_n_params, np)
    push!(resnet_errors, err)
    push!(resnet_train_errors, train_err)
    println("ResNet width=$w  params=$np  peak_err=$(round(err; sigdigits=3))  train_err=$(round(train_err; sigdigits=3))")
end

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(size=(800, 380))
ax = Axis(fig[1, 1];
    xlabel="Number of parameters",
    ylabel=L"\max_{t\in[0,\,100]}\,|[H(t)-H(0)]\,/\,H(0)|",
    title=L"\text{Peak energy error on test IC $(q_0/2, p_0/2)$ vs. network size}",
    xscale=log10,
    yscale=log10)

scatterlines!(ax, symp_n_params, symp_errors;
    color=:orange, linewidth=2, markersize=10,
    label=L"\text{SympNet — test IC  ($n_\mathrm{layers}$ varies)}")
scatterlines!(ax, symp_n_params, symp_train_errors;
    color=:orange, linewidth=2, markersize=10, linestyle=:dash,
    label="SympNet — train IC")
scatterlines!(ax, resnet_n_params, resnet_errors;
    color=:purple, linewidth=2, markersize=10,
    label=L"\text{ResNet — test IC  ($n_\mathrm{layers}$ varies)}")
scatterlines!(ax, resnet_n_params, resnet_train_errors;
    color=:purple, linewidth=2, markersize=10, linestyle=:dash,
    label="ResNet — train IC")

axislegend(ax; position=:lb)
CairoMakie.save(joinpath(plotdir, "ho_scaling_energy_error.png"), fig)
println("Saved → ho_scaling_energy_error.png")
