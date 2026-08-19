using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!
import GeometricProblems.Pendulum.q₀ as _q₀
import GeometricProblems.Pendulum.p₀ as _p₀

seed!(123)

const timestep = 0.1
const tspan    = (0.0, 50.0)

# Figures are written next to this script, i.e. into the talk folder.
const outdir   = @__DIR__
const animbase = joinpath(outdir, "Animations")
# Set to true to also render one frame per time step for the animated slides.
# (Disabled for the initial training run — enable once SympNet quality is confirmed.)
const make_animation = false

# ---- Ensemble configuration ------------------------------------------------
# Three pendulum lengths × six initial angles (p₀ = 0, starting at rest).
# These 18 reference trajectories are used for evaluation / plotting only.
const l_values    = [0.5, 1.0, 2.0]
const ic_angles   = [acos(0.8), acos(0.5), acos(0.2), acos(-0.1), acos(-0.4), acos(-0.7)]
const params_list = [(l = l, m = 1.0, g = 1.0) for l in l_values]

# ---- Training IC grid (l = 1.0, broader phase-space coverage) --------------
# A 2-D grid in (θ, p_angular) to cover the libration region uniformly, not
# just the zero-velocity curve (p = 0).  Angles span from small oscillations
# near the stable equilibrium (θ ≈ π) out to large-amplitude oscillations
# near the separatrix; momenta are sampled at 0, ±40 %, and ±75 % of the
# separatrix momentum at each angle.
const train_n_angles     = 10
const train_n_moms       = 5
const train_angles_grid  = range(π - 2.5, π - 0.15, length = train_n_angles)
const mom_fracs_grid     = [0.0, 0.40, 0.75, -0.40, -0.75]
const train_tspan        = (0.0, 30.0)   # shorter per-IC; compensated by 5× more ICs

# ---- Helper functions ------------------------------------------------------

# Angular → Euclidean (canonical, m = g = 1):
#   q_eucl = l · (sinθ, cosθ)
#   p_eucl = (p_angular / l) · (cosθ, -sinθ)    [p_angular = l²·θ̇  for m = 1]
function angular_to_euclidean(θ::Vector, p_ang::Vector, l::Float64)
    q_eucl = l    .* hcat(sin.(θ), cos.(θ))'
    p_eucl = (1/l) .* hcat(cos.(θ), -sin.(θ))' .* p_ang'
    return q_eucl, p_eucl
end

# Euclidean → angular:  θ = atan(x, y),  θ̇ = (y·px - x·py) / (x² + y²)
# Valid for any l; the denominator (= l²) naturally cancels the l-scaling of
# p_eucl so the result is the angular velocity θ̇ (not the canonical momentum).
function convert_euclidean_to_angular(q_eucl::Matrix, p_eucl::Matrix)
    x,  y  = q_eucl[1, :], q_eucl[2, :]
    px, py = p_eucl[1, :], p_eucl[2, :]
    θ  = atan.(x, y)
    θ̇ = (y .* px .- x .* py) ./ (x .^ 2 .+ y .^ 2)
    return θ, θ̇
end

# Euclidean Hamiltonian H = ½‖p_eucl‖² + y  (m = g = 1, l-independent).
# Derivation: T = ½m‖v‖² = ‖p_eucl‖²/(2m) → ½‖p_eucl‖² for m = 1;
#             V = mgl·cosθ = g·q_eucl[2] → q_eucl[2] for g = 1.
function hamiltonian_euclidean(q_eucl::Matrix, p_eucl::Matrix)
    0.5 .* (p_eucl[1, :] .^ 2 .+ p_eucl[2, :] .^ 2) .+ q_eucl[2, :]
end

compute_energy_error(h) = (h .- h[1]) ./ abs.(h[1])

# ---- Generate ensemble reference trajectories ------------------------------
# For each IC angle, create a HODEEnsemble over the three lengths, then
# integrate all three trajectories at once.

println("Generating reference trajectories ...")

# Each entry: (label, l, q₀_angle, nsteps, t, θ, θ̇, q_eucl, p_eucl)
ref_trajectories = []

for q₀_angle in ic_angles
    ens  = hodeensemble([q₀_angle], [0.0], params_list;
                        timespan = tspan, timestep = timestep)
    sols = integrate(ens, Gauss(2))

    for (j, l) in enumerate(l_values)
        sol     = sols[j]
        ns      = length(sol.t)
        t_vec   = collect(sol.t)
        θ_vec   = [sol.q[i][1] for i in 0:ns-1]
        pang    = [sol.p[i][1] for i in 0:ns-1]   # canonical p = l²·θ̇ (m=1)
        θ̇_vec  = pang ./ l^2                       # angular velocity

        q_eucl, p_eucl = angular_to_euclidean(θ_vec, pang, l)

        label = "l$(l)_q$(round(q₀_angle; digits=2))"
        push!(ref_trajectories, (
            label  = label,
            l      = l,
            q₀     = q₀_angle,
            nsteps = ns,
            t      = t_vec,
            θ      = θ_vec,
            θ̇     = θ̇_vec,
            q_eucl = q_eucl,
            p_eucl = p_eucl,
        ))
    end
end

println("Generated $(length(ref_trajectories)) reference trajectories.")

# ---- Build training DataLoader (Euclidean coordinates) ---------------------
# We train only on l=1.0: the time-step flow map depends on l (pendulum
# l=0.5 oscillates at √2× the frequency of l=1.0), so mixing lengths is
# ill-posed for a single-network integrator.
#
# The training IC grid covers both angle and initial momentum, giving the
# SympNet phase-space coverage throughout the libration region rather than
# just along the zero-velocity curve.
println("Generating training trajectories (l=1.0, phase-space grid) ...")
train_q = Matrix{Float64}(undef, 2, 0)
train_p = Matrix{Float64}(undef, 2, 0)

for θ₀ in train_angles_grid
    # Canonical angular momentum at the separatrix (l=1, m=1):  p_sep = sqrt(2*(1-cosθ))
    p_sep = sqrt(max(2.0 * (1.0 - cos(θ₀)), 0.0))
    for frac in mom_fracs_grid
        p₀_ang = p_sep * frac
        ens    = hodeensemble([θ₀], [p₀_ang], [(l = 1.0, m = 1.0, g = 1.0)];
                              timespan = train_tspan, timestep = timestep)
        sol    = integrate(ens, Gauss(2))[1]
        ns     = length(sol.t)
        θ_v    = [sol.q[i][1] for i in 0:ns-1]
        p_ang  = [sol.p[i][1] for i in 0:ns-1]
        q_e, p_e = angular_to_euclidean(θ_v, p_ang, 1.0)
        train_q  = hcat(train_q, q_e)
        train_p  = hcat(train_p, p_e)
    end
end

dl = DataLoader((q = train_q, p = train_p))
println("Training data: $(size(train_q, 2)) samples ($(train_n_angles * train_n_moms) ICs, l=1.0).")

# ---- Train the networks ----------------------------------------------------
n_epochs = 100000
batch    = Batch(100)

println("Training SympNet ...")
arch1 = GSympNet(4; n_layers = 8)
nn1   = NeuralNetwork(arch1)
o1    = Optimizer(AdamOptimizer(), nn1)
loss1 = o1(nn1, dl, batch, n_epochs)
println("SympNet parameters: ", parameterlength(nn1))

println("Training ResNet ...")
arch2 = ResNet(4, 4, tanh)
nn2   = NeuralNetwork(arch2)
o2    = Optimizer(AdamOptimizer(), nn2)
loss2 = o2(nn2, dl, batch, n_epochs)
println("ResNet parameters:  ", parameterlength(nn2))

# ---- Loss comparison plot --------------------------------------------------
fig_loss = Figure(size = (800, 400))
ax_loss  = Axis(fig_loss[1, 1], xlabel = "Epoch", ylabel = "Loss",
                title = "Training Loss Comparison (Ensemble)")
lines!(ax_loss, loss1;
       label = L"SympNet; $n_\mathrm{params}$=%$(parameterlength(nn1))",
       color = :orange, linewidth = 2)
lines!(ax_loss, loss2;
       label = L"ResNet; $n_\mathrm{params}$=%$(parameterlength(nn2))",
       color = :purple, linewidth = 2)
axislegend(ax_loss, position = :rt)
save(joinpath(outdir, "pendulum_loss_comparison.png"), fig_loss)

# ---- Roll out, plot and animate per ensemble case --------------------------
for tr in ref_trajectories
    label  = tr.label
    l      = tr.l
    ns     = tr.nsteps
    t_vec  = tr.t
    θ_ref  = tr.θ
    θ̇_ref = tr.θ̇
    h_ref  = hamiltonian_euclidean(tr.q_eucl, tr.p_eucl)

    # Initial conditions in Euclidean space (p₀_angular = 0 → p_eucl = 0)
    ics = (q = l .* [sin(tr.q₀), cos(tr.q₀)], p = zeros(2))

    pred1 = iterate(nn1, ics; n_points = ns)
    pred2 = iterate(nn2, ics; n_points = ns)

    θ1, θ̇1 = convert_euclidean_to_angular(pred1.q, pred1.p)
    θ2, θ̇2 = convert_euclidean_to_angular(pred2.q, pred2.p)
    h1      = hamiltonian_euclidean(pred1.q, pred1.p)
    h2      = hamiltonian_euclidean(pred2.q, pred2.p)

    # --- Solution plots (static snapshots of the full trajectory) -----------
    println("Saving solution plots for $label ...")
    save(joinpath(outdir, "pendulum_solution_reference_$(label).png"),
         plot_solution(ns, t_vec, θ_ref, θ̇_ref, h_ref, labels_hamiltonian))
    save(joinpath(outdir, "pendulum_solution_sympnet_$(label).png"),
         plot_solution(ns, t_vec, θ1, θ̇1, h1, labels_hamiltonian))
    save(joinpath(outdir, "pendulum_solution_resnet_$(label).png"),
         plot_solution(ns, t_vec, θ2, θ̇2, h2, labels_hamiltonian))

    # --- Energy error plot --------------------------------------------------
    fig_e = Figure(size = (800, 400))
    ax_e  = Axis(fig_e[1, 1], xlabel = "t",
                 ylabel = L"[H(t) - H(0)] / |H(0)|",
                 title  = "Energy Error — $(label)")
    lines!(ax_e, t_vec, compute_energy_error(h_ref);
           label = "Reference", color = :black,  linewidth = 2)
    lines!(ax_e, t_vec, compute_energy_error(h1);
           label = "SympNet",   color = :orange, linewidth = 2)
    lines!(ax_e, t_vec, compute_energy_error(h2);
           label = "ResNet",    color = :purple, linewidth = 2)
    axislegend(ax_e, position = :lt)
    save(joinpath(outdir, "pendulum_energy_error_$(label).png"), fig_e)

    # --- Animation frames (one PNG per time step) ---------------------------
    if make_animation
        for (method, qc, pc, hc) in [
                ("reference", θ_ref, θ̇_ref, h_ref),
                ("sympnet",   θ1,    θ̇1,    h1),
                ("resnet",    θ2,    θ̇2,    h2),
            ]
            dir = joinpath(animbase, "$(method)_$(label)")
            mkpath(dir)
            for f in filter(f -> endswith(f, ".png"), readdir(dir; join = true))
                rm(f)
            end
            println("Animating $(method)_$(label) ($ns frames) → $dir")
            for i in 1:ns
                save(joinpath(dir, "pendulum_$(lpad(string(i), 4, '0')).png"),
                     plot_solution(i, t_vec, qc, pc, hc, labels_hamiltonian))
            end
        end
    end
end

println("All plots saved to $(outdir)")
