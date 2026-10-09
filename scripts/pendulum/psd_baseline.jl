# psd_baseline.jl — proper symplectic decomposition on the paper's pendulum data
#
# The linear baseline for Section 7.2.5 of the symplectic-autoencoder paper. `PSDArch(4, 2)` is the
# cotangent-lift PSD of GeometricMachineLearning, A = diag(Φ, Φ) with Φ ∈ St(1, 2), and `solve!`
# fits it by SVD; no training is involved. It is evaluated on the same fifteen level sets and with
# the same metrics as `latent_plot.jl` evaluates the SAE, so the two tables can be read side by side:
#
#   action      ∮ pθ dθ, analytic
#   latent      ∮ p dq of the latent loop A⁺x, which equals ∮ p·dq of the decoded loop because A is
#               symplectic -- checked in the last column
#   gap         |latent - action| / action
#   recon       relative L2 reconstruction error ‖AA⁺x - x‖ / ‖x‖ on the level set
#
# The data set is the base grid as the paper's run integrated it, not as `train_sae.jl` does: ten
# angles in [π - 5/2, π - 3/20], the eight momentum fractions {0, ±2/5, ±3/4, -1, -2, -3} of the
# separatrix momentum, integrated to t = 10 with Δt = 1/10. That is the grid the paper's SAE numbers
# were measured on; `SAE_FRACS`, `SAE_TSPAN` do not apply here.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/psd_baseline.jl
#
# Needs GeometricMachineLearning, PoincareInvariants and GMLDatasets; no weights file.

using GeometricMachineLearning, PoincareInvariants, Printf, LinearAlgebra

import GMLDatasets: angular_to_euclidean, pendulum, pendulum_energy

const l = 1.0

# ---- the base grid, as the paper's run integrated it ---------------------------------------------
solution = pendulum(; qmin = [π - 5 / 2], qmax = [π - 3 / 20], qsamples = [10],
    momentum_fractions = [0, 2 / 5, -2 / 5, 3 / 4, -3 / 4, -1, -2, -3],
    timespan = (0.0, 10.0), timestep = 0.1)
energies = pendulum_energy(solution)[1, :]
data = angular_to_euclidean(solution)                  # 4 × 101 × 80
println("$(size(data, 3)) trajectories, $(size(data, 2)) snapshots each, H ∈ [",
    round(minimum(energies); digits = 3), ", ", round(maximum(energies); digits = 3), "]")

# ---- PSD ----------------------------------------------------------------------------------------
psd = NeuralNetwork(PSDArch(4, 2), CPU(), Float64)
training_loss = solve!(psd, data)
enc, dec = encoder(psd), decoder(psd)
Φ = GeometricMachineLearning.params(psd)[1].weight.A                          # the 2 × 1 Stiefel factor of A = diag(Φ, Φ)
A = [Φ zeros(2, 1); zeros(2, 1) Φ]
@printf("\nPSD fitted by SVD: Φ = (%.6f, %.6f), training loss (AutoEncoderLoss) %.4f\n",
    Φ[1], Φ[2], training_loss)
@printf("symplecticity of A:  ‖AᵀJ₄A - J₂‖ = %.1e\n",
    norm(A' * [0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0] * A - [0 1; -1 0]))
# The reduced Hamiltonian is H∘A = ½|Φ p_r|² + (Φ q_r)₂ = ½ p_r² + Φ₂ q_r: a particle in a uniform
# field whenever Φ₂ ≠ 0 and a free particle when Φ₂ = 0. Either way no orbit is closed.
@printf("reduced Hamiltonian: H(Az) = ½ p_r² + (%.6f) q_r\n", Φ[2])

# ---- the fifteen level sets of latent_plot.jl ---------------------------------------------------
lift(θ, pθ) = [l * sin(θ), l * cos(θ), cos(θ) * pθ / l, -sin(θ) * pθ / l]

"A level set of H, sampled uniformly in its own angle; rotating orbits take the p < 0 branch."
function level_set(H; n)
    s = 2π .* range(0, 1; length = n + 1)[1:(end - 1)]
    if H < 1
        k = sqrt((1 + H) / 2)
        (π .+ 2 .* asin.(clamp.(k .* sin.(s), -1, 1)), 2 .* k .* cos.(s),
         2 .* k .* cos.(s) ./ sqrt.(max.(1 .- (k .* sin.(s)) .^ 2, eps())))
    else
        (collect(s), -sqrt.(2 .* (H .- cos.(s))), fill(1.0, n))
    end
end

pinv2(Z) = compute!(CanonicalFirstPI{Float64, 2}(length(Z)), Matrix(reduce(hcat, Z)'))
pinv4(X) = compute!(CanonicalFirstPI{Float64, 4}(length(X)), Matrix(reduce(hcat, X)'))

function relerr(θs, pθs)
    num = 0.0; den = 0.0
    for (a, b) in zip(θs, pθs)
        x = lift(a, b); r = dec(enc(x))
        num += sum(abs2, r .- x); den += sum(abs2, x)
    end
    sqrt(num / den)
end

function table(title, Hs; n = 16_000)
    println("\n", title)
    println("      H   regime    action ∮pθdθ   latent ∮p dq   action gap   recon (rel L2)   sympl.")
    println("  " * "─"^90)
    gaps = Float64[]; recs = Float64[]
    for H in Hs
        θs, pθs, dθ = level_set(H; n)
        action = abs(sum(pθs .* dθ) * (2π / n))
        X = [lift(a, b) for (a, b) in zip(θs, pθs)]
        Z = [enc(x) for x in X]
        latent = abs(pinv2(Z)); decoded = abs(pinv4([dec(z) for z in Z]))
        gap = abs(latent - action) / action; rec = relerr(θs[1:8:end], pθs[1:8:end])
        push!(gaps, gap); push!(recs, rec)
        @printf("  %6.2f  %s  %12.5f  %13.5f   %8.1f%%   %11.1f%%   %8.1e\n",
            H, H < 1 ? "librate" : "rotate ", action, latent, 100gap, 100rec,
            latent == 0 ? abs(decoded) : abs(latent - decoded) / latent)
    end
    gaps, recs
end

const H_levels = collect(range(-0.4, 1.4; length = 15))     # as in latent_plot.jl
gaps, recs = table("THE PAPER'S FIFTEEN LEVEL SETS (Figure 6)", H_levels)
lib = H_levels .< 1
@printf("\n  recon:       librating %.1f%%-%.1f%%   rotating %.1f%%-%.1f%%\n",
    100minimum(recs[lib]), 100maximum(recs[lib]), 100minimum(recs[.!lib]), 100maximum(recs[.!lib]))
@printf("  action gap:  librating %.1f%%-%.1f%%   rotating %.1f%%-%.1f%%\n",
    100minimum(gaps[lib]), 100maximum(gaps[lib]), 100minimum(gaps[.!lib]), 100maximum(gaps[.!lib]))

# The rotating energies the training data actually cover, where the paper quotes the SAE at 9.5%-14%.
gc, rc = table("ROTATING, p_theta < 0, AT ENERGIES THE DATA COVER",
    [1.60, 1.75, 2.00, 2.50, 4.00, 8.00, 16.91])
@printf("\n  recon:       %.1f%%-%.1f%%   action gap: %.1f%%-%.1f%%\n",
    100minimum(rc), 100maximum(rc), 100minimum(gc), 100maximum(gc))
