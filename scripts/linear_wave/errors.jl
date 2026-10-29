# Projection and reduction errors of the linear-wave experiment, for the PSD and for every trained
# symplectic autoencoder, at the test wave speeds LW_TEST_MU (see `common.jl`).
#
#     julia --project=scripts --threads=auto scripts/linear_wave/errors.jl
#
# Run `train_sae.jl` first. For a test value μ, with FOM snapshots x⁽ᵗ⁾, t = 0, …, n_t, encoder 𝒫
# and decoder ℛ,
#
#     e_proj(μ) = √( Σₜ ‖x⁽ᵗ⁾ - ℛ(𝒫(x⁽ᵗ⁾))‖² / Σₜ ‖x⁽ᵗ⁾‖² ),
#     e_red(μ)  = √( Σₜ ‖x⁽ᵗ⁾ - ℛ(x_r⁽ᵗ⁾)‖² / Σₜ ‖x⁽ᵗ⁾‖² ),
#
# where x_r is the solution of the reduced model `HRedSys` (the Hamiltonian system of H∘ℛ) from
# 𝒫(x⁽⁰⁾), integrated with the implicit midpoint rule on the time grid of the FOM. These are
# `GeometricMachineLearning.projection_error` and `reduction_error`.
#
# The reduced Hamiltonian H∘ℛ is a first integral of the reduced model, so its relative drift
# along x_r, `hred_drift`, is reported with each reduction error: a large drift means the reduced
# integration and not the reduction is what the error measures.
#
# Writes `$(LW_OUTPUT_DIR)/errors.csv`, one row per (method, n, seed, μ), and `errors.h5`, the same
# columns with the settings and package versions as attributes. The PSD rows have seed 0.

using CUDA
using GeometricIntegrators: ImplicitMidpoint, integrate

include(joinpath(@__DIR__, "common.jl"))

import GeometricMachineLearning: integrate_reduced_system, projection_error, reduction_error

data = training_data()

# The FOM solution of each test value, integrated once.
full_solutions = Dict(μ => integrate(full_problem(μ), ImplicitMidpoint()) for μ in TEST_MU)
fom_drift = Dict(μ => begin
                     s = full_solutions[μ]
                     H = [LinearWave.hamiltonian(0.0, s.q[k], s.p[k], (μ = μ,))
                          for k in eachindex(s.t)]
                     maximum(abs, H .- first(H)) / abs(first(H))
                 end for μ in TEST_MU)

# Every reduction, as (method, n, seed, network). The PSD has no seed.
reductions = Tuple{String, Int, Int, Any}[]
for n in HALF_REDUCED_DIMS
    push!(reductions, ("PSD", n, 0, fit_psd(data, n)))
    for seed in SEEDS
        path = sae_path(n, seed)
        isfile(path) || error("$path is missing; run train_sae.jl first")
        push!(reductions, ("SAE", n, seed, load(NeuralNetwork, path, architecture(n))))
    end
end

function evaluate(method, n, seed, network, μ)
    problem = full_problem(μ)
    sol_full = full_solutions[μ]
    rs = HRedSys(problem, encoder(network), decoder(network); integrator = ImplicitMidpoint())
    e_proj = projection_error(rs, sol_full)
    e_red, drift = try
        sol_reduced = integrate_reduced_system(rs)
        # H∘ℛ evaluated directly: `rs.h_reduced` calls the decoder on (q, p) as two arguments, which
        # the decoder does not accept.
        dec = decoder(network)
        H = map(eachindex(sol_reduced.t)) do k
            z = dec(vcat(sol_reduced.q[k], sol_reduced.p[k]))
            LinearWave.hamiltonian(0.0, z[1:N], z[(N + 1):end], (μ = μ,))
        end
        (reduction_error(rs, sol_full, sol_reduced),
            maximum(abs, H .- first(H)) / abs(first(H)))
    catch err
        @warn "reduced integration failed" method n seed μ exception = err
        (NaN, NaN)
    end
    (method = method, n = n, seed = seed, mu = μ, projection_error = e_proj,
        reduction_error = e_red, hred_drift = drift, fom_drift = fom_drift[μ])
end

jobs = [(r..., μ) for r in reductions for μ in TEST_MU]
rows = Vector{Any}(undef, length(jobs))
Threads.@threads for i in eachindex(jobs)
    rows[i] = evaluate(jobs[i]...)
    r = rows[i]
    @printf("%-3s n = %2d seed %d μ = %.3f: e_proj = %.3e, e_red = %.3e, H∘ℛ drift %.1e\n",
        r.method, r.n, r.seed, r.mu, r.projection_error, r.reduction_error, r.hred_drift)
end
sort!(rows; by = r -> (r.method, r.n, r.seed, r.mu))

mkpath(OUTPUT_DIR)
columns = keys(first(rows))
open(errors_path("csv"), "w") do io
    println(io, join(columns, ','))
    for r in rows
        println(io, join((v isa AbstractFloat ? repr(v) : string(v) for v in values(r)), ','))
    end
end
HDF5.h5open(errors_path("h5"), "w") do file
    for c in columns
        file[string(c)] = [getproperty(r, c) for r in rows]
    end
    write_metadata(file)
end
println("wrote ", errors_path("csv"), " and ", errors_path("h5"), " ($(length(rows)) rows)")
