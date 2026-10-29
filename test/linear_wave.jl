using GMLDatasets
using GeometricMachineLearning: DataLoader
using Test

import GeometricProblems.LinearWave as LinearWave

# A small lattice and a short run keep this cheap; the defaults are checked separately below without
# integrating them.
const N = 16
const μs = [5 / 12, 1 / 2, 2 / 3]
const n_time_steps = 21

solution = linear_wave(; N = N, μ = μs, n_time_steps = n_time_steps)

@testset "dimensions" begin
    @test length(solution) == length(μs)
    data = linear_wave_snapshots(solution)
    @test size(data) == (2N, n_time_steps, length(μs))
    @test linear_wave_parameters(solution) == μs

    single = linear_wave_snapshots(solution.s[2])
    @test single == data[:, :, 2]

    dl = DataLoader(data; autoencoder = true, suppress_info = true)
    @test dl.input_dim == 2N
    @test dl.input_time_steps == n_time_steps
    @test dl.n_params == length(μs)
end

@testset "initial conditions" begin
    data = linear_wave_snapshots(solution)
    for (k, μ) in pairs(μs)
        ic = LinearWave.initial_condition(μ, N)
        @test data[1:N, 1, k] == ic.q
        @test data[(N + 1):end, 1, k] == ic.p
    end
end

@testset "energy conservation" begin
    # The Hamiltonian is quadratic, so the implicit midpoint rule conserves it to round-off.
    for (k, s) in enumerate(solution)
        params = (μ = μs[k],)
        H = [LinearWave.hamiltonian(0.0, s.q[n], s.p[n], params) for n in eachindex(s.t)]
        @test maximum(abs, H .- first(H)) ≤ 1e-10 * first(H)
    end
end

@testset "one hand-checked wave speed" begin
    # μ = 1/2 at N = 16: the grid is ξᵢ = i/17 - 1/2, and the pulse h(20μ|ξ + μ/2|) is centered at
    # ξ = -1/4 with support |ξ + 1/4| ≤ 1/5. ξ₄ = 4/17 - 1/2 = -9/34 lies inside it, at
    # s = 10 · |-9/34 + 1/4| = 10/68 = 5/34.
    data = linear_wave_snapshots(solution)
    s = 5 / 34
    @test data[4, 1, 2] ≈ 1 - 3 / 2 * s^2 + 3 / 4 * s^3
    # ξ₁₆ = 15/34 is outside the support, so q and p vanish there initially.
    @test data[N, 1, 2] == 0
    @test data[2N, 1, 2] == 0
    # Before the pulse reaches a boundary its total "mass" Σ qᵢ is transported, not created: a
    # coarse lattice disperses the pulse, but the sum stays close to its initial value.
    @test sum(data[1:N, end, 2]) ≈ sum(data[1:N, 1, 2]) rtol = 0.1
end

@testset "arguments" begin
    @test_throws ArgumentError linear_wave(; μ = Float64[])
    @test_throws ArgumentError linear_wave(; n_time_steps = 1)
end
