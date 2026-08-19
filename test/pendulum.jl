using GMLDatasets
using GeometricMachineLearning
using Test

import GeometricProblems.Pendulum as Pendulum

# Deliberately not the defaults: every quantity below is wrong by a factor of `l`, `m` or `g` if the
# parameters are dropped anywhere between `pendulum` and the coordinate conversions.
const parameters = (l=2.0, m=3.0, g=9.81)

@testset "Coordinate lift" begin
    θ = [-1.2, 0.0, 0.8, 3.0]
    pθ = [0.2, -0.4, 0.7, 0.0]

    for l in (1, 2.0)
        q, p = angular_to_euclidean(θ, pθ; l=l)
        @test size(q) == (2, 4)
        @test size(p) == (2, 4)

        # The image is the tangent bundle of the circle of radius `l`: ‖q‖ = l and q ⟂ p.
        @test vec(sum(abs2, q; dims=1)) ≈ fill(l^2, 4)
        @test vec(sum(q .* p; dims=1)) ≈ zeros(4) atol = 1e-14

        θ_back, pθ_back = euclidean_to_angular(q, p; l=l)
        @test θ_back ≈ θ
        @test pθ_back ≈ pθ
    end

    @test_throws ArgumentError angular_to_euclidean(θ, pθ; l=0)
    @test_throws ArgumentError euclidean_to_angular(zeros(2, 3), zeros(2, 3); l=-1)
    @test_throws DimensionMismatch angular_to_euclidean(θ, pθ[1:2])
    @test_throws DimensionMismatch euclidean_to_angular(zeros(3, 4), zeros(3, 4))
    @test_throws DimensionMismatch euclidean_to_angular(zeros(2, 4), zeros(2, 3))
end

@testset "Hamiltonian" begin
    θ = [-1.2, 0.0, 0.8, 3.0]
    pθ = [0.2, -0.4, 0.7, 0.0]

    # The canonical method must stay `GeometricProblems`' Hamiltonian rather than a copy of it, and
    # the Euclidean one must be the same function written in the lifted coordinates.
    @test pendulum_energy(θ, pθ, parameters) ==
          Pendulum.hamiltonian.(0.0, θ, pθ, Ref(parameters))

    q, p = angular_to_euclidean(θ, pθ; l=parameters.l)
    @test pendulum_energy(q, p, parameters) ≈ pendulum_energy(θ, pθ, parameters)
    @test pendulum_energy(vcat(q, p), parameters) ≈ pendulum_energy(θ, pθ, parameters)

    @test_throws DimensionMismatch pendulum_energy(zeros(3, 4), zeros(3, 4))
    @test_throws DimensionMismatch pendulum_energy(zeros(2, 4), zeros(2, 3))
    @test_throws DimensionMismatch pendulum_energy(zeros(3, 4))
end

@testset "Ensemble" begin
    solution = pendulum(; qmin=[2.0], qmax=[4.0], pmin=[0.0], pmax=[0.5],
        qsamples=[3], psamples=[2], parameters=parameters,
        timespan=(0.0, 2.0), timestep=0.1)
    @test length(solution.s) == 6
    @test length(solution.t) == 21

    # One array, rows (q₁, q₂, p₁, p₂), one column per time step and one slice per trajectory.
    data = angular_to_euclidean(solution)
    @test size(data) == (4, 21, 6)

    # `l` is read off the problem, not assumed to be 1.
    @test vec(sum(abs2, data[1:2, :, :]; dims=1)) ≈ fill(parameters.l^2, 21 * 6)
    @test vec(sum(data[1:2, :, :] .* data[3:4, :, :]; dims=1)) ≈ zeros(21 * 6) atol = 1e-14

    # The energy of the lifted data agrees with the energy of the canonical solution, which is the
    # check that the parameters survive the lift, and Gauss collocation is symplectic, so it barely
    # drifts over the run.
    energy = pendulum_energy(solution)
    @test size(energy) == (21, 6)
    @test pendulum_energy(data, parameters) ≈ energy
    @test maximum(abs, energy .- energy[1:1, :]) < 1e-4

    @test angular_to_euclidean(first(solution.s)) ≈ data[:, :, 1:1]
    @test pendulum_energy(first(solution.s)) ≈ energy[:, 1]
end

@testset "DataLoader and SymplecticAutoencoder" begin
    solution = pendulum(; qmin=[2.0], qmax=[4.0], pmin=[0.0], pmax=[0.5],
        qsamples=[3], psamples=[2], timespan=(0.0, 2.0), timestep=0.1)
    data = angular_to_euclidean(solution)

    # No adapter of our own: this is `GeometricMachineLearning`'s own tensor constructor.
    loader = DataLoader(data; autoencoder=true, suppress_info=true)
    @test loader.input_dim == 4
    @test loader.input_time_steps == 21
    @test loader.n_params == 6

    # The four rows are (q₁, q₂, p₁, p₂), i.e. q first and p second, which is the layout every
    # symplectic architecture in `GeometricMachineLearning` assumes.
    architecture = SymplecticAutoencoder(loader.input_dim, 2;
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_decoder_output_layers=2,
        sympnet_upscale=2,
    )
    network = NeuralNetwork(architecture, CPU(), eltype(loader))
    input = data[:, :, 1]
    prediction = network(input, network.params)
    @test size(prediction) == size(input)
    @test all(isfinite, prediction)

    # One epoch, to pin down that the data set as handed over is actually trainable — the layout is
    # the whole point, and `AutoEncoderLoss` is picky about what it is given.
    optimizer = Optimizer(Adam(), network; step_size=1f-3)
    losses = optimizer(network, loader, Batch(32), 1)
    @test length(losses) == 1
    @test isfinite(first(losses))
end
