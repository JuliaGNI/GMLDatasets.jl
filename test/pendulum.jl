using Test
using GMLDatasets
using GeometricMachineLearning

@testset "Pendulum coordinate conversion" begin
    angle = [-1.2, 0.0, 0.8]
    momentum = [0.2, -0.4, 0.7]
    q, p = angular_to_euclidean(angle, momentum; length=2.0)
    @test size(q) == (2, 3)
    @test size(p) == (2, 3)
    @test vec(sum(abs2, q; dims=1)) ≈ fill(4.0, 3)
    angle_back, momentum_back = euclidean_to_angular(q, p; length=2.0)
    @test angle_back ≈ angle
    @test momentum_back ≈ momentum
end

@testset "Pendulum trajectory" begin
    trajectory = pendulum_trajectory(; timespan=(0.0, 0.4), timestep=0.1, angle=0.6, momentum=0.2)
    @test trajectory isa PendulumTrajectory
    @test length(trajectory) == 5
    @test size(trajectory.q) == (2, 5)
    @test size(trajectory.p) == (2, 5)
    @test trajectory.t == collect(0.0:0.1:0.4)
    @test trajectory.q[:, 1] ≈ vec(first(angular_to_euclidean([0.6], [0.2])))
    reference_energy = pendulum_energy(trajectory.angle, trajectory.momentum)
    @test maximum(abs.(reference_energy .- first(reference_energy))) < 1e-5
    @test pendulum_energy(trajectory.q, trajectory.p) ≈ reference_energy
end

@testset "Pendulum dataset layout" begin
    dataset = pendulum_dataset([(0.2, 0.0), (0.8, 0.1)]; timespan=(0.0, 0.2), timestep=0.1)
    @test dataset isa PendulumDataset
    @test length(dataset) == 2
    matrices = pendulum_matrix(dataset; concatenate=false)
    @test length(matrices) == 2
    @test size(first(matrices)) == (4, 3)
    @test size(pendulum_matrix(dataset)) == (4, 6)
    @test_throws ArgumentError pendulum_dataset([]; timespan=(0.0, 0.2), timestep=0.1)
    @test_throws ArgumentError pendulum_trajectory(; length=0)
end

@testset "Pendulum DataLoader and SAE" begin
    dataset = pendulum_dataset([(0.2, 0.0), (0.8, 0.1)]; timespan=(0.0, 0.2), timestep=0.1)
    loader = pendulum_data_loader(dataset; suppress_info=true)
    @test loader.input_dim == 4
    @test size(loader.input) == (4, 1, 6)

    architecture = SymplecticAutoencoder(4, 2;
        n_encoder_blocks=2,
        n_decoder_blocks=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_decoder_output_layers=2,
        sympnet_upscale=2,
    )
    network = NeuralNetwork(architecture, CPU(), Float64)
    prediction = network(pendulum_matrix(dataset), network.params)
    @test size(prediction) == (4, 6)
    @test all(isfinite, prediction)

    optimizer = Optimizer(Adam(), network; step_size=1f-3)
    losses = optimizer(network, loader, Batch(6), 1)
    @test length(losses) == 1
    @test isfinite(first(losses))
end
