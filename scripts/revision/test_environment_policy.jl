using Test

include(joinpath(@__DIR__, "environment_policy.jl"))

function validation_error(julia_version, gml_version, go_version, nnp_version)
    try
        validate_environment_versions(julia_version, gml_version, go_version, nnp_version)
    catch exception
        return exception
    end
    return nothing
end

@testset "revision environment version policy" begin
    @test isnothing(validate_environment_versions(
        v"1.11.0",
        REQUIRED_GML_VERSION,
        REQUIRED_GO_VERSION,
        REQUIRED_NNP_VERSION,
    ))
    @test isnothing(validate_environment_versions(
        v"1.12.6",
        REQUIRED_GML_VERSION,
        REQUIRED_GO_VERSION,
        REQUIRED_NNP_VERSION,
    ))

    rejected_stacks = (
        (v"1.10.10", REQUIRED_GML_VERSION, REQUIRED_GO_VERSION, REQUIRED_NNP_VERSION, "Julia"),
        (v"1.11.0", v"0.6.2", REQUIRED_GO_VERSION, REQUIRED_NNP_VERSION, "GeometricMachineLearning"),
        (v"1.11.0", REQUIRED_GML_VERSION, v"0.6.1", REQUIRED_NNP_VERSION, "GeometricOptimizers"),
        (v"1.11.0", REQUIRED_GML_VERSION, REQUIRED_GO_VERSION, v"0.2.4", "NeuralNetworkParameters"),
    )
    for (julia_version, gml_version, go_version, nnp_version, rejected_name) in rejected_stacks
        exception = validation_error(julia_version, gml_version, go_version, nnp_version)
        @test exception isa ErrorException
        @test occursin(rejected_name, exception.msg)
        @test occursin("scripts/Manifest.toml", exception.msg)
    end
end
