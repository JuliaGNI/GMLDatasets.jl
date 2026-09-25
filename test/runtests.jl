using SafeTestsets

@safetestset "MNIST utilities and the classification DataLoader                              " begin
    include("mnist_utils.jl")
end

@safetestset "Docstrings                                                                     " begin
    include("docstrings.jl")
end

@safetestset "Pendulum dataset                                                               " begin
    include("pendulum.jl")
end

# The record schemas the revision harness writes and validates. They live under `scripts/` because
# that is where the harness is, and they need nothing but `SHA` and `Base`, so they run here: a test
# that only CI never runs is a test that rots. The two script suites that additionally need
# `GeometricOptimizers` run from `.github/workflows/Scripts.yml` against `scripts/Project.toml`,
# which is the escape hatch `CI.yml`'s own header prescribes for a repository-specific job.
@safetestset "Revision harness: run artifact schema                                          " begin
    include(joinpath(@__DIR__, "..", "scripts", "revision", "test_run_artifact_schema.jl"))
end

@safetestset "Revision harness: retraction record schema                                     " begin
    include(joinpath(@__DIR__, "..", "scripts", "revision", "test_retraction_record_schema.jl"))
end
