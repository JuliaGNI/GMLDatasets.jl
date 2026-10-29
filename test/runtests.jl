using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "MNIST utilities and the classification DataLoader" include("mnist_utils.jl")
    @safetestset "Docstrings" include("integration/docstrings.jl")
    @safetestset "Pendulum dataset" include("pendulum.jl")
    @safetestset "Linear wave dataset" include("linear_wave.jl")
    @safetestset "MNIST tutorial figure" include("quality/mnist_figure.jl")
    @safetestset "Revision harness: run artifact schema" include("../scripts/revision/test_run_artifact_schema.jl")
    @safetestset "Revision harness: retraction record schema" include("../scripts/revision/test_retraction_record_schema.jl")
end
if "doctests" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
