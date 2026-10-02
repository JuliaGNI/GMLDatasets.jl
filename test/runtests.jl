using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "MNIST utilities and the classification DataLoader" include("mnist_utils.jl")
    @safetestset "Docstrings" include("docstrings.jl")
    @safetestset "Pendulum dataset" include("pendulum.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
