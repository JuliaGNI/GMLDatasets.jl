using SafeTestsets

@safetestset "MNIST utilities and the classification DataLoader                              " begin
    include("mnist_utils.jl")
end

@safetestset "Docstrings                                                                     " begin
    include("docstrings.jl")
end

@safetestset "Pendulum dataset                                                              " begin
    include("pendulum.jl")
end
