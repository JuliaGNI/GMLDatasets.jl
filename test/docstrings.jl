using GMLDatasets
using Test

# The examples in the docstrings of `onehotbatch` and `split_and_flatten` are `jldoctest` blocks, so
# Documenter checks them when the documentation is built. They are checked here as well so that
# `Pkg.test()` catches a change to either function without a docs build.
@testset "Docstring examples" begin
    @test onehotbatch([0]) == reshape([1, zeros(Int, 9)...], 10, 1, 1)

    input = [ 1  2  3  4  5  6;
              7  8  9 10 11 12;
             13 14 15 16 17 18;
             19 20 21 22 23 24;
             25 26 27 28 29 30;
             31 32 33 34 35 36]

    expected_patches = [ 1 19  4 22;
                         7 25 10 28;
                        13 31 16 34;
                         2 20  5 23;
                         8 26 11 29;
                        14 32 17 35;
                         3 21  6 24;
                         9 27 12 30;
                        15 33 18 36]

    @test split_and_flatten(input; patch_length = 3, number_of_patches = 4) == expected_patches
end
