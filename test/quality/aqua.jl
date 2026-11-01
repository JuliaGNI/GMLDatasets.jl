using Aqua
using GMLDatasets
using Test

# `piracies` fails and is marked broken, with its issue: the `DataLoader` constructor for labelled
# images in `src/data_loader.jl` adds a method to a `GeometricMachineLearning` type for argument
# types that this package does not own.
Aqua.test_all(GMLDatasets;
    piracies = (broken = true,))   # issue #29
