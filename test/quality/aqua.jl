using Aqua
using GMLDatasets
using Test

# Two checks fail and are marked broken, each with its issue:
#
# - `piracies`: the `DataLoader` constructor for labelled images in `src/data_loader.jl` adds a
#   method to a `GeometricMachineLearning` type for argument types that this package does not own.
# - `persistent_tasks`: Aqua 0.8.18 throws before the check runs, because the dependency
#   `InternedStrings` (through `MLDatasets` and `Pickle`) has no `Project.toml`.
Aqua.test_all(GMLDatasets;
    piracies = (broken = true,),           # issue #29
    persistent_tasks = (broken = true,))   # issue #30
