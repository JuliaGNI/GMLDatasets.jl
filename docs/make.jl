using GMLDatasets
using GeometricMachineLearning
using Documenter
using DocumenterCitations
import Bibliography

bib = CitationBibliography(joinpath(@__DIR__, "src", "GMLDatasets.bib"))
Bibliography.sort_bibliography!(bib.entries, :nyt)  # name-year-title

# The MNIST tutorial loads the data set at build time, which prompts through DataDeps unless this is
# set. It is set here rather than only in CI so that a local build behaves the same way.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# The trained weights the MNIST tutorial reads live in `docs/artifacts/` rather than under
# `docs/src/`, because Documenter copies everything in `src` verbatim into `build` and `deploydocs`
# then publishes it: 3 MB of HDF5 that only the build reads and no reader of the site ever
# downloads. They are inputs to the build, not part of it.
#
# The path travels through the environment because an `@setup` block cannot see anything defined
# here, and because the alternative -- a path relative to the block's working directory, which is
# the built page's own directory -- would silently break if the tutorial ever moved.
ENV["GMLDATASETS_DOCS_ARTIFACTS"] = joinpath(@__DIR__, "artifacts")

DocMeta.setdocmeta!(GMLDatasets, :DocTestSetup, :(using GMLDatasets); recursive=true)

makedocs(;
    plugins=[bib],
    modules=[GMLDatasets],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/GMLDatasets.jl/blob/{commit}{path}#{line}",
    sitename="GMLDatasets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/GMLDatasets.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "MNIST Tutorial" => "mnist/mnist_tutorial.md",
        "The Numerical Experiment on Homogeneous Spaces" => "homogeneous_spaces_experiment.md",
        "Running the Experiments" => "running_the_experiments.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/GMLDatasets.jl",
    devurl="latest",
    devbranch="main",
)
