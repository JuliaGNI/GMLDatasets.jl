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
        "Pendulum Dataset" => "pendulum.md",
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
