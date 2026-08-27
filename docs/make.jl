using GMLDatasets
using GeometricMachineLearning
using Documenter
using DocumenterCitations
using DocumenterInterLinks
import Bibliography

bib = CitationBibliography(joinpath(@__DIR__, "src", "GMLDatasets.bib"))
Bibliography.sort_bibliography!(bib.entries, :nyt)  # name-year-title

# `@extref` links to the upstream API documentation. Each entry is the root URL of the deployed
# documentation, the inventory to read the anchors from, and a committed copy of that inventory to
# fall back on when the network is unavailable.
#
# The **trailing slash on the root URL is required**. `DocumenterInterLinks` concatenates the root
# with the relative URI the inventory records, so without it `DataLoader`'s
# `data_loader/data_loader/#$` becomes `.../latestdata_loader/data_loader/#...`, which is a 404 that
# nothing in the build checks: `@extref` resolution only verifies that the *name* is in the
# inventory, never that the assembled URL exists.
#
# Refresh the fallbacks with
#     julia -e 'using DocInventories; save("docs/inventories/<Name>.toml", Inventory(<url>; root_url = ""))'
links = InterLinks(
    "GeometricMachineLearning" => (
        "https://juliagni.github.io/GeometricMachineLearning.jl/latest/",
        "https://juliagni.github.io/GeometricMachineLearning.jl/latest/objects.inv",
        joinpath(@__DIR__, "inventories", "GeometricMachineLearning.toml")
    ),
    "GeometricSolutions" => (
        "https://juliagni.github.io/GeometricSolutions.jl/stable/",
        "https://juliagni.github.io/GeometricSolutions.jl/stable/objects.inv",
        joinpath(@__DIR__, "inventories", "GeometricSolutions.toml")
    ),
)

# The MNIST tutorial loads the data set at build time, which prompts through DataDeps unless this is
# set. It is set here rather than only in CI so that a local build behaves the same way.
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

DocMeta.setdocmeta!(GMLDatasets, :DocTestSetup, :(using GMLDatasets); recursive=true)

makedocs(;
    plugins=[bib, links],
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
        "The Pendulum Data Set" => "pendulum.md",
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
