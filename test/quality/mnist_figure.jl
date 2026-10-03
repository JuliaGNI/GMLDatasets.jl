using GMLDatasets
using Test

# The MNIST tutorial draws its preprocessing figure with CairoMakie during the docs build, once per
# Documenter theme. These tests read the tutorial's source, so they run without CairoMakie and
# without the MNIST download: they check that both themes are drawn and shown, each in the theme's
# own container, and that no figure source or built image sits beside the tutorial.

const MNIST_DOCS = joinpath(pkgdir(GMLDatasets), "docs", "src", "mnist")
const TUTORIAL = read(joinpath(MNIST_DOCS, "mnist_tutorial.md"), String)

# The Markdown between `<div class="docs-<theme>-only">` and the next `</div>`.
function theme_containers(text, theme)
    pattern = Regex("<div class=\"docs-$(theme)-only\">(.*?)</div>", "s")
    return [m.captures[1] for m in eachmatch(pattern, text)]
end

# The code of every `@setup` block of the named sandbox.
function setup_blocks(text, sandbox)
    pattern = Regex("```@setup $(sandbox)\\n(.*?)```", "s")
    return [m.captures[1] for m in eachmatch(pattern, text)]
end

@testset "The MNIST figure is drawn for both themes" begin
    blocks = setup_blocks(TUTORIAL, "mnist_visualization")
    @test length(blocks) == 1
    code = only(blocks)
    @test occursin("include(\"mnist_visualization.jl\")", code)
    @test occursin("(:light, :dark)", code)
    @test occursin("mnist_visualization_\$(theme).png", code)
    @test isfile(joinpath(MNIST_DOCS, "mnist_visualization.jl"))
end

@testset "The tutorial shows one image per theme" begin
    for (theme, other) in (("light", "dark"), ("dark", "light"))
        containers = theme_containers(TUTORIAL, theme)
        figure = filter(c -> occursin("mnist_visualization_", c), containers)
        @test length(figure) == 1
        @test occursin("](mnist_visualization_$(theme).png)", only(figure))
        @test !occursin("mnist_visualization_$(other).png", only(figure))

        loss = filter(c -> occursin("mnist_training_loss_", c), containers)
        @test length(loss) == 1
        @test occursin("](mnist_training_loss_$(theme).png)", only(loss))
    end

    # Every Markdown image of a themed figure sits inside the container of its theme.
    for name in ("mnist_visualization", "mnist_training_loss"), theme in ("light", "dark")

        image = "]($(name)_$(theme).png)"
        @test count(image, TUTORIAL) ==
              count(image, join(theme_containers(TUTORIAL, theme)))
    end
end

@testset "No figure source and no image beside the tutorial" begin
    files = readdir(MNIST_DOCS)
    @test !any(f -> endswith(f, ".tex"), files)
    @test "Makefile" ∉ files
    @test !any(f -> any(e -> endswith(f, e), (".png", ".pdf", ".svg")), files)
end
