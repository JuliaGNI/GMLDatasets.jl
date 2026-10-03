using GMLDatasets
using Test

# The MNIST tutorial draws its preprocessing figure with CairoMakie during the docs build, once per
# Documenter theme. These tests read the tutorial's source, so they run without CairoMakie and
# without the MNIST download: they check that the tutorial calls the drawing function once per
# theme, with that theme, and shows each image in the theme's own container. They do not draw the
# figure; the docs build does.

const MNIST_DOCS = joinpath(pkgdir(GMLDatasets), "docs", "src", "mnist")
const TUTORIAL = read(joinpath(MNIST_DOCS, "mnist_tutorial.md"), String)

# The Markdown between `<div class="docs-<theme>-only">` and the next `</div>`.
function theme_containers(text, theme)
    pattern = Regex("<div class=\"docs-$(theme)-only\">(.*?)</div>", "s")
    return [m.captures[1] for m in eachmatch(pattern, text)]
end

# The code of every `@setup` block of the named sandbox.
function setup_blocks(text, sandbox)
    pattern = Regex("```@setup $(sandbox)\\r?\\n(.*?)```", "s")
    return [m.captures[1] for m in eachmatch(pattern, text)]
end

@testset "The tutorial calls the MNIST figure once per theme, with that theme" begin
    blocks = setup_blocks(TUTORIAL, "mnist_visualization")
    @test length(blocks) == 1
    code = only(blocks)
    @test occursin("include(\"mnist_visualization.jl\")", code)
    @test occursin("for theme in (:light, :dark)", code)
    @test occursin("mnist_visualization_\$(theme).png", code)
    @test occursin("mnist_visualization(train_x[:, :, 8]; theme)", code)
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
