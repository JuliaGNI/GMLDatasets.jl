using GMLDatasets
using CairoMakie

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

#MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
image_dim = 28
patch_length = 7
n_heads = 7
n_layers = 5
patch_number = (image_dim÷patch_length)^2

train_x, train_y = mnist(:train)

#preprocessing steps 
first_image = train_x[:, :, 8]

function split_image(image::AbstractMatrix, pl)
    n, m = size(image)
    @assert n == m
    @assert n%pl == 0
    #square root of patch number
    pnsq = n ÷ pl
    small_images = Tuple(map(i -> zeros(eltype(image), pl, pl), 1:(pnsq ^ 2)))
    for i in 1:pnsq
        for j in 1:pnsq
            small_images[pnsq * (j - 1) + i] .= image[(pl * (i - 1) + 1):(pl * i), (pl * (j - 1) + 1):(pl * j)]
        end
    end
    #Tuple(vcat(map(j -> map(i -> image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j,1]), 1:pnsq),1:pnsq)...)
    small_images
end

processed_image₁ = split_image(first_image, patch_length)
processed_image₂ = Tuple(map(i -> reshape(processed_image₁[i], 49, 1), 1:16))

fully_processed_image = split_and_flatten(
    first_image; patch_length = patch_length, number_of_patches = patch_number)

#see https://github.com/JuliaImages/ImageView.jl/issues/28
function plot_image!(fig::Figure, pic::AbstractMatrix)
    first_axis, second_axis = axes(pic)
    ax = Axis(fig[1, 1],
        backgroundcolor = :transparent,
        aspect = DataAspect(),
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = false,
        xautolimitmargin = (zero(Float32), zero(Float32)),
        yautolimitmargin = (zero(Float32), zero(Float32))
    )
    heatmap!(ax, first_axis, second_axis, pic; colormap = :oslo)
    ax
end

"""
    force_render!(figure)

Render `figure` once, discarding the result.

Each figure below is saved as the *child scene* of its `Axis` rather than as the figure, which is
what makes the images tightly cropped and free of figure padding. A child scene of a figure that has
never been rendered has no content, so `CairoMakie.save` on it writes a fully transparent image —
the render pass is what this call is for, and it has to happen before the `Axis` is added.

This used to be `display(fig)`, from when the script ran under GLMakie and displaying a figure and
then drawing into the live window was the idiom. CairoMakie has no interactive backend, so `display`
falls through to the file-based show stack, which writes a temporary image and hands it to the system
viewer: 34 Preview windows on macOS on every documentation build, every one of them blank, because
the call sits between `Figure()` and the `plot_image!` that fills it.

`colorbuffer` is the same render pass without the display stack, and is pixel-identical to what
`display` produced — verified over all 34 images. `Makie.update_state_before_display!` is *not* a
substitute despite the name: it leaves the child scenes empty and the saved images blank.
"""
force_render!(figure) = (colorbuffer(figure); nothing)

fig = Figure(; backgroundcolor = :transparent)
force_render!(fig)
filename = "original/image.png"
ax = plot_image!(fig, first_image')
CairoMakie.save(filename, fig.content[1, 1].scene)

for i in 1:16
    global fig = Figure(; backgroundcolor = :transparent)
    force_render!(fig)
    p_small = processed_image₁[i]
    file_name = "split/"*string(i)*".png"
    global ax = plot_image!(fig, p_small')
    CairoMakie.save(file_name, ax.scene)
end

for i in 1:16
    global fig = Figure(; backgroundcolor = :transparent)
    force_render!(fig)
    p_small = processed_image₂[i]
    file_name = "flatten/"*string(i)*".png"
    global ax = plot_image!(fig, p_small')
    CairoMakie.save(file_name, ax.scene)
end

fig = Figure(; backgroundcolor = :transparent)
force_render!(fig)
p_final = fully_processed_image
filename = "final/image.png"
ax = plot_image!(fig, p_final')
CairoMakie.save(filename, ax.scene)
