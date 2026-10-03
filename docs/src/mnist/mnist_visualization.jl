# The figure of the MNIST tutorial that shows how an image becomes the input of the classification
# transformer: the original digit with its 4 × 4 grid of patches, the 16 patches, the 16 flattened
# patches, and the flattened image with a red grid, which enters the transformer block drawn above
# it.
#
# The tutorial includes this file in an `@setup` block and saves one figure per Documenter theme.
# Every length below is in TeX points, and the figure maps one point to one unit, so line widths and
# font sizes are in points as well.

using CairoMakie
using GMLDatasets: split_and_flatten

const MNIST_FIGURE_COLORS = (
    orange = RGBf(255 / 255, 127 / 255, 14 / 255),
    blue = RGBf(31 / 255, 119 / 255, 180 / 255),
    red = RGBf(214 / 255, 39 / 255, 40 / 255),
    purple = RGBf(148 / 255, 103 / 255, 189 / 255)
)

# `color` mixed with white, as TikZ's `color!percent`.
tint(color, percent) = percent / 100 * color + (1 - percent / 100) * RGBf(1, 1, 1)

# A rectangle with rounded corners, as the vertices of a polygon.
function rounded_rectangle(xmin, ymin, xmax, ymax; radius = 4)
    corners = ((xmax - radius, ymax - radius, 0), (xmin + radius, ymax - radius, π / 2),
        (xmin + radius, ymin + radius, π), (xmax - radius, ymin + radius, 3π / 2))
    return [Point2f(cx + radius * cos(a0 + t), cy + radius * sin(a0 + t))
            for (cx, cy, a0) in corners for t in range(0, π / 2; length = 8)]
end

# A polyline with a stealth arrow head at its last point.
function arrow!(ax, points; color, linewidth, headlength = 6, headwidth = 4.5)
    tip = Point2f(points[end])
    direction = tip - Point2f(points[end - 1])
    direction = direction / sqrt(sum(abs2, direction))
    normal = Point2f(-direction[2], direction[1])
    base = tip - headlength * direction
    lines!(ax, Point2f.(vcat(points[1:(end - 1)], [tip - 0.6 * headlength * direction]));
        color, linewidth)
    head = Point2f[tip, base + headwidth / 2 * normal, tip - 0.6 * headlength * direction,
        base - headwidth / 2 * normal]
    poly!(ax, head; color, strokewidth = 0)
    return nothing
end

# The point where the segment from `from` to the centre of a box leaves the box.
function box_border(from, center, halfwidth, halfheight)
    d = Point2f(from) - Point2f(center)
    s = min(halfwidth / max(abs(d[1]), eps(Float32)), halfheight /
                                                      max(abs(d[2]), eps(Float32)))
    return Point2f(Point2f(center) + s * d)
end

# A module of the transformer block: a rounded box with a label, as in the TikZ original.
function module_box!(ax, center, label; fill, stroke, height)
    halfwidth, halfheight = 32.3, height / 2
    poly!(ax,
        rounded_rectangle(center[1] - halfwidth, center[2] - halfheight,
            center[1] + halfwidth, center[2] + halfheight);
        color = fill, strokecolor = stroke, strokewidth = 1.2)
    text!(ax, center[1], center[2]; text = label, align = (:center, :center),
        justification = :center, color = stroke, fontsize = 10)
    return (center = Point2f(center), halfwidth, halfheight)
end

"""
    mnist_visualization(image; theme, patch_length = 7)

Draw the preprocessing figure of the MNIST tutorial for `image`, a square matrix of grey values in
``[0, 1]``, in the Documenter theme `theme`, `:light` or `:dark`.
"""
function mnist_visualization(image::AbstractMatrix; theme::Symbol, patch_length::Integer = 7)
    theme in (:light, :dark) ||
        throw(ArgumentError("theme must be :light or :dark, not $theme"))
    colors = MNIST_FIGURE_COLORS
    fg = theme == :dark ? RGBf(1, 1, 1) : RGBf(0, 0, 0)
    percent = theme == :dark ? 70 : 40

    n = size(image, 1)
    patches_per_side = n ÷ patch_length
    number_of_patches = patches_per_side^2
    flattened = split_and_flatten(image; patch_length, number_of_patches)
    heatmap_options = (colormap = :oslo, colorrange = (0, 1))

    # The lengths of the TikZ original, in points.
    cm = 28.45
    sep = 3.33                       # TikZ's default inner sep around a node
    original_width = 60.0
    patch_width = 15.0
    column_width = patch_width / patch_length
    patch_pitch = patch_width + 2sep + 0.05cm
    final_cell = 17column_width / number_of_patches

    fig = Figure(; backgroundcolor = :transparent, figure_padding = 0)
    ax = Axis(fig[1, 1]; backgroundcolor = :transparent, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    # The cell edges of a heatmap of `count` cells of width `width` from `start`.
    edges(start, width, count) = range(start, start + count * width; length = count + 1)

    # The original image, with its 4 × 4 grid of patches.
    cell = original_width / patches_per_side
    heatmap!(ax, edges(0, original_width / n, n), edges(0, original_width / n, n), image';
        heatmap_options...)
    for k in 0:patches_per_side
        lines!(ax, [0, original_width], [k * cell, k * cell]; color = colors.red,
            linewidth = 0.4)
        lines!(ax, [k * cell, k * cell], [0, original_width]; color = colors.red,
            linewidth = 0.4)
    end

    # The patches, in one row, each with its flattened patch below it.
    patch_left = -6.5cm - sep - patch_width
    patch_bottom = -6.5cm - sep - patch_width
    column_top = patch_bottom - 2sep - 0.2cm
    column_height = patch_length^2 * column_width
    centers = Point2f[]
    for k in 1:number_of_patches
        x0 = patch_left + (k - 1) * patch_pitch
        xc = x0 + patch_width / 2
        push!(centers, Point2f(xc, column_top - column_height / 2))
        patch = reshape(flattened[:, k], patch_length, patch_length)
        heatmap!(ax, edges(x0, column_width, patch_length),
            edges(patch_bottom, column_width, patch_length), patch'; heatmap_options...)
        heatmap!(ax, edges(xc - column_width / 2, column_width, 1),
            edges(column_top - column_height, column_width, patch_length^2),
            reshape(flattened[:, k], 1, :); heatmap_options...)

        # The patch's cell in the original: column `c` from the left, row `r` from the bottom.
        c, r = divrem(k - 1, patches_per_side)
        from = Point2f((c + 0.5) * cell, (r + 0.5) * cell)
        patch_center = Point2f(xc, patch_bottom + patch_width / 2)
        arrow!(ax,
            [
                from, box_border(from, patch_center, patch_width / 2 + sep,
                    patch_width / 2 + sep)];
            color = colors.orange,
            linewidth = 0.4)
        arrow!(ax, [Point2f(xc, patch_bottom - sep), Point2f(xc, column_top + sep)];
            color = colors.orange, linewidth = 0.4)
    end

    # The flattened image, with one red column per patch.
    final_left = centers[end][1] + column_width / 2 + 2sep + 1cm
    final_width = number_of_patches * final_cell
    final_height = patch_length^2 * final_cell
    final_center = Point2f(final_left + final_width / 2, centers[end][2])
    final_bottom = final_center[2] - final_height / 2
    heatmap!(ax, edges(final_left, final_cell, number_of_patches),
        edges(final_bottom, final_cell, patch_length^2), flattened'; heatmap_options...)
    for k in 0:number_of_patches
        x = final_left + k * final_cell
        lines!(ax, [x, x], [final_bottom, final_bottom + final_height]; color = colors.red,
            linewidth = 1.6)
    end
    for y in (final_bottom, final_bottom + final_height)
        lines!(ax, [final_left - 0.8, final_left + final_width + 0.8], [y, y];
            color = colors.red, linewidth = 1.6)
    end

    # The double arrow from the last flattened patch to the flattened image.
    start = centers[end][1] + column_width / 2 + sep
    stop = final_left - sep
    for offset in (-1.0, 1.0)
        lines!(ax, [start, stop - 5], fill(final_center[2] + offset, 2);
            color = colors.orange, linewidth = 0.4)
    end
    arrow!(ax, [Point2f(stop - 6, final_center[2]), Point2f(stop, final_center[2])];
        color = colors.orange, linewidth = 0.4)

    # The transformer block above the flattened image.
    two_lines, one_line = 31.0, 17.0
    x = final_center[1]
    attention = module_box!(ax, (x, final_center[2] + 3cm), "Multihead\nAttention";
        fill = tint(colors.orange, percent), stroke = fg, height = two_lines)
    feedforward = module_box!(ax, (x, attention.center[2] + 1.3cm), "Feed\nForward";
        fill = tint(colors.blue, percent), stroke = fg, height = two_lines)
    add = module_box!(ax, (x, feedforward.center[2] + 1cm), "Add";
        fill = tint(colors.purple, percent), stroke = fg, height = one_line)
    classification = module_box!(ax, (x, add.center[2] + 1cm), "Classification";
        fill = tint(colors.red, percent), stroke = fg, height = one_line)
    output = Point2f(x, classification.center[2] + 1cm)
    text!(
        ax, output[1], output[2]; text = "Output", align = (:center, :center), color = fg,
        fontsize = 10)

    bottom(box) = box.center - Point2f(0, box.halfheight)
    top(box) = box.center + Point2f(0, box.halfheight)
    final_north = Point2f(x, final_center[2] + 1.8cm)
    arrow!(ax, [final_north, bottom(attention)]; color = fg, linewidth = 0.8)
    arrow!(ax, [top(attention), bottom(feedforward)]; color = fg, linewidth = 0.8)
    arrow!(ax, [top(feedforward), bottom(add)]; color = fg, linewidth = 0.8)
    arrow!(ax, [top(add), bottom(classification)]; color = fg, linewidth = 0.8)
    arrow!(ax, [top(classification), output - Point2f(0, 6)]; color = fg, linewidth = 0.8)

    # The residual connection around the feed-forward module, and the three inputs of the
    # attention module.
    attention_residual = (bottom(attention) + final_north) / 2
    feedforward_residual = (bottom(feedforward) + top(attention)) / 2
    right_of_add = x + 1.5cm
    arrow!(ax,
        [feedforward_residual, Point2f(right_of_add, feedforward_residual[2]),
            Point2f(right_of_add, add.center[2]), add.center + Point2f(add.halfwidth, 0)];
        color = fg, linewidth = 0.8)
    fork = (bottom(attention) + attention_residual) / 2
    for side in (-1, 1)
        target = bottom(attention) + Point2f(side * attention.halfwidth / 2, 0)
        arrow!(ax, [fork, Point2f(target[1], fork[2]), target]; color = fg, linewidth = 0.8)
    end

    # The box around the transformer layer, repeated 16 times.
    encoder = (xmin = x - attention.halfwidth - sep, xmax = right_of_add + sep,
        ymin = attention_residual[2] - sep, ymax = top(add)[2] + sep)
    lines!(ax,
        [rounded_rectangle(encoder.xmin, encoder.ymin, encoder.xmax, encoder.ymax);
         rounded_rectangle(encoder.xmin, encoder.ymin, encoder.xmax, encoder.ymax)[1:1]];
        color = fg, linewidth = 1.6)
    text!(ax, encoder.xmin - sep, (encoder.ymin + encoder.ymax) / 2; text = "16×",
        align = (:right, :center), color = fg, fontsize = 10)

    # The canvas: the extent of everything drawn, with a margin.
    xmin, xmax = patch_left - 5, encoder.xmax + 5
    ymin, ymax = column_top - column_height - 5, original_width + 5
    limits!(ax, xmin, xmax, ymin, ymax)
    resize!(fig.scene, round(Int, xmax - xmin), round(Int, ymax - ymin))
    return fig
end
