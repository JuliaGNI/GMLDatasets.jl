# The projection and reduction errors of PSD and of the symplectic autoencoders against n, half the
# reduced dimension, one panel per test wave speed, from the numbers `errors.jl` saved.
#
#     julia --project=scripts scripts/linear_wave/plot_errors.jl
#
# Reads `$(LW_OUTPUT_DIR)/errors.h5` and writes `errors.pdf`, drawn at its printed width
# (`LW_WIDTH_PT`, default a 16 cm text width) with 10 pt text. The errors are on a logarithmic scale,
# shared by all panels. The SAE markers are the median over the seeds and the bars span the smallest
# to the largest seed; PSD is deterministic. The panels are labeled below; what the markers mean
# goes in the caption.

using CairoMakie
using Statistics

include(joinpath(@__DIR__, "common.jl"))

const WIDTH_PT = parse(Float64, get(ENV, "LW_WIDTH_PT", string(16 / 2.54 * 72)))

columns = HDF5.h5open(joinpath(OUTPUT_DIR, "errors.h5"), "r") do file
    Dict(name => read(file, name) for name in keys(file))
end
rows = [(method = columns["method"][i], n = columns["n"][i], mu = columns["mu"][i],
            projection_error = columns["projection_error"][i],
            reduction_error = columns["reduction_error"][i])
        for i in eachindex(columns["method"])]
mus = sort(unique(r.mu for r in rows))
ns = sort(unique(r.n for r in rows))

select(method, μ, n) = filter(r -> r.method == method && r.mu == μ && r.n == n, rows)

orange, green = Makie.wong_colors()[2], Makie.wong_colors()[3]
fig = Figure(; size = (WIDTH_PT, 0.8WIDTH_PT), fontsize = 10, figure_padding = 4)
n_cols = 2
axes = Axis[]
for (k, μ) in pairs(mus)
    row, col = fldmod1(k, n_cols)
    ax = Axis(fig[2row - 1, col]; yscale = log10, xticks = ns[1]:2:ns[end],
        yticks = ([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"]),
        xlabel = L"n", ylabel = "relative error")
    push!(axes, ax)
    Label(fig[2row, col], L"(%$('a' + k - 1))\;\mu = %$(μ)"; tellwidth = false)

    for (field, marker) in ((:projection_error, :xcross), (:reduction_error, :circle))
        psd = [only(select("PSD", μ, n))[field] for n in ns]
        scatter!(ax, ns .- 0.12, psd; color = orange, marker = marker, markersize = 7)

        sae = [[r[field] for r in select("SAE", μ, n)] for n in ns]
        mid = median.(sae)
        rangebars!(ax, ns .+ 0.12, minimum.(sae), maximum.(sae); color = green,
            whiskerwidth = 4)
        scatter!(ax, ns .+ 0.12, mid; color = green, marker = marker, markersize = 7)
    end
end
linkyaxes!(axes...)
for ax in axes[1:(end - n_cols)]
    hidexdecorations!(ax; grid = false, ticks = false, ticklabels = false)
end
for (k, ax) in pairs(axes)
    iseven(k) && hideydecorations!(ax; grid = false, ticks = false, ticklabels = false)
end

legend_entries = [
    MarkerElement(; color = orange, marker = :xcross, markersize = 7),
    MarkerElement(; color = orange, marker = :circle, markersize = 7),
    MarkerElement(; color = green, marker = :xcross, markersize = 7),
    MarkerElement(; color = green, marker = :circle, markersize = 7)]
Legend(fig[2 * cld(length(mus), n_cols) + 1, 1:n_cols], legend_entries,
    ["PSD projection", "PSD reduction", "SAE projection", "SAE reduction"];
    orientation = :horizontal, nbanks = 2, framevisible = false,
    padding = (0, 0, 0, 0))
rowgap!(fig.layout, 2)

CairoMakie.save(joinpath(OUTPUT_DIR, "errors.pdf"), fig; pt_per_unit = 1)
println("wrote ", joinpath(OUTPUT_DIR, "errors.pdf"))
