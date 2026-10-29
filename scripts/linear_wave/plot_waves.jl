# The FOM solutions of the linear wave equation for three wave speeds at three times: q over the
# interior grid, one row per time, earliest on top.
#
#     julia --project=scripts scripts/linear_wave/plot_waves.jl
#
# The wave speeds are the two ends and the midpoint of [5/12, 2/3]; the times are the snapshots
# t = 0, 67/199 ≈ 0.34 and 133/199 ≈ 0.67 of the training grid. Writes
# `$(LW_OUTPUT_DIR)/wave_plot.h5` (the plotted values, with the settings and package versions) and
# `wave_plot.pdf`, drawn at its printed width (`LW_WIDTH_PT`, default 0.7 of a 16 cm text width)
# with 10 pt text. No titles: the rows are explained in the caption.

using CairoMakie

include(joinpath(@__DIR__, "common.jl"))

const PLOT_MU = [MU_MIN, (MU_MIN + MU_MAX) / 2, MU_MAX]
const PLOT_STEPS = SMOKE ? [0, 3, 6] : [0, 67, 133]
const WIDTH_PT = parse(Float64, get(ENV, "LW_WIDTH_PT", string(0.7 * 16 / 2.54 * 72)))

solution = linear_wave(;
    N = N, μ = PLOT_MU, timespan = TIMESPAN, n_time_steps = N_TIME_STEPS)
data = linear_wave_snapshots(solution)
times = collect(solution.s[1].t)
ξ = collect(LinearWave.interior_domain(N))
# q at the plotted steps, with the boundary values q(±1/2) = 0 added at both ends.
ξ_closed = [-0.5; ξ; 0.5]
curves = [[0.0; data[1:N, k + 1, j]; 0.0] for k in PLOT_STEPS, j in eachindex(PLOT_MU)]

mkpath(OUTPUT_DIR)
HDF5.h5open(joinpath(OUTPUT_DIR, "wave_plot.h5"), "w") do file
    file["xi"] = ξ_closed
    file["mu"] = PLOT_MU
    file["t"] = times[PLOT_STEPS .+ 1]
    file["q"] = cat((reduce(hcat, curves[i, :]) for i in eachindex(PLOT_STEPS))...; dims = 3)
    write_metadata(file)
end

fig = Figure(; size = (WIDTH_PT, 0.75WIDTH_PT), fontsize = 10, figure_padding = (
    4, 12, 4, 4))
colors = Makie.wong_colors()[4:6]
axes = [Axis(fig[i, 1]; ylabel = L"q", yticks = [0, 0.5, 1],
            xticks = -0.5:0.25:0.5, xtickformat = "{:.2f}", limits = (
                (-0.5, 0.5), (-0.1, 1.1)))
        for i in eachindex(PLOT_STEPS)]
for (i, ax) in pairs(axes), j in eachindex(PLOT_MU)

    lines!(ax, ξ_closed, curves[i, j]; color = colors[j], linewidth = 1.2,
        label = L"\mu = %$(round(PLOT_MU[j]; digits = 3))")
end
linkxaxes!(axes...)
for ax in axes[1:(end - 1)]
    hidexdecorations!(ax; grid = false, ticks = false)
end
axes[end].xlabel = L"\xi"
Legend(fig[0, 1], axes[1]; orientation = :horizontal, framevisible = false,
    padding = (0, 0, 0, 0))
rowgap!(fig.layout, 4)

CairoMakie.save(joinpath(OUTPUT_DIR, "wave_plot.pdf"), fig; pt_per_unit = 1)
println("wrote ", joinpath(OUTPUT_DIR, "wave_plot.pdf"))
