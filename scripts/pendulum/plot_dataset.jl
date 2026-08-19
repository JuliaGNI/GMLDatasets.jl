using GMLDatasets
using CairoMakie

const output = get(ENV, "PENDULUM_PLOT_OUTPUT", "pendulum_phase_portrait.png")
isfile(output) && error("refusing to overwrite existing output: $output")

trajectory = pendulum_trajectory(timespan=(0.0, 10.0), timestep=0.1, angle=0.8, momentum=0.0)
figure = Figure(size=(700, 500))
axis = Axis(figure[1, 1], xlabel="q₁", ylabel="p₁", title="Pendulum phase portrait")
lines!(axis, trajectory.q[1, :], trajectory.p[1, :])
mkpath(dirname(abspath(output)))
CairoMakie.save(output, figure)
println("Wrote $output")
