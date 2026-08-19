# Draw the pendulum data set: what it looks like in canonical and in Euclidean coordinates, and how
# well the integrator conserves the Hamiltonian on it.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/plot_dataset.jl
#
# This regenerates the data rather than reading a file, which is the opposite of what
# `scripts/gml/plot_mnist_results.jl` does and for the opposite reason: `pendulum()` is a fifth of a
# second of Gauss collocation, not four configurations of 500 epochs, and it is deterministic, so
# there is no run to read. The figures below are of exactly the data set `train_sae.jl` trains on.

using CairoMakie

import GeometricProblems.Pendulum: default_parameters
import GMLDatasets: angular_to_euclidean, euclidean_to_angular, pendulum, pendulum_energy

const parameters = default_parameters()

solution = pendulum()
data = angular_to_euclidean(solution)
energy = pendulum_energy(data, parameters)

# `data` is 4 × time steps × trajectories; flattened, its columns are the samples the autoencoder
# sees, and `euclidean_to_angular` takes them back to the angle and its conjugate momentum.
const n_steps, n_trajectories = size(data, 2), size(data, 3)
flat = reshape(data, 4, :)
θ, pθ = euclidean_to_angular(flat[1:2, :], flat[3:4, :]; l = parameters.l)

# The chrome of `scripts/gml/plot_mnist_results.jl`: recessive, no top or right spine. The
# trajectories are coloured by their energy with a perceptually uniform, colourblind-safe map, since
# the energy is what separates the two kinds of motion.
CairoMakie.activate!(type = "png", px_per_unit = 2)

const INK = "#898781"
const GRID = (INK, 0.3)
const COLORMAP = :viridis

# Below mgl the pendulum swings back and forth about θ = π, above it the pendulum goes over the top.
const SEPARATRIX = parameters.m * parameters.g * parameters.l

"An axis with recessive chrome, matching the MNIST figures."
function plain_axis(position; kwargs...)
    Axis(position; backgroundcolor = :transparent,
         xgridvisible = false, ygridcolor = GRID, ygridwidth = 1,
         topspinevisible = false, rightspinevisible = false,
         leftspinecolor = INK, bottomspinecolor = INK, xtickcolor = INK, ytickcolor = INK,
         xticklabelcolor = INK, yticklabelcolor = INK, xlabelcolor = INK, ylabelcolor = INK,
         titlecolor = INK, kwargs...)
end

figure = Figure(size = (960, 420), backgroundcolor = :transparent, fontsize = 14)

# Left: the canonical phase portrait. Scattered rather than drawn as lines because the angle is
# wrapped into one period and a line would jump across the plot at every wrap.
canonical = plain_axis(figure[1, 1]; xlabel = "θ", ylabel = "pθ",
                       title = "canonical coordinates")
scatter!(canonical, mod2pi.(θ), pθ; color = vec(energy), colormap = COLORMAP, markersize = 1.5)

# Right: the same data in two of the four Euclidean rows. This is the projection the autoencoder is
# given — the full picture is the tangent bundle of a circle, a two-dimensional surface in ℝ⁴.
euclidean = plain_axis(figure[1, 2]; xlabel = "q₁", ylabel = "p₁",
                       title = "Euclidean coordinates")
for i in 1:n_trajectories
    lines!(euclidean, data[1, :, i], data[3, :, i];
           color = energy[:, i], colormap = COLORMAP, colorrange = extrema(energy), linewidth = 1)
end

Colorbar(figure[1, 3]; colormap = COLORMAP, limits = extrema(energy), label = "H",
         labelcolor = INK, ticklabelcolor = INK, tickcolor = INK,
         topspinecolor = INK, bottomspinecolor = INK, leftspinecolor = INK, rightspinecolor = INK)

CairoMakie.save("pendulum_dataset.png", figure)

# Gauss collocation is symplectic, so the energy of each trajectory oscillates within a bounded band
# instead of drifting away from its initial value — which is the property that makes this data set
# worth handing to a *symplectic* autoencoder in the first place.
drift = Figure(size = (760, 420), backgroundcolor = :transparent, fontsize = 14)
axis = plain_axis(drift[1, 1]; xlabel = "t", ylabel = "|H(t) − H(0)|", yscale = log10,
                  title = "energy error, Gauss(2)")
for i in 1:n_trajectories
    lines!(axis, collect(solution.t), abs.(energy[:, i] .- energy[1, i]) .+ eps();
           color = (INK, 0.4), linewidth = 1)
end

CairoMakie.save("pendulum_energy_drift.png", drift)

println("wrote pendulum_dataset.png and pendulum_energy_drift.png: ",
        "$(n_trajectories) trajectories × $(n_steps) time steps, ",
        "H ∈ $(round.(extrema(energy); digits = 3)), separatrix at $(SEPARATRIX), ",
        "maximum energy error $(maximum(abs, energy .- energy[1:1, :]))")
