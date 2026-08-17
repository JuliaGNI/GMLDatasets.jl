# Draw the loss-curve figures of the MNIST comparison from a finished run.
#
# Run from the repository root, after `transformer_mnist.jl` (or `transformer_fashion_mnist.jl`) has
# written its `.jld2`:
#
#   julia --project=scripts scripts/gml/plot_mnist_results.jl [results.jld2]
#
# The default is `mnist_parameters.jld2`; the output names are derived from whatever is passed, so
# `fashion_mnist_parameters.jld2` produces `fashion_mnist_*.png` and does not overwrite the MNIST
# figures.
#
# This reads a run rather than performing one, which is the same split as
# `scripts/geometric_optimizers/distill_mnist_results.jl` and the same reason
# `docs/src/homogeneous_spaces_experiment.md` plots from checked-in CSVs: the training is four
# configurations of 500 epochs, and redrawing a figure must not cost a rerun of it. Everything the
# figures and the summary need — the four loss arrays, the four wall-clock times and the four test
# accuracies — is already in the `.jld2`; only the network weights in it are unused here.
#
# This file used to be `transformer_bfgs.jl`, which trained the four configurations itself in order
# to plot them. Its first two curves were `BFGSOptimizer()`, GML's own manifold BFGS — a
# gradient-only, fixed-step method driven by `optimization_step!` — which was deleted when the
# optimizer layer moved to `GeometricOptimizers` on the stated grounds that GO's `BFGS` replaces it.
# It does not: GO's is a quasi-Newton method whose cache holds an inverse-Hessian approximation sized
# by the *flattened* parameters, and GML's per-leaf update path cannot drive it (see the GML
# changelog). There is therefore no BFGS run to plot, and the four configurations are the ones
# `transformer_mnist.jl` produces.

using CairoMakie
import JLD2

const results_file = isempty(ARGS) ? "mnist_parameters.jld2" : ARGS[1]
const prefix = replace(basename(results_file), r"_parameters\.jld2$" => "", r"\.jld2$" => "")

isfile(results_file) || error("no such file: $results_file — run `transformer_mnist.jl` first, " *
                              "or pass the path of a `.jld2` it wrote")

# Read only the datasets that are needed. `JLD2.load` would pull in the `nn*weights` too, and
# reconstructing a `NeuralNetworkParameters` of `StiefelManifold`s needs `GeometricMachineLearning`
# loaded just to be thrown away again.
losses, times, accuracies = JLD2.jldopen(results_file, "r") do file
    ([file["loss_array$i"] for i in 1:4],
     [file["total_time$i"] for i in 1:4],
     [file["accuracy_score$i"] for i in 1:4])
end

# The index order `transformer_mnist.jl` saves in. Only the projections of the attention layers are
# constrained; the feedforward and classification weights are unconstrained in all four.
const RUNS = ("regular weights, Adam", "Stiefel weights, Adam",
              "Stiefel weights, gradient", "Stiefel weights, momentum")

# The palette of `docs/src/homogeneous_spaces_experiment.md`: recessive chrome, colorblind-safe, and
# a series that keeps its color across both figures.
CairoMakie.activate!(type = "png", px_per_unit = 2)

const INK = "#898781"
const GRID = (INK, 0.3)
const REGULAR  = "#eda100"
const ADAM     = "#2a78d6"
const GRADIENT = "#e87ba4"
const MOMENTUM = "#008300"

"An axis with recessive chrome: no top or right spine, horizontal gridlines only."
function loss_axis(figure)
    Axis(figure[1, 1]; xlabel = "epoch", ylabel = "training loss",
         limits = (nothing, (0., 1.4)), backgroundcolor = :transparent,
         xgridvisible = false, ygridcolor = GRID, ygridwidth = 1,
         topspinevisible = false, rightspinevisible = false,
         leftspinecolor = INK, bottomspinecolor = INK, xtickcolor = INK, ytickcolor = INK,
         xticklabelcolor = INK, yticklabelcolor = INK, xlabelcolor = INK, ylabelcolor = INK)
end

"A figure with one axis, one line per `(loss, color, label)`, and a horizontal legend below."
function loss_figure(series)
    figure = Figure(size = (760, 420), backgroundcolor = :transparent, fontsize = 14)
    ax = loss_axis(figure)
    for (loss, color, label) in series
        lines!(ax, loss; color = color, linewidth = 2, label = label)
    end
    Legend(figure[2, 1], ax; orientation = :horizontal, framevisible = false,
           labelcolor = INK, padding = (0, 0, 0, 0))
    figure
end

# Constraining the attention projections is what makes the network trainable at all: the
# unconstrained baseline plateaus at √1.8 ≈ 1.34, a trivial prediction.
CairoMakie.save("$(prefix)_regular_vs_stiefel.png",
    loss_figure([(losses[1], REGULAR, "Regular weights"),
                 (losses[2], ADAM, "Weights on Stiefel manifold")]))

# The three optimizers, all with the projections on the Stiefel manifold.
CairoMakie.save("$(prefix)_optimizer_comparison.png",
    loss_figure([(losses[2], ADAM, "Adam"),
                 (losses[3], GRADIENT, "Gradient"),
                 (losses[4], MOMENTUM, "Momentum")]))

text_string = "epochs: " * string(length(losses[1])) * "\n" *
    join(("$(rpad(RUNS[i], 26)) time: $(times[i]) s   classification accuracy: $(accuracies[i])"
          for i in 1:4), "\n") * "\n"

print(text_string)

open("$(prefix)_summary.txt", "w") do file
    write(file, text_string)
end
