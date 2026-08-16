# The Numerical Experiment on Homogeneous Spaces

`GeometricOptimizers` generalizes Adam to homogeneous spaces; [Optimization on Homogeneous Spaces](https://juliagni.github.io/GeometricOptimizers.jl/latest/manifold_optimizers/) derives how. This page is the numerical evidence, reproducing [brantner2023generalizing](@cite).

## The numerical experiment

The paper trains a vision transformer on MNIST and Fashion-MNIST: 16 transformer blocks, 7 attention heads, ``49 \times 16`` inputs (``28 \times 28`` images cut into 16 patches of ``7 \times 7``), batch size 2048, 500 epochs, `Float32`, no hyperparameter tuning. The projection matrices ``W^Q_i, W^K_i, W^V_i`` of the multi-head attention layers are the weights that are put on ``St(7, 49)``; the feedforward and classification weights stay unconstrained. Four configurations are compared:

| weights | optimizer | outcome |
|---|---|---|
| unconstrained | Adam | does not learn — loss stuck at ``\approx 1.34`` |
| Stiefel | gradient descent | learns |
| Stiefel | momentum | learns, second best |
| Stiefel | Adam | learns, **best** |

Constraining the projections to the Stiefel manifold is what makes the network trainable at all. The unconstrained baseline suffers from **vanishing gradients**: with 16 blocks and none of the usual remedies — layer normalization, dropout, regularization, pre-training — the gradient that reaches the early blocks dies, the network collapses onto a *trivial prediction* ``e_i`` and stays there. The plateau is exactly ``\sqrt{\tfrac{9}{10} \cdot 2} \approx 1.34``, the ``L_2`` distance between a one-hot guess and a one-hot target that disagree on 9 of 10 digits. An orthonormal ``Y`` neither amplifies nor damps what passes through a block, which is what removes the problem.

Two conclusions follow. First, hard geometric constraints can replace the heuristics a transformer normally needs, and they do so without adding a hyperparameter — unlike a soft orthogonality penalty ``\tilde{L} = L + \mu\Sigma``, which adds ``\mu`` and guarantees nothing. Second, among the Stiefel optimizers, Adam gives speed-ups over gradient descent and momentum of the same order as it does in the vector space case, which is the point of generalizing it rather than settling for a first-order method.

## Reproducing the experiment

The scripts in `scripts/geometric_optimizers/` run that experiment against `GeometricOptimizers` itself — `mnist.jl` on the CPU, `mnist_cuda.jl` on an NVIDIA GPU, `mnist_metal.jl` on Apple silicon. [Running the Experiments](@ref) says what each of them does and how to start it. Everything below is one run of `mnist_cuda.jl` on an RTX 4090: the same four configurations, 500 epochs of 29 batches each at a batch size of 2048, `Float32`, the Cayley retraction — none of the scripts passes `retraction`, so they all take the default — 6 h 53 min for all four.

The three series the figures plot are checked in under `docs/src/data/` — 540 rows in total, distilled from the run by `scripts/geometric_optimizers/distill_mnist_results.jl`. The figures are therefore rebuilt with the documentation and need neither a GPU nor a rerun.

```@setup mnist
using GMLDatasets
using CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 2)

# One palette serves both documentation themes, so the ink is a mid gray that clears 3:1 against
# a white and against a dark background, and the figure background is transparent. The four
# series colors are a colorblind-safe set: the worst pair separates by ΔE 13.0 (OKLab ×100)
# under simulated protanopia, well over the 8.0 target, and all four clear 3:1 against the dark
# background. Only `Adam` reaches 2.2:1 against the white one; the table below the first figure
# is what carries those numbers for a reader who cannot resolve the line.
const INK = "#898781"
const GRID = (INK, 0.3)

# Slot order of the palette, assigned to the series in the order the legends list them. It is
# fixed: a series keeps its color across all three figures, so `Adam + Stiefel` is blue whether
# or not the unconstrained baseline appears beside it.
const COLORS = ["adam_stiefel" => "#2a78d6", "adam_regular" => "#eda100",
                "gradient_stiefel" => "#e87ba4", "momentum_stiefel" => "#008300"]
const LABELS = Dict("adam_stiefel" => "Adam + Stiefel", "adam_regular" => "Adam",
                    "gradient_stiefel" => "Gradient + Stiefel",
                    "momentum_stiefel" => "Momentum + Stiefel")

"Columns of one of the checked-in CSVs, by name."
function read_series(name)
    rows = readlines(joinpath(pkgdir(GMLDatasets), "docs", "src", "data", name))
    columns = split(popfirst!(rows), ',')
    values = [parse.(Float64, split(row, ',')) for row in rows]
    Dict(column => [value[i] for value in values] for (i, column) in enumerate(columns))
end

loss = read_series("mnist_training_loss.csv")
accuracy = read_series("mnist_test_accuracy.csv")
drift = read_series("mnist_manifold_drift.csv")

"An axis with recessive chrome: no top or right spine, horizontal gridlines only."
function axis(figure; kwargs...)
    Axis(figure[1, 1]; xlabel = "epoch", backgroundcolor = :transparent,
         xgridvisible = false, ygridcolor = GRID, ygridwidth = 1,
         topspinevisible = false, rightspinevisible = false,
         leftspinecolor = INK, bottomspinecolor = INK, xtickcolor = INK, ytickcolor = INK,
         xticklabelcolor = INK, yticklabelcolor = INK, xlabelcolor = INK, ylabelcolor = INK,
         kwargs...)
end

"""
    series_figure(plot!, data, series; kwargs...)

A figure with one axis, `plot!` called once per column named in `series`, and a horizontal legend
below. Returns the figure and the axis, the latter so a caller can annotate it.
"""
function series_figure(plot!, data, series; height = 420, kwargs...)
    figure = Figure(size = (760, height), backgroundcolor = :transparent, fontsize = 14)
    ax = axis(figure; kwargs...)
    for (column, color) in COLORS
        column in series && plot!(ax, data["epoch"], data[column], color, LABELS[column])
    end
    Legend(figure[2, 1], ax; orientation = :horizontal, framevisible = false,
           labelcolor = INK, padding = (0, 0, 0, 0))
    figure, ax
end

loss_figure, _ = series_figure(loss, first.(COLORS); ylabel = "training loss",
                               limits = ((0, 500), (0, 1.45)),
                               xticks = 0:100:500) do ax, x, y, color, label
    lines!(ax, x, y; color = color, linewidth = 2, label = label)
end

accuracy_figure, _ = series_figure(accuracy, first.(COLORS); ylabel = "test accuracy",
                                   limits = ((0, 510), (0, 1)),
                                   xticks = 0:100:500) do ax, x, y, color, label
    scatterlines!(ax, x, y; color = color, linewidth = 2, markersize = 8, label = label)
end

# Both axes are logarithmic here, unlike the two figures above: the claim the figure has to
# settle is whether the drift grows like a power of the step count and with which exponent, and
# only on a log-log axis is that a straight line whose slope can be read off. A linear y axis
# would also hide the two lower curves under the top one, which is a factor of twenty away.
drift_figure, drift_axis = series_figure(drift, ["adam_stiefel", "gradient_stiefel",
                                                 "momentum_stiefel"];
                                         height = 400, ylabel = "‖YᵀY - I‖",
                                         xscale = log10, yscale = log10,
                                         limits = ((20, 700), (1e-5, 3e-2)),
                                         xticks = ([25, 50, 100, 200, 500],
                                                   ["25", "50", "100", "200", "500"])) do ax, x, y, color, label
    scatterlines!(ax, x, y; color = color, linewidth = 2, markersize = 8, label = label)
end

# The bar `mnist_cuda.jl` checks the drift against, so the figure shows how much room is left,
# and a slope-one guide, so the growth can be compared against linear by eye rather than by
# trusting the sentence below the figure.
hlines!(drift_axis, 1e-2; color = INK, linewidth = 1, linestyle = :dash)
text!(drift_axis, 21, 1.15e-2; text = "tolerance of the script, 1e-2",
      color = INK, fontsize = 12, align = (:left, :bottom))
lines!(drift_axis, [25, 500], [1.4e-5, 2.8e-4]; color = INK, linewidth = 1, linestyle = :dot)
text!(drift_axis, 520, 2.8e-4; text = "slope 1", color = INK, fontsize = 12,
      align = (:left, :center))
nothing
```

### The training loss

```@example mnist
loss_figure # hide
```

The curves reproduce the MNIST panel of the paper's training-loss figure, and closely: the run ends at a loss of **0.234** for Adam on the Stiefel manifold, **0.693** for momentum, **0.726** for gradient descent and **1.342** for unconstrained Adam. The ordering Adam ``<`` momentum ``<`` gradient descent ``<`` unconstrained is the paper's, and the plateau of the unconstrained baseline is the trivial prediction ``\sqrt{1.8} \approx 1.342`` derived above — it is reached within three epochs and held for the remaining 497.

| weights | optimizer | loss, epoch 1 | loss, epoch 500 | test accuracy | ``\|Y^TY-\mathbb{I}\|`` | time |
|---|---|---|---|---|---|---|
| Stiefel | Adam | 0.990 | **0.234** | **0.8658** | 8.6e-03 | 1:35:25 |
| unconstrained | Adam | 1.277 | 1.342 | 0.0980 | — | 1:30:30 |
| Stiefel | gradient | 1.043 | 0.726 | 0.6013 | 4.3e-04 | 1:47:35 |
| Stiefel | momentum | 1.020 | 0.693 | 0.6392 | 3.5e-04 | 1:54:33 |

### The test accuracy

```@example mnist
accuracy_figure # hide
```

The paper reports the training loss only, so the accuracies are this repository's addition rather than a reproduction. They say what the loss curves imply: the unconstrained network sits at 0.098, the accuracy of always guessing the same digit, for all 500 epochs, while the three Stiefel runs separate in the same order as their losses. Adam is also the only one of the three that has converged — it reaches 0.86 by epoch 100 and then moves within ``\pm 0.01``, whereas gradient descent and momentum are still climbing at epoch 500.

### Drift off the manifold

```@example mnist
drift_figure # hide
```

A retraction maps onto the manifold by construction, so in infinite-precision real arithmetic ``Y^TY = \mathbb{I}`` would hold after every one of the 14500 steps and this figure would be a flat line at zero. What it actually plots is the rounding error of a finite format — the experiment runs in `Float32`, as the paper's does — accumulated over those steps.

Two things are worth noting. The growth is **linear in the step count** rather than a random walk: between epoch 25 and epoch 500 the three curves grow by factors of 19.6, 19.6 and 16.7 while the step count grows by 20, which is an exponent of 0.99, 0.99 and 0.94 against the slope-one guide. So the departure is a systematic accumulation and not noise, and it is set by the number of steps rather than by anything in the geometry. A random walk would give an exponent near 0.5 and, at epoch 500, a drift an order of magnitude smaller. And Adam departs about twenty times faster than gradient descent or momentum, which is what its normalized update ``\hat{m}/(\sqrt{\hat{v}} + \varepsilon)`` predicts: its steps are far larger than a raw gradient's, and the error a retraction leaves behind grows with the size of the step it is given.

In absolute terms none of this matters at this length. After 500 epochs Adam is at ``8.6\cdot10^{-3}``, which is still orthonormal to two digits, and the other two are two orders of magnitude below that. It is worth knowing rather than worth fixing: linearity means the drift is predictable, so a run several times longer, or one that needs more than two digits, would want the section rebuilt at intervals — a cost that scales with the number of steps and not with the size of the network. Moving to `Float64` would push the whole figure down by roughly the ratio of the two epsilons and change nothing about its shape.

## References

```@bibliography
Pages = ["homogeneous_spaces_experiment.md"]
Canonical = false

brantner2023generalizing
```
