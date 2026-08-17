# The loss-curve figures of the MNIST comparison, written against `GeometricMachineLearning`.
#
# This is `transformer_mnist.jl` with figures instead of a `.jld2`: the same four configurations, and
# the two plots the comparison is usually shown as — regular weights against Stiefel weights, and the
# three optimizers against each other with the weights on the Stiefel manifold.
#
# The file is called `transformer_bfgs.jl` because its first two curves used to be `BFGSOptimizer()`,
# GML's own manifold BFGS: a gradient-only, fixed-step method driven by `optimization_step!`. That
# method was deleted when the optimizer layer moved to `GeometricOptimizers`, on the stated grounds
# that GO's `BFGS` replaces it. It does not — GO's is a quasi-Newton method whose cache holds an
# inverse-Hessian approximation sized by the *flattened* parameters, and GML's per-leaf update path
# cannot drive it (see the GML changelog). So there is no BFGS to plot here, the runs below are
# `Adam`, `GradientMethod` and `MomentumMethod`, and the labels say so rather than saying `BFGS`.

using CairoMakie, CUDA, GeometricMachineLearning, GMLDatasets

# MNIST images are 28×28, so a sequence length of 16 = 4² means the image patches are of size 7² = 49
const patch_length = 7
const n_heads = 7
const L = 16
const batch_size = 2048
const n_epochs = 500
const add_connection = false

# Use the GPU where there is one and fall back to the host where there is not, so that the script
# runs on a laptop at a reduced `n_epochs` as well as on a workstation.
backend, to_device = CUDA.functional() ? (CUDABackend(), cu) : (CPU(), identity)

# `transform` is applied to the images and the labels before the `DataLoader` splits them into
# patches, so the splitting and the one-hot encoding run on the device too.
dl = mnist_data_loader(:train; patch_length = patch_length, transform = to_device)
dl_test = mnist_data_loader(:test; patch_length = patch_length, transform = to_device)
const T = eltype(dl)

# the difference between the first and the second model is that we put the weights on the Stiefel manifold in the second case
model1 = ClassificationTransformer(dl;
                                    n_heads = n_heads,
                                    L = L,
                                    add_connection = add_connection,
                                    Stiefel = false)
model2 = ClassificationTransformer(dl;
                                    n_heads = n_heads,
                                    L = L,
                                    add_connection = add_connection,
                                    Stiefel = true)

batch = Batch(batch_size, dl)

# The optimizer *method* only produces a direction; the step size is `Optimizer`'s `step_size`
# keyword, which is why `GradientMethod()` and `MomentumMethod(α)` take no learning rate.
function transformer_training(model::GeometricMachineLearning.Architecture;
                              n_epochs = 100, method = AdamOptimizer(T), step_size = T(0.001))
    nn = NeuralNetwork(model, backend, T)
    optimizer_instance = Optimizer(method, nn; step_size = step_size)

    println("initial test accuracy: ", GeometricMachineLearning.accuracy(nn, dl_test), "\n")

    # use the `time` function to get the system time.
    init_time = time()

    loss_array = optimizer_instance(nn, dl, batch, n_epochs, FeedForwardLoss())

    total_time = time() - init_time

    accuracy_score = GeometricMachineLearning.accuracy(nn, dl_test)
    println("final test accuracy: ", accuracy_score, "\n")

    loss_array, nn, total_time, accuracy_score
end

loss_array1, nn1, total_time1, accuracy_score1 =
    transformer_training(model1; n_epochs = n_epochs)
loss_array2, nn2, total_time2, accuracy_score2 =
    transformer_training(model2; n_epochs = n_epochs)
loss_array3, nn3, total_time3, accuracy_score3 =
    transformer_training(model2; n_epochs = n_epochs, method = GradientOptimizer())
loss_array4, nn4, total_time4, accuracy_score4 =
    transformer_training(model2; n_epochs = n_epochs, method = MomentumOptimizer(T(0.5)))

# The figure style of `docs/src/homogeneous_spaces_experiment.md`: recessive chrome, a
# colorblind-safe palette, and a series that keeps its color across both figures.
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

CairoMakie.save("mnist_regular_vs_stiefel.png",
    loss_figure([(loss_array1, REGULAR, "Regular weights"),
                 (loss_array2, ADAM, "Weights on Stiefel manifold")]))

CairoMakie.save("mnist_optimizer_comparison.png",
    loss_figure([(loss_array2, ADAM, "Adam"),
                 (loss_array3, GRADIENT, "Gradient"),
                 (loss_array4, MOMENTUM, "Momentum")]))

text_string =
    "n_epochs: " * string(n_epochs) * "\n" *
    "Regular weights,  Adam:     time: " * string(total_time1) * " classification accuracy: " * string(accuracy_score1) * "\n" *
    "Stiefel weights,  Adam:     time: " * string(total_time2) * " classification accuracy: " * string(accuracy_score2) * "\n" *
    "Stiefel weights,  gradient: time: " * string(total_time3) * " classification accuracy: " * string(accuracy_score3) * "\n" *
    "Stiefel weights,  momentum: time: " * string(total_time4) * " classification accuracy: " * string(accuracy_score4) * "\n"

print(text_string)

open("measure_times" * string(backend), "w") do file
    write(file, text_string)
end
