using CUDA, GeometricMachineLearning, GMLDatasets, JLD2

# MNIST images are 28×28, so a sequence_length of 16 = 4² means the image patches are of size 7² = 49
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
function transformer_training(Ψᵉ::GeometricMachineLearning.Architecture;
                              n_epochs=100, opt=AdamOptimizer(T), step_size=T(0.001))
    nn = NeuralNetwork(Ψᵉ, backend, T)
    optimizer_instance = Optimizer(opt, nn; step_size=step_size)

    println("initial test accuracy: ", GeometricMachineLearning.accuracy(nn, dl_test), "\n")

    # use the `time` function to get the system time.
    init_time = time()

    loss_array = optimizer_instance(nn, dl, batch, n_epochs, FeedForwardLoss())

    total_time = time() - init_time

    accuracy_score = GeometricMachineLearning.accuracy(nn, dl_test)
    println("final test accuracy: ", accuracy_score, "\n")

    loss_array, nn, total_time, accuracy_score
end

loss_array2, nn2, total_time2, accuracy_score2 = transformer_training(model2; n_epochs=n_epochs)
loss_array1, nn1, total_time1, accuracy_score1 = transformer_training(model1; n_epochs=n_epochs)
loss_array3, nn3, total_time3, accuracy_score3 = transformer_training(model2; n_epochs=n_epochs, opt=GradientOptimizer())
loss_array4, nn4, total_time4, accuracy_score4 = transformer_training(model2; n_epochs=n_epochs, opt=MomentumOptimizer(T(0.5)))

const mtc = GeometricMachineLearning.map_to_cpu

# `JLD2.save` and not a bare `save`: `GeometricMachineLearning` and `AbstractNeuralNetworks` export a
# `save` of their own, so the unqualified name is ambiguous and raises an `UndefVarError`.
JLD2.save("mnist_parameters.jld2",
        "loss_array1", loss_array1, "nn1weights", nn1.params |> mtc, "total_time1", total_time1, "accuracy_score1", accuracy_score1,
        "loss_array2", loss_array2, "nn2weights", nn2.params |> mtc, "total_time2", total_time2, "accuracy_score2", accuracy_score2,
        "loss_array3", loss_array3, "nn3weights", nn3.params |> mtc, "total_time3", total_time3, "accuracy_score3", accuracy_score3,
        "loss_array4", loss_array4, "nn4weights", nn4.params |> mtc, "total_time4", total_time4, "accuracy_score4", accuracy_score4,        
        )

text_string =
    "n_epochs: " * string(n_epochs) * "\n" *
    "Regular weights:   time: " * string(total_time1) * " classification accuracy: " * string(accuracy_score1) * "\n" *
    "Stiefel weights:   time: " * string(total_time2) * " classification accuracy: " * string(accuracy_score2) * "\n" *
    "GradientOptimizer: time: " * string(total_time3) * " classification accuracy: " * string(accuracy_score3) * "\n" *
    "MomentumOptimizer: time: " * string(total_time4) * " classification accuracy: " * string(accuracy_score4) * "\n"

print(text_string)
