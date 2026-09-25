# The preflight of a revision run: report the environment, then probe the two properties a
# version number cannot express.
#
# There is deliberately no assertion on the resolved dependency versions. `scripts/Project.toml`
# states the bounds and `scripts/Manifest.toml` in the bundle records what a run actually
# resolved; a second copy of those numbers in Julia can only disagree with them.

using CUDA
using GeometricMachineLearning
using GeometricOptimizers
using LinearAlgebra: qr!
using NeuralNetworkParameters
using Pkg

required_name = get(ENV, "GML_REQUIRED_GPU", "RTX 4090")
allow_any_gpu = parse(Bool, get(ENV, "GML_ALLOW_ANY_GPU", "0"))
allow_no_cuda = parse(Bool, get(ENV, "GML_ALLOW_NO_CUDA", "0"))
functional = CUDA.functional()
functional || allow_no_cuda || error("CUDA.functional() is false")
device_name = functional ? CUDA.name(CUDA.device()) : "none"
functional && !allow_any_gpu && !occursin(required_name, device_name) &&
    error(
        "expected a GPU containing `$required_name`, found `$device_name`; " *
        "set GML_ALLOW_ANY_GPU=1 only for deliberate testing")

println("julia_version=", VERSION)
println("active_project=", Base.active_project())
println("device=", device_name)
println("driver_version=", functional ? CUDA.driver_version() : "unavailable")
println("runtime_version=", functional ? CUDA.runtime_version() : "unavailable")
println("threads=", Threads.nthreads())
println("geometric_machine_learning_version=", pkgversion(GeometricMachineLearning))
println("geometric_optimizers_version=", pkgversion(GeometricOptimizers))
println("neural_network_parameters_version=", pkgversion(NeuralNetworkParameters))

# Two properties of a *device-resident* manifold parameter set, which is what no version number
# can carry. Both cost two calls on a 4 × 2 point and both once cost a run its pendulum stage
# after the image stages had already succeeded — the image trainer keeps its parameters in a host
# container and copies to the device inside the objective, so nothing before the pendulum stage
# builds a device-resident cache.
#
#  1. the optimizer cache and state can be built at all. `similar` of a horizontal lift used to
#     allocate on the host, and because the four-argument cache constructors bind their three
#     gradient blocks to one type that was a `MethodError` at optimizer construction rather than
#     a wrong number (run 20260903T125418Z_smoke).
#  2. the Riemannian gradient of a device-resident point lands on the device. The pullback hands
#     `rgrad` an ambient gradient that stayed on the host, and `∇L' * Y.A` is then a CPU `gemm!`
#     on a device pointer (runs 20260903T185459Z_smoke and 20260903T191704Z_smoke).
#
# The second is the property the harness needs, not the mechanism that currently provides it: a
# temporary shim in `GeometricOptimizers` moves the gradient across today, and when
# JuliaGNI/GeometricMachineLearning.jl#258 and JuliaGNI/AbstractNeuralNetworks.jl#39 close it will
# arrive on the device to begin with. Either way this must hold, and a failure here says the
# harness needs updating rather than that the check has expired.
if functional
    let Q = Matrix(qr!(randn(Float32, 4, 2)).Q)[:, 1:2], Y = StiefelManifold(CUDA.cu(Q)),
        ps = NetworkParameters((weight = Y,))

        GeometricOptimizers.OptimizerCache(Adam(Float32), ps)
        GeometricOptimizers.OptimizerState(Adam(Float32), ps)
        println("geometric_optimizers_device_cache=true")

        rgrad(Y, randn(Float32, 4, 2)) isa CUDA.CuArray || error(
            "rgrad returned a host array for a device-resident point; the Riemannian gradient " *
            "must reach the device before the retraction does")
        println("geometric_optimizers_device_gradient=true")
    end
end
Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
functional && CUDA.versioninfo()
