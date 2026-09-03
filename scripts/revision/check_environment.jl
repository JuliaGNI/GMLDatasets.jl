using CUDA
using GeometricMachineLearning
using GeometricOptimizers
using LinearAlgebra: qr!
using NeuralNetworkParameters
using Pkg

include(joinpath(@__DIR__, "environment_policy.jl"))

required_name = get(ENV, "GML_REQUIRED_GPU", "RTX 4090")
allow_any_gpu = parse(Bool, get(ENV, "GML_ALLOW_ANY_GPU", "0"))
allow_no_cuda = parse(Bool, get(ENV, "GML_ALLOW_NO_CUDA", "0"))
functional = CUDA.functional()
functional || allow_no_cuda || error("CUDA.functional() is false")
device_name = functional ? CUDA.name(CUDA.device()) : "none"
functional && !allow_any_gpu && !occursin(required_name, device_name) &&
    error("expected a GPU containing `$required_name`, found `$device_name`; set GML_ALLOW_ANY_GPU=1 only for deliberate testing")

println("julia_version=", VERSION)
println("active_project=", Base.active_project())
println("device=", device_name)
println("driver_version=", functional ? CUDA.driver_version() : "unavailable")
println("runtime_version=", functional ? CUDA.runtime_version() : "unavailable")
println("threads=", Threads.nthreads())
gml_version = pkgversion(GeometricMachineLearning)
go_version = pkgversion(GeometricOptimizers)
nnp_version = pkgversion(NeuralNetworkParameters)
println("geometric_machine_learning_version=", gml_version)
println("geometric_optimizers_version=", go_version)
println("neural_network_parameters_version=", nnp_version)

validate_environment_versions(VERSION, gml_version, go_version, nnp_version)
isdefined(GeometricOptimizers, :PhaseTimer) || error(
    "GeometricOptimizers must provide PhaseTimer from JuliaGNI/GeometricOptimizers.jl#78")
println("geometric_optimizers_phase_timer=true")

# The optimizer cache and state of a *device-resident* manifold parameter set, which is the one
# thing the preflight cannot infer from a version number. `similar` of a horizontal lift allocated
# on the host before JuliaGNI/GeometricOptimizers.jl#79, and because the four-argument cache
# constructors bind their three gradient blocks to a single type, that was a `MethodError` at
# optimizer construction rather than a wrong number. It cost run 20260903T125418Z_smoke its pendulum
# stage after the image stages had already succeeded: they keep their parameters in a host container
# and copy to the device inside the objective, so nothing before the pendulum stage builds a
# device-resident cache. Two constructor calls on a 4 × 2 point cost nothing and fail here instead.
if functional
    let Q = Matrix(qr!(randn(Float32, 4, 2)).Q)[:, 1:2],
        ps = NetworkParameters((weight = StiefelManifold(CUDA.cu(Q)),))

        GeometricOptimizers.OptimizerCache(Adam(Float32), ps)
        GeometricOptimizers.OptimizerState(Adam(Float32), ps)
    end
    println("geometric_optimizers_device_cache=true")
end
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
functional && CUDA.versioninfo()
