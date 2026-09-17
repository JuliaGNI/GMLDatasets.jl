#!/usr/bin/env julia

module RetractionRecords

using GeometricOptimizers
using GeometricOptimizers: AugmentedPade, NativePade, ScaledSquaring, SkewSymMatrix,
                           StiefelLieAlgHorMatrix, geodesic
using LinearAlgebra: I, norm
import Random

include("arguments.jl")
include("records.jl")
using .RunRecords: capture_source, validate_records, write_retraction_records

const USAGE = """usage: retraction_records.jl --go-repo DIR [options]

Write schema-v1 CSV records for the GeometricOptimizers retraction benchmark.

  --output FILE          CSV destination (default: retraction-records.csv)
  --patch-output FILE    exact GO patch destination (default: beside the CSV)
  --backend cpu|cuda     execution backend (default: cpu)
  --precision TYPE       Float32 or Float64 (default: Float64)
  --rows N               full square lift size (default: 20)
  --columns N            Stiefel columns (default: 3)
  --scales LIST          comma-separated upstream lift scales
  --repetitions N        measured calls after one recorded warm-up (default: 20)
  --seed N               deterministic lift seed (default: 1234)

CPU mode records ScaledSquaring, NativePade, and AugmentedPade on the host. CUDA mode
records ScaledSquaring and NativePade on the GPU and AugmentedPade on the host; GPU outputs
are transferred before comparison with the host AugmentedPade result. The Julia load path must
make CUDA available for CUDA mode while this script runs with the GO repository as its project.
"""

const DEFAULTS = (output = "retraction-records.csv", patch_output = "", go_repo = "",
    backend = "cpu", precision = "Float64", rows = 20, columns = 3,
    scales = "0.1,1.0,3.0,6.0,12.0,30.0,60.0,120.0", repetitions = 20, seed = 1234)

"""Parse and check the command line, returning the benchmark's settings."""
function parse_options(args)
    parsed = parse_arguments(args, DEFAULTS; usage = USAGE)
    parsed === nothing && return nothing

    isempty(parsed.go_repo) && throw(ArgumentError("--go-repo is required"))
    parsed.backend in ("cpu", "cuda") || throw(ArgumentError("backend must be cpu or cuda"))
    parsed.precision in ("Float32", "Float64") ||
        throw(ArgumentError("precision must be Float32 or Float64"))
    parsed.rows > 0 || throw(ArgumentError("--rows must be positive"))
    0 < parsed.columns <= parsed.rows ||
        throw(ArgumentError("--columns must be positive and no greater than --rows"))
    scales = parse.(Float64, split(parsed.scales, ','; keepempty = false))
    !isempty(scales) && all(scale -> isfinite(scale) && scale >= 0, scales) ||
        throw(ArgumentError("--scales must contain finite nonnegative values"))
    parsed.repetitions > 0 || throw(ArgumentError("--repetitions must be positive"))
    parsed.seed >= 0 || throw(ArgumentError("--seed must be nonnegative"))

    merge(parsed,
        (backend = Symbol(parsed.backend),
            precision = parsed.precision == "Float32" ? Float32 : Float64, scales = scales))
end

function verify_loaded_source(go_root::AbstractString)
    loaded_root = realpath(joinpath(dirname(pathof(GeometricOptimizers)), ".."))
    loaded_root == go_root || throw(ArgumentError(
        "loaded GeometricOptimizers from $loaded_root, not requested repository $go_root"))
end

# Julia 1.12 forbids calling methods that were added after the running frame started, so the
# import has to finish in `main` before the benchmark frame is entered through `invokelatest`.
# Loading CUDA also brings in package extensions, whose methods are subject to the same rule.
function import_cuda()
    @eval import CUDA
    @eval device_allocated(f) = CUDA.@allocated f()
    nothing
end

function loaded_cuda()
    cuda = getglobal(@__MODULE__, :CUDA)
    cuda.functional(true) || throw(ArgumentError("CUDA is not functional"))
    cuda
end

function cuda_lift(host, cuda)
    A = SkewSymMatrix(cuda.CuArray(copy(parent(host.A))), host.n)
    B = cuda.CuArray(copy(host.B))
    StiefelLieAlgHorMatrix(A, B, host.N, host.n)
end

sanitize_error(error) = replace(sprint(showerror, error), r"\s+" => " ")
float_string(value) = string(Float64(value))

function base_record(;
        algorithm, backend, device, precision, rows, stiefel_columns, lift_norm,
        memory_metric, repetition, warmup, seed, scale, source, patch_file)
    Dict(
        "schema_version" => "1",
        "algorithm" => algorithm,
        "backend" => backend,
        "device" => string(device),
        "precision" => string(precision),
        "matrix_rows" => string(rows),
        "matrix_columns" => string(rows),
        "stiefel_columns" => string(stiefel_columns),
        "lift_norm" => float_string(lift_norm),
        "memory_metric" => memory_metric,
        "repetition" => string(repetition),
        "warmup" => string(warmup),
        "seed" => string(seed),
        "scale" => float_string(scale),
        "go_sha" => source.sha,
        "go_dirty" => string(source.dirty),
        "go_patch_file" => patch_file,
        "go_patch_sha256" => source.patch_sha256
    )
end

function record_invocation(base, operation, synchronize_backend, allocated_bytes, to_host,
        agreement_reference, forward_reference)
    started = time_ns()
    try
        synchronize_backend()
        started = time_ns()
        result = operation()
        synchronize_backend()
        elapsed = (time_ns() - started) / 1.0e9

        memory = allocated_bytes(operation)
        synchronize_backend()
        output = Matrix{Float64}(to_host(result))
        merge(base,
            Dict(
                "agreement_error" => float_string(
                    norm(output - agreement_reference) / norm(agreement_reference)),
                "forward_error" => float_string(
                    norm(output - forward_reference) / norm(forward_reference)),
                "manifold_constraint_error" => float_string(norm(output' * output - I)),
                "runtime_seconds" => float_string(elapsed),
                "memory_bytes" => string(memory),
                "success" => "true",
                "error_type" => "",
                "error_message" => ""
            ))
    catch error
        try
            synchronize_backend()
        catch
        end
        merge(base,
            Dict(
                "agreement_error" => "NaN",
                "forward_error" => "NaN",
                "manifold_constraint_error" => "NaN",
                "runtime_seconds" => float_string(max(0.0, (time_ns() - started) / 1.0e9)),
                "memory_bytes" => "0",
                "success" => "false",
                "error_type" => string(typeof(error)),
                "error_message" => sanitize_error(error)
            ))
    end
end

# The host path: allocation is Julia's own, nothing needs synchronizing, and every algorithm
# runs on it. `AugmentedPade` uses dense LAPACK and stays here even in CUDA mode.
function host_path(name, algorithm, lift)
    (name = name, backend = "CPU", device = Sys.CPU_NAME,
        algorithm = algorithm, input = lift, synchronize_backend = () -> nothing,
        allocated_bytes = operation -> @allocated(operation()),
        to_host = result -> Matrix(result), memory_metric = "host_allocated_bytes")
end

function algorithm_paths(options, host_lift, cuda)
    if options.backend == :cpu
        return [host_path("ScaledSquaring", ScaledSquaring(), host_lift),
            host_path("NativePade", NativePade(), host_lift),
            host_path("AugmentedPade", AugmentedPade(), host_lift)]
    end

    gpu_lift = cuda_lift(host_lift, cuda)
    device_path(name, algorithm) = (name = name, backend = "CUDA",
        device = string(cuda.name(cuda.device())), algorithm = algorithm, input = gpu_lift,
        synchronize_backend = () -> cuda.synchronize(),
        allocated_bytes = operation -> Base.invokelatest(device_allocated, operation),
        to_host = result -> Array(parent(result)),
        memory_metric = "device_allocated_bytes")
    [device_path("ScaledSquaring", ScaledSquaring()),
        device_path("NativePade", NativePade()),
        host_path("AugmentedPade", AugmentedPade(), host_lift)]
end

function run_benchmark(options)
    patch_path = isempty(options.patch_output) ?
                 joinpath(dirname(abspath(options.output)),
        "geometricoptimizers-retraction.patch") : abspath(options.patch_output)
    source = capture_source(options.go_repo, patch_path)
    verify_loaded_source(source.root)
    patch_file = relpath(patch_path, dirname(abspath(options.output)))
    cuda = options.backend == :cuda ? loaded_cuda() : nothing
    records = Dict{String, String}[]
    rng = Random.Xoshiro(options.seed)

    for scale in options.scales
        host_lift = options.precision(scale) * rand(rng,
            StiefelLieAlgHorMatrix{options.precision}, options.rows, options.columns)
        dense_lift = Matrix(host_lift)
        lift_norm = norm(dense_lift)
        agreement_reference = Matrix{Float64}(Matrix(geodesic(host_lift, AugmentedPade())))
        forward_reference = exp(Matrix{Float64}(dense_lift))

        for path in algorithm_paths(options, host_lift, cuda)
            operation = () -> geodesic(path.input, path.algorithm)
            for repetition in 0:(options.repetitions)
                base = base_record(; algorithm = path.name, backend = path.backend,
                    device = path.device, precision = options.precision,
                    rows = options.rows, stiefel_columns = options.columns, lift_norm,
                    memory_metric = path.memory_metric, repetition, warmup = repetition ==
                                                                             0,
                    seed = options.seed, scale, source, patch_file)
                push!(records,
                    record_invocation(base, operation, path.synchronize_backend,
                        path.allocated_bytes, path.to_host, agreement_reference,
                        forward_reference))
            end
        end
    end
    write_retraction_records(options.output, records)
end

function main(args = ARGS)
    options = parse_options(args)
    options === nothing && return 0
    options.backend == :cuda && import_cuda()
    Base.invokelatest(run_benchmark, options)
    device_backend = options.backend == :cuda ? "CUDA" : "CPU"
    required_paths = [("AugmentedPade", "CPU"), ("ScaledSquaring", device_backend),
        ("NativePade", device_backend)]
    summary = validate_records(options.output; required_paths, go_repo = options.go_repo)
    println("wrote $(summary.rows) retraction records ($(summary.warmups) warm-up, " *
            "$(summary.steady_state) steady-state) to $(abspath(options.output))")
    summary.failures == 0 ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(RetractionRecords.main())
    catch error
        println(stderr, "retraction record benchmark failed: ", sprint(showerror, error))
        exit(1)
    end
end
