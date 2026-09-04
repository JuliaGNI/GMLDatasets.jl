#!/usr/bin/env julia

module RetractionRecords

using GeometricOptimizers
using GeometricOptimizers: AugmentedPade, NativePade, ScaledSquaring, SkewSymMatrix,
    StiefelLieAlgHorMatrix, geodesic
using LinearAlgebra: I, norm
import Random

include("retraction_record_schema.jl")
using .RetractionRecordSchema: capture_source, validate_records, write_records

Base.@kwdef struct Options
    output::String = "retraction-records.csv"
    patch_output::String = ""
    go_repo::String = ""
    backend::Symbol = :cpu
    precision::DataType = Float64
    rows::Int = 20
    stiefel_columns::Int = 3
    scales::Vector{Float64} = [0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0]
    repetitions::Int = 20
    seed::Int = 1234
end

function usage(io::IO=stdout)
    println(io, """usage: retraction_records.jl --go-repo DIR [options]

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
make CUDA available for CUDA mode while this script runs with the GO repository as its project.""")
end

function take_value(args, index, option)
    index == length(args) && throw(ArgumentError("missing value for $option"))
    args[index + 1], index + 2
end

function parse_options(args)
    values = Dict{Symbol,Any}()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            usage()
            return nothing
        elseif arg == "--output"
            values[:output], index = take_value(args, index, arg)
        elseif arg == "--patch-output"
            values[:patch_output], index = take_value(args, index, arg)
        elseif arg == "--go-repo"
            values[:go_repo], index = take_value(args, index, arg)
        elseif arg == "--backend"
            value, index = take_value(args, index, arg)
            value in ("cpu", "cuda") || throw(ArgumentError("backend must be cpu or cuda"))
            values[:backend] = Symbol(value)
        elseif arg == "--precision"
            value, index = take_value(args, index, arg)
            value in ("Float32", "Float64") ||
                throw(ArgumentError("precision must be Float32 or Float64"))
            values[:precision] = value == "Float32" ? Float32 : Float64
        elseif arg == "--rows"
            value, index = take_value(args, index, arg)
            values[:rows] = parse(Int, value)
        elseif arg == "--columns"
            value, index = take_value(args, index, arg)
            values[:stiefel_columns] = parse(Int, value)
        elseif arg == "--scales"
            value, index = take_value(args, index, arg)
            values[:scales] = parse.(Float64, split(value, ','))
        elseif arg == "--repetitions"
            value, index = take_value(args, index, arg)
            values[:repetitions] = parse(Int, value)
        elseif arg == "--seed"
            value, index = take_value(args, index, arg)
            values[:seed] = parse(Int, value)
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end

    options = Options(; values...)
    isempty(options.go_repo) && throw(ArgumentError("--go-repo is required"))
    options.rows > 0 || throw(ArgumentError("--rows must be positive"))
    0 < options.stiefel_columns <= options.rows ||
        throw(ArgumentError("--columns must be positive and no greater than --rows"))
    !isempty(options.scales) && all(isfinite(s) && s >= 0 for s in options.scales) ||
        throw(ArgumentError("--scales must contain finite nonnegative values"))
    options.repetitions > 0 || throw(ArgumentError("--repetitions must be positive"))
    options.seed >= 0 || throw(ArgumentError("--seed must be nonnegative"))
    options
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

function base_record(; algorithm, backend, device, precision, rows, stiefel_columns, lift_norm,
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
        "go_patch_sha256" => source.patch_sha256,
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
        agreement = norm(output - agreement_reference) / norm(agreement_reference)
        forward = norm(output - forward_reference) / norm(forward_reference)
        manifold_error = norm(output' * output - I)
        merge(base, Dict(
            "agreement_error" => float_string(agreement),
            "forward_error" => float_string(forward),
            "manifold_constraint_error" => float_string(manifold_error),
            "runtime_seconds" => float_string(elapsed),
            "memory_bytes" => string(memory),
            "success" => "true",
            "error_type" => "",
            "error_message" => "",
        ))
    catch error
        try
            synchronize_backend()
        catch
        end
        elapsed = max(0.0, (time_ns() - started) / 1.0e9)
        merge(base, Dict(
            "agreement_error" => "NaN",
            "forward_error" => "NaN",
            "manifold_constraint_error" => "NaN",
            "runtime_seconds" => float_string(elapsed),
            "memory_bytes" => "0",
            "success" => "false",
            "error_type" => string(typeof(error)),
            "error_message" => sanitize_error(error),
        ))
    end
end

function algorithm_paths(options, host_lift, cuda)
    if options.backend == :cpu
        synchronize_backend = () -> nothing
        allocated_bytes = operation -> @allocated operation()
        to_host = result -> Matrix(result)
        return [
            (name="ScaledSquaring", backend="CPU", device=Sys.CPU_NAME,
                algorithm=ScaledSquaring(), input=host_lift, synchronize_backend,
                allocated_bytes, to_host, memory_metric="host_allocated_bytes"),
            (name="NativePade", backend="CPU", device=Sys.CPU_NAME,
                algorithm=NativePade(), input=host_lift, synchronize_backend,
                allocated_bytes, to_host, memory_metric="host_allocated_bytes"),
            (name="AugmentedPade", backend="CPU", device=Sys.CPU_NAME,
                algorithm=AugmentedPade(), input=host_lift, synchronize_backend,
                allocated_bytes, to_host, memory_metric="host_allocated_bytes"),
        ]
    end

    gpu_lift = cuda_lift(host_lift, cuda)
    gpu_sync = () -> cuda.synchronize()
    gpu_allocated = operation -> Base.invokelatest(device_allocated, operation)
    gpu_to_host = result -> Array(parent(result))
    cpu_sync = () -> nothing
    cpu_allocated = operation -> @allocated operation()
    cpu_to_host = result -> Matrix(result)
    device = string(cuda.name(cuda.device()))
    [
        (name="ScaledSquaring", backend="CUDA", device,
            algorithm=ScaledSquaring(), input=gpu_lift, synchronize_backend=gpu_sync,
            allocated_bytes=gpu_allocated, to_host=gpu_to_host,
            memory_metric="device_allocated_bytes"),
        (name="NativePade", backend="CUDA", device,
            algorithm=NativePade(), input=gpu_lift, synchronize_backend=gpu_sync,
            allocated_bytes=gpu_allocated, to_host=gpu_to_host,
            memory_metric="device_allocated_bytes"),
        (name="AugmentedPade", backend="CPU", device=Sys.CPU_NAME,
            algorithm=AugmentedPade(), input=host_lift, synchronize_backend=cpu_sync,
            allocated_bytes=cpu_allocated, to_host=cpu_to_host,
            memory_metric="host_allocated_bytes"),
    ]
end

function run_benchmark(options::Options)
    patch_path = isempty(options.patch_output) ?
        joinpath(dirname(abspath(options.output)), "geometricoptimizers-retraction.patch") :
        abspath(options.patch_output)
    source = capture_source(options.go_repo, patch_path)
    verify_loaded_source(source.root)
    patch_file = relpath(patch_path, dirname(abspath(options.output)))
    cuda = options.backend == :cuda ? loaded_cuda() : nothing
    records = Dict{String,String}[]
    rng = Random.Xoshiro(options.seed)

    for scale in options.scales
        host_lift = options.precision(scale) * rand(rng,
            StiefelLieAlgHorMatrix{options.precision}, options.rows, options.stiefel_columns)
        dense_lift = Matrix(host_lift)
        lift_norm = norm(dense_lift)
        agreement_reference = Matrix{Float64}(Matrix(geodesic(host_lift, AugmentedPade())))
        forward_reference = exp(Matrix{Float64}(dense_lift))

        for path in algorithm_paths(options, host_lift, cuda)
            operation = () -> geodesic(path.input, path.algorithm)
            for repetition in 0:options.repetitions
                base = base_record(; algorithm=path.name, backend=path.backend, device=path.device,
                    precision=options.precision, rows=options.rows,
                    stiefel_columns=options.stiefel_columns, lift_norm,
                    memory_metric=path.memory_metric, repetition, warmup=repetition == 0,
                    seed=options.seed, scale, source, patch_file)
                push!(records, record_invocation(base, operation, path.synchronize_backend,
                    path.allocated_bytes, path.to_host, agreement_reference, forward_reference))
            end
        end
    end
    write_records(options.output, records)
    records
end

function main(args=ARGS)
    options = parse_options(args)
    options === nothing && return 0
    options.backend == :cuda && import_cuda()
    records = Base.invokelatest(run_benchmark, options)
    required_paths = options.backend == :cuda ?
        [("ScaledSquaring", "CUDA"), ("NativePade", "CUDA"), ("AugmentedPade", "CPU")] :
        [(name, "CPU") for name in ("ScaledSquaring", "NativePade", "AugmentedPade")]
    summary = validate_records(options.output; required_paths, go_repo=options.go_repo)
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
