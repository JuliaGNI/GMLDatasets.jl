using Test
using SHA: sha256

include("retraction_record_schema.jl")
using .RetractionRecordSchema: HEADER, validate_records, write_records

function fixture_row(; algorithm="ScaledSquaring", repetition=0, warmup=true, success=true,
    patch_sha, error_type="", error_message="")
    Dict(
        "schema_version" => "1",
        "algorithm" => algorithm,
        "backend" => "CPU",
        "device" => "fixture-cpu",
        "precision" => "Float64",
        "matrix_rows" => "6",
        "matrix_columns" => "6",
        "stiefel_columns" => "2",
        "lift_norm" => "0.5",
        "agreement_error" => success ? "1.0e-15" : "NaN",
        "forward_error" => success ? "2.0e-15" : "NaN",
        "manifold_constraint_error" => success ? "3.0e-15" : "NaN",
        "runtime_seconds" => "0.001",
        "memory_bytes" => success ? "1024" : "0",
        "memory_metric" => "host_allocated_bytes",
        "repetition" => string(repetition),
        "warmup" => string(warmup),
        "success" => string(success),
        "error_type" => error_type,
        "error_message" => error_message,
        "seed" => "1234",
        "scale" => "0.1",
        "go_sha" => repeat("a", 40),
        "go_dirty" => "true",
        "go_patch_file" => "go.patch",
        "go_patch_sha256" => patch_sha,
    )
end

@testset "retraction record schema" begin
    mktempdir() do directory
        patch = Vector{UInt8}("fixture patch\n")
        write(joinpath(directory, "go.patch"), patch)
        patch_sha = bytes2hex(sha256(patch))
        records = Dict{String,String}[]
        for algorithm in ("ScaledSquaring", "NativePade", "AugmentedPade")
            push!(records, fixture_row(; algorithm, patch_sha))
            push!(records, fixture_row(; algorithm, repetition=1, warmup=false, patch_sha))
        end
        path = joinpath(directory, "records.csv")
        write_records(path, records)
        summary = validate_records(path; required_paths=[
            ("ScaledSquaring", "CPU"), ("NativePade", "CPU"), ("AugmentedPade", "CPU")])
        @test summary == (rows=6, warmups=3, steady_state=3, failures=0)
        @test split(readline(path), ',') == HEADER

        failure_records = [
            fixture_row(; patch_sha),
            fixture_row(; repetition=1, warmup=false, success=false, patch_sha,
                error_type="ErrorException", error_message="intentional, quoted failure"),
        ]
        failure_path = joinpath(directory, "failure.csv")
        write_records(failure_path, failure_records)
        failure_summary = validate_records(failure_path; allow_failures=true,
            require_failure=true)
        @test failure_summary.failures == 1
        @test_throws ArgumentError validate_records(failure_path;
            required_paths=[("ScaledSquaring", "CPU")], allow_failures=true)
        @test_throws ArgumentError validate_records(failure_path)

        malformed = read(path, String)
        write(joinpath(directory, "malformed.csv"),
            replace(malformed, "host_allocated_bytes" => "peak_bytes"; count=1))
        @test_throws ArgumentError validate_records(joinpath(directory, "malformed.csv"))
    end
end
