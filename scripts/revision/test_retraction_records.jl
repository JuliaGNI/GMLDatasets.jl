using Test
using SHA: sha256

include("retraction_records.jl")

@testset "retraction benchmark failure records" begin
    mktempdir() do directory
        patch = Vector{UInt8}("fixture patch\n")
        patch_path = joinpath(directory, "go.patch")
        write(patch_path, patch)
        source = (sha=repeat("a", 40), dirty=true,
            patch_sha256=bytes2hex(sha256(patch)))
        reference = [1.0 0.0; 0.0 1.0]

        warmup_base = RetractionRecords.base_record(; algorithm="ScaledSquaring", backend="CPU",
            device="fixture-cpu", precision=Float64, rows=2, stiefel_columns=1, lift_norm=0.0,
            memory_metric="host_allocated_bytes", repetition=0, warmup=true, seed=1234,
            scale=0.0, source, patch_file=basename(patch_path))
        warmup = RetractionRecords.record_invocation(warmup_base, () -> reference, () -> nothing,
            _ -> 16, identity, reference, reference)

        failure_base = merge(warmup_base,
            Dict("repetition" => "1", "warmup" => "false"))
        failure = RetractionRecords.record_invocation(failure_base,
            () -> error("intentional, quoted failure"), () -> nothing, _ -> 16, identity,
            reference, reference)
        @test failure["success"] == "false"
        @test failure["agreement_error"] == "NaN"
        @test failure["memory_bytes"] == "0"
        @test failure["error_type"] == "ErrorException"
        @test failure["error_message"] == "intentional, quoted failure"

        records_path = joinpath(directory, "records.csv")
        RetractionRecords.write_records(records_path, [warmup, failure])
        summary = RetractionRecords.validate_records(records_path; allow_failures=true,
            require_failure=true)
        @test summary == (rows=2, warmups=1, steady_state=1, failures=1)
    end
end
