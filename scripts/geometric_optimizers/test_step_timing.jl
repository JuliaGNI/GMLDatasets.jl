# Focused regression for the exclusive phase timer and schema-v4 timing fields.

using Test

include(joinpath(@__DIR__, "step_timing.jl"))

@testset "exclusive nested step timing" begin
    # The first pair is a compilation/warm-up sample. The remaining timestamps describe one
    # optimizer phase with two gradient segments, one nested inside retraction/application.
    timestamps = UInt64[0, 10, 100, 110, 130, 140, 145, 155, 170, 200]
    clock_index = Ref(0)
    clock() = timestamps[(clock_index[] += 1)]
    synchronizations = Ref(0)
    synchronize() = (synchronizations[] += 1; nothing)
    timer = ExclusiveStepTimer(synchronize; clock=clock)

    timer(:optimizer_state_direction, :enter)
    timer(:optimizer_state_direction, :exit)
    @test timer.optimizer_ns == 10
    reset_step_timing!(timer)

    timer(:optimizer_state_direction, :enter) # 100
    timer(:gradient, :enter)                  # 110: optimizer +10
    timer(:gradient, :exit)                   # 130: gradient +20
    timer(:retraction_application, :enter)    # 140: optimizer +10
    timer(:gradient, :enter)                  # 145: retraction +5
    timer(:gradient, :exit)                   # 155: gradient +10
    timer(:retraction_application, :exit)     # 170: retraction +15
    timer(:optimizer_state_direction, :exit)  # 200: optimizer +30

    timing = step_timing(timer)
    @test synchronizations[] == length(timestamps)
    @test timer.gradient_calls == 2
    @test timer.retraction_calls == 1
    @test timer.optimizer_calls == 1
    @test timing.timed_steps == 1
    @test timing.gradient_ad_seconds_total ≈ 30e-9
    @test timing.optimizer_state_direction_seconds_total ≈ 50e-9
    @test timing.retraction_application_seconds_total ≈ 20e-9
    @test timing.gradient_ad_seconds_per_step == timing.gradient_ad_seconds_total
    @test timing.optimizer_state_direction_seconds_per_step ==
          timing.optimizer_state_direction_seconds_total
    @test timing.retraction_application_seconds_per_step ==
          timing.retraction_application_seconds_total
    @test all(isfinite, step_timing_csv_values(timing))
    @test all(>=(0), step_timing_csv_values(timing))
end

@testset "warm-up reset and schema-v4 columns" begin
    timer = ExclusiveStepTimer()
    timing = step_timing(timer)
    @test timing.timed_steps == 0
    @test all(iszero, step_timing_csv_values(timing))
    @test MNIST_RUN_SCHEMA_VERSION == 4
    @test STEP_TIMING_CSV_COLUMNS == (
        "timed_steps",
        "gradient_ad_seconds_total",
        "gradient_ad_seconds_per_step",
        "optimizer_state_direction_seconds_total",
        "optimizer_state_direction_seconds_per_step",
        "retraction_application_seconds_total",
        "retraction_application_seconds_per_step",
    )
end

@testset "device synchronization selection" begin
    synchronizations = Ref(0)
    synchronize() = (synchronizations[] += 1; nothing)
    gpu_synchronize = step_timing_synchronizer(true, synchronize)
    cpu_synchronize = step_timing_synchronizer(false, synchronize)
    gpu_synchronize()
    @test synchronizations[] == 1
    cpu_synchronize()
    @test synchronizations[] == 1
end

@testset "invalid event sequences are rejected" begin
    timer = ExclusiveStepTimer()
    @test_throws ArgumentError timer(:gradient, :exit)
    @test_throws ArgumentError timer(:unknown, :enter)
    timer(:gradient, :enter)
    @test_throws ArgumentError timer(:objective, :exit)
    @test_throws ArgumentError reset_step_timing!(timer)
    timer(:gradient, :exit)
end
