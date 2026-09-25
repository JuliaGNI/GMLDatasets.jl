# Focused regression for the schema-v4 timing adapter.
#
# What this file is *not* about: whether `GeometricOptimizers.PhaseTimer` nests its phases correctly,
# pauses an enclosing phase while an inner one runs, synchronises at every boundary, or rejects an
# `:exit` with no matching `:enter`. Those are upstream's semantics and
# `GeometricOptimizers/test/optimizer_observer.jl` pins them; re-deriving them here from a synthetic
# clock means this suite fails for upstream's reasons, about code it does not own.
#
# What is local is the adapter: the seven schema-v4 fields, their order, the relation between each
# total and its per-step value, the `0` and `NaN` sentinels, and the refusal to infer completed steps
# from attempted ones.
#
#   julia --project=scripts scripts/geometric_optimizers/test_step_timing.jl

using Test
using GeometricOptimizers: observe_optimizer_phase

include(joinpath(@__DIR__, "..", "revision", "records.jl"))
using .RunRecords: IMAGE_RECORD_HEADER, MNIST_RUN_SCHEMA_VERSION, STEP_TIMING_COLUMNS
include(joinpath(@__DIR__, "step_timing.jl"))

@testset "the seven schema-v4 fields, in the order the header names them" begin
    # `step_timing` returns a `NamedTuple` whose keys *are* the columns, which is what lets
    # `step_timing_values` be `values` of it and lets `step_timing.jl` know nothing about
    # `headers.jl`. This is the one place the two have to meet.
    timing = step_timing(ExclusiveStepTimer(), 0)
    @test collect(string.(keys(timing))) == STEP_TIMING_COLUMNS
    @test step_timing_values(timing) === values(timing)

    @test MNIST_RUN_SCHEMA_VERSION == 4
    # The trainer writes these seven under exactly these names and the validator reads them from the
    # same constant; the last three columns of `IMAGE_RECORD_HEADER` follow them, so a column
    # inserted between the two would show up here.
    @test IMAGE_RECORD_HEADER[(end - 9):(end - 3)] == STEP_TIMING_COLUMNS
end

@testset "each total divided by the completed steps is its per-step value" begin
    timer = ExclusiveStepTimer()
    for _ in 1:4
        observe_optimizer_phase(timer, :optimizer_state_direction) do
            observe_optimizer_phase(timer, :gradient) do
            end
            observe_optimizer_phase(timer, :retraction_application) do
            end
        end
    end

    timing = step_timing(timer, 4)
    @test timing.timed_steps == 4
    @test all(isfinite, step_timing_values(timing))
    @test all(>=(0), step_timing_values(timing))
    for phase in ("gradient_ad", "optimizer_state_direction", "retraction_application")
        total = getproperty(timing, Symbol(phase, "_seconds_total"))
        per_step = getproperty(timing, Symbol(phase, "_seconds_per_step"))
        @test per_step ≈ total / 4
    end
end

@testset "a zero-step result is zero and not a division by zero" begin
    timing = step_timing(ExclusiveStepTimer(), 0)
    @test timing.timed_steps == 0
    @test all(iszero, step_timing_values(timing))
    @test_throws ArgumentError step_timing(ExclusiveStepTimer(), -1)
end

@testset "completed steps are not inferred from attempted steps" begin
    # A step that threw was attempted and not completed, and the two counts diverging is a result
    # whose per-step values would be silently wrong. It is an error rather than a smaller divisor.
    timer = ExclusiveStepTimer()
    @test_throws ErrorException observe_optimizer_phase(timer, :optimizer_state_direction) do
        error("synthetic step failure")
    end
    @test timer.calls[:optimizer_state_direction] == 1
    @test_throws ArgumentError step_timing(timer, 0)
end

@testset "the device is synchronised only where there is a device" begin
    # The CPU path has to be independent of CUDA, and the boundary calls have to be the same ones
    # either way — which is why the choice is outside the timer rather than inside it.
    synchronizations = Ref(0)
    synchronize() = (synchronizations[] += 1; nothing)
    step_timing_synchronizer(true, synchronize)()
    @test synchronizations[] == 1
    step_timing_synchronizer(false, synchronize)()
    @test synchronizations[] == 1
end
