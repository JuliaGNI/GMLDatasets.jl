# Exclusive timing for one optimizer step.
#
# `GeometricOptimizers` emits properly nested phase events. This observer synchronizes the active
# device before taking every boundary timestamp and pauses an outer phase while a nested phase is
# running. The three reported categories are therefore disjoint; objective/merit evaluations are
# observed only so that they are not mislabeled as optimizer-state work.
#
# This file contains definitions only so the focused regression can exercise the same timer and
# schema helpers as the trainer without loading data or starting an experiment.

const MNIST_RUN_SCHEMA_VERSION = 4
const STEP_TIMING_CSV_COLUMNS = (
    "timed_steps",
    "gradient_ad_seconds_total",
    "gradient_ad_seconds_per_step",
    "optimizer_state_direction_seconds_total",
    "optimizer_state_direction_seconds_per_step",
    "retraction_application_seconds_total",
    "retraction_application_seconds_per_step",
)

"""
    step_timing_synchronizer(device_active, synchronize)

Return `synchronize` when a device is active and a no-op otherwise. Keeping this choice outside
the timer makes the CPU path independent of CUDA while preserving the same boundary calls.
"""
step_timing_synchronizer(device_active::Bool, synchronize) =
    device_active ? synchronize : (() -> nothing)

mutable struct ExclusiveStepTimer{S,C}
    synchronize::S
    clock::C
    stack::Vector{Symbol}
    started_ns::UInt64
    gradient_ns::UInt64
    objective_ns::UInt64
    optimizer_ns::UInt64
    retraction_ns::UInt64
    gradient_calls::Int
    objective_calls::Int
    optimizer_calls::Int
    retraction_calls::Int
end

function ExclusiveStepTimer(synchronize=(() -> nothing); clock=time_ns)
    ExclusiveStepTimer(synchronize, clock, Symbol[], UInt64(0), UInt64(0), UInt64(0),
        UInt64(0), UInt64(0), 0, 0, 0, 0)
end

function _add_phase_time!(timer::ExclusiveStepTimer, phase::Symbol, elapsed_ns::UInt64)
    if phase === :gradient
        timer.gradient_ns += elapsed_ns
    elseif phase === :objective
        timer.objective_ns += elapsed_ns
    elseif phase === :optimizer_state_direction
        timer.optimizer_ns += elapsed_ns
    elseif phase === :retraction_application
        timer.retraction_ns += elapsed_ns
    else
        throw(ArgumentError("unknown optimizer phase: $phase"))
    end
    nothing
end

function _complete_phase!(timer::ExclusiveStepTimer, phase::Symbol)
    if phase === :gradient
        timer.gradient_calls += 1
    elseif phase === :objective
        timer.objective_calls += 1
    elseif phase === :optimizer_state_direction
        timer.optimizer_calls += 1
    elseif phase === :retraction_application
        timer.retraction_calls += 1
    else
        throw(ArgumentError("unknown optimizer phase: $phase"))
    end
    nothing
end

function (timer::ExclusiveStepTimer)(phase::Symbol, event::Symbol)
    if event === :enter
        phase in (:gradient, :objective, :optimizer_state_direction,
            :retraction_application) || throw(ArgumentError("unknown optimizer phase: $phase"))
        timer.synchronize()
        now = UInt64(timer.clock())
        isempty(timer.stack) ||
            _add_phase_time!(timer, last(timer.stack), now - timer.started_ns)
        push!(timer.stack, phase)
        timer.started_ns = now
    elseif event === :exit
        isempty(timer.stack) && throw(ArgumentError("cannot exit $phase: no timing phase is active"))
        last(timer.stack) === phase || throw(ArgumentError(
            "cannot exit $phase while $(last(timer.stack)) is active"))
        timer.synchronize()
        now = UInt64(timer.clock())
        _add_phase_time!(timer, phase, now - timer.started_ns)
        pop!(timer.stack)
        _complete_phase!(timer, phase)
        timer.started_ns = now
    else
        throw(ArgumentError("unknown optimizer phase event: $event"))
    end
    nothing
end

"""
    reset_step_timing!(timer)

Clear every accumulated duration and call count. The orchestrator performs compilation warm-up in
an unrecorded step and reconstructs the seeded training state; this reset is the boundary between
that discarded sample and the measured run.
"""
function reset_step_timing!(timer::ExclusiveStepTimer)
    isempty(timer.stack) || throw(ArgumentError("cannot reset timing while a phase is active"))
    timer.started_ns = UInt64(0)
    timer.gradient_ns = UInt64(0)
    timer.objective_ns = UInt64(0)
    timer.optimizer_ns = UInt64(0)
    timer.retraction_ns = UInt64(0)
    timer.gradient_calls = 0
    timer.objective_calls = 0
    timer.optimizer_calls = 0
    timer.retraction_calls = 0
    timer
end

"""
    step_timing(timer)

Return schema-v4 totals and per-completed-step values in seconds. Objective evaluations are
intentionally absent: observing them makes the three requested categories exclusive, but does not
turn forward/merit evaluation into optimizer-state work.
"""
function step_timing(timer::ExclusiveStepTimer)
    isempty(timer.stack) || throw(ArgumentError("cannot snapshot timing while a phase is active"))
    steps = timer.optimizer_calls
    divisor = max(steps, 1)
    gradient_seconds = Float64(timer.gradient_ns) * 1.0e-9
    optimizer_seconds = Float64(timer.optimizer_ns) * 1.0e-9
    retraction_seconds = Float64(timer.retraction_ns) * 1.0e-9
    (
        timed_steps=steps,
        gradient_ad_seconds_total=gradient_seconds,
        gradient_ad_seconds_per_step=gradient_seconds / divisor,
        optimizer_state_direction_seconds_total=optimizer_seconds,
        optimizer_state_direction_seconds_per_step=optimizer_seconds / divisor,
        retraction_application_seconds_total=retraction_seconds,
        retraction_application_seconds_per_step=retraction_seconds / divisor,
    )
end

step_timing_csv_values(timing::NamedTuple) = Tuple(getproperty(timing, Symbol(column))
    for column in STEP_TIMING_CSV_COLUMNS)
