# Schema-v4 adapter for GeometricOptimizers' exclusive phase timer.
#
# `GeometricOptimizers.PhaseTimer` synchronizes the active device before taking every boundary
# timestamp and pauses an outer phase while a nested phase is running. The three reported
# categories are therefore disjoint; objective/merit evaluations are observed only so that they
# are not mislabeled as optimizer-state work.
#
# This file contains definitions only so the focused regression can exercise the same timer without
# loading data or starting an experiment.
#
# It deliberately does **not** know the column names. `step_timing` returns a `NamedTuple` whose keys
# are the seven schema-v4 columns *in order*, so `step_timing_values` is `values` of it and nothing
# here has to be kept in step with `headers.jl` — which is where those names live, and where the
# validators read them from. `test_step_timing.jl` asserts that the keys and the constant agree,
# which is the one place the two have to meet.

using GeometricOptimizers: PhaseTimer

const STEP_TIMING_PHASES = (
    :gradient, :objective, :optimizer_state_direction, :retraction_application)

"""
    step_timing_synchronizer(device_active, synchronize)

Return `synchronize` when a device is active and a no-op otherwise. Keeping this choice outside
the timer makes the CPU path independent of CUDA while preserving the same boundary calls.
"""
function step_timing_synchronizer(device_active::Bool, synchronize)
    device_active ? synchronize : (() -> nothing)
end

"""Construct the upstream exclusive timer with the phases used by schema v4."""
function ExclusiveStepTimer(synchronize = (() -> nothing); clock = time_ns)
    PhaseTimer(; phases = STEP_TIMING_PHASES, synchronize, clock)
end

"""
    reset_step_timing!(timer)

Clear every accumulated duration and call count. The orchestrator performs compilation warm-up in
an unrecorded step and reconstructs the seeded training state; this reset is the boundary between
that discarded sample and the measured run.
"""
reset_step_timing!(timer::PhaseTimer) = empty!(timer)

"""
    step_timing(timer, completed_steps)

Return schema-v4 totals and per-completed-step values in seconds. Objective evaluations are
intentionally absent: observing them makes the three requested categories exclusive, but does not
turn forward/merit evaluation into optimizer-state work. `completed_steps` comes from the loss
series rather than `PhaseTimer.calls`, whose entry count also includes a step that later throws.
"""
function step_timing(timer::PhaseTimer, completed_steps::Integer)
    completed_steps >= 0 || throw(ArgumentError("completed_steps must be nonnegative"))
    attempted_steps = get(timer.calls, :optimizer_state_direction, 0)
    attempted_steps == completed_steps || throw(ArgumentError(
        "timing recorded $attempted_steps optimizer-step attempts for " *
        "$completed_steps completed steps"))
    steps = Int(completed_steps)
    divisor = max(steps, 1)
    seconds(phase) = Float64(get(timer.exclusive, phase, UInt64(0))) * 1.0e-9
    gradient_seconds = seconds(:gradient)
    optimizer_seconds = seconds(:optimizer_state_direction)
    retraction_seconds = seconds(:retraction_application)
    (
        timed_steps = steps,
        gradient_ad_seconds_total = gradient_seconds,
        gradient_ad_seconds_per_step = gradient_seconds / divisor,
        optimizer_state_direction_seconds_total = optimizer_seconds,
        optimizer_state_direction_seconds_per_step = optimizer_seconds / divisor,
        retraction_application_seconds_total = retraction_seconds,
        retraction_application_seconds_per_step = retraction_seconds / divisor
    )
end

"""The seven values in schema-v4 column order, which is the order `step_timing` builds them in."""
step_timing_values(timing::NamedTuple) = values(timing)
