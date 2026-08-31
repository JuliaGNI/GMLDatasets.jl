#!/usr/bin/env bash
#
# Run `scripts/geometric_optimizers/mnist_cuda.jl` on the RTX 4090 workstation.
#
#   screen -S mnist                     # then, inside it:
#   scripts/geometric_optimizers/run_mnist_cuda.sh --smoke   # two epochs each, ~1/250 of the full schedule
#   scripts/geometric_optimizers/run_mnist_cuda.sh           # the real thing, 500 epochs each
#
# Detach with `C-a d`, log out, come back with `screen -r mnist`. Or start it detached in one
# go, which is what an ssh session is for:
#
#   screen -dmS mnist scripts/geometric_optimizers/run_mnist_cuda.sh
#
# Either way the run leaves four files behind in `results/`, and they are the point of it:
#
#   <stamp>_report.txt      the environment, the gradient checks, one line per epoch, the
#                           per-run summaries and the verdict, flushed as it goes
#   <stamp>_losses.csv      one row per optimizer step: run, epoch, batch, step, loss
#   <stamp>_parameters.jld2 the trained parameters, the losses, the timings and the accuracies,
#                           rewritten after every configuration
#   <stamp>_stdout.txt      everything the process wrote, including a backtrace if it died
#
# `<stamp>` is the start time, so a second run never overwrites the first. Copy all four off
# the machine when it is done — they are self-contained.
#
# Run this from the repository root; it does not care where it is invoked from otherwise.

set -euo pipefail

cd "$(dirname "$0")/.."

julia_bin="${JULIA:-julia}"
mode="full"

case "${1:-}" in
    --smoke) mode="smoke" ;;
    "") ;;
    *) echo "usage: $0 [--smoke]" >&2; exit 2 ;;
esac

stamp="$(date +%Y%m%d_%H%M%S)_${mode}"
mkdir -p results

export MNIST_REPORT="results/${stamp}_report.txt"
export MNIST_LOSSES="results/${stamp}_losses.csv"
export MNIST_OUTPUT="results/${stamp}_parameters.jld2"
stdout_log="results/${stamp}_stdout.txt"

if [ "$mode" = "smoke" ]; then
    # Two epochs, evaluated every epoch: enough to see the gradient check pass, every code path
    # of all four configurations execute, and the report and the `.jld2` be written — and it
    # gives the per-epoch time the full schedule should be extrapolated from. Two hundred and
    # fifty times this is the full run.
    export MNIST_N_EPOCHS=2
    export MNIST_ACCURACY_EVERY=1
fi

# The progress line is a carriage return per batch. It is what tells you the first run is
# moving, so it goes to the terminal, but not into the log that `tee` writes.
export MNIST_PROGRESS=0

echo "mode              $mode"
echo "julia             $("$julia_bin" --version)"
echo "report            $MNIST_REPORT"
echo "loss curves       $MNIST_LOSSES"
echo "parameters        $MNIST_OUTPUT"
echo "stdout            $stdout_log"
echo "follow it with    tail -f $MNIST_REPORT"
echo

# The checked-in manifest pins the reviewed experiment stack. Instantiating it here means an
# unavailable dependency fails before MNIST is downloaded; it is a no-op once the environment
# exists.
"$julia_bin" --project=scripts --startup-file=no -e 'using Pkg; Pkg.instantiate()'
echo

# `--startup-file=no` so that nothing in the user's `startup.jl` reaches a run of this length.
# `stdbuf` keeps the redirected output line buffered, so `tail -f` on it is worth something;
# it is not on every machine, hence the fallback.
> "$stdout_log"

# `set +e` around the pipeline: with `errexit` and `pipefail` a julia that dies would take this
# script down with it before it could say where it left its output, which is the one thing it
# has to get right.
set +e
if command -v stdbuf > /dev/null; then
    stdbuf -oL -eL "$julia_bin" --project=scripts --startup-file=no scripts/geometric_optimizers/mnist_cuda.jl 2>&1 | tee "$stdout_log"
else
    "$julia_bin" --project=scripts --startup-file=no scripts/geometric_optimizers/mnist_cuda.jl 2>&1 | tee "$stdout_log"
fi
status="${PIPESTATUS[0]}"
set -e

echo
if [ "$status" -eq 0 ]; then
    echo "done. Copy these off the machine:"
    echo "    $MNIST_REPORT"
    echo "    $MNIST_LOSSES"
    echo "    $MNIST_OUTPUT"
    echo "    $stdout_log"
    echo "e.g. from the other machine:"
    echo "    scp '<user>@<host>:$(pwd)/results/${stamp}_*' ."
else
    echo "julia exited with $status — the report is complete up to where it stopped:"
    echo "    $MNIST_REPORT"
    echo "and the backtrace is at the end of:"
    echo "    $stdout_log"
fi

exit "$status"
