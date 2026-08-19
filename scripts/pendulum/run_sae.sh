#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
julia_bin="${JULIA:-julia}"
output="${PENDULUM_SAE_OUTPUT:-$repo_root/pendulum_sae.h5}"

if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 2
fi

echo "project  $repo_root"
echo "julia    $("$julia_bin" --version)"
echo "seed     ${PENDULUM_SEED:-123}"
echo "epochs   ${PENDULUM_N_EPOCHS:-10}"
echo "output   $output"
echo

"$julia_bin" --project="$repo_root" --startup-file=no \
    "$repo_root/scripts/pendulum/train_sae.jl"
