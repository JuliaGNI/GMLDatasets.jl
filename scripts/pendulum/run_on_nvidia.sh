#!/usr/bin/env bash
# Run the pendulum symplectic-autoencoder pipeline on a CUDA workstation.
#
# Syncs this repository to the remote, instantiates the scripts environment there, trains, learns
# the reduced dynamics, and then reports the result PER BRANCH of the cylinder -- which is the step
# that matters, and the one an aggregate reconstruction error hides.
#
# Usage, from the repository root:
#
#   bash scripts/pendulum/run_on_nvidia.sh
#
# Everything a run varies is an environment variable, so a sweep needs no file edits on the remote:
#
#   SAE_SEED=123 SAE_FRACS=one SAE_TSPAN=40 SAE_EPOCHS=12000 SAE_UPSCALE=20 \
#     SAE_ETA=1e-4 SAE_OUT=pendulum_sae.h5 \
#     bash scripts/pendulum/run_on_nvidia.sh
#
# SAE_FRACS=both puts rotating data on BOTH branches of the cylinder. Read the header of
# train_sae.jl before reading the result of that: one of the two rotating families is then forced
# into a shrinking nest whose enclosed area is bounded while its action is not, so its reconstruction
# cannot be fixed by capacity. Step 3 is what tells that apart from a poor fit.
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/run_on_nvidia.sh), where it synced and ran two scripts that
# both hardcoded CPU() and never synced the one CUDA script -- so every "GPU run" was a CPU run on
# another machine. Both scripts pick the device themselves now and say which in their first line.

set -euo pipefail

REMOTE="${REMOTE:-benbradmin@pc-benbr-2}"
REMOTE_DIR="${REMOTE_DIR:-SciCade26/GMLDatasets}"   # relative to the remote home
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Forwarded to the scripts on the remote; see train_sae.jl.
SAE_SEED="${SAE_SEED:-123}"
SAE_FRACS="${SAE_FRACS:-one}"
SAE_TSPAN="${SAE_TSPAN:-40}"
SAE_EPOCHS="${SAE_EPOCHS:-12000}"
SAE_UPSCALE="${SAE_UPSCALE:-20}"
SAE_ETA="${SAE_ETA:-1e-4}"
SAE_OUT="${SAE_OUT:-pendulum_sae.h5}"
RUN_STEP2="${RUN_STEP2:-1}"

echo "==> Syncing $REPO_ROOT to ${REMOTE}:${REMOTE_DIR}"
ssh "$REMOTE" "mkdir -p '${REMOTE_DIR}'"
# Outputs are excluded in both directions: the remote keeps its own out/, which is what comes back.
rsync -az --delete --info=progress2 \
    --exclude='.git' --exclude='docs/build*' --exclude='out' \
    --exclude='*.h5' --exclude='*.png' --exclude='plots' --exclude='Animations' \
    "${REPO_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"

echo "==> Instantiating the scripts environment on the remote"
# CUDA installs (though non-functional) on machines without a device, so it is a plain dependency of
# scripts/Project.toml and needs no special casing here.
ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" bash <<'ENDSSH'
    set -euo pipefail
    cd "$HOME/$REMOTE_DIR"
    julia --project=scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
ENDSSH

echo "==> Step 1: train the SAE"
echo "    seed=${SAE_SEED} fracs=${SAE_FRACS} tspan=${SAE_TSPAN} epochs=${SAE_EPOCHS}"
echo "    upscale=${SAE_UPSCALE} eta=${SAE_ETA} -> ${SAE_OUT}"
# The script says CUDA or CPU in its first line of output. If that says CPU on the workstation then
# CUDA.jl did not load; fix that before reading any timing.
ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" \
    SAE_SEED="$SAE_SEED" SAE_FRACS="$SAE_FRACS" SAE_TSPAN="$SAE_TSPAN" \
    SAE_EPOCHS="$SAE_EPOCHS" SAE_UPSCALE="$SAE_UPSCALE" SAE_ETA="$SAE_ETA" \
    SAE_OUT="$SAE_OUT" bash <<'ENDSSH'
    set -euo pipefail
    cd "$HOME/$REMOTE_DIR"
    export GML_OUTDIR="$PWD/out"; mkdir -p "$GML_OUTDIR"
    julia --project=scripts scripts/pendulum/train_sae.jl \
        2>&1 | tee "$GML_OUTDIR/log_train_sae.txt"
ENDSSH

if [ "$RUN_STEP2" = "1" ]; then
    echo "==> Step 2: reduced dynamics"
    ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SAE_OUT="$SAE_OUT" bash <<'ENDSSH'
        set -euo pipefail
        cd "$HOME/$REMOTE_DIR"
        export GML_OUTDIR="$PWD/out"
        export SAE_WEIGHTS="$GML_OUTDIR/$SAE_OUT"
        julia --project=scripts scripts/pendulum/reduced_networks.jl \
            2>&1 | tee "$GML_OUTDIR/log_reduced.txt"
ENDSSH
fi

echo "==> Step 3: the per-branch report, and the latent figure"
# Not the training loss and not an aggregate reconstruction error. This table: the action and the
# latent invariant per regime AND per sign of p_theta, with the three checks -- sign-constant,
# nested, simple -- that say whether the latent space is still a chart on each family.
ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SAE_OUT="$SAE_OUT" SAE_UPSCALE="$SAE_UPSCALE" bash <<'ENDSSH'
    set -euo pipefail
    cd "$HOME/$REMOTE_DIR"
    export GML_OUTDIR="$PWD/out"
    export SAE_WEIGHTS="$GML_OUTDIR/$SAE_OUT"
    julia --project=scripts scripts/pendulum/branch_report.jl \
        2>&1 | tee "$GML_OUTDIR/log_branch_report.txt"
    julia --project=scripts scripts/pendulum/latent_plot.jl \
        2>&1 | tee "$GML_OUTDIR/log_latent_plot.txt"
ENDSSH

echo "==> Fetching results"
DEST="${GML_OUTDIR:-$REPO_ROOT/out}"
mkdir -p "$DEST"
rsync -az --info=progress2 "${REMOTE}:${REMOTE_DIR}/out/" "${DEST}/"

echo "==> Done. Weights, plots and logs are in ${DEST}"
echo "    Read log_branch_report.txt before anything else."
