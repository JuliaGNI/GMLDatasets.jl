#!/usr/bin/env bash
# Run the pendulum symplectic-autoencoder pipeline on a CUDA workstation, detached.
#
# The whole pipeline -- train, reduced dynamics, per-branch report, latent figure -- runs inside a
# `screen` session ON THE WORKSTATION, so it survives this laptop sleeping, the network going, and
# the terminal being closed. Nothing is held open here except the sync, which takes seconds.
#
# Usage, from the repository root:
#
#   bash scripts/pendulum/run_on_nvidia.sh            # sync, then start it detached
#   bash scripts/pendulum/run_on_nvidia.sh --status   # is it still going, and where is it
#   bash scripts/pendulum/run_on_nvidia.sh --attach   # watch it live (Ctrl-a d to leave it running)
#   bash scripts/pendulum/run_on_nvidia.sh --fetch    # bring out/ back when it says DONE
#   bash scripts/pendulum/run_on_nvidia.sh --stop     # kill the session
#   bash scripts/pendulum/run_on_nvidia.sh --foreground   # run here and wait; for smoke tests
#
# Everything a run varies is an environment variable, so a sweep needs no file edits on the remote:
#
#   SAE_SEED=123 SAE_FRACS=one SAE_TSPAN=40 SAE_EPOCHS=12000 SAE_UPSCALE=20 \
#     SAE_ETA=1e-4 SAE_OUT=pendulum_sae.h5 SESSION=sae \
#     bash scripts/pendulum/run_on_nvidia.sh
#
# Give each concurrent run its own SESSION and SAE_OUT, and its own GML_OUTDIR for the fetch.
#
# SAE_FRACS=both puts rotating data on BOTH branches of the cylinder. Read the header of
# train_sae.jl before reading the result of that: one of the two rotating families is then forced
# into a shrinking nest whose enclosed area is bounded while its action is not, so its reconstruction
# cannot be fixed by capacity. The report is what tells that apart from a poor fit.
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/run_on_nvidia.sh), where it synced and ran two scripts that
# both hardcoded CPU() and never synced the one CUDA script -- so every "GPU run" was a CPU run on
# another machine. Both scripts pick the device themselves now and say which in their first line.

set -euo pipefail

REMOTE="${REMOTE:-benbradmin@pc-benbr-2}"
REMOTE_DIR="${REMOTE_DIR:-SciCade26/GMLDatasets}"   # relative to the remote home
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-sae}"

# Forwarded to the scripts on the remote; see train_sae.jl.
SAE_SEED="${SAE_SEED:-123}"
SAE_FRACS="${SAE_FRACS:-one}"
SAE_TSPAN="${SAE_TSPAN:-40}"
SAE_EPOCHS="${SAE_EPOCHS:-12000}"
SAE_UPSCALE="${SAE_UPSCALE:-20}"
SAE_ETA="${SAE_ETA:-1e-4}"
SAE_BATCH="${SAE_BATCH:-256}"
SAE_OUT="${SAE_OUT:-pendulum_sae.h5}"
RUN_STEP2="${RUN_STEP2:-1}"

DEST="${GML_OUTDIR:-$REPO_ROOT/out}"
MODE="${1:-start}"

# ---------------------------------------------------------------------------------------------
# The subcommands that do not start anything
# ---------------------------------------------------------------------------------------------
case "$MODE" in
  --status)
      ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SESSION="$SESSION" bash <<'ENDSSH'
        cd "$HOME/$REMOTE_DIR" 2>/dev/null || { echo "no $REMOTE_DIR on the remote"; exit 1; }
        if [ -f out/STATUS ]; then echo "STATUS: $(cat out/STATUS)"; else echo "STATUS: never started"; fi
        echo "session: $(screen -ls 2>/dev/null | grep -c "\.${SESSION}[[:space:]]" || true) live"
        screen -ls 2>/dev/null | sed -n '2,$p' | sed 's/^/  /' || true
        echo "--- last 15 lines of the newest log ---"
        ls -t out/log_*.txt 2>/dev/null | head -1 | xargs -r tail -15
ENDSSH
      exit 0 ;;
  --attach)
      echo "Attaching to '$SESSION' on $REMOTE. Ctrl-a d detaches and LEAVES IT RUNNING."
      exec ssh -t "$REMOTE" "screen -d -r '$SESSION'" ;;
  --stop)
      ssh "$REMOTE" "screen -S '$SESSION' -X quit || true; echo stopped"
      exit 0 ;;
  --fetch)
      mkdir -p "$DEST"
      rsync -az --info=progress2 "${REMOTE}:${REMOTE_DIR}/out/" "${DEST}/"
      echo "==> out/ is in ${DEST}"
      echo "    Read log_branch_report.txt before anything else."
      exit 0 ;;
  start|--start|--foreground) ;;
  *)  echo "unknown option: $MODE"; sed -n '7,14p' "$0"; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------------------------
echo "==> Syncing $REPO_ROOT to ${REMOTE}:${REMOTE_DIR}"
ssh "$REMOTE" "mkdir -p '${REMOTE_DIR}/out'"
# Outputs are excluded in both directions: the remote keeps its own out/, which is what comes back.
rsync -az --delete --info=progress2 \
    --exclude='.git' --exclude='docs/build*' --exclude='out' \
    --exclude='*.h5' --exclude='*.png' --exclude='plots' --exclude='Animations' \
    "${REPO_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"

# ---------------------------------------------------------------------------------------------
# Build the pipeline script HERE, where the quoting is visible, and ship it. Generating it inside a
# heredoc inside an ssh inside a screen is the version of this that silently mis-expands one
# variable and is found out three hours in.
# ---------------------------------------------------------------------------------------------
PIPE="$(mktemp -t sae_pipeline)"
trap 'rm -f "$PIPE"' EXIT
cat > "$PIPE" <<EOF
#!/usr/bin/env bash
# Generated by run_on_nvidia.sh. Runs on the workstation, under screen. Do not edit here.
set -uo pipefail
cd "\$HOME/${REMOTE_DIR}"

export GML_OUTDIR="\$PWD/out"
mkdir -p "\$GML_OUTDIR"
export SAE_SEED="${SAE_SEED}" SAE_FRACS="${SAE_FRACS}" SAE_TSPAN="${SAE_TSPAN}"
export SAE_EPOCHS="${SAE_EPOCHS}" SAE_UPSCALE="${SAE_UPSCALE}" SAE_ETA="${SAE_ETA}"
export SAE_BATCH="${SAE_BATCH}" SAE_OUT="${SAE_OUT}"

say () { echo "[\$(date '+%F %T')] \$*" | tee -a "\$GML_OUTDIR/log_pipeline.txt"; }
fail () { echo "FAILED at \$1" > "\$GML_OUTDIR/STATUS"; say "FAILED at \$1"; exit 1; }

echo "RUNNING" > "\$GML_OUTDIR/STATUS"
say "instantiate"
julia --project=scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \\
    > "\$GML_OUTDIR/log_instantiate.txt" 2>&1 || fail instantiate

say "train  (seed=${SAE_SEED} fracs=${SAE_FRACS} tspan=${SAE_TSPAN} epochs=${SAE_EPOCHS} upscale=${SAE_UPSCALE} eta=${SAE_ETA})"
julia --project=scripts scripts/pendulum/train_sae.jl \\
    > "\$GML_OUTDIR/log_train_sae.txt" 2>&1 || fail train
# The backend is the first line. Say it here too, so --status shows it without opening the log.
say "  \$(head -1 "\$GML_OUTDIR/log_train_sae.txt")"

export SAE_WEIGHTS="\$GML_OUTDIR/${SAE_OUT}"

if [ "${RUN_STEP2}" = "1" ]; then
    say "reduced dynamics"
    julia --project=scripts scripts/pendulum/reduced_networks.jl \\
        > "\$GML_OUTDIR/log_reduced.txt" 2>&1 || fail reduced
fi

say "branch report"
julia --project=scripts scripts/pendulum/branch_report.jl \\
    > "\$GML_OUTDIR/log_branch_report.txt" 2>&1 || fail report

say "latent figure"
julia --project=scripts scripts/pendulum/latent_plot.jl \\
    > "\$GML_OUTDIR/log_latent_plot.txt" 2>&1 || fail latent_plot

echo "DONE" > "\$GML_OUTDIR/STATUS"
say "DONE"
EOF

# Catch a quoting mistake here rather than three hours into a run.
bash -n "$PIPE" || { echo "generated pipeline does not parse; not launching"; exit 1; }
scp -q "$PIPE" "${REMOTE}:${REMOTE_DIR}/out/run_pipeline.sh"

# ---------------------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------------------
if [ "$MODE" = "--foreground" ]; then
    echo "==> Running in the foreground. A dropped connection kills this; use the default for long runs."
    ssh -t "$REMOTE" "cd '$REMOTE_DIR' && bash out/run_pipeline.sh"
    mkdir -p "$DEST"
    rsync -az --info=progress2 "${REMOTE}:${REMOTE_DIR}/out/" "${DEST}/"
    echo "==> out/ is in ${DEST}"
    exit 0
fi

# screen if it is there, tmux if it is not, and setsid+nohup if neither -- all three detach from
# the ssh session, which is the only property that matters here.
ssh "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SESSION="$SESSION" bash <<'ENDSSH'
    set -euo pipefail
    cd "$HOME/$REMOTE_DIR"
    if screen -ls 2>/dev/null | grep -q "\.${SESSION}[[:space:]]"; then
        echo "a session named '$SESSION' is already running; stop it first or pass another SESSION" >&2
        exit 1
    fi
    if command -v screen >/dev/null 2>&1; then
        screen -dmS "$SESSION" bash out/run_pipeline.sh
        echo "launched under screen as '$SESSION'"
    elif command -v tmux >/dev/null 2>&1; then
        tmux new-session -d -s "$SESSION" "bash out/run_pipeline.sh"
        echo "screen is not installed; launched under tmux as '$SESSION'"
    else
        setsid nohup bash out/run_pipeline.sh > out/log_nohup.txt 2>&1 < /dev/null &
        echo "neither screen nor tmux is installed; launched with setsid+nohup (no attach)"
    fi
ENDSSH

cat <<EOF

==> Started on ${REMOTE}, detached. You can close this terminal.

    check on it     bash scripts/pendulum/run_on_nvidia.sh --status
    watch it live   bash scripts/pendulum/run_on_nvidia.sh --attach     (Ctrl-a d to leave it running)
    bring it back   bash scripts/pendulum/run_on_nvidia.sh --fetch      (once --status says DONE)
    kill it         bash scripts/pendulum/run_on_nvidia.sh --stop

    Read log_branch_report.txt before anything else.
EOF
