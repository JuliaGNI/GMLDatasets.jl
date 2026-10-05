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
#   bash scripts/pendulum/run_on_nvidia.sh --resume   # sync, reuse saved SAE, run remaining stages
#   bash scripts/pendulum/run_on_nvidia.sh --status   # is it still going, and where is it
#   bash scripts/pendulum/run_on_nvidia.sh --attach   # watch it live (Ctrl-a d to leave it running)
#   bash scripts/pendulum/run_on_nvidia.sh --fetch    # bring out/ back when it says DONE
#   bash scripts/pendulum/run_on_nvidia.sh --stop     # kill the session
#   bash scripts/pendulum/run_on_nvidia.sh --foreground   # run here and wait; for smoke tests
#
# Everything a run varies is an environment variable, so a sweep needs no file edits on the remote:
#
#   SAE_GRID=paper SAE_SEED=123 SAE_FRACS=one SAE_TSPAN=40 SAE_EPOCHS=3000 SAE_UPSCALE=20 \
#     SAE_ETA=1e-3 SAE_BATCH=2048 SAE_OUT=pendulum_sae.h5 SESSION=sae \
#     bash scripts/pendulum/run_on_nvidia.sh
#
# Give each concurrent run its own SESSION and SAE_OUT, and its own GML_OUTDIR for the fetch.
# --resume reuses out/$SAE_OUT on the remote and leaves the SAE training log intact.
# Set SAE_OUT to the original filename if the run used a nondefault checkpoint name.
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

# An ssh host alias, NOT a hostname: this repository is public, so the machine's name and the
# account on it live in ~/.ssh/config, which is not. One-time setup on a new laptop:
#
#   Host sae-gpu
#       HostName <the workstation>
#       User     <your account there>
#
# REMOTE=user@host still works for a one-off run that is not worth a config entry.
REMOTE="${REMOTE:-sae-gpu}"
# Relative to the remote home, and the name `git clone` gives this repository, so a checkout made
# by hand and the one this script syncs to are the same
# directory rather than two.
REMOTE_DIR="${REMOTE_DIR:-GMLDatasets.jl}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-sae}"

# Forwarded to the scripts on the remote, with train_sae.jl's defaults; see its header.
SAE_GRID="${SAE_GRID:-paper}"
SAE_SEED="${SAE_SEED:-123}"
SAE_FRACS="${SAE_FRACS:-one}"
SAE_TSPAN="${SAE_TSPAN:-40}"
SAE_EPOCHS="${SAE_EPOCHS:-3000}"
SAE_UPSCALE="${SAE_UPSCALE:-20}"
SAE_ETA="${SAE_ETA:-1e-3}"
SAE_BATCH="${SAE_BATCH:-2048}"
SAE_CHECK_EVERY="${SAE_CHECK_EVERY:-100}"
SAE_NSAMP="${SAE_NSAMP:-1600}"
SAE_PATIENCE="${SAE_PATIENCE:-500}"
SAE_MIN_GAIN="${SAE_MIN_GAIN:-0.01}"
SAE_OUT="${SAE_OUT:-pendulum_sae.h5}"
RUN_STEP2="${RUN_STEP2:-1}"
RUN_TRAIN=1

DEST="${GML_OUTDIR:-$REPO_ROOT/out}"
MODE="${1:-start}"

# `ssh -G` reports the effective config. An alias with no Host block resolves to itself, which is
# the "you have not set this up yet" case and is worth catching before four connections try it.
if [ "$REMOTE" = "sae-gpu" ] \
   && [ "$(ssh -G sae-gpu 2>/dev/null | awk '/^hostname /{print $2}')" = "sae-gpu" ]; then
    cat >&2 <<'MSG'
No `Host sae-gpu` block in ~/.ssh/config, and no REMOTE set.

The workstation's name is deliberately not in this file -- the repository is public. Add to
~/.ssh/config:

    Host sae-gpu
        HostName <the workstation>
        User     <your account there>

or run this once with REMOTE=user@host.
MSG
    exit 2
fi

# One password for the whole invocation instead of one per connection. A start opens four -- the
# mkdir, the rsync, the scp of the pipeline, the launch -- and without this each one authenticates
# again. The first to run becomes the master; the rest ride its socket.
#
# Nothing here calls `ssh -O exit`: --attach replaces this process with exec, so its cleanup would
# never run anyway, and a cleanup in one terminal would tear down a --attach riding the same socket
# in another. ControlPersist expires it instead, a minute after the last client leaves.
#
# SSH_MUX=0 turns it off, for a remote whose sshd refuses multiplexing (MaxSessions 1).
if [ "${SSH_MUX:-1}" = "1" ]; then
    SSH_OPTS=(-o ControlMaster=auto
              -o ControlPath="${TMPDIR:-/tmp}/sae-ssh-%r@%h-%p"
              -o ControlPersist=60)
else
    SSH_OPTS=(-o ControlMaster=no)
fi
# rsync takes the remote shell as one string; none of the options above contain a space.
RSH="ssh ${SSH_OPTS[*]}"

# macOS 15 ships openrsync ("rsync version 2.6.9 compatible"), which has --progress but not
# --info=progress2. Probe rather than branch on uname: a Homebrew rsync on the same laptop does
# take it, and the remote's flavour does not enter into it -- these flags are read locally.
if rsync --info=progress2 --version >/dev/null 2>&1; then
    RSYNC_PROGRESS="--info=progress2"
else
    RSYNC_PROGRESS="--progress"
fi

# ---------------------------------------------------------------------------------------------
# The subcommands that do not start anything
# ---------------------------------------------------------------------------------------------
case "$MODE" in
  --status)
      ssh "${SSH_OPTS[@]}" "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SESSION="$SESSION" bash <<'ENDSSH'
        cd "$HOME/$REMOTE_DIR" 2>/dev/null || { echo "no $REMOTE_DIR on the remote"; exit 1; }
        if [ -f out/STATUS ]; then echo "STATUS: $(cat out/STATUS)"; else echo "STATUS: never started"; fi
        echo "session: $(screen -ls 2>/dev/null | grep -c "\.${SESSION}[[:space:]]" || true) live"
        screen -ls 2>/dev/null | sed -n '2,$p' | sed 's/^/  /' || true
        echo "--- last 50 lines of log_pipeline.txt ---"
        tail -50 out/log_pipeline.txt 2>/dev/null || true
ENDSSH
      exit 0 ;;
  --attach)
      echo "Attaching to '$SESSION' on $REMOTE. Ctrl-a d detaches and LEAVES IT RUNNING."
      exec ssh -t "${SSH_OPTS[@]}" "$REMOTE" \
          "screen -d -r '$SESSION' || { echo 'The session has ended (finished or failed):'; \
           cat '$REMOTE_DIR/out/STATUS'; tail -50 '$REMOTE_DIR/out/log_pipeline.txt'; }" ;;
  --stop)
      ssh "${SSH_OPTS[@]}" "$REMOTE" "screen -S '$SESSION' -X quit || true; echo stopped"
      exit 0 ;;
  --fetch)
      mkdir -p "$DEST"
      rsync -az -e "$RSH" "$RSYNC_PROGRESS" "${REMOTE}:${REMOTE_DIR}/out/" "${DEST}/"
      echo "==> out/ is in ${DEST}"
      if [ -f "$DEST/log_branch_report.txt" ]; then
          echo "    Read log_branch_report.txt before interpreting the results."
      else
          echo "    No branch report was fetched; check STATUS and log_pipeline.txt."
      fi
      exit 0 ;;
  start|--start|--foreground) ;;
  --resume) RUN_TRAIN=0 ;;
  *)  echo "unknown option: $MODE"; sed -n '8,17p' "$0"; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------------------------
echo "==> Syncing $REPO_ROOT to ${REMOTE}:${REMOTE_DIR}"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '${REMOTE_DIR}/out'"
# Outputs are excluded in both directions: the remote keeps its own out/, which is what comes back.
rsync -az --delete -e "$RSH" "$RSYNC_PROGRESS" \
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
: > "\$GML_OUTDIR/log_pipeline.txt"
export SAE_SEED="${SAE_SEED}" SAE_FRACS="${SAE_FRACS}" SAE_TSPAN="${SAE_TSPAN}"
export SAE_EPOCHS="${SAE_EPOCHS}" SAE_UPSCALE="${SAE_UPSCALE}" SAE_ETA="${SAE_ETA}"
export SAE_BATCH="${SAE_BATCH}" SAE_OUT="${SAE_OUT}" SAE_GRID="${SAE_GRID}"
export SAE_CHECK_EVERY="${SAE_CHECK_EVERY}" SAE_NSAMP="${SAE_NSAMP}" SAE_PATIENCE="${SAE_PATIENCE}" SAE_MIN_GAIN="${SAE_MIN_GAIN}"
${JULIA:+export JULIA="${JULIA}"}

say () { echo "[\$(date '+%F %T')] \$*" | tee -a "\$GML_OUTDIR/log_pipeline.txt"; }
# The step's own log goes into the pipeline log on failure, so --status shows the actual error
# rather than only the name of the step that had one. From the first ERROR line and not the tail:
# a Julia stack trace ends in the outermost frames, and with this network's type parameters the
# last twenty lines are four frames of Chain{...} and never the error itself. Lines are cut for
# the same reason. Match ERROR anywhere on the line: terminal colour codes or progress output can
# precede it. Status and attach retain 50 lines so they include this entire 31-line error excerpt.
# The tail is the fallback for a failure that is not a Julia exception.
fail () {
    echo "FAILED at \$1" > "\$GML_OUTDIR/STATUS"
    say "FAILED at \$1"
    if [ -n "\${2:-}" ] && [ -f "\$GML_OUTDIR/\$2" ]; then
        { echo "--- \$2, from its first ERROR (lines cut at 300 characters) ---"
          { grep -m1 -A30 'ERROR' "\$GML_OUTDIR/\$2" || tail -20 "\$GML_OUTDIR/\$2"; } \\
              | cut -c1-300; } \\
            | tee -a "\$GML_OUTDIR/log_pipeline.txt"
    fi
    exit 1
}

echo "RUNNING" > "\$GML_OUTDIR/STATUS"

# screen starts a non-interactive shell, which does not read ~/.bashrc -- where juliaup puts
# itself on PATH. JULIA=/path/to/julia overrides the search.
JULIA="\${JULIA:-\$(command -v julia || true)}"
for c in "\$HOME/.juliaup/bin/julia" "\$HOME/.local/bin/julia" /usr/local/bin/julia; do
    [ -n "\$JULIA" ] && break
    [ -x "\$c" ] && JULIA="\$c"
done
[ -n "\$JULIA" ] || { say "no julia on PATH or in ~/.juliaup/bin; set JULIA=/path/to/julia"; fail julia; }

# The manifest is the Mac's -- gitignored, but rsync does not read .gitignore -- and records the
# Julia it was resolved with. Another minor version cannot instantiate it: the stdlibs differ
# (Zstd_jll is one on 1.13 and a registry package on 1.12), and the failure surfaces as a precompile
# error deep inside JLD2. So run that version, through juliaup's +channel if the default is another.
JULIA_CMD=("\$JULIA")
WANT="\$(sed -n 's/^julia_version = "\([0-9]*\.[0-9]*\).*/\1/p' scripts/Manifest.toml 2>/dev/null)"
if [ -n "\$WANT" ]; then
    have () { "\${JULIA_CMD[@]}" -e 'print(VERSION.major, ".", VERSION.minor)' 2>/dev/null; }
    if [ "\$(have)" != "\$WANT" ] && [ -x "\$(dirname "\$JULIA")/juliaup" ]; then
        "\$(dirname "\$JULIA")/juliaup" add "\$WANT" >/dev/null 2>&1 || true   # fails if already there
        JULIA_CMD=("\$JULIA" "+\$WANT")
    fi
    [ "\$(have)" = "\$WANT" ] || {
        say "scripts/Manifest.toml was resolved with julia \$WANT, and \$JULIA is not that and has no juliaup beside it"
        fail julia
    }
fi
say "julia: \${JULIA_CMD[*]} (\$("\${JULIA_CMD[@]}" --version 2>&1))"

say "instantiate"
"\${JULIA_CMD[@]}" --project=scripts -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \\
    > "\$GML_OUTDIR/log_instantiate.txt" 2>&1 || fail instantiate log_instantiate.txt

export SAE_WEIGHTS="\$GML_OUTDIR/${SAE_OUT}"
if [ "${RUN_TRAIN}" = "1" ]; then
    say "train  (grid=${SAE_GRID} seed=${SAE_SEED} fracs=${SAE_FRACS} tspan=${SAE_TSPAN} epochs=${SAE_EPOCHS} upscale=${SAE_UPSCALE} eta=${SAE_ETA} batch=${SAE_BATCH})"
    "\${JULIA_CMD[@]}" --project=scripts scripts/pendulum/train_sae.jl \\
        > "\$GML_OUTDIR/log_train_sae.txt" 2>&1 || fail train log_train_sae.txt
    # Progress output can precede the buffered backend line in the redirected log.
    say "  \$(grep -m1 '^Backend:' "\$GML_OUTDIR/log_train_sae.txt")"
else
    [ -s "\$SAE_WEIGHTS" ] || { say "no saved SAE weights at \$SAE_WEIGHTS"; fail resume; }
    say "reuse SAE weights: \$SAE_WEIGHTS (skipping training)"
fi

if [ "${RUN_STEP2}" = "1" ]; then
    say "reduced dynamics"
    "\${JULIA_CMD[@]}" --project=scripts scripts/pendulum/reduced_networks.jl \\
        > "\$GML_OUTDIR/log_reduced.txt" 2>&1 || fail reduced log_reduced.txt
fi

say "branch report"
"\${JULIA_CMD[@]}" --project=scripts scripts/pendulum/branch_report.jl \\
    > "\$GML_OUTDIR/log_branch_report.txt" 2>&1 || fail report log_branch_report.txt

say "latent figure"
"\${JULIA_CMD[@]}" --project=scripts scripts/pendulum/latent_plot.jl \\
    > "\$GML_OUTDIR/log_latent_plot.txt" 2>&1 || fail latent_plot log_latent_plot.txt

echo "DONE" > "\$GML_OUTDIR/STATUS"
say "DONE"
EOF

# Catch a quoting mistake here rather than three hours into a run.
bash -n "$PIPE" || { echo "generated pipeline does not parse; not launching"; exit 1; }
scp -q "${SSH_OPTS[@]}" "$PIPE" "${REMOTE}:${REMOTE_DIR}/out/run_pipeline.sh"

# ---------------------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------------------
if [ "$MODE" = "--foreground" ]; then
    echo "==> Running in the foreground. A dropped connection kills this; use the default for long runs."
    ssh -t "${SSH_OPTS[@]}" "$REMOTE" "cd '$REMOTE_DIR' && bash out/run_pipeline.sh"
    mkdir -p "$DEST"
    rsync -az -e "$RSH" "$RSYNC_PROGRESS" "${REMOTE}:${REMOTE_DIR}/out/" "${DEST}/"
    echo "==> out/ is in ${DEST}"
    exit 0
fi

# screen if it is there, tmux if it is not, and setsid+nohup if neither -- all three detach from
# the ssh session, which is the only property that matters here.
ssh "${SSH_OPTS[@]}" "$REMOTE" REMOTE_DIR="$REMOTE_DIR" SESSION="$SESSION" bash <<'ENDSSH'
    set -euo pipefail
    cd "$HOME/$REMOTE_DIR"
    if screen -ls 2>/dev/null | grep -q "\.${SESSION}[[:space:]]"; then
        echo "a session named '$SESSION' is already running; stop it first or pass another SESSION" >&2
        exit 1
    fi
    if command -v screen >/dev/null 2>&1; then
        screen -dmS "$SESSION" bash -l out/run_pipeline.sh
        echo "launched under screen as '$SESSION'"
    elif command -v tmux >/dev/null 2>&1; then
        tmux new-session -d -s "$SESSION" "bash -l out/run_pipeline.sh"
        echo "screen is not installed; launched under tmux as '$SESSION'"
    else
        setsid nohup bash -l out/run_pipeline.sh > out/log_nohup.txt 2>&1 < /dev/null &
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
