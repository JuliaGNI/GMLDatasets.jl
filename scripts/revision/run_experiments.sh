#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

julia_bin="${JULIA:-julia}"
mode="full"
stages="mnist,fashion-mnist,pendulum,retraction"
seeds="1234,1235,1236,1237,1238,1239,1240,1241,1242,1243"
configurations="${MNIST_CONFIGURATIONS:-all}"
output_root="${GML_RESULTS_ROOT:-$repo_root/results/revision}"
resume_dir=""
allow_dirty=0
allow_any_gpu=0
allow_no_cuda=0
retraction_repo="${GEOMETRIC_OPTIMIZERS_REPO:-$repo_root/../GeometricOptimizers}"

usage() {
    cat <<USAGE
usage: $0 [--smoke|--full] [--stages LIST] [--seeds LIST]
          [--configurations LIST] [--output-dir DIR] [--resume-dir DIR] [--allow-dirty]
          [--allow-any-gpu] [--allow-no-cuda] [--retraction-repo DIR]

MNIST and Fashion-MNIST run records use schema 4 and report exclusive gradient/AD,
optimizer-state/direction, and retraction/application timing. See scripts/revision/README.md.
The retraction stage writes validated schema-1 CSV plus an exact GeometricOptimizers patch.
Full mode runs five image configurations and the four intrinsic pendulum SAE configurations,
each over exactly seeds 1234:1243.
Smoke mode validates all CSV records, archive members, the checksum, and restart metadata.
USAGE
}

while (( $# )); do
    case "$1" in
        --smoke) mode="smoke"; shift ;;
        --full) mode="full"; shift ;;
        --stages) stages="${2:?missing stage list}"; shift 2 ;;
        --seeds) seeds="${2:?missing seed list}"; shift 2 ;;
        --configurations) configurations="${2:?missing configuration list}"; shift 2 ;;
        --output-dir) output_root="${2:?missing output directory}"; shift 2 ;;
        --resume-dir) resume_dir="${2:?missing run directory}"; shift 2 ;;
        --allow-dirty) allow_dirty=1; shift ;;
        --allow-any-gpu) allow_any_gpu=1; shift ;;
        --allow-no-cuda) allow_no_cuda=1; shift ;;
        --retraction-repo) retraction_repo="${2:?missing repository path}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The three environment files every bundle must carry. They are checked before anything runs,
# because a bundle without them cannot be reproduced and finding that out afterwards wastes the
# whole run.
required_archive_inputs=(Project.toml scripts/Project.toml scripts/Manifest.toml)
for required_input in "${required_archive_inputs[@]}"; do
    [[ -s "$required_input" ]] || {
        # `scripts/Manifest.toml` is deliberately untracked, so this is the first thing a fresh
        # clone meets and it has to say what to do about it rather than only what is missing.
        echo "missing required archive input: $repo_root/$required_input" >&2
        echo "instantiate the scripts environment first:" >&2
        echo "  julia --project=scripts -e 'using Pkg; Pkg.update(); Pkg.precompile()'" >&2
        echo "see scripts/revision/README.md" >&2
        exit 1
    }
done

IFS=',' read -r -a seed_array <<< "$seeds"
(( ${#seed_array[@]} > 0 )) || { echo "no seeds supplied" >&2; exit 2; }
seen_seeds=","
for seed_value in "${seed_array[@]}"; do
    [[ "$seed_value" =~ ^[0-9]+$ ]] || {
        echo "seeds must be comma-separated nonnegative integers, got '$seed_value'" >&2
        exit 2
    }
    [[ "$seen_seeds" != *",$seed_value,"* ]] || {
        echo "seed list contains duplicate value $seed_value" >&2
        exit 2
    }
    seen_seeds+="$seed_value,"
done
if [[ "$mode" == full && ${#seed_array[@]} -ne 10 ]]; then
    echo "full mode requires exactly 10 seeds, got ${#seed_array[@]}" >&2
    exit 2
fi

IFS=',' read -r -a stage_array <<< "$stages"
(( ${#stage_array[@]} > 0 )) || { echo "no stages supplied" >&2; exit 2; }
for stage_value in "${stage_array[@]}"; do
    case "$stage_value" in
        mnist|fashion-mnist|pendulum|retraction|none) ;;
        *) echo "unknown stage: $stage_value" >&2; exit 2 ;;
    esac
done
if [[ ",$stages," == *",none,"* && ${#stage_array[@]} -ne 1 ]]; then
    echo "the test-only 'none' stage cannot be combined with experiment stages" >&2
    exit 2
fi

repetitions="${#seed_array[@]}"
epochs=500
sae_epochs=1000
experiment_backend="cuda"
if [[ "$mode" == smoke ]]; then
    seed_array=("${seed_array[0]}")
    repetitions=1
    seeds="${seed_array[0]}"
    epochs=2
    sae_epochs=2
fi
[[ "$allow_no_cuda" -eq 1 ]] && experiment_backend="cpu"

# The pendulum SAE has no unconstrained Adam ablation: PSD layers must retain Stiefel weights for
# the network to stay symplectic. `all` is therefore the four intrinsic methods, even though it is
# five methods for the image stages.
if [[ "$configurations" == "all" ]]; then
    pendulum_configuration_array=(geometric-adam-cayley scalar-moment-adam gradient momentum)
else
    IFS=',' read -r -a pendulum_configuration_array <<< "$configurations"
fi

if [[ -n "$resume_dir" ]]; then
    [[ -d "$resume_dir" ]] || {
        echo "resume directory does not exist: $resume_dir" >&2
        exit 1
    }
    run_dir="$(cd "$(dirname "$resume_dir")" && pwd)/$(basename "$resume_dir")"
    output_root="$(dirname "$run_dir")"
else
    stamp="$(date -u +%Y%m%dT%H%M%SZ)_${mode}"
    mkdir -p "$output_root" || exit 1
    output_root="$(cd "$output_root" && pwd)"
    run_dir="$output_root/$stamp"
fi
mkdir -p "$run_dir" || exit 1
[[ -w "$run_dir" ]] || { echo "output directory is not writable: $run_dir" >&2; exit 1; }
log="$run_dir/run.log"
status_file="$run_dir/stages.csv"
[[ -s "$status_file" ]] || printf 'stage,status,started_utc,finished_utc,command\n' > "$status_file"

validator=("$julia_bin" --startup-file=no --project=scripts
    scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir")

contains_stage() { [[ ",$stages," == *",$1,"* ]]; }

is_git_repository() { git -C "$1" rev-parse --git-dir >/dev/null 2>&1; }

# `stage` and `status` are the first two columns and neither can contain a comma, so splitting on
# commas is safe here. It is *not* safe on the record CSVs, whose configuration column is a quoted
# field containing one — those are read by the Julia validator instead.
stage_succeeded() {
    awk -F, -v requested_stage="$1" '
        NR > 1 && $1 == requested_stage { status = $2 }
        END { exit status == "ok" ? 0 : 1 }
    ' "$status_file"
}

# The one provenance capture, for whichever repository a bundle has to record. This used to be
# written twice — here in bash and as `capture_source` in Julia — the same algorithm in two
# languages, each carrying the same comment about `git diff --no-index` exiting 1, and only the Julia
# one computing the SHA-256 and cross-checking the dirty flag against the patch. Two copies of a
# provenance mechanism is the one kind of duplication that undermines the thing it exists to provide.
#
# It prints four `key=value` lines — `sha`, `dirty`, `patch_file`, `patch_sha256` — and writes
# `<prefix>.patch` and `<prefix>.status` into the run directory.
capture_source() {
    "$julia_bin" --startup-file=no --project=scripts scripts/revision/capture_source.jl \
        --repo "$1" --prefix "$2" --out "$run_dir"
}

# One `key=value` line out of a `capture_source` block.
captured() { printf '%s\n' "$2" | sed -n "s/^$1=//p"; }

# Every `MNIST_*`, `SAE_*` and `RETRACTION_*` variable that is set, plus the Julia ones. A prefix
# sweep rather than a list of names: a new override added to a trainer is carried into the restart
# command automatically, where a list would have to be remembered and would silently drop it.
write_restart_command() {
    local variable
    local -a command=(env)
    for variable in JULIA JULIA_DEPOT_PATH JULIA_LOAD_PATH \
            $(compgen -v | grep -E '^(MNIST|SAE|RETRACTION)_' | sort); do
        [[ -n "${!variable:-}" ]] && command+=("$variable=${!variable}")
    done
    command+=("$0" "--$mode" --resume-dir "$run_dir" --stages "$stages"
        --seeds "$seeds" --configurations "$configurations"
        --retraction-repo "$retraction_repo")
    [[ "$allow_dirty" -eq 1 ]] && command+=(--allow-dirty)
    [[ "$allow_any_gpu" -eq 1 ]] && command+=(--allow-any-gpu)
    [[ "$allow_no_cuda" -eq 1 ]] && command+=(--allow-no-cuda)
    printf '%q ' "${command[@]}" > "$run_dir/restart-command.txt"
    printf '\n' >> "$run_dir/restart-command.txt"
}

write_run_configuration() {
    {
        printf 'mode=%s\n' "$mode"
        printf 'stages=%s\n' "$stages"
        printf 'seeds=%s\n' "$seeds"
        printf 'configurations=%s\n' "$configurations"
        printf 'allow_dirty=%s\n' "$allow_dirty"
        printf 'allow_any_gpu=%s\n' "$allow_any_gpu"
        printf 'allow_no_cuda=%s\n' "$allow_no_cuda"
        printf 'retraction_repo=%s\n' "$retraction_repo"
    } > "$run_dir/run-configuration.txt"
}

# The bundle has to contain the run directory, whole. Comparing the archive against the directory
# is what says so; a hand-maintained list of expected members would be a second copy of this
# script's control flow, and would go stale the first time a stage gained an output.
verify_archive_contents() {
    local archive="$1" listing expected missing status=0
    listing="$(mktemp "${TMPDIR:-/tmp}/gmldatasets-archive-list.XXXXXX")" || return 1
    expected="$(mktemp "${TMPDIR:-/tmp}/gmldatasets-archive-expected.XXXXXX")" || return 1
    if tar -tzf "$archive" | sed 's:/$::' | LC_ALL=C sort > "$listing" &&
            (cd "$output_root" && find "$(basename "$run_dir")" | LC_ALL=C sort) > "$expected"
    then
        missing="$(comm -23 "$expected" "$listing")"
        if [[ -n "$missing" ]]; then
            echo "archive is missing members:" >&2
            echo "$missing" >&2
            status=1
        fi
    else
        status=1
    fi
    rm -f "$listing" "$expected"
    return "$status"
}

archive_results() {
    local exit_code="$1"
    local archive_failed=0
    local archive_path="$run_dir.tar.gz"
    local checksum_path="$run_dir.tar.gz.sha256"
    local validation_started validation_finished validation_command_text validation_status
    set +e
    mkdir -p "$run_dir/environments/root" "$run_dir/environments/scripts"
    cp Project.toml "$run_dir/environments/root/" || archive_failed=1
    cp scripts/Project.toml scripts/Manifest.toml "$run_dir/environments/scripts/" ||
        archive_failed=1
    capture_source "$repo_root" gmldatasets > "$run_dir/gmldatasets.source" || archive_failed=1
    if is_git_repository "$retraction_repo"; then
        capture_source "$retraction_repo" geometricoptimizers \
            > "$run_dir/geometricoptimizers.source" || archive_failed=1
    fi
    write_restart_command || archive_failed=1
    write_run_configuration || archive_failed=1

    if [[ "$exit_code" -eq 0 && "$archive_failed" -eq 0 ]]; then
        validation_started="$(date -u +%FT%TZ)"
        validation_command=("${validator[@]}" --mode "$mode" --stages "$stages" --seeds "$seeds"
            --configurations "$configurations" --image-epochs "$epochs"
            --pendulum-epochs "$sae_epochs" --backend "$experiment_backend"
            --retraction-repo "$retraction_repo")
        [[ "$mode" == smoke ]] && validation_command+=(--allow-validation-failures)
        printf -v validation_command_text '%q ' "${validation_command[@]}"
        "${validation_command[@]}" > "$run_dir/artifact-validation.txt" 2>&1
        validation_status=$?
        validation_finished="$(date -u +%FT%TZ)"
        if [[ "$validation_status" -eq 0 ]]; then
            printf '%s,ok,%s,%s,"%s"\n' artifact-validation "$validation_started" \
                "$validation_finished" "$validation_command_text" >> "$status_file"
        else
            printf '%s,failed:%d,%s,%s,"%s"\n' artifact-validation "$validation_status" \
                "$validation_started" "$validation_finished" "$validation_command_text" \
                >> "$status_file"
            archive_failed=1
        fi
    else
        printf '%s\n' "artifact validation skipped because an earlier stage failed" \
            > "$run_dir/artifact-validation.txt"
    fi

    tar -C "$output_root" -czf "$archive_path" "$(basename "$run_dir")" || archive_failed=1
    verify_archive_contents "$archive_path" || archive_failed=1
    local checksum=(sha256sum)
    command -v sha256sum >/dev/null 2>&1 || checksum=(shasum -a 256)
    (cd "$output_root" && "${checksum[@]}" "$(basename "$archive_path")") > "$checksum_path" ||
        archive_failed=1
    (cd "$output_root" && "${checksum[@]}" -c "$(basename "$checksum_path")") || archive_failed=1
    echo "artifact: $archive_path"
    echo "checksum: $checksum_path"
    if [[ "$archive_failed" -ne 0 && "$exit_code" -eq 0 ]]; then
        exit_code=1
    fi
    trap - EXIT
    exit "$exit_code"
}
trap 'archive_results $?' EXIT

exec > >(tee -a "$log") 2>&1

if [[ "$allow_dirty" -ne 1 && -n "$(git status --porcelain=v1)" ]]; then
    echo "working tree is dirty; commit/stash changes or pass --allow-dirty" >&2
    exit 1
fi

export GML_ALLOW_ANY_GPU="$allow_any_gpu"
export GML_ALLOW_NO_CUDA="$allow_no_cuda"
export GML_REQUIRED_GPU="${GML_REQUIRED_GPU:-RTX 4090}"
"$julia_bin" --project=scripts scripts/revision/check_environment.jl \
    > "$run_dir/environment.txt" 2>&1 || exit 1
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$run_dir/nvidia-smi.txt" 2>&1 || exit 1
elif [[ "$allow_no_cuda" -eq 1 ]]; then
    printf '%s\n' 'nvidia-smi unavailable; CPU smoke explicitly allowed' > "$run_dir/nvidia-smi.txt"
else
    echo "nvidia-smi is unavailable" >&2
    exit 1
fi

run_stage() {
    local stage="$1"; shift
    local started finished command_text
    started="$(date -u +%FT%TZ)"
    printf -v command_text '%q ' "$@"
    echo "[$started] starting $stage"
    if "$@" > >(tee -a "$run_dir/${stage}.stdout.txt") 2> >(tee -a "$run_dir/${stage}.stderr.txt" >&2); then
        finished="$(date -u +%FT%TZ)"
        printf '%s,ok,%s,%s,"%s"\n' "$stage" "$started" "$finished" "$command_text" >> "$status_file"
    else
        local code=$?
        finished="$(date -u +%FT%TZ)"
        printf '%s,failed:%d,%s,%s,"%s"\n' "$stage" "$code" "$started" "$finished" "$command_text" >> "$status_file"
        return "$code"
    fi
}

# --------------------------------------------------------------------- image stages ---

run_image_dataset() {
    local dataset="$1"
    local prefix="$run_dir/$dataset"
    local validation_stage="${dataset}-record-validation"
    local -a common_environment=(
        MNIST_DATASET="$dataset"
        MNIST_CONFIGURATIONS="$configurations"
    )
    if [[ "$mode" == smoke ]]; then
        local smoke_samples="${MNIST_SMOKE_SAMPLES:-32}"
        common_environment+=(
            MNIST_BATCH_SIZE="${MNIST_BATCH_SIZE:-$smoke_samples}"
            MNIST_TRAINING_SAMPLES="${MNIST_TRAINING_SAMPLES:-$smoke_samples}"
            MNIST_TEST_SAMPLES="${MNIST_TEST_SAMPLES:-$smoke_samples}"
        )
    fi
    local -a validation_command=("${validator[@]}" --image "$dataset" --seeds "$seeds"
        --configurations "$configurations" --image-epochs "$epochs"
        --backend "$experiment_backend")
    [[ "$mode" == smoke ]] && validation_command+=(--allow-validation-failures)

    # The trainer does not resume inside its own matrix, so a partial matrix is rerun whole.
    if stage_succeeded "$validation_stage" && [[ -s "$prefix.jld2" ]]; then
        if "${validation_command[@]}"; then
            echo "skipping previously validated $dataset outputs"
            return 0
        fi
        echo "existing $dataset outputs failed validation; rerunning the complete dataset stage"
    fi

    if [[ "$mode" == full ]]; then
        run_stage "${dataset}-warmup" env "${common_environment[@]}" MNIST_REPETITIONS=1 \
            MNIST_SEEDS="${seed_array[0]}" MNIST_N_EPOCHS=1 MNIST_ACCURACY_EVERY=1 \
            MNIST_REPORT="$prefix-warmup-report.txt" MNIST_LOSSES="$prefix-warmup-losses.csv" \
            MNIST_RECORDS="$prefix-warmup-runs.csv" MNIST_OUTPUT="$prefix-warmup.jld2" \
            "$julia_bin" --project=scripts \
            scripts/geometric_optimizers/mnist_cuda_repetitions.jl || return
        run_stage "${dataset}-warmup-record-validation" "${validator[@]}" --image "$dataset" \
            --artifact-prefix "$dataset-warmup" --seeds "${seed_array[0]}" \
            --configurations "$configurations" --image-epochs 1 \
            --backend "$experiment_backend" --allow-validation-failures || return
    fi
    run_stage "$dataset" env "${common_environment[@]}" MNIST_REPETITIONS="$repetitions" \
        MNIST_SEEDS="$seeds" MNIST_N_EPOCHS="$epochs" MNIST_REPORT="$prefix-report.txt" \
        MNIST_LOSSES="$prefix-losses.csv" MNIST_RECORDS="$prefix-runs.csv" \
        MNIST_OUTPUT="$prefix.jld2" "$julia_bin" --project=scripts \
        scripts/geometric_optimizers/mnist_cuda_repetitions.jl || return
    run_stage "$validation_stage" "${validation_command[@]}"
}

contains_stage mnist && { run_image_dataset mnist || exit $?; }
contains_stage fashion-mnist && { run_image_dataset fashion-mnist || exit $?; }

# ------------------------------------------------------------------- pendulum stage ---

if contains_stage pendulum; then
    records="$run_dir/pendulum-runs.csv"
    complete_jobs="$run_dir/pendulum-complete.txt"
    : > "$complete_jobs"
    # One process per configuration and seed appends its own row, so the record file describes
    # whatever finished before the interruption. Revalidate it as a partial table and let the
    # validator list the jobs that are complete: the runner cannot read `configuration_key`,
    # `repetition` and `seed` out of the CSV itself, because the display name between them is a
    # quoted field containing a comma.
    if [[ -s "$records" ]]; then
        "${validator[@]}" --pendulum --allow-partial --seeds "$seeds" \
            --pendulum-epochs "$sae_epochs" --configurations "$configurations" \
            --backend "$experiment_backend" --list-complete "$complete_jobs" || exit $?
    fi
    require_cuda=1
    [[ "$allow_no_cuda" -eq 1 ]] && require_cuda=0

    for configuration_key in "${pendulum_configuration_array[@]}"; do
        if [[ "$mode" == full ]]; then
            warmup_checkpoint="$run_dir/pendulum-$configuration_key-warmup.h5"
            if stage_succeeded "pendulum-$configuration_key-warmup" &&
                    [[ -s "$warmup_checkpoint" ]]; then
                echo "skipping completed pendulum $configuration_key warm-up"
            else
                run_stage "pendulum-$configuration_key-warmup" env SAE_REQUIRE_CUDA=1 \
                    SAE_CONFIGURATION="$configuration_key" SAE_SEED="${seed_array[0]}" \
                    SAE_N_EPOCHS=1 SAE_OUTPUT="$warmup_checkpoint" \
                    "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
            fi
        fi
        repetition=0
        for seed_value in "${seed_array[@]}"; do
            repetition=$((repetition + 1))
            checkpoint="$run_dir/pendulum-$configuration_key-seed-${seed_value}.h5"
            if [[ -s "$checkpoint" ]] &&
                    grep -Fqx "$configuration_key,$repetition,$seed_value" "$complete_jobs"; then
                echo "skipping validated pendulum checkpoint $checkpoint"
                continue
            fi
            run_stage "pendulum-$configuration_key-seed-${seed_value}" env \
                SAE_REQUIRE_CUDA="$require_cuda" SAE_CONFIGURATION="$configuration_key" \
                SAE_SEED="$seed_value" SAE_REPETITION="$repetition" SAE_N_EPOCHS="$sae_epochs" \
                SAE_OUTPUT="$checkpoint" SAE_RECORD="$records" \
                SAE_LOSSES="$run_dir/pendulum-losses.csv" \
                "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
        done
    done
    rm -f "$complete_jobs"
    run_stage pendulum-record-validation "${validator[@]}" --pendulum --seeds "$seeds" \
        --configurations "$configurations" --pendulum-epochs "$sae_epochs" \
        --backend "$experiment_backend" || exit $?
fi

# ----------------------------------------------------------------- retraction stage ---

if contains_stage retraction; then
    # The benchmark itself lives in `GeometricOptimizers`, beside the algorithms, the reference and
    # the seeded lift sweep it measures — this harness supplies the provenance stamp and validates
    # what comes back, which is the part that belongs to *this* bundle's contract.
    upstream_benchmark="$retraction_repo/scripts/retraction_records.jl"
    [[ -f "$upstream_benchmark" ]] || {
        echo "missing upstream retraction benchmark: $upstream_benchmark" >&2
        echo "it arrived in GeometricOptimizers 0.8; update the checkout at $retraction_repo" >&2
        exit 1
    }
    records="$run_dir/retraction-runs.csv"
    source_patch="$run_dir/geometricoptimizers-retraction.patch"
    retraction_backend="cuda"
    retraction_precision="${RETRACTION_PRECISION:-Float32}"
    retraction_scales="${RETRACTION_SCALES:-0.1,1,3,6,12,30,60,120}"
    retraction_repetitions="${RETRACTION_REPETITIONS:-20}"
    if [[ "$allow_no_cuda" -eq 1 ]]; then
        retraction_backend="cpu"
        retraction_precision="${RETRACTION_PRECISION:-Float64}"
    fi
    if [[ "$mode" == smoke ]]; then
        retraction_scales="${RETRACTION_SCALES:-0.1}"
        retraction_repetitions="${RETRACTION_REPETITIONS:-1}"
    fi

    device_backend=CPU
    [[ "$retraction_backend" == cuda ]] && device_backend=CUDA
    validation_command=("$julia_bin" --startup-file=no --project=scripts
        scripts/revision/validate_retraction_records.jl --input "$records"
        --go-repo "$retraction_repo" --require "AugmentedPade:CPU"
        --require "ScaledSquaring:$device_backend" --require "NativePade:$device_backend")

    if stage_succeeded retraction-record-validation && [[ -s "$records" && -e "$source_patch" ]] &&
            "${validation_command[@]}"; then
        echo "skipping previously validated retraction outputs"
    else
        # The stamp every row carries, taken here and not by the benchmark: the patch beside the CSV
        # is this bundle's, and the validator re-checks the four values against it.
        retraction_source="$(capture_source "$retraction_repo" geometricoptimizers-retraction)" ||
            exit 1
        # The scripts environment stays on the load path for `CUDA`, which `GeometricOptimizers`
        # does not depend on and the CUDA arm of the benchmark imports.
        run_stage retraction env JULIA_LOAD_PATH="$retraction_repo:$repo_root/scripts:@stdlib" \
            "$julia_bin" --startup-file=no --project="$retraction_repo" \
            "$upstream_benchmark" \
            --output "$records" --backend "$retraction_backend" \
            --precision "$retraction_precision" --rows "${RETRACTION_ROWS:-20}" \
            --columns "${RETRACTION_COLUMNS:-3}" --scales "$retraction_scales" \
            --repetitions "$retraction_repetitions" --seed "${RETRACTION_SEED:-1234}" \
            --source-sha "$(captured sha "$retraction_source")" \
            --source-dirty "$(captured dirty "$retraction_source")" \
            --source-patch-file "$(captured patch_file "$retraction_source")" \
            --source-patch-sha256 "$(captured patch_sha256 "$retraction_source")" || exit $?
        run_stage retraction-record-validation "${validation_command[@]}" || exit $?
    fi
fi

echo "all requested stages completed"
trap - EXIT
archive_results 0
