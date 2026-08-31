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
Full mode defaults to all five optimizer configurations and exactly seeds 1234:1243.
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

required_archive_inputs=(
    "Project.toml"
    "scripts/Project.toml"
    "scripts/Manifest.toml"
)
for required_input in "${required_archive_inputs[@]}"; do
    if [[ ! -s "$required_input" ]]; then
        echo "missing required archive input: $repo_root/$required_input" >&2
        exit 1
    fi
done

IFS=',' read -r -a seed_array <<< "$seeds"
(( ${#seed_array[@]} > 0 )) || { echo "no seeds supplied" >&2; exit 2; }
seen_seeds_csv=","
for seed_value in "${seed_array[@]}"; do
    [[ "$seed_value" =~ ^[0-9]+$ ]] || {
        echo "seeds must be comma-separated nonnegative integers, got '$seed_value'" >&2
        exit 2
    }
    [[ "$seen_seeds_csv" != *",$seed_value,"* ]] || {
        echo "seed list contains duplicate value $seed_value" >&2
        exit 2
    }
    seen_seeds_csv+="$seed_value,"
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
sae_epochs=12000
experiment_backend="cuda"
if [[ "$mode" == smoke ]]; then
    seed_array=("${seed_array[0]}")
    repetitions=1
    seeds="${seed_array[0]}"
    epochs=2
    sae_epochs=2
fi
[[ "$allow_no_cuda" -eq 1 ]] && experiment_backend="cpu"

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

contains_stage() { [[ ",$stages," == *",$1,"* ]]; }

is_git_repository() {
    git -C "$1" rev-parse --git-dir >/dev/null 2>&1
}

stage_succeeded() {
    local requested_stage="$1"
    awk -F, -v requested_stage="$requested_stage" '
        NR > 1 && $1 == requested_stage { status = $2 }
        END { exit status == "ok" ? 0 : 1 }
    ' "$status_file"
}

capture_repository() {
    local repository="$1"
    local prefix="$2"
    local patch_file="$run_dir/$prefix.patch"
    git -C "$repository" rev-parse HEAD > "$run_dir/$prefix.sha" || return 1
    git -C "$repository" status --porcelain=v1 > "$run_dir/$prefix.status" || return 1
    git -C "$repository" diff --binary HEAD -- > "$patch_file" || return 1
    while IFS= read -r -d '' untracked_path; do
        git -C "$repository" diff --binary --no-index -- /dev/null "$untracked_path" \
            >> "$patch_file"
        local status=$?
        [[ "$status" -eq 1 ]] || return "$status"
    done < <(git -C "$repository" ls-files --others --exclude-standard -z)
}

write_restart_command() {
    local variable assignment
    local -a command=(env)
    for variable in JULIA JULIA_DEPOT_PATH JULIA_LOAD_PATH MNIST_BATCH_SIZE \
            MNIST_TRAINING_SAMPLES MNIST_TEST_SAMPLES MNIST_SMOKE_SAMPLES \
            MNIST_SCALAR_MOMENT_ADAM_LEARNING_RATE MNIST_SCALAR_MOMENT_ADAM_AMBIENT_NORM \
            RETRACTION_PRECISION RETRACTION_ROWS RETRACTION_COLUMNS RETRACTION_SCALES \
            RETRACTION_REPETITIONS RETRACTION_SEED; do
        if [[ -n "${!variable:-}" ]]; then
            assignment="$variable=${!variable}"
            command+=("$assignment")
        fi
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

write_required_archive_members() {
    local dataset seed_value
    local -a members=(
        artifact-validation.txt archive-required-members.txt environment.txt
        environments/root/Project.toml environments/scripts/Project.toml
        environments/scripts/Manifest.toml gmldatasets.patch gmldatasets.sha
        gmldatasets.status nvidia-smi.txt restart-command.txt run-configuration.txt
        run.log stages.csv
    )
    if is_git_repository "$retraction_repo"; then
        members+=(geometricoptimizers.patch geometricoptimizers.sha geometricoptimizers.status)
    fi
    for dataset in mnist fashion-mnist; do
        contains_stage "$dataset" || continue
        members+=("$dataset-report.txt" "$dataset-losses.csv" "$dataset-runs.csv"
            "$dataset.jld2" "$dataset.stdout.txt" "$dataset.stderr.txt"
            "$dataset-record-validation.stdout.txt" "$dataset-record-validation.stderr.txt")
        if [[ "$mode" == full ]]; then
            members+=("$dataset-warmup-report.txt" "$dataset-warmup-losses.csv"
                "$dataset-warmup-runs.csv" "$dataset-warmup.jld2"
                "$dataset-warmup.stdout.txt" "$dataset-warmup.stderr.txt"
                "$dataset-warmup-record-validation.stdout.txt"
                "$dataset-warmup-record-validation.stderr.txt")
        fi
    done
    if contains_stage pendulum; then
        members+=(pendulum-runs.csv pendulum-record-validation.stdout.txt
            pendulum-record-validation.stderr.txt)
        [[ "$mode" == full ]] && members+=(pendulum-warmup.h5 pendulum-warmup.stdout.txt
            pendulum-warmup.stderr.txt)
        for seed_value in "${seed_array[@]}"; do
            members+=("pendulum-seed-$seed_value.h5" "pendulum-seed-$seed_value.stdout.txt"
                "pendulum-seed-$seed_value.stderr.txt")
        done
    fi
    if contains_stage retraction; then
        members+=(geometricoptimizers-retraction.patch retraction-runs.csv
            retraction.stdout.txt retraction.stderr.txt
            retraction-record-validation.stdout.txt retraction-record-validation.stderr.txt)
    fi
    printf '%s\n' "${members[@]}" | LC_ALL=C sort > "$run_dir/archive-required-members.txt"
}

verify_required_files() {
    local member
    while IFS= read -r member; do
        [[ -e "$run_dir/$member" ]] || {
            echo "missing required archive member: $member" >&2
            return 1
        }
    done < "$run_dir/archive-required-members.txt"
}

verify_archive_members() {
    local archive="$1"
    local listing member expected
    listing="$(mktemp "${TMPDIR:-/tmp}/gmldatasets-archive-list.XXXXXX")" || return 1
    if ! tar -tzf "$archive" > "$listing"; then
        rm -f "$listing"
        return 1
    fi
    while IFS= read -r member; do
        expected="$(basename "$run_dir")/$member"
        if ! grep -Fqx "$expected" "$listing"; then
            echo "archive is missing required member: $expected" >&2
            rm -f "$listing"
            return 1
        fi
    done < "$run_dir/archive-required-members.txt"
    rm -f "$listing"
}

archive_results() {
    local exit_code="$1"
    local archive_failed=0
    local validation_started validation_finished validation_command_text
    local archive_path="$run_dir.tar.gz"
    local checksum_path="$run_dir.tar.gz.sha256"
    set +e
    mkdir -p "$run_dir/environments/root" "$run_dir/environments/scripts"
    if ! cp Project.toml "$run_dir/environments/root/"; then
        echo "failed to archive required input: $repo_root/Project.toml" >&2
        archive_failed=1
    fi
    for required_input in scripts/Project.toml scripts/Manifest.toml; do
        if ! cp "$required_input" "$run_dir/environments/scripts/"; then
            echo "failed to archive required input: $repo_root/$required_input" >&2
            archive_failed=1
        fi
    done
    capture_repository "$repo_root" gmldatasets || archive_failed=1
    if is_git_repository "$retraction_repo"; then
        capture_repository "$retraction_repo" geometricoptimizers || archive_failed=1
    fi
    write_restart_command || archive_failed=1
    write_run_configuration || archive_failed=1
    write_required_archive_members || archive_failed=1

    if [[ "$exit_code" -eq 0 && "$archive_failed" -eq 0 ]]; then
        validation_started="$(date -u +%FT%TZ)"
        validation_command=("$julia_bin" --startup-file=no --project=scripts
            scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir" --mode "$mode"
            --stages "$stages" --seeds "$seeds" --configurations "$configurations"
            --image-epochs "$epochs" --pendulum-epochs "$sae_epochs"
            --backend "$experiment_backend" --retraction-repo "$retraction_repo")
        [[ "$mode" == smoke ]] && validation_command+=(--allow-validation-failures)
        printf -v validation_command_text '%q ' "${validation_command[@]}"
        if "${validation_command[@]}" > "$run_dir/artifact-validation.txt" 2>&1; then
            validation_finished="$(date -u +%FT%TZ)"
            printf '%s,ok,%s,%s,"%s"\n' artifact-validation "$validation_started" \
                "$validation_finished" "$validation_command_text" >> "$status_file"
        else
            validation_finished="$(date -u +%FT%TZ)"
            printf '%s,failed:1,%s,%s,"%s"\n' artifact-validation "$validation_started" \
                "$validation_finished" "$validation_command_text" >> "$status_file"
            archive_failed=1
        fi
    else
        printf '%s\n' "artifact validation skipped because an earlier stage failed" \
            > "$run_dir/artifact-validation.txt"
    fi

    verify_required_files || archive_failed=1
    tar -C "$output_root" -czf "$archive_path" "$(basename "$run_dir")" || archive_failed=1
    verify_archive_members "$archive_path" || archive_failed=1
    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$output_root" && sha256sum "$(basename "$archive_path")") > "$checksum_path" ||
            archive_failed=1
        (cd "$output_root" && sha256sum -c "$(basename "$checksum_path")") || archive_failed=1
    else
        (cd "$output_root" && shasum -a 256 "$(basename "$archive_path")") > "$checksum_path" ||
            archive_failed=1
        (cd "$output_root" && shasum -a 256 -c "$(basename "$checksum_path")") ||
            archive_failed=1
    fi
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
"$julia_bin" --project=scripts scripts/revision/check_environment.jl > "$run_dir/environment.txt" 2>&1 || exit 1
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
    local -a validation_command=("$julia_bin" --startup-file=no --project=scripts
        scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir" --image "$dataset"
        --seeds "$seeds" --configurations "$configurations" --image-epochs "$epochs"
        --backend "$experiment_backend")
    [[ "$mode" == smoke ]] && validation_command+=(--allow-validation-failures)

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
            "$julia_bin" --project=scripts scripts/geometric_optimizers/mnist_cuda_repetitions.jl ||
            return
        local -a warmup_validation=("$julia_bin" --startup-file=no --project=scripts
            scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir" --image "$dataset"
            --artifact-prefix "$dataset-warmup"
            --seeds "${seed_array[0]}" --configurations "$configurations" --image-epochs 1
            --backend "$experiment_backend" --allow-validation-failures)
        run_stage "${dataset}-warmup-record-validation" "${warmup_validation[@]}" || return
    fi
    run_stage "$dataset" env "${common_environment[@]}" MNIST_REPETITIONS="$repetitions" \
        MNIST_SEEDS="$seeds" MNIST_N_EPOCHS="$epochs" MNIST_REPORT="$prefix-report.txt" \
        MNIST_LOSSES="$prefix-losses.csv" MNIST_RECORDS="$prefix-runs.csv" \
        MNIST_OUTPUT="$prefix.jld2" "$julia_bin" --project=scripts \
        scripts/geometric_optimizers/mnist_cuda_repetitions.jl || return
    run_stage "$validation_stage" "${validation_command[@]}"
}

if contains_stage mnist; then
    run_image_dataset mnist || exit $?
fi
if contains_stage fashion-mnist; then
    run_image_dataset fashion-mnist || exit $?
fi

if contains_stage pendulum; then
    records="$run_dir/pendulum-runs.csv"
    if [[ "$mode" == full ]]; then
        if stage_succeeded pendulum-warmup && [[ -s "$run_dir/pendulum-warmup.h5" ]]; then
            echo "skipping completed pendulum warm-up"
        else
            run_stage pendulum-warmup env SAE_REQUIRE_CUDA=1 SAE_SEED="${seed_array[0]}" \
                SAE_N_EPOCHS=1 SAE_OUTPUT="$run_dir/pendulum-warmup.h5" \
                "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
        fi
    fi
    if [[ -s "$records" ]]; then
        "$julia_bin" --startup-file=no --project=scripts \
            scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir" --pendulum \
            --allow-partial --seeds "$seeds" --pendulum-epochs "$sae_epochs" \
            --backend "$experiment_backend" || exit $?
    fi
    repetition=0
    for seed_value in ${seeds//,/ }; do
        repetition=$((repetition + 1))
        checkpoint="$run_dir/pendulum-seed-${seed_value}.h5"
        if [[ -s "$checkpoint" ]] && awk -F, -v repetition="$repetition" -v seed="$seed_value" '
                NR > 1 && $4 == repetition && $5 == seed && $6 == "ok" { found = 1 }
                END { exit found ? 0 : 1 }
            ' "$records"; then
            echo "skipping validated checkpoint $checkpoint"
            continue
        fi
        require_cuda=1
        [[ "$allow_no_cuda" -eq 1 ]] && require_cuda=0
        run_stage "pendulum-seed-${seed_value}" env SAE_REQUIRE_CUDA="$require_cuda" SAE_SEED="$seed_value" SAE_REPETITION="$repetition" SAE_N_EPOCHS="$sae_epochs" SAE_OUTPUT="$checkpoint" SAE_RECORD="$records" "$julia_bin" --project=scripts scripts/pendulum/train_sae.jl || exit $?
    done
    run_stage pendulum-record-validation "$julia_bin" --startup-file=no --project=scripts \
        scripts/revision/validate_run_artifacts.jl --run-dir "$run_dir" --pendulum \
        --seeds "$seeds" --pendulum-epochs "$sae_epochs" --backend "$experiment_backend" ||
        exit $?
fi

if contains_stage retraction; then
    benchmark="$repo_root/scripts/revision/retraction_records.jl"
    validator="$repo_root/scripts/revision/validate_retraction_records.jl"
    upstream_benchmark="$retraction_repo/scripts/retraction_accuracy.jl"
    [[ -f "$upstream_benchmark" ]] || {
        echo "missing upstream retraction benchmark: $upstream_benchmark" >&2
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

    load_path="$retraction_repo:$repo_root/scripts:@stdlib"
    required_paths=("AugmentedPade:CPU")
    if [[ "$retraction_backend" == cuda ]]; then
        required_paths+=("ScaledSquaring:CUDA" "NativePade:CUDA")
    else
        required_paths+=("ScaledSquaring:CPU" "NativePade:CPU")
    fi
    validation_command=("$julia_bin" --startup-file=no --project=scripts "$validator"
        --input "$records" --go-repo "$retraction_repo")
    for required_path in "${required_paths[@]}"; do
        validation_command+=(--require "$required_path")
    done
    if stage_succeeded retraction-record-validation && [[ -s "$records" && -e "$source_patch" ]] &&
            "${validation_command[@]}"; then
        echo "skipping previously validated retraction outputs"
    else
        run_stage retraction env JULIA_LOAD_PATH="$load_path" "$julia_bin" --startup-file=no \
            --project="$retraction_repo" "$benchmark" --go-repo "$retraction_repo" \
            --output "$records" --patch-output "$source_patch" --backend "$retraction_backend" \
            --precision "$retraction_precision" --rows "${RETRACTION_ROWS:-20}" \
            --columns "${RETRACTION_COLUMNS:-3}" --scales "$retraction_scales" \
            --repetitions "$retraction_repetitions" --seed "${RETRACTION_SEED:-1234}" || exit $?
        run_stage retraction-record-validation "${validation_command[@]}" || exit $?
    fi
fi

echo "all requested stages completed"
trap - EXIT
archive_results 0
