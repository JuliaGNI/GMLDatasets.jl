#!/usr/bin/env bash
set -euo pipefail

runner="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_experiments.sh"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/gmldatasets-required-inputs.XXXXXX")"

cleanup() {
    case "$fixture_root" in
        "${TMPDIR:-/tmp}"/gmldatasets-required-inputs.*) rm -rf "$fixture_root" ;;
        *) echo "refusing to remove unexpected fixture path: $fixture_root" >&2; exit 1 ;;
    esac
}
trap cleanup EXIT

mkdir -p "$fixture_root/scripts/revision"
cp "$runner" "$fixture_root/scripts/revision/run_experiments.sh"

write_required_inputs() {
    printf '%s\n' 'name = "Fixture"' > "$fixture_root/Project.toml"
    printf '%s\n' '[deps]' > "$fixture_root/scripts/Project.toml"
    printf '%s\n' 'manifest_format = "2.0"' > "$fixture_root/scripts/Manifest.toml"
}

for missing_input in Project.toml scripts/Project.toml scripts/Manifest.toml; do
    write_required_inputs
    rm "$fixture_root/$missing_input"
    output_file="$fixture_root/${missing_input//\//_}.txt"
    set +e
    (
        cd "$fixture_root"
        scripts/revision/run_experiments.sh \
            --smoke --allow-dirty --allow-any-gpu --allow-no-cuda
    ) > "$output_file" 2>&1
    status=$?
    set -e

    if [[ "$status" -eq 0 ]]; then
        echo "runner unexpectedly accepted a missing $missing_input" >&2
        exit 1
    fi
    if ! grep -Fq 'missing required archive input:' "$output_file" ||
            ! grep -Fq "/$missing_input" "$output_file"; then
        echo "runner did not report the missing $missing_input clearly" >&2
        sed -n '1,20p' "$output_file" >&2
        exit 1
    fi
done

echo "all missing required archive inputs rejected as expected"
