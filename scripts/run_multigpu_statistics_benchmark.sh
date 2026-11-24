#!/usr/bin/env bash
set -euo pipefail

gpu_count="${1:-2}"
if [[ "$gpu_count" != "1" && "$gpu_count" != "2" ]]; then
    echo "Usage: $0 [1|2]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build-mgpu-stats-256-$gpu_count"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_dir="$repo_dir/benchmark-results/multigpu-stats-256-${gpu_count}gpu-$timestamp"
jobs="$(nproc)"

modes=(solver profiles volumetric combined)
mkdir -p -- "$result_dir"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "gpu_count=$gpu_count"
    echo "host=$(hostname)"
    echo "kernel=$(uname -r)"
    nvcc --version | tail -n 1
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "$result_dir/environment.txt"

printf 'mode,wall_mlups,device_mlups\n' > "$result_dir/summary.csv"

for mode in "${modes[@]}"; do
    config="$repo_dir/cases/multigpu_stats_256_${mode}.toml"
    mode_dir="$result_dir/$mode"
    mkdir -p -- "$mode_dir"

    echo
    echo "=== $mode | $gpu_count GPU(s) ==="
    cmake -S "$repo_dir" -B "$build_dir" \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DLBM_CONFIG_FILE="$config" \
        -DLBM_NUM_GPUS_OVERRIDE="$gpu_count" \
        2>&1 | tee "$mode_dir/configure.log"
    cmake --build "$build_dir" --target glbm -j"$jobs" \
        2>&1 | tee "$mode_dir/build.log"

    (
        cd -- "$mode_dir"
        "$build_dir/glbm" 2>&1 | tee run.log
    )

    normalized_log="$mode_dir/run.normalized.log"
    tr '\r' '\n' < "$mode_dir/run.log" > "$normalized_log"
    wall_mlups="$(sed -n 's/.*Wall MLUPS[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$normalized_log" | tail -n 1)"
    device_mlups="$(sed -n 's/.*Device MLUPS[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$normalized_log" | tail -n 1)"
    if [[ -z "$wall_mlups" || -z "$device_mlups" ]]; then
        echo "Could not extract MLUPS for mode '$mode'" >&2
        exit 1
    fi
    printf '%s,%s,%s\n' "$mode" "$wall_mlups" "$device_mlups" \
        >> "$result_dir/summary.csv"
done

echo
echo "Benchmark completed. Summary:"
column -s, -t "$result_dir/summary.csv" 2>/dev/null || \
    cat "$result_dir/summary.csv"
echo
echo "Results: $result_dir"
