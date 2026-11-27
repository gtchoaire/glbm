#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/checkpoint_re3200_turbulence_256_pilot.glbm" >&2
    exit 2
fi

checkpoint="$(realpath -- "$1")"
run_dir="$(dirname -- "$(dirname -- "$checkpoint")")"
statistics="$run_dir/statistics"
if [[ ! -f "$checkpoint" || ! -f "$statistics/bmbc_blocks.bin" ]]; then
    echo "Checkpoint or BMBC statistics not found: $checkpoint" >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build-re3200-turbulence-256-pilot"
config="$repo_dir/cases/re3200_turbulence_profiles_256_extend.toml"
cp "$config" "$run_dir/case.toml"

cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DLBM_CONFIG_FILE="$run_dir/case.toml" \
    -DLBM_NUM_GPUS_OVERRIDE=2 \
    -DLBM_RESTART_PATH_OVERRIDE="$checkpoint"
cmake --build "$build_dir" --target glbm -j"$(nproc)"
cp "$build_dir/glbm" "$run_dir/glbm"

(
    cd -- "$run_dir"
    ./glbm 2>&1 | tee -a extension_run.log
)

snapshot="$statistics/bmbc_blocks.bin"
if [[ ! -f "$snapshot" ]]; then
    echo "Extension completed, but final BMBC outputs were not found." >&2
    exit 1
fi

(
    cd -- "$run_dir"
    python3 "$repo_dir/post/process_statistics.py"
    MPLCONFIGDIR=/tmp/glbm-matplotlib python3 "$repo_dir/post/plot_statistics.py"
)

echo
echo "Extension completed. Send these summaries:"
echo "  $statistics/derived/data/experiment_summary.dat"
echo "  $statistics/derived/data/line_statistics.dat"
