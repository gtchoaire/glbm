#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build-re3200-turbulence-256-pilot"
config="$repo_dir/cases/re3200_turbulence_profiles_256_pilot.toml"
run_dir="$repo_dir/run-re3200-turbulence-256-pilot"
restart_path="${1:-}"

mkdir -p "$run_dir"
cp "$config" "$run_dir/case.toml"

cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DLBM_CONFIG_FILE="$run_dir/case.toml" \
    -DLBM_NUM_GPUS_OVERRIDE=2 \
    -DLBM_RESTART_PATH_OVERRIDE="$restart_path"
cmake --build "$build_dir" --target glbm -j"$(nproc)"
cp "$build_dir/glbm" "$run_dir/glbm"

(
    cd -- "$run_dir"
    ./glbm 2>&1 | tee -a pilot_run.log
)

statistics="$run_dir/statistics"
snapshot="$statistics/bmbc_blocks.bin"

if [[ ! -f "$snapshot" ]]; then
    echo "Simulation completed, but final BMBC outputs were not found." >&2
    exit 1
fi

(
    cd -- "$run_dir"
    python3 "$repo_dir/post/process_statistics.py"
    MPLCONFIGDIR=/tmp/glbm-matplotlib python3 "$repo_dir/post/plot_statistics.py"
)

echo
echo "Pilot completed. Run directory: $run_dir"
echo "Send only these two small files for review:"
echo "  $statistics/derived/data/experiment_summary.dat"
echo "  $statistics/derived/data/line_statistics.dat"
