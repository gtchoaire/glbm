#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build-re3200-turbulence-256-pilot"
config="$repo_dir/cases/re3200_turbulence_profiles_256_pilot.toml"
restart_path="${1:-}"

cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DLBM_CONFIG_FILE="$config" \
    -DLBM_NUM_GPUS_OVERRIDE=2 \
    -DLBM_RESTART_PATH_OVERRIDE="$restart_path"
cmake --build "$build_dir" --target glbm -j"$(nproc)"

(
    cd -- "$build_dir"
    ./glbm 2>&1 | tee -a pilot_run.log
)

run_dir="$(find "$build_dir/output" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
csv="$(find "$run_dir" -maxdepth 1 -type f -name 'bmbc_profiles_step_*.csv' \
    -printf '%f\n' | sort -V | tail -n 1)"
snapshot="$run_dir/checkpoint_re3200_turbulence_256_pilot.glbm.bmbc"

if [[ -z "$csv" || ! -f "$snapshot" ]]; then
    echo "Simulation completed, but final BMBC outputs were not found." >&2
    exit 1
fi

MPLCONFIGDIR=/tmp/glbm-matplotlib python3 \
    "$repo_dir/post/analyze_turbulence_profiles.py" \
    "$run_dir/$csv" --snapshot "$snapshot"

echo
echo "Pilot completed. Run directory: $run_dir"
echo "Send only these two small files for review:"
echo "  $run_dir/${csv%.csv}_turbulence_analysis/experimental_comparison_summary.csv"
echo "  $run_dir/${csv%.csv}_turbulence_analysis/turbulence_bmbc_summary.csv"
