#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_file="${GLBM_CASE_FILE:-${project_dir}/case.toml}"
gpu_index="${GLBM_GPU_INDEX:-0}"
tuning_steps="${GLBM_AUTOTUNE_STEPS:-3000}"
repetitions="${GLBM_AUTOTUNE_REPETITIONS:-2}"
run_after_tuning=1
force_retune=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--tune-only] [--retune]" \
        "" \
        "Environment variables:" \
        "  GLBM_CASE_FILE             Case file (default: case.toml)" \
        "  GLBM_GPU_INDEX             Physical GPU index (default: 0)" \
        "  GLBM_AUTOTUNE_STEPS        Steps per measurement (default: 3000)" \
        "  GLBM_AUTOTUNE_REPETITIONS  Measurements per candidate (default: 2)" \
        "  GLBM_AUTOTUNE_CANDIDATES   Space-separated BXxBYxBZ list"
}

for argument in "$@"; do
    case "${argument}" in
        --tune-only) run_after_tuning=0 ;;
        --retune) force_retune=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "${argument}" >&2; usage >&2; exit 2 ;;
    esac
done

for command_name in cmake nvidia-smi nvcc sha256sum awk sed; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "${command_name}" >&2
        exit 1
    fi
done

if [[ ! -f "${case_file}" ]]; then
    printf 'Case file not found: %s\n' "${case_file}" >&2
    exit 1
fi

read_case_value() {
    local key="$1"
    awk -F= -v requested_key="${key}" '
        {
            line=$0
            sub(/[[:space:]]*#.*/, "", line)
            split(line, fields, "=")
            candidate=fields[1]
            gsub(/[[:space:]]/, "", candidate)
            if (candidate == requested_key) {
                value=substr(line, index(line, "=") + 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                gsub(/^"|"$/, "", value)
                print value
                exit
            }
        }
    ' "${case_file}"
}

nx="$(read_case_value n_size)"
sar="$(read_case_value aspect_ratio)"
num_gpus="$(read_case_value num_gpus)"
precision="$(read_case_value precision)"
nz=$((nx * sar / num_gpus))

if [[ "${precision}" == "double" ]]; then
    scalar_bytes=8
else
    scalar_bytes=4
fi

gpu_record="$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid,name,compute_cap,driver_version --format=csv,noheader,nounits)"
cuda_record="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
source_record="$({
    sha256sum "${project_dir}/CMakeLists.txt"
    find "${project_dir}/include" "${project_dir}/src" -type f \( -name '*.h' -o -name '*.cuh' -o -name '*.cu' -o -name '*.cpp' \) -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | awk '{print $1}')"
case_record="$(sha256sum "${case_file}" | awk '{print $1}')"
cache_key="$(printf '%s\n%s\n%s\n%s\n' "${gpu_record}" "${cuda_record}" "${source_record}" "${case_record}" | sha256sum | awk '{print substr($1,1,16)}')"
cache_dir="${project_dir}/.autotune/${cache_key}"
result_file="${cache_dir}/result.env"
build_root="${project_dir}/build-autotune/${cache_key}"
mkdir -p "${cache_dir}" "${build_root}"

printf 'GPU: %s\n' "${gpu_record}"
printf 'Case: %s (domain %sx%sx%s, %s)\n' "${case_file}" "${nx}" "${nx}" "${nz}" "${precision}"

if [[ -f "${result_file}" && ${force_retune} -eq 0 ]]; then
    # The file is generated locally below and contains integer assignments only.
    source "${result_file}"
    printf 'Cached block: %sx%sx%s (%s MLUPS)\n' "${BEST_BLOCK_X}" "${BEST_BLOCK_Y}" "${BEST_BLOCK_Z}" "${BEST_MLUPS}"
else
    candidates_string="${GLBM_AUTOTUNE_CANDIDATES:-32x2x2 32x4x1 32x2x4 32x4x2 32x8x1 64x1x2 64x2x1 64x2x2 32x4x4 32x8x2}"
    read -r -a candidates <<< "${candidates_string}"
    best_mlups=0
    best_block=""

    printf 'Autotuning %s candidate(s), %s repetition(s) of %s steps...\n' "${#candidates[@]}" "${repetitions}" "${tuning_steps}"
    for candidate in "${candidates[@]}"; do
        if [[ ! "${candidate}" =~ ^([0-9]+)x([0-9]+)x([0-9]+)$ ]]; then
            printf 'Skipping malformed candidate: %s\n' "${candidate}" >&2
            continue
        fi
        bx="${BASH_REMATCH[1]}"
        by="${BASH_REMATCH[2]}"
        bz="${BASH_REMATCH[3]}"
        threads=$((bx * by * bz))
        shared_bytes=$(((bx + 2) * (by + 2) * (bz + 2) * 10 * scalar_bytes))

        # 48 KiB is the portable static shared-memory limit for these kernels.
        if ((threads > 1024 || nx % bx != 0 || nx % by != 0 || nz % bz != 0 || shared_bytes > 49152)); then
            printf 'Skipping %s (threads=%s, shared=%s bytes, or incompatible domain)\n' "${candidate}" "${threads}" "${shared_bytes}"
            continue
        fi

        candidate_build="${build_root}/bench-${candidate}"
        cmake -S "${project_dir}" -B "${candidate_build}" -DCMAKE_BUILD_TYPE=Release \
            -DLBM_CONFIG_FILE="${case_file}" \
            -DLBM_BLOCK_X_OVERRIDE="${bx}" \
            -DLBM_BLOCK_Y_OVERRIDE="${by}" \
            -DLBM_BLOCK_Z_OVERRIDE="${bz}" \
            -DLBM_SIMULATION_STEPS_OVERRIDE="${tuning_steps}" \
            -DLBM_PRINT_INTERVAL_OVERRIDE="${tuning_steps}" \
            -DLBM_SAVE_VTI_OVERRIDE=false >/dev/null
        cmake --build "${candidate_build}" --parallel >/dev/null

        total_mlups=0
        valid_runs=0
        for ((run=1; run<=repetitions; ++run)); do
            if ! output="$(CUDA_VISIBLE_DEVICES="${gpu_index}" "${candidate_build}/glbm" 2>&1)"; then
                printf 'Candidate %s failed during run %s.\n%s\n' \
                    "${candidate}" "${run}" "${output}" >&2
                exit 1
            fi
            mlups="$(awk '
                /^Step / {
                    for (i=1; i<=NF; ++i)
                        if ($i == "MLUPS") value=$(i+1)
                }
                /^Completed / {
                    for (i=1; i<=NF; ++i)
                        if ($i == "solver") value=$(i+1)
                }
                END {if (value != "") print value}
            ' <<< "${output}")"
            if [[ ! "${mlups}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                printf 'Candidate %s did not report MLUPS.\n%s\n' "${candidate}" "${output}" >&2
                exit 1
            fi
            printf '  %s run %s: %s MLUPS\n' "${candidate}" "${run}" "${mlups}"
            total_mlups="$(awk -v a="${total_mlups}" -v b="${mlups}" 'BEGIN {printf "%.9f", a+b}')"
            valid_runs=$((valid_runs + 1))
        done
        average_mlups="$(awk -v total="${total_mlups}" -v count="${valid_runs}" 'BEGIN {printf "%.3f", total/count}')"
        printf '  %s average: %s MLUPS\n' "${candidate}" "${average_mlups}"
        if awk -v candidate_value="${average_mlups}" -v best_value="${best_mlups}" 'BEGIN {exit !(candidate_value > best_value)}'; then
            best_mlups="${average_mlups}"
            best_block="${candidate}"
        fi
    done

    if [[ -z "${best_block}" ]]; then
        printf 'No valid block configuration was found.\n' >&2
        exit 1
    fi
    IFS=x read -r BEST_BLOCK_X BEST_BLOCK_Y BEST_BLOCK_Z <<< "${best_block}"
    BEST_MLUPS="${best_mlups}"
    printf 'BEST_BLOCK_X=%s\nBEST_BLOCK_Y=%s\nBEST_BLOCK_Z=%s\nBEST_MLUPS=%s\n' \
        "${BEST_BLOCK_X}" "${BEST_BLOCK_Y}" "${BEST_BLOCK_Z}" "${BEST_MLUPS}" > "${result_file}"
    printf 'Selected block: %sx%sx%s (%s MLUPS)\n' "${BEST_BLOCK_X}" "${BEST_BLOCK_Y}" "${BEST_BLOCK_Z}" "${BEST_MLUPS}"
fi

production_build="${build_root}/selected"
cmake -S "${project_dir}" -B "${production_build}" -DCMAKE_BUILD_TYPE=Release \
    -DLBM_CONFIG_FILE="${case_file}" \
    -DLBM_BLOCK_X_OVERRIDE="${BEST_BLOCK_X}" \
    -DLBM_BLOCK_Y_OVERRIDE="${BEST_BLOCK_Y}" \
    -DLBM_BLOCK_Z_OVERRIDE="${BEST_BLOCK_Z}" >/dev/null
cmake --build "${production_build}" --parallel >/dev/null
printf 'Tuned executable: %s\n' "${production_build}/glbm"

if ((run_after_tuning)); then
    exec env CUDA_VISIBLE_DEVICES="${gpu_index}" "${production_build}/glbm"
fi
