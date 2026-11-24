cmake_minimum_required(VERSION 3.24)

# Detect the native architectures only when creating a build directory. The
# cached value avoids invoking nvidia-smi on every case.toml reconfiguration.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi REQUIRED)
    execute_process(
        COMMAND "${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader,nounits
        RESULT_VARIABLE NVIDIA_SMI_STATUS
        OUTPUT_VARIABLE GPU_COMPUTE_CAPABILITIES
        ERROR_VARIABLE NVIDIA_SMI_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT NVIDIA_SMI_STATUS EQUAL 0 OR GPU_COMPUTE_CAPABILITIES STREQUAL "")
        message(FATAL_ERROR "Could not detect the GPU architecture: ${NVIDIA_SMI_ERROR}")
    endif()

    string(REPLACE "\n" ";" GPU_COMPUTE_CAPABILITIES "${GPU_COMPUTE_CAPABILITIES}")
    set(DETECTED_CUDA_ARCHITECTURES)
    foreach(capability IN LISTS GPU_COMPUTE_CAPABILITIES)
        string(STRIP "${capability}" capability)
        string(REPLACE "." "" capability "${capability}")
        list(APPEND DETECTED_CUDA_ARCHITECTURES "${capability}-real")
    endforeach()
    list(REMOVE_DUPLICATES DETECTED_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures detected from the installed GPU(s)")
endif()

project(glbm LANGUAGES CXX CUDA)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(LBM_CONFIG_FILE "${CMAKE_SOURCE_DIR}/case.toml" CACHE FILEPATH
    "Simulation configuration file")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${LBM_CONFIG_FILE}")

if(NOT EXISTS "${LBM_CONFIG_FILE}")
    message(FATAL_ERROR "Configuration file not found: ${LBM_CONFIG_FILE}")
endif()

file(STRINGS "${LBM_CONFIG_FILE}" LBM_CONFIG)

function(read_config key variable)
    foreach(line IN LISTS LBM_CONFIG)
        string(REGEX REPLACE "[ \t]*#.*$" "" line "${line}")
        if(line MATCHES "^[ \t]*${key}[ \t]*=[ \t]*(.*)$")
            set(value "${CMAKE_MATCH_1}")
            string(STRIP "${value}" value)
            string(REGEX REPLACE "^\"(.*)\"$" "\\1" value "${value}")
            set(${variable} "${value}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Missing '${key}' in ${LBM_CONFIG_FILE}")
endfunction()

read_config(num_gpus NUM_GPUS)
read_config(n_size N_SIZE)
read_config(aspect_ratio SAR)
read_config(steps SIMULATION_STEPS)
read_config(print_interval PRINT_INTERVAL)
read_config(precision PRECISION)
read_config(reynolds_number RE_NUMBER)
read_config(u_max U_MAX)
read_config(stencil STENCIL)
read_config(reconstruction_order ORDER)
read_config(block_x BLOCK_X)
read_config(block_y BLOCK_Y)
read_config(block_z BLOCK_Z)
read_config(vti SAVE_VTI)
read_config(interval OUTPUT_INTERVAL)
read_config(binary SAVE_BINARY)
read_config(binary_interval BINARY_OUTPUT_INTERVAL)
read_config(kinetic_energy SAVE_KINETIC_ENERGY)
read_config(kinetic_energy_interval KINETIC_ENERGY_INTERVAL)
read_config(checkpoint_interval CHECKPOINT_INTERVAL)
read_config(checkpoint_path CHECKPOINT_PATH)
read_config(restart_path RESTART_PATH)

# Temporal statistics
read_config(enabled STATISTICS_ENABLED)
read_config(start_t_star STATISTICS_START_T_STAR)
read_config(end_t_star STATISTICS_END_T_STAR)
read_config(sample_interval STATISTICS_SAMPLE_INTERVAL)

# BMBC configuration
read_config(bmbc_enabled BMBC_ENABLED)
read_config(bmbc_sample_interval BMBC_SAMPLE_INTERVAL)
read_config(bmbc_turbulence_profiles BMBC_TURBULENCE_PROFILES)

if(BMBC_ENABLED)
    set(BMBC_ENABLED 1)
else()
    set(BMBC_ENABLED 0)
endif()
if(BMBC_TURBULENCE_PROFILES)
    set(BMBC_TURBULENCE_PROFILES 1)
else()
    set(BMBC_TURBULENCE_PROFILES 0)
endif()
read_config(bmbc_block_samples BMBC_BLOCK_SAMPLES)
read_config(bmbc_initial_capacity_blocks BMBC_INITIAL_CAPACITY_BLOCKS)
read_config(bmbc_report_interval_blocks BMBC_REPORT_INTERVAL_BLOCKS)
read_config(bmbc_export_interval_blocks BMBC_EXPORT_INTERVAL_BLOCKS)

foreach(variable BMBC_SAMPLE_INTERVAL BMBC_BLOCK_SAMPLES BMBC_INITIAL_CAPACITY_BLOCKS)
    if(NOT "${${variable}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${variable} must be a positive integer")
    endif()
endforeach()

foreach(variable BMBC_REPORT_INTERVAL_BLOCKS BMBC_EXPORT_INTERVAL_BLOCKS)
    if(NOT "${${variable}}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${variable} must be a non-negative integer")
    endif()
endforeach()


if(STATISTICS_ENABLED)
    set(STATISTICS_ENABLED 1)
else()
    set(STATISTICS_ENABLED 0)
endif()

if(NOT STATISTICS_SAMPLE_INTERVAL MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "statistics.sample_interval must be a positive integer")
endif()

foreach(variable
        NUM_GPUS SIMULATION_STEPS PRINT_INTERVAL
        BLOCK_X BLOCK_Y BLOCK_Z SAVE_VTI SAVE_BINARY SAVE_KINETIC_ENERGY
        KINETIC_ENERGY_INTERVAL CHECKPOINT_INTERVAL CHECKPOINT_PATH RESTART_PATH)
    if(DEFINED LBM_${variable}_OVERRIDE)
        set(${variable} "${LBM_${variable}_OVERRIDE}")
    endif()
endforeach()

if(SAVE_VTI)
    set(SAVE_VTI 1)
else()
    set(SAVE_VTI 0)
endif()

if(SAVE_BINARY)
    set(SAVE_BINARY 1)
else()
    set(SAVE_BINARY 0)
endif()

if(SAVE_KINETIC_ENERGY)
    set(SAVE_KINETIC_ENERGY 1)
else()
    set(SAVE_KINETIC_ENERGY 0)
endif()

if(NOT BINARY_OUTPUT_INTERVAL MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "binary_interval must be a positive integer")
endif()

if(NOT KINETIC_ENERGY_INTERVAL MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "kinetic_energy_interval must be a positive integer")
endif()

set(GLBM_CORE_SOURCES
    src/checkpoint.cpp
    src/lbmState.cu
    src/mlbm.cu
    src/multiGpu.cu
    src/nodeType.cu
    src/temporalAverages.cu
    src/bmbc.cu
    src/bmbcIO.cu
)

# These files include CUDA headers and device-side declarations.
set_source_files_properties(src/checkpoint.cpp src/output.cpp
    PROPERTIES LANGUAGE CUDA)

add_library(glbm_core OBJECT ${GLBM_CORE_SOURCES})
add_library(glbm_output OBJECT 
    src/output.cpp)
add_library(glbm_simulation OBJECT src/simulation.cu)
add_executable(glbm
    src/main.cu
    $<TARGET_OBJECTS:glbm_core>
    $<TARGET_OBJECTS:glbm_output>
    $<TARGET_OBJECTS:glbm_simulation>
)

set(GLBM_COMPILE_TARGETS glbm_core glbm_output glbm_simulation)
foreach(target IN LISTS GLBM_COMPILE_TARGETS)
    target_include_directories(${target} PRIVATE include)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--restrict>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
    target_compile_definitions(${target} PRIVATE
        NUM_GPUS=${NUM_GPUS}
        N_SIZE=${N_SIZE}
        SAR=${SAR}
        RE_NUMBER=${RE_NUMBER}
        U_MAX_VALUE=${U_MAX}
        STENCIL=${STENCIL}
        ORDER=${ORDER}
        BLOCK_NX_VALUE=${BLOCK_X}
        BLOCK_NY_VALUE=${BLOCK_Y}
        BLOCK_NZ_VALUE=${BLOCK_Z}
        BMBC_TURBULENCE_PROFILES_VALUE=${BMBC_TURBULENCE_PROFILES}
    )
    if(PRECISION STREQUAL "double")
        target_compile_definitions(${target} PRIVATE double_precision)
    endif()
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# Operational settings do not affect the numerical kernels. Keeping them on
# their actual consumers avoids rebuilding every CUDA translation unit when
# only output cadence, step count, or restart settings change.
target_compile_definitions(glbm_simulation PRIVATE
    SIMULATION_STEPS=${SIMULATION_STEPS}
    PRINT_INTERVAL_VALUE=${PRINT_INTERVAL}
    SAVE_VTI_OUTPUT_VALUE=${SAVE_VTI}
    OUTPUT_INTERVAL=${OUTPUT_INTERVAL}
    SAVE_BINARY_OUTPUT_VALUE=${SAVE_BINARY}
    BINARY_OUTPUT_INTERVAL=${BINARY_OUTPUT_INTERVAL}
    SAVE_KINETIC_ENERGY_OUTPUT_VALUE=${SAVE_KINETIC_ENERGY}
    KINETIC_ENERGY_OUTPUT_INTERVAL=${KINETIC_ENERGY_INTERVAL}
    CHECKPOINT_INTERVAL_VALUE=${CHECKPOINT_INTERVAL}
    STATISTICS_ENABLED_VALUE=${STATISTICS_ENABLED}
    STATISTICS_START_T_STAR_VALUE=${STATISTICS_START_T_STAR}
    STATISTICS_END_T_STAR_VALUE=${STATISTICS_END_T_STAR}
    STATISTICS_SAMPLE_INTERVAL_VALUE=${STATISTICS_SAMPLE_INTERVAL}
    CHECKPOINT_PATH_VALUE=\"${CHECKPOINT_PATH}\"
    RESTART_PATH_VALUE=\"${RESTART_PATH}\"
    BMBC_BLOCK_SAMPLES_VALUE=${BMBC_BLOCK_SAMPLES}
    BMBC_SAMPLE_INTERVAL_VALUE=${BMBC_SAMPLE_INTERVAL}
    BMBC_INITIAL_CAPACITY_BLOCKS_VALUE=${BMBC_INITIAL_CAPACITY_BLOCKS}
    BMBC_REPORT_INTERVAL_BLOCKS_VALUE=${BMBC_REPORT_INTERVAL_BLOCKS}
    BMBC_EXPORT_INTERVAL_BLOCKS_VALUE=${BMBC_EXPORT_INTERVAL_BLOCKS}
    BMBC_ENABLED_VALUE=${BMBC_ENABLED}
)

target_compile_definitions(glbm_output PRIVATE
    SIMULATION_STEPS=${SIMULATION_STEPS}
    SAVE_KINETIC_ENERGY_OUTPUT_VALUE=${SAVE_KINETIC_ENERGY}
    KINETIC_ENERGY_OUTPUT_INTERVAL=${KINETIC_ENERGY_INTERVAL}
    CHECKPOINT_PATH_VALUE=\"${CHECKPOINT_PATH}\"
)

set_target_properties(glbm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(CTest)
if(BUILD_TESTING)
    add_executable(cuda_topology_probe tests/cuda_topology_probe.cu)
    set_target_properties(cuda_topology_probe PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

    add_executable(bmbc_statistics_test
        tests/bmbc_statistics_test.cu
        $<TARGET_OBJECTS:glbm_core>
    )
    target_include_directories(bmbc_statistics_test PRIVATE include)
    target_compile_definitions(bmbc_statistics_test PRIVATE
        NUM_GPUS=${NUM_GPUS}
        N_SIZE=${N_SIZE}
        SAR=${SAR}
        RE_NUMBER=${RE_NUMBER}
        U_MAX_VALUE=${U_MAX}
        STENCIL=${STENCIL}
        ORDER=${ORDER}
        BLOCK_NX_VALUE=${BLOCK_X}
        BLOCK_NY_VALUE=${BLOCK_Y}
        BLOCK_NZ_VALUE=${BLOCK_Z}
        BMBC_TURBULENCE_PROFILES_VALUE=${BMBC_TURBULENCE_PROFILES}
    )
    if(PRECISION STREQUAL "double")
        target_compile_definitions(bmbc_statistics_test PRIVATE double_precision)
    endif()
    set_target_properties(bmbc_statistics_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME bmbc_statistics COMMAND bmbc_statistics_test)
endif()
