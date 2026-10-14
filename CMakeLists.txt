cmake_minimum_required(VERSION 3.24)

# Detect every installed GPU and emit native machine code only.  The "-real"
# suffix avoids PTX JIT compatibility problems with older driver branches.
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

string(REPLACE "\n" ";" GPU_COMPUTE_CAPABILITIES
    "${GPU_COMPUTE_CAPABILITIES}")
set(DETECTED_CUDA_ARCHITECTURES)
foreach(capability IN LISTS GPU_COMPUTE_CAPABILITIES)
    string(STRIP "${capability}" capability)
    string(REPLACE "." "" capability "${capability}")
    list(APPEND DETECTED_CUDA_ARCHITECTURES "${capability}-real")
endforeach()
list(REMOVE_DUPLICATES DETECTED_CUDA_ARCHITECTURES)

# FORCE replaces stale architecture values in existing build directories.
set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures detected from the installed GPU(s)" FORCE)

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

foreach(variable
        NUM_GPUS SIMULATION_STEPS PRINT_INTERVAL
        BLOCK_X BLOCK_Y BLOCK_Z SAVE_VTI SAVE_BINARY SAVE_KINETIC_ENERGY
        KINETIC_ENERGY_INTERVAL CHECKPOINT_INTERVAL)
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

add_executable(glbm
    src/checkpoint.cpp
    src/main.cu
    src/lbmState.cu
    src/mlbm.cu
    src/multiGpu.cu
    src/nodeType.cu
    src/output.cpp
    src/simulation.cu
)

# These files include CUDA headers and device-side declarations.
set_source_files_properties(src/checkpoint.cpp src/output.cpp
    PROPERTIES LANGUAGE CUDA)

target_include_directories(glbm PRIVATE include)

target_compile_options(glbm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--restrict>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_definitions(glbm PRIVATE
    NUM_GPUS=${NUM_GPUS}
    N_SIZE=${N_SIZE}
    SAR=${SAR}
    RE_NUMBER=${RE_NUMBER}
    U_MAX_VALUE=${U_MAX}
    SIMULATION_STEPS=${SIMULATION_STEPS}
    PRINT_INTERVAL_VALUE=${PRINT_INTERVAL}
    STENCIL=${STENCIL}
    ORDER=${ORDER}
    BLOCK_NX_VALUE=${BLOCK_X}
    BLOCK_NY_VALUE=${BLOCK_Y}
    BLOCK_NZ_VALUE=${BLOCK_Z}
    SAVE_VTI_OUTPUT_VALUE=${SAVE_VTI}
    OUTPUT_INTERVAL=${OUTPUT_INTERVAL}
    SAVE_BINARY_OUTPUT_VALUE=${SAVE_BINARY}
    BINARY_OUTPUT_INTERVAL=${BINARY_OUTPUT_INTERVAL}
    SAVE_KINETIC_ENERGY_OUTPUT_VALUE=${SAVE_KINETIC_ENERGY}
    KINETIC_ENERGY_OUTPUT_INTERVAL=${KINETIC_ENERGY_INTERVAL}
    CHECKPOINT_INTERVAL_VALUE=${CHECKPOINT_INTERVAL}
    CHECKPOINT_PATH_VALUE=\"${CHECKPOINT_PATH}\"
    RESTART_PATH_VALUE=\"${RESTART_PATH}\"
)

if(PRECISION STREQUAL "double")
    target_compile_definitions(glbm PRIVATE double_precision)
endif()

set_target_properties(glbm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
