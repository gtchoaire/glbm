cmake_minimum_required(VERSION 3.24)

find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi REQUIRED)
execute_process(
    COMMAND "${NVIDIA_SMI_EXECUTABLE}"
            --query-gpu=compute_cap
            --format=csv,noheader,nounits
    RESULT_VARIABLE NVIDIA_SMI_RESULT
    OUTPUT_VARIABLE NVIDIA_COMPUTE_CAPABILITIES
    ERROR_VARIABLE NVIDIA_SMI_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NVIDIA_SMI_RESULT EQUAL 0 OR NVIDIA_COMPUTE_CAPABILITIES STREQUAL "")
    message(FATAL_ERROR
        "Could not detect the CUDA architecture with nvidia-smi: "
        "${NVIDIA_SMI_ERROR}")
endif()

string(REPLACE "\r" "" NVIDIA_COMPUTE_CAPABILITIES
       "${NVIDIA_COMPUTE_CAPABILITIES}")
string(REPLACE "\n" ";" NVIDIA_COMPUTE_CAPABILITIES
       "${NVIDIA_COMPUTE_CAPABILITIES}")

set(DETECTED_CUDA_ARCHITECTURES)
foreach(COMPUTE_CAPABILITY IN LISTS NVIDIA_COMPUTE_CAPABILITIES)
    string(STRIP "${COMPUTE_CAPABILITY}" COMPUTE_CAPABILITY)
    if(NOT COMPUTE_CAPABILITY MATCHES "^([0-9]+)\\.([0-9]+)$")
        message(FATAL_ERROR
            "Unexpected compute capability reported by nvidia-smi: "
            "'${COMPUTE_CAPABILITY}'")
    endif()
    list(APPEND DETECTED_CUDA_ARCHITECTURES
         "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
endforeach()
list(REMOVE_DUPLICATES DETECTED_CUDA_ARCHITECTURES)

# Replace stale values in an existing build cache with the architecture(s)
# detected from the local GPU(s).
set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures detected from the local GPU(s)" FORCE)

project(glbm LANGUAGES CXX CUDA)

message(STATUS "GLBM CUDA architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(LBM_CONFIG_FILE "${CMAKE_SOURCE_DIR}/case.toml")

if(NOT EXISTS "${LBM_CONFIG_FILE}")
    message(FATAL_ERROR "Simulation configuration not found: ${LBM_CONFIG_FILE}")
endif()

# Reconfigure automatically when the selected case file changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LBM_CONFIG_FILE}")
file(STRINGS "${LBM_CONFIG_FILE}" LBM_CONFIG_LINES)

function(read_toml_value key output_variable)
    unset(value)

    foreach(line IN LISTS LBM_CONFIG_LINES)
        # The current case format uses scalar values only. Sections are informative.
        string(REGEX REPLACE "[ \t]*#.*$" "" line_without_comment "${line}")

        if(line_without_comment MATCHES "^[ \t]*${key}[ \t]*=[ \t]*(.+)[ \t]*$")
            set(value "${CMAKE_MATCH_1}")
            string(STRIP "${value}" value)
            string(REGEX REPLACE "^\"(.*)\"$" "\\1" value "${value}")
            break()
        endif()
    endforeach()

    if(NOT DEFINED value)
        message(FATAL_ERROR "Missing '${key}' in ${LBM_CONFIG_FILE}")
    endif()

    set(${output_variable} "${value}" PARENT_SCOPE)
endfunction()

read_toml_value(num_gpus LBM_NUM_GPUS)
read_toml_value(n_size LBM_N_SIZE)
read_toml_value(aspect_ratio LBM_SAR)
read_toml_value(steps LBM_SIMULATION_STEPS)
read_toml_value(print_interval LBM_PRINT_INTERVAL)
read_toml_value(precision LBM_PRECISION)
read_toml_value(reynolds_number LBM_RE_NUMBER)
read_toml_value(u_max LBM_U_MAX)
read_toml_value(stencil LBM_STENCIL)
read_toml_value(reconstruction_order LBM_ORDER)
read_toml_value(vti LBM_SAVE_VTI)
read_toml_value(interval LBM_OUTPUT_INTERVAL)

foreach(parameter
        LBM_NUM_GPUS
        LBM_N_SIZE
        LBM_SAR
        LBM_SIMULATION_STEPS
        LBM_PRINT_INTERVAL
        LBM_ORDER
        LBM_OUTPUT_INTERVAL)
    if(NOT "${${parameter}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${parameter} must be a positive integer")
    endif()
endforeach()

if(NOT LBM_RE_NUMBER MATCHES "^[0-9]+([.][0-9]+)?$")
    message(FATAL_ERROR "reynolds_number must be a positive number")
endif()

if(NOT LBM_U_MAX MATCHES "^[0-9]+([.][0-9]+)?$")
    message(FATAL_ERROR "u_max must be a positive number")
endif()

if(NOT LBM_PRECISION STREQUAL "float" AND NOT LBM_PRECISION STREQUAL "double")
    message(FATAL_ERROR "precision must be either 'float' or 'double'")
endif()

if(NOT LBM_STENCIL MATCHES "^D[1-3]Q[0-9]+$")
    message(FATAL_ERROR "Invalid stencil name: ${LBM_STENCIL}")
endif()

if(NOT LBM_SAVE_VTI STREQUAL "true" AND NOT LBM_SAVE_VTI STREQUAL "false")
    message(FATAL_ERROR "vti must be either true or false")
endif()

if(LBM_SAVE_VTI STREQUAL "true")
    set(LBM_SAVE_VTI_VALUE 1)
else()
    set(LBM_SAVE_VTI_VALUE 0)
endif()

message(STATUS "GLBM case file: ${LBM_CONFIG_FILE}")
message(STATUS "GLBM domain: ${LBM_N_SIZE} x ${LBM_N_SIZE} x (${LBM_SAR} * ${LBM_N_SIZE})")
message(STATUS "GLBM GPUs: ${LBM_NUM_GPUS}")
message(STATUS "GLBM precision: ${LBM_PRECISION}")
message(STATUS "GLBM progress interval: ${LBM_PRINT_INTERVAL}")
message(STATUS "GLBM VTI output: ${LBM_SAVE_VTI}")

add_executable(glbm
    src/main.cu
    src/lbmState.cu
    src/mlbm.cu
    src/nodeType.cu
    src/output.cpp
    src/simulation.cu
)

set_source_files_properties(src/output.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(glbm PRIVATE include)

target_compile_options(glbm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--restrict>
)

target_compile_definitions(glbm PRIVATE
    NUM_GPUS=${LBM_NUM_GPUS}
    N_SIZE=${LBM_N_SIZE}
    SAR=${LBM_SAR}
    RE_NUMBER=${LBM_RE_NUMBER}
    U_MAX_VALUE=${LBM_U_MAX}
    SIMULATION_STEPS=${LBM_SIMULATION_STEPS}
    PRINT_INTERVAL_VALUE=${LBM_PRINT_INTERVAL}
    STENCIL=${LBM_STENCIL}
    ORDER=${LBM_ORDER}
    SAVE_VTI_OUTPUT_VALUE=${LBM_SAVE_VTI_VALUE}
    OUTPUT_INTERVAL=${LBM_OUTPUT_INTERVAL}
)

if(LBM_PRECISION STREQUAL "double")
    target_compile_definitions(glbm PRIVATE double_precision)
endif()

set_target_properties(glbm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
