#ifndef GLBM_BMBC_IO_CUH
#define GLBM_BMBC_IO_CUH

#include <cstddef>
#include <filesystem>

#include "bmbc.cuh"

std::filesystem::path writeBMBCSnapshot(
    const BMBCProfiles &monitor,
    std::size_t completed_step,
    std::size_t start_step,
    std::size_t sample_interval,
    const std::filesystem::path &statistics_directory);

void restoreBMBCSnapshot(
    BMBCProfiles &monitor,
    std::size_t expected_step,
    std::size_t expected_samples,
    std::size_t expected_start_step,
    std::size_t expected_sample_interval,
    const std::filesystem::path &statistics_directory);

#endif
