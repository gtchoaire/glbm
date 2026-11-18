#ifndef GLBM_BMBC_IO_CUH
#define GLBM_BMBC_IO_CUH

#include <cstddef>
#include <filesystem>

#include "bmbc.cuh"

// Single-GPU BMBC sidecar accompanies a .glbm checkpoint.
// Example: checkpoint.glbm -> checkpoint.glbm.bmbc
// Call on the device owning the BMBC buffers, once the sampling stream is synchronized.
void writeBMBCSnapshot(
    const BMBCProfiles &monitor,
    std::size_t completed_step,
    const std::filesystem::path &glbm_checkpoint_file);

// Requires a monitor allocated with the same block_samples as the snapshot.
// Validates checkpoint step and total samples. Does not touch the solver state.
void restoreBMBCSnapshot(
    BMBCProfiles &monitor,
    std::size_t expected_step,
    std::size_t expected_samples,
    const std::filesystem::path &glbm_checkpoint_file);

// Runs the existing GPU BMBC statistics for powers-of-two groupings,
// then exports ALL points of ux(y) and uy(x) to a single CSV.
// Output: output_directory/bmbc_profiles_step_<step>.csv
void exportBMBCProfilesCSV(
    BMBCProfiles &monitor,
    std::size_t completed_step,
    const std::filesystem::path &output_directory);

#endif
