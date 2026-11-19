#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "globalStructs.cuh"

// Statistics in field-major AoSoA layout on the host, one vector per GPU.
// Only physical nodes are serialized, so checkpoint files are block-size independent.
struct CheckpointStatistics
{
    bool present = false;
    size_t start_step = 0;
    size_t end_step = 0;
    size_t sample_interval = 1;
    size_t samples = 0;
    std::vector<std::vector<double>> means;
};

void writeCheckpoint(
    const std::vector<LBMState *> &states,
    size_t completed_step,
    const std::string &path,
    const std::filesystem::path &output_directory,
    const CheckpointStatistics *statistics = nullptr);

size_t loadCheckpoint(
    const std::vector<std::unique_ptr<DeviceDomain>> &domains,
    const std::vector<LBMState *> &states,
    const std::string &path,
    std::filesystem::path &output_directory,
    CheckpointStatistics *statistics = nullptr);

#endif
