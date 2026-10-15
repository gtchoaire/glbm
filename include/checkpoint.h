#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "globalStructs.cuh"

void writeCheckpoint(
    const std::vector<LBMState *> &states,
    size_t completed_step,
    const std::string &path,
    const std::filesystem::path &output_directory);

size_t loadCheckpoint(
    const std::vector<std::unique_ptr<DeviceDomain>> &domains,
    const std::vector<LBMState *> &states,
    const std::string &path,
    std::filesystem::path &output_directory);

#endif
