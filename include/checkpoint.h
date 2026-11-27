#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "globalStructs.cuh"

std::filesystem::path writeCheckpoint(
    const std::vector<LBMState *> &states,
    size_t completed_step,
    const std::string &path);

size_t loadCheckpoint(
    const std::vector<std::unique_ptr<DeviceDomain>> &domains,
    const std::vector<LBMState *> &states,
    const std::string &path);

#endif
