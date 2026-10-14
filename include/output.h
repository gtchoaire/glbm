#ifndef OUTPUT_H
#define OUTPUT_H

#include <filesystem>
#include <vector>

#include "globalStructs.cuh"

// States must be ordered by increasing global Z subdomain.
bool createOutputDirectory(std::filesystem::path &output_directory);
bool writeSimulationInfo(const std::filesystem::path &output_directory);
bool writeVti(const std::vector<LBMState *> &states, int step,
              const std::filesystem::path &output_directory);
// Writes raw physical fields in global z-y-x order, without a file header.
bool writeBinaryFields(const std::vector<LBMState *> &states, int step,
                       const std::filesystem::path &output_directory);

#endif
