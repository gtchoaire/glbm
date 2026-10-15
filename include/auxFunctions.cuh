#ifndef AUX_FUNCTIONS_CUH
#define AUX_FUNCTIONS_CUH

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "globalStructs.cuh"
#include "globalFunctions.cuh"
#include PROPERTIES
#include "model.h"

inline void reconcileKineticEnergyOutput(const std::filesystem::path &output_directory, const size_t completed_step)
{
    const std::filesystem::path path = output_directory / "total_kinetic_energy.dat";
    if (!std::filesystem::exists(path))
        return;

    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Failed to open kinetic energy history: " + path.string());

    std::filesystem::path temporary = path;
    temporary += ".restart.tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output)
        throw std::runtime_error("Failed to create temporary kinetic energy history: " + temporary.string());

    std::unordered_set<size_t> retained_steps;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;

        std::istringstream parser(line);
        size_t step = 0;
        double energy = 0.0;
        if (!(parser >> step >> energy))
            throw std::runtime_error("Invalid kinetic energy history line: " + line);

        std::string extra_value;
        if (parser >> extra_value)
            throw std::runtime_error("Unexpected value in kinetic energy history line: " + line);

        if (step <= completed_step && retained_steps.insert(step).second)
            output << line << '\n';
    }
    if (!input.eof())
        throw std::runtime_error("Failed while reading kinetic energy history: " + path.string());

    output.close();
    if (!output)
        throw std::runtime_error("Failed while rewriting kinetic energy history: " + temporary.string());

    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error)
        throw std::runtime_error("Failed to publish reconciled kinetic energy history: " + error.message());
}

inline void totalKineticEnergy(const std::vector<LBMState *> &states, const size_t step, const std::filesystem::path &output_directory)
{
    double velocity_squared_sum = 0.0;
    for (const LBMState *state : states)
    {
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                {
                    const size_t index = idxScalarBlock(
                        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

                    const double ux = static_cast<double>(state->host(M_UX_INDEX, index) / F_M_I_SCALE);
                    const double uy = static_cast<double>(state->host(M_UY_INDEX, index) / F_M_I_SCALE);
                    const double uz = static_cast<double>(state->host(M_UZ_INDEX, index) / F_M_I_SCALE);

                    velocity_squared_sum += ux * ux + uy * uy + uz * uz;
                }
    }

    const double total_nodes = static_cast<double>(states.size()) * static_cast<double>(NUMBER_LBM_NODES);
    constexpr double reference_velocity = static_cast<double>(U_MAX);
    const double normalized_energy = 0.5 * velocity_squared_sum / (total_nodes * reference_velocity * reference_velocity);

    const std::filesystem::path path = output_directory / "total_kinetic_energy.dat";
    std::ofstream file(path, std::ios_base::app);

    file << step << ' ' << std::setprecision(std::numeric_limits<double>::max_digits10) << normalized_energy << '\n';
}

#endif
