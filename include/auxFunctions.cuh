#ifndef AUX_FUNCTIONS_CUH
#define AUX_FUNCTIONS_CUH

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>
#include "globalStructs.cuh"
#include "globalFunctions.cuh"
#include PROPERTIES
#include "model.h"

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
    const double reference_velocity = static_cast<double>(U_MAX);
    const double normalized_energy = 0.5 * velocity_squared_sum / (total_nodes * reference_velocity * reference_velocity);

    const std::filesystem::path path = output_directory / "total_kinetic_energy.dat";
    std::ofstream file(path, std::ios_base::app);

    file << step << ' ' << std::setprecision(std::numeric_limits<double>::max_digits10) << normalized_energy << '\n';
}

#endif
