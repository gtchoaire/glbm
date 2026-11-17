
#include "../include/strainRate.cuh"
#include "../include/globalFunctions.cuh"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

bool writeSxyValidationProfile(const std::vector<LBMState *> &states, const int global_x, const int global_z, const size_t step, const std::filesystem::path &output_directory)
{
    if (states.size() != NUM_GPUS ||
        NX < 3 || NY < 3 ||
        global_x < 0 || global_x >= GLOBAL_NX ||
        global_z < 0 || global_z >= GLOBAL_NZ)
        return false;

    const int gpu = global_z / LOCAL_NZ;
    const int z = global_z % LOCAL_NZ;
    const LBMState &state = *states[gpu];

    auto index = [&](int x, int y)
    {
        return idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
    };

    auto ux = [&](int x, int y) -> dfloat
    {
        return state.host(M_UX_INDEX, index(x, y)) / F_M_I_SCALE;
    };

    auto uy = [&](int x, int y) -> dfloat
    {
        return state.host(M_UY_INDEX, index(x, y)) / F_M_I_SCALE;
    };

    std::ostringstream filename;
    filename << "sxy_x" << global_x
             << "_z" << global_z
             << "_step" << std::setw(8)
             << std::setfill('0') << step << ".csv";

    std::ofstream file(output_directory / filename.str());

    if (!file)
        return false;

    file << std::setprecision(std::numeric_limits<dfloat>::max_digits10);

    // file << "y,y_star,dux_dy,duy_dx,sxy_moment,sxy_fd\n";
    file << "y,y_star,dux_dy,duy_dx,"
            "sxy_moment,sxy_fd,sxy_explicit,sxy_difference\n";

    for (int y = 1; y < NY - 1; ++y)
    {
        const size_t idx = index(global_x, y);

        const dfloat sxy_moment = computeSxy(state.host(M_UX_INDEX, idx), state.host(M_UY_INDEX, idx), state.host(M_MXY_INDEX, idx));
        const dfloat sxy_explicit = computeSxyExplicit(state.host(M_UX_INDEX, idx), state.host(M_UY_INDEX, idx), state.host(M_MXY_INDEX, idx));
        const dfloat sxy_difference = sxy_explicit - sxy_moment;
        const dfloat dux_dy = (ux(global_x, y + 1) - ux(global_x, y - 1)) / static_cast<dfloat>(2);

        dfloat duy_dx = 0;

        if (global_x == NX - 1)
        {
            duy_dx = (static_cast<dfloat>(3) * uy(NX - 1, y) - static_cast<dfloat>(4) * uy(NX - 2, y) + uy(NX - 3, y)) / static_cast<dfloat>(2);
        }
        else if (global_x == 0)
        {
            duy_dx = (-static_cast<dfloat>(3) * uy(0, y) + static_cast<dfloat>(4) * uy(1, y) - uy(2, y)) / static_cast<dfloat>(2);
        }
        else
        {
            duy_dx = (uy(global_x + 1, y) - uy(global_x - 1, y)) / static_cast<dfloat>(2);
        }

        const dfloat sxy_fd = static_cast<dfloat>(0.5) * (dux_dy + duy_dx);
        const dfloat y_star = static_cast<dfloat>(y) / static_cast<dfloat>(GLOBAL_NY - 1);

        // file << y << ","
        //      << y_star << ","
        //      << dux_dy << ","
        //      << duy_dx << ","
        //      << sxy_moment << ","
        //      << sxy_fd << "\n";

        file << y << ","
             << y_star << ","
             << dux_dy << ","
             << duy_dx << ","
             << sxy_moment << ","
             << sxy_fd << ","
             << sxy_explicit << ","
             << sxy_difference << "\n";
    }

    return static_cast<bool>(file);
}
