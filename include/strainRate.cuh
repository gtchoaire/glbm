
#ifndef STRAIN_RATE_CUH
#define STRAIN_RATE_CUH

#include <cstddef>
#include <filesystem>
#include <vector>

#include "globalStructs.cuh"
#include PROPERTIES

// Sxy from scaled, post-collision second-order moments.
__host__ __device__ inline dfloat computeSxy(
    const dfloat ux_scaled,
    const dfloat uy_scaled,
    const dfloat mxy_scaled)
{
    const dfloat mxy_neq = mxy_scaled - ux_scaled * uy_scaled;
    const dfloat factor = -OMEGA / (static_cast<dfloat>(2) * cs2 * T_OMEGA * F_M_IJ_SCALE);

    return factor * mxy_neq;
}

bool writeSxyValidationProfile(const std::vector<LBMState *> &states, int global_x, int global_z, size_t step, const std::filesystem::path &output_directory);

__host__ __device__ inline dfloat computeSxyExplicit(const dfloat ux_scaled, const dfloat uy_scaled, const dfloat mxy_scaled)
{

    const dfloat ux = ux_scaled / F_M_I_SCALE;
    const dfloat uy = uy_scaled / F_M_I_SCALE;
    const dfloat mxy = mxy_scaled / F_M_IJ_SCALE;
    const dfloat sxy = static_cast<dfloat>(0.5) * as2 * OMEGA * (ux * uy - mxy);

    return sxy / T_OMEGA;
}

#endif
