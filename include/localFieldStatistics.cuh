
#ifndef LOCAL_FIELD_STATISTICS_CUH
#define LOCAL_FIELD_STATISTICS_CUH
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>
#include "globalStructs.cuh"
#include "globalFunctions.cuh"
#include PROPERTIES

constexpr int NUM_LOCAL_FIELDS = 16;

enum LocalFieldIndex
{
    LOCAL_FIELD_UX = 0,
    LOCAL_FIELD_UY,
    LOCAL_FIELD_UZ,
    LOCAL_FIELD_SXX,
    LOCAL_FIELD_SXY,
    LOCAL_FIELD_SXZ,
    LOCAL_FIELD_SYY,
    LOCAL_FIELD_SYZ,
    LOCAL_FIELD_SZZ,
    LOCAL_FIELD_UX2,
    LOCAL_FIELD_UY2,
    LOCAL_FIELD_UZ2,
    LOCAL_FIELD_UXUY,
    LOCAL_FIELD_UXUZ,
    LOCAL_FIELD_UYUZ,
    LOCAL_FIELD_S2
};

__device__ inline void evaluateLocalFields(
    const LBMState &state, const size_t index, double *values)
{
    const double ux = static_cast<double>(state.device(M_UX_INDEX, index)) / static_cast<double>(F_M_I_SCALE);
    const double uy = static_cast<double>(state.device(M_UY_INDEX, index)) / static_cast<double>(F_M_I_SCALE);
    const double uz = static_cast<double>(state.device(M_UZ_INDEX, index)) / static_cast<double>(F_M_I_SCALE);

    const double mxx = static_cast<double>(state.device(M_MXX_INDEX, index)) / static_cast<double>(F_M_II_SCALE);
    const double mxy = static_cast<double>(state.device(M_MXY_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
    const double mxz = static_cast<double>(state.device(M_MXZ_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
    const double myy = static_cast<double>(state.device(M_MYY_INDEX, index)) / static_cast<double>(F_M_II_SCALE);
    const double myz = static_cast<double>(state.device(M_MYZ_INDEX, index)) / static_cast<double>(F_M_IJ_SCALE);
    const double mzz = static_cast<double>(state.device(M_MZZ_INDEX, index)) / static_cast<double>(F_M_II_SCALE);

    const double factor = 0.5 * static_cast<double>(as2) *
        static_cast<double>(OMEGA) / static_cast<double>(T_OMEGA);

    const double Sxx = factor * (ux * ux - mxx);
    const double Sxy = factor * (ux * uy - mxy);
    const double Sxz = factor * (ux * uz - mxz);
    const double Syy = factor * (uy * uy - myy);
    const double Syz = factor * (uy * uz - myz);
    const double Szz = factor * (uz * uz - mzz);

    values[LOCAL_FIELD_UX] = ux;
    values[LOCAL_FIELD_UY] = uy;
    values[LOCAL_FIELD_UZ] = uz;
    values[LOCAL_FIELD_SXX] = Sxx;
    values[LOCAL_FIELD_SXY] = Sxy;
    values[LOCAL_FIELD_SXZ] = Sxz;
    values[LOCAL_FIELD_SYY] = Syy;
    values[LOCAL_FIELD_SYZ] = Syz;
    values[LOCAL_FIELD_SZZ] = Szz;
    values[LOCAL_FIELD_UX2] = ux * ux;
    values[LOCAL_FIELD_UY2] = uy * uy;
    values[LOCAL_FIELD_UZ2] = uz * uz;
    values[LOCAL_FIELD_UXUY] = ux * uy;
    values[LOCAL_FIELD_UXUZ] = ux * uz;
    values[LOCAL_FIELD_UYUZ] = uy * uz;
    values[LOCAL_FIELD_S2] = Sxx * Sxx + Syy * Syy + Szz * Szz +
        2.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);
}

struct LocalFieldStatistics
{
    // Long recurrent averages stagnate in float once 1/n approaches machine
    // precision. Keep statistics in double independently of solver precision.
    double *d_means = nullptr;
    size_t samples = 0;
};

void copyLocalFieldStatisticsToHost(
    const LocalFieldStatistics &statistics,
    std::vector<double> &host_means);

void allocateLocalFieldStatistics(LocalFieldStatistics &statistics);

void accumulateLocalFieldStatistics(
    LocalFieldStatistics &statistics,
    const LBMState &state,
    dim3 grid,
    dim3 block,
    cudaStream_t stream);

void freeLocalFieldStatistics(LocalFieldStatistics &statistics);

void restoreLocalFieldStatisticsFromHost(
    LocalFieldStatistics &statistics,
    const std::vector<double> &host_means,
    size_t samples);

#endif
