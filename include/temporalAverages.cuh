
#ifndef TEMPORAL_AVERAGES_CUH
#define TEMPORAL_AVERAGES_CUH
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>
#include "globalStructs.cuh"

constexpr int NUM_AVERAGE_FIELDS = 10;

enum AverageField
{
    AVG_RHO = 0,
    AVG_UX,
    AVG_UY,
    AVG_UZ,
    AVG_SXX,
    AVG_SXY,
    AVG_SXZ,
    AVG_SYY,
    AVG_SYZ,
    AVG_SZZ
};

struct TemporalAverages
{
    dfloat *d_means = nullptr;
    size_t samples = 0;
};

void copyTemporalAveragesToHost(
    const TemporalAverages &averages,
    std::vector<dfloat> &host_means);

void allocateTemporalAverages(TemporalAverages &averages);

void accumulateTemporalAverages(
    TemporalAverages &averages,
    const LBMState &state,
    dim3 grid,
    dim3 block,
    cudaStream_t stream);

void copyTemporalAveragesProbe(
    const TemporalAverages &averages,
    int x, int y, int z,
    dfloat *values);

void freeTemporalAverages(TemporalAverages &averages);

void restoreTemporalAveragesFromHost(
    TemporalAverages &averages,
    const std::vector<dfloat> &host_means,
    size_t samples);

#endif
